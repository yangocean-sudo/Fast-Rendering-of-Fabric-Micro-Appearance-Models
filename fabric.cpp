#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/texture.h>

#include "principledhelpers.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Fabric final : public BSDF<Float, Spectrum>
{
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)

    Fabric(const Properties &props) : Base(props)
    {
        m_base_color = props.texture<Texture>("base_color", 0.5f);
        m_roughness = props.texture<Texture>("roughness", 0.5f);
        m_has_anisotropic = get_flag("anisotropic", props);
        m_anisotropic = props.texture<Texture>("anisotropic", 0.0f);
        m_has_spec_trans = get_flag("spec_trans", props);
        m_spec_trans = props.texture<Texture>("spec_trans", 0.0f);
        m_has_sheen = get_flag("sheen", props);
        m_sheen = props.texture<Texture>("sheen", 0.0f);
        m_has_sheen_tint = get_flag("sheen_tint", props);
        m_sheen_tint = props.texture<Texture>("sheen_tint", 0.0f);
        m_has_flatness = get_flag("flatness", props);
        m_flatness = props.texture<Texture>("flatness", 0.0f);
        m_has_spec_tint = get_flag("spec_tint", props);
        m_spec_tint = props.texture<Texture>("spec_tint", 0.0f);
        m_eta_thin = props.texture<Texture>("eta", 1.5f);
        m_has_diff_trans = get_flag("diff_trans", props);
        m_diff_trans = props.texture<Texture>("diff_trans", 0.0f);
        m_fiber_density = props.texture<Texture>("fiber_density", 1.0f);
        m_fiber_radius = props.texture<Texture>("fiber_radius", 0.5f);
        m_fiber_variation = props.texture<Texture>("fiber_variation", 0.3f);
        m_spec_refl_srate =
            props.get("specular_reflectance_sampling_rate", 1.0f);
        m_spec_trans_srate =
            props.get("specular_transmittance_sampling_rate", 1.0f);
        m_diff_trans_srate =
            props.get("diffuse_transmittance_sampling_rate", 1.0f);
        m_diff_refl_srate =
            props.get("diffuse_reflectance_sampling_rate", 1.0f);

        initialize_lobes();
    }

    void initialize_lobes()
    {
        // Diffuse reflection lobe
        m_components.push_back(BSDFFlags::DiffuseReflection |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);
        // Specular diffuse lobe
        m_components.push_back(BSDFFlags::DiffuseTransmission |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);

        // Specular transmission lobe
        if (m_has_spec_trans)
        {
            uint32_t f = BSDFFlags::GlossyTransmission | BSDFFlags::FrontSide |
                         BSDFFlags::BackSide;
            if (m_has_anisotropic)
                f = f | BSDFFlags::Anisotropic;
            m_components.push_back(f);
        }

        // Main specular reflection lobe
        uint32_t f = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide |
                     BSDFFlags::BackSide;
        if (m_has_anisotropic)
            f = f | BSDFFlags::Anisotropic;
        m_components.push_back(f);

        for (auto c : m_components)
            m_flags |= c;
        dr::set_attr(this, "flags", m_flags);
    }

    Spectrum compute_visibility(const Vector3f &omega, const float &k,
                                float light_frequency)
    {
        float high_frequency_threshold = 0.0f;
        if (light_frequency > high_frequency_threshold)
        {
            // High frequency light
            return compute_ssdf(omega, k);
        }
        else
        {
            // Low frequency light
            return compute_sg(omega, k);
        }
    }

    // Function to compute indirect illumination contribution using IIRTF
    Spectrum computeIIRTFIndirectContribution(const SurfaceInteraction3f &si,
                                              const Vector3f &wo) const
    {
        static_cast<void>(si);
        static_cast<void>(wo);
        Spectrum iirtfContribution = Spectrum(0.0f);

        return iirtfContribution;
    }

    Eigen::MatrixXf pcaCompressSSDF(const Eigen::MatrixXf &ssdf,
                                    int numComponents)
    {
        // Convert SSDF to floating point for PCA
        Eigen::MatrixXf ssdfFloat = ssdf.cast<float>();

        // Normalize by subtracting the mean
        Eigen::VectorXf mean = ssdfFloat.colwise().mean();
        Eigen::MatrixXf ssdfCentered = ssdfFloat.rowwise() - mean.transpose();

        // Compute the covariance matrix
        Eigen::MatrixXf covarianceMatrix =
            (ssdfCentered.adjoint() * ssdfCentered) /
            double(ssdfFloat.rows() - 1);

        // Compute eigenvalues and eigenvectors of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigenSolver(
            covarianceMatrix);
        if (eigenSolver.info() != Eigen::Success)
            abort(); // Handle failure

        // Sort eigenvectors by eigenvalues (in descending order)
        Eigen::MatrixXf eigenvectors = eigenSolver.eigenvectors();
        // This step assumes eigenvectors are already sorted by eigenvalues in
        // descending order

        // Pick the top 'numComponents' eigenvectors
        Eigen::MatrixXf principalComponents =
            eigenvectors.rightCols(numComponents);

        // Transform the original centered data
        Eigen::MatrixXf transformedData = ssdfCentered * principalComponents;

        // Return the transformed data in integer form
        return transformedData;
    }

    int
    computeDistanceToNearestOcclusion(int x, int y,
                                      const Eigen::MatrixXi &visibilityImage)
    {
        // distance computation
        int shortestDistance = INT_MAX; // Use a large initial value

        // Scan through the entire image to find the nearest occluded pixel
        for (int i = 0; i < visibilityImage.rows(); ++i)
        {
            for (int j = 0; j < visibilityImage.cols(); ++j)
            {
                if (visibilityImage(i, j) == 0)
                { // Found an occluded pixel
                    int distance =
                        std::sqrt(std::pow(x - i, 2) + std::pow(y - j, 2));
                    if (distance < shortestDistance)
                    {
                        shortestDistance = distance;
                    }
                }
            }
        }

        return shortestDistance;
    }

    int computeOccludedDistance(int x, int y,
                                const Eigen::MatrixXi &visibilityImage)
    {
        x = x + 1;
        y = y + 1;
        static_cast<void>(visibilityImage);

        return -1;
    }

    Eigen::MatrixXi
    computeSSDFFromVisibilityImage(const Eigen::MatrixXi &visibilityImage)
    {
        int rows = visibilityImage.rows();
        int cols = visibilityImage.cols();
        Eigen::MatrixXi ssdf =
            Eigen::MatrixXi::Zero(rows, cols); // Initialize SSDF matrix

        // Loop through each pixel in the visibility image
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                if (visibilityImage(i, j) == 1)
                {
                    // For visible pixels, compute the distance to the nearest
                    // occluded pixel
                    int distance = computeDistanceToNearestOcclusion(
                        i, j, visibilityImage);
                    ssdf(i, j) = distance;
                }
                else
                {
                    // For occluded pixels, you might want to use a different
                    // method to compute the SSDF value, possibly involving
                    // negative distances or another marker
                    ssdf(i, j) = computeOccludedDistance(i, j, visibilityImage);
                }
            }
        }

        return ssdf;
    }

    Spectrum convertToSpectrum(const Eigen::MatrixXf &compressedSSDF)
    {
        double sum = compressedSSDF.sum();
        double maxPossibleValue = compressedSSDF.size() * 255.0;
        double normalizedSum = sum / maxPossibleValue;

        // Assuming Spectrum can be constructed from a single scalar value
        Spectrum spectrumValue = normalizedSum;

        return spectrumValue;
    }

    Spectrum compute_ssdf(const Vector3f &omega, const float &k)
    {
        // Locate the nearest copy of the exemplar block
        static_cast<void>(omega);
        static_cast<void>(k);
        // Render a 128x128 binary visibility image
        Eigen::MatrixXi visibility_image(128, 128);
        // perform ray casting and determine
        for (int theta_idx = 0; theta_idx < 128; ++theta_idx)
        {
            for (int phi_idx = 0; phi_idx < 128; ++phi_idx)
            {
                // Compute theta and phi for the current pixel
                // Cast a ray in this direction and determine visibility
                bool visible = true;
                visibility_image(theta_idx, phi_idx) = visible ? 1 : 0;
            }
        }

        // Compute the SSDF from the visibility image
        Eigen::MatrixXi ssdf = computeSSDFFromVisibilityImage(visibility_image);

        // Compress the SSDF using PCA
        Eigen::MatrixXf compressed_ssdf =
            pcaCompressSSDF(ssdf.cast<float>(), 48);

        // Convert the compressed SSDF into a format usable by your rendering
        // system This could involve mapping the PCA components to a spectrum,
        Spectrum ssdf_spectrum = convertToSpectrum(compressed_ssdf);

        return ssdf_spectrum;
    }

    Spectrum compute_sg(const Vector3f &omega, const float &k)
    {
        // SG axis
        Vector3f sg_axis = Vector3f(0.0f, 1.0f, 0.0f);
        static_cast<void>(k);
        // SG sharpness
        float sg_sharpness = 100.0f;

        // Compute the SG contribution using the formula G(omega; xi, lambda) =
        // exp(lambda(omega . xi - 1))
        auto dot_product = dot(omega, sg_axis);
        Spectrum sg_contribution = exp(sg_sharpness * (dot_product - 1.0f));

        return sg_contribution;
    }

    // Chapter 5.1 Main rendering function for shell-mapped fabric volume
    struct FiberSegment
    {
        Point3f p0; // Start point of the fiber segment
        Point3f p1; // End point of the fiber segment

        // Constructor for ease of use
        FiberSegment(const Point3f &start, const Point3f &end)
            : p0(start), p1(end) {}
    };

    bool intersectRayWithFiber(const Ray3f &ray, const FiberSegment &fiber,
                               Float &t)
    {
        // Define variables for calculations
        auto p = fiber.p0 - ray.o;
        auto d = fiber.p1 - fiber.p0;
        auto r = ray.d;

        // Compute coefficients for quadratic equation
        auto a = dot(d, d);
        auto b = dot(d, r);
        auto c = dot(p, p) - dot(p, d) * dot(p, d) / a;
        auto e = dot(p, r);
        auto f = dot(r, r);
        auto D = b * b - a * f;

        // Initialize intersection distance to infinity
        t = dr::Infinity<Float>;

        // Check if lines are almost parallel
        auto parallel_mask = D < std::numeric_limits<Float>::epsilon();

        Float tc;
        if (dr::none(parallel_mask))
        {
            // Not parallel, compute the intersection point
            tc = (b * e - c * f) / D;
        }
        else
        {
            // Lines are parallel, no intersection
            return false;
        }

        // Compute the intersection point on the ray
        auto point_on_ray = ray.o + tc * ray.d;

        // Compute the projection of the point on the segment line to find
        // its parameterized position
        auto t_on_segment = dot(point_on_ray - fiber.p0, d) / dot(d, d);

        // Check if the intersection point is within the segment bounds [0,
        // 1]
        auto within_segment_bounds = t_on_segment >= 0 && t_on_segment <= 1;

        // Update the intersection distance `t` if within bounds
        if (dr::any(within_segment_bounds))
        {
            t = tc;
            return true;
        }
        return false;
    }

    // chapter 5.2.1
    bool computeGlobalVisibility(const SurfaceInteraction3f &si,
                                 const Vector3f &lightDir) const
    {
        float ShadowEpsilon = 1e-4f;
        auto wavelengths = si.wavelengths;
        Ray3f shadowRay(si.p + si.n * ShadowEpsilon, lightDir, ShadowEpsilon,
                        dr::Infinity<Float>, wavelengths);
        bool isOccluded = false;
        return !isOccluded;
    }

    struct Segment
    {
        Point3f p0, p1;       // Endpoints of the segment
        Vector3f orientation; // Direction or orientation of the segment
        float opacity;        // Opacity of the segment

        // Constructor for ease of use
        Segment(const Point3f &start, const Point3f &end, const Vector3f &dir,
                float opac)
            : p0(start), p1(end), orientation(dir), opacity(opac) {}
    };

    Float computeLocalVisibility(const SurfaceInteraction3f &si,
                                 const Vector3f &incomingLightDir) const
    {
        // Initial visibility is set to fully visible.
        Float visibility = 1.0f;

        // Example loop over a hypothetical list of fiber segments influencing
        // the point `si.p`
        std::vector<Segment> microStructureSegments;
        for (const auto &segment : microStructureSegments)
        {
            // Compute the angle between the incoming light direction and the
            // fiber segment's orientation
            Float angle = dot(normalize(incomingLightDir), segment.orientation);
            Float zeroValue = 0.0f;
            Float occlusionFactor = angle - zeroValue;

            // Adjust visibility based on the occlusion factor and the segment's
            // opacity.
            visibility *= (1.0f - segment.opacity * occlusionFactor);
        }

        return visibility;
    }

    Spectrum evaluateBCSDF(const SurfaceInteraction3f &si,
                           const Vector3f &incomingLightDir,
                           const Vector3f &lightDir) const
    {
        float diffuseReflectivity =
            0.8f; // Fraction of light diffusely reflected
        float specularReflectivity =
            0.2f;               // Fraction of light specularly reflected
        float roughness = 0.5f; // Roughness of the fabric surface

        // Normalize light directions
        Vector3f nIncomingLightDir = normalize(incomingLightDir);
        Vector3f nLightDir = normalize(lightDir);
        Vector3f nSurfaceNormal = normalize(si.n);

        // Diffuse reflection component
        auto cosThetaI = dot(nIncomingLightDir, nSurfaceNormal);
        auto cosThetaO = dot(nLightDir, nSurfaceNormal);
        Spectrum diffuseComponent =
            cosThetaI * cosThetaO * diffuseReflectivity * dr::InvPi<Float>;

        // Specular reflection component8
        Vector3f halfVector = normalize(nIncomingLightDir + nLightDir);
        auto specAngle = dot(halfVector, nSurfaceNormal);
        Spectrum specularComponent =
            pow(specAngle, roughness * 100) * specularReflectivity;

        // Combine diffuse and specular components
        Spectrum result = diffuseComponent + specularComponent;

        return result;
    }

    Vector3f sampleHemisphere(int sampleIdx, int numSamples) const
    {
        return Vector3f(0, 1, 0);
    }

    Spectrum computeSingleScattering(const SurfaceInteraction3f &si,
                                     const Vector3f &lightDir) const
    {
        Spectrum outgoingRadiance = 0.0f;

        // Compute global visibility
        bool globalVisibility = computeGlobalVisibility(si, lightDir);

        if (!globalVisibility)
        {
            return outgoingRadiance; // Early exit if the light is occluded
                                     // globally
        }

        // Iterate over a hemisphere to sample incoming light directions
        // For simplicity, this code uses a fixed number of samples
        int numSamples =
            16; // Example: Number of samples for hemisphere integration
        for (int i = 0; i < numSamples; ++i)
        {
            Vector3f incomingLightDir = sampleHemisphere(i, numSamples);

            // Compute local visibility for this direction
            Float localVisibility =
                computeLocalVisibility(si, incomingLightDir);

            // Evaluate the BCSDF for this direction
            Spectrum bcsdfValue = evaluateBCSDF(si, incomingLightDir, lightDir);

            // Accumulate contribution
            outgoingRadiance += localVisibility * bcsdfValue;
        }

        // Normalize by the number of samples
        outgoingRadiance /= numSamples;

        return outgoingRadiance;
    }

    // chapter 5.2.2
    struct SGLight
    {
        Vector3f direction; // Light direction
        Spectrum intensity; // Light intensity
        float sharpness;    // Control the spread of the light
    };

    Spectrum evaluateBCSDF(const SurfaceInteraction3f &si,
                           const Vector3f &incomingLightDir,
                           const SGLight &sgLight) const
    {
        // This is a placeholder implementation. You would replace this with
        // your actual BCSDF evaluation logic that accounts for SG light
        // characteristics.
        Spectrum result = Spectrum(0.5f); // Placeholder result
        return result;
    }

    Vector3f sampleHemisphereUniformly(int sampleIdx, int numSamples) const
    {
        // Convert linear sample index to 2D
        float u1 = (sampleIdx + 0.5f) / numSamples;
        float u2 = sampleIdx + 1.5f;
        // Convert uniform sample [0,1]^2 to spherical coordinates
        float z = u1; // Cosine of the angle with the up vector
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        float phi = 2 * 3.14159265358979323846 *
                    u2; // Use Pi constant with higher precision
        float x = r * std::cos(phi);
        float y = r * std::sin(phi);

        return Vector3f(x, y, z);
    }

    Spectrum computeSingleScatteringSG(const SurfaceInteraction3f &si,
                                       const SGLight &sgLight) const
    {
        Spectrum totalScattering = Spectrum(0.0f);

        int numSamples = 16; // Number of samples for the integration
        for (int i = 0; i < numSamples; ++i)
        {
            Vector3f sampledDir = sampleHemisphereUniformly(i, numSamples);
            // Evaluate the BCSDF for the sampled direction and the SG light
            Spectrum scattering = evaluateBCSDF(si, sampledDir, sgLight);
            // Accumulate the scattering contribution
            totalScattering += scattering;
        }
        // Normalize the accumulated scattering by the number of samples
        totalScattering /= numSamples;
        return totalScattering;
    }

    // Chapter 5.3
    struct SVF
    {
        std::vector<std::vector<float>> segmentVisibilityTextures;

        auto directionToParameter(const Vector3f &direction) const
        {
            // Normalize the direction vector to ensure it's a unit vector
            Vector3f normalizedDirection = normalize(direction);

            // Compute the azimuthal angle (phi) of the direction vector in
            // spherical coordinates
            auto phi = normalizedDirection.y() + normalizedDirection.x();

            // Normalize phi to the range [0, 1]
            auto normalizedPhi = (phi + M_PI) / (2.0 * M_PI);

            return normalizedPhi;
        }

        auto sampleVisibility(size_t segmentIndex,
                              const Vector3f &direction) const
        {
            auto parameter = directionToParameter(direction);
            return parameter;
        }
    };

    struct IIRTF
    {
        std::vector<Spectrum> grid;
        size_t resolution[3];
        IIRTF(size_t resX, size_t resY, size_t resZ)
            : resolution{resX, resY, resZ}
        {
            grid.resize(resX * resY * resZ);
        }
        // Function to sample the IIRTF at a given position within the fabric
        // volume.
        // `position`: position within the fabric, expected to be normalized [0,
        // 1] in each dimension. Returns a Spectrum indicating the indirect
        // illumination contribution.
        Spectrum sampleIIRTF(const Vector3f &position) const
        {
            // Convert position to grid coordinates.
            auto x = (position.x() * resolution[0]) + resolution[0] - 1;
            auto y = (position.y() * resolution[1]) + resolution[1] - 1;
            auto z = (position.z() * resolution[2]) + resolution[2] - 1;

            // Access the grid cell corresponding to the position.
            return grid[resolution[1]];
        }
    };

    void traverse(TraversalCallback *callback) override
    {
        callback->put_object("eta", m_eta_thin.get(),
                             ParamFlags::Differentiable |
                                 ParamFlags::Discontinuous);
        callback->put_object("roughness", m_roughness.get(),
                             ParamFlags::Differentiable |
                                 ParamFlags::Discontinuous);
        callback->put_object("diff_trans", m_diff_trans.get(),
                             +ParamFlags::Differentiable);
        callback->put_parameter("specular_reflectance_sampling_rate",
                                m_spec_refl_srate,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("diffuse_reflectance_sampling_rate",
                                m_diff_refl_srate,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("diffuse_transmittance_sampling_rate",
                                m_diff_trans_srate,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("specular_transmittance_sampling_rate",
                                m_spec_trans_srate,
                                +ParamFlags::NonDifferentiable);
        callback->put_object("base_color", m_base_color.get(),
                             +ParamFlags::Differentiable);
        callback->put_object("anisotropic", m_anisotropic.get(),
                             +ParamFlags::Differentiable);
        callback->put_object("spec_tint", m_spec_tint.get(),
                             +ParamFlags::Differentiable);
        callback->put_object("sheen", m_sheen.get(),
                             +ParamFlags::Differentiable);
        callback->put_object("sheen_tint", m_sheen_tint.get(),
                             +ParamFlags::Differentiable);
        callback->put_object("spec_trans", m_spec_trans.get(),
                             +ParamFlags::Differentiable);
        callback->put_object("flatness", m_flatness.get(),
                             +ParamFlags::Differentiable);
    }

    void
    parameters_changed(const std::vector<std::string> &keys = {}) override
    {
        // In case the parameters are changed from zero to something else
        // boolean flags need to be changed also.
        if (string::contains(keys, "spec_trans"))
            m_has_spec_trans = true;
        if (string::contains(keys, "diff_trans"))
            m_has_diff_trans = true;
        if (string::contains(keys, "sheen"))
            m_has_sheen = true;
        if (string::contains(keys, "sheen_tint"))
            m_has_sheen_tint = true;
        if (string::contains(keys, "anisotropic"))
            m_has_anisotropic = true;
        if (string::contains(keys, "flatness"))
            m_has_flatness = true;
        if (string::contains(keys, "spec_tint"))
            m_has_spec_tint = true;

        initialize_lobes();
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override
    {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        // Ignoring perfectly grazing incoming rays
        active &= dr::neq(cos_theta_i, 0.0f);

        if (unlikely(dr::none_or<false>(active)))
            return {bs, 0.0f};

        // Store the weights.
        Float anisotropic =
                  m_has_anisotropic ? m_anisotropic->eval_1(si, active) : 0.0f,
              roughness = m_roughness->eval_1(si, active),
              spec_trans =
                  m_has_spec_trans ? m_spec_trans->eval_1(si, active) : 0.0f;
        /* Diffuse transmission weight. Normally, its range is 0-2, we
               make it 0-1 here. */
        Float diff_trans =
            m_has_diff_trans ? m_diff_trans->eval_1(si, active) / 2.0f : 0.0f;

        // There is no negative incoming angle for a thin surface, so we
        // change the direction for back_side case. The direction change is
        // taken into account after sampling the outgoing direction.
        Vector3f wi = dr::mulsign(si.wi, cos_theta_i);

        // Probability for each minor lobe
        Float prob_spec_reflect =
            m_has_spec_trans ? spec_trans * m_spec_refl_srate / 2.0f : 0.0f;
        Float prob_spec_trans =
            m_has_spec_trans ? spec_trans * m_spec_trans_srate / 2.0f : 0.0f;
        Float prob_coshemi_reflect =
            m_diff_refl_srate * (1.0f - spec_trans) * (1.0f - diff_trans);
        Float prob_coshemi_trans =
            m_has_diff_trans
                ? m_diff_trans_srate * (1.0f - spec_trans) * (diff_trans)
                : 0.0f;

        // Normalizing the probabilities for the specular minor lobes
        Float rcp_total_prob =
            dr::rcp(prob_spec_reflect + prob_spec_trans + prob_coshemi_reflect +
                    prob_coshemi_trans);

        prob_spec_reflect *= rcp_total_prob;
        prob_spec_trans *= rcp_total_prob;
        prob_coshemi_reflect *= rcp_total_prob;

        // Sampling masks
        Float curr_prob(0.0f);
        Mask sample_spec_reflect =
            m_has_spec_trans && active && (sample1 < prob_spec_reflect);
        curr_prob += prob_spec_reflect;
        Mask sample_spec_trans = m_has_spec_trans && active &&
                                 (sample1 >= curr_prob) &&
                                 (sample1 < curr_prob + prob_spec_trans);
        curr_prob += prob_spec_trans;
        Mask sample_coshemi_reflect =
            active && (sample1 >= curr_prob) &&
            (sample1 < curr_prob + prob_coshemi_reflect);
        curr_prob += prob_coshemi_reflect;
        Mask sample_coshemi_trans =
            m_has_diff_trans && active && (sample1 >= curr_prob);

        // Thin model is just a  2D surface, both mediums have the same
        // index of refraction
        bs.eta = 1.0f;

        // Microfacet reflection lobe
        if (m_has_spec_trans && dr::any_or<true>(sample_spec_reflect))
        {
            // Defining the Microfacet Distribution.
            auto [ax, ay] =
                calc_dist_params(anisotropic, roughness, m_has_anisotropic);
            MicrofacetDistribution spec_reflect_distr(MicrofacetType::GGX, ax,
                                                      ay);
            Normal3f m_spec_reflect =
                std::get<0>(spec_reflect_distr.sample(wi, sample2));

            // Sampling
            Vector3f wo = reflect(wi, m_spec_reflect);
            dr::masked(bs.wo, sample_spec_reflect) = wo;
            dr::masked(bs.sampled_component, sample_spec_reflect) = 3;
            dr::masked(bs.sampled_type, sample_spec_reflect) =
                +BSDFFlags::GlossyReflection;

            // Filter the cases where macro and micro SURFACES do not agree
            // on the same side and the ray is not reflected.
            Mask reflect = Frame3f::cos_theta(wo) > 0.0f;
            active &= (!sample_spec_reflect ||
                       (mac_mic_compatibility(Vector3f(m_spec_reflect), wi, wo,
                                              wi.z(), true) &&
                        reflect));
        }
        // Specular transmission lobe
        if (m_has_spec_trans && dr::any_or<true>(sample_spec_trans))
        {
            // Relative index of refraction.
            Float eta_t = m_eta_thin->eval_1(si, active);

            // Defining the scaled distribution for thin specular
            // transmission Scale roughness based on IOR (Burley 2015,
            // Figure 15).
            Float roughness_scaled = (0.65f * eta_t - 0.35f) * roughness;
            auto [ax_scaled, ay_scaled] = calc_dist_params(
                anisotropic, roughness_scaled, m_has_anisotropic);
            MicrofacetDistribution spec_trans_distr(MicrofacetType::GGX,
                                                    ax_scaled, ay_scaled);
            Normal3f m_spec_trans =
                std::get<0>(spec_trans_distr.sample(wi, sample2));

            // Here, we are reflecting and turning the ray to the other side
            // since there is no bending on thin surfaces.
            Vector3f wo = reflect(wi, m_spec_trans);
            wo.z() = -wo.z();
            dr::masked(bs.wo, sample_spec_trans) = wo;
            dr::masked(bs.sampled_component, sample_spec_trans) = 2;
            dr::masked(bs.sampled_type, sample_spec_trans) =
                +BSDFFlags::GlossyTransmission;

            // Filter the cases where macro and micro SURFACES do not agree
            // on the same side and the ray is not refracted.
            Mask transmission = Frame3f::cos_theta(wo) < 0.0f;
            active &= (!sample_spec_trans ||
                       (mac_mic_compatibility(Vector3f(m_spec_trans), wi, wo,
                                              wi.z(), false) &&
                        transmission));
        }
        // Cosine hemisphere reflection for  reflection lobes (diffuse,
        //  retro reflection)
        if (dr::any_or<true>(sample_coshemi_reflect))
        {
            dr::masked(bs.wo, sample_coshemi_reflect) =
                warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.sampled_component, sample_coshemi_reflect) = 0;
            dr::masked(bs.sampled_type, sample_coshemi_reflect) =
                +BSDFFlags::DiffuseReflection;
        }
        // Diffuse transmission lobe
        if (m_has_diff_trans && dr::any_or<true>(sample_coshemi_trans))
        {
            dr::masked(bs.wo, sample_coshemi_trans) =
                -1.0f * warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.sampled_component, sample_coshemi_trans) = 1;
            dr::masked(bs.sampled_type, sample_coshemi_trans) =
                +BSDFFlags::DiffuseTransmission;
        }

        /* The direction is changed once more. (Because it was changed in
           the beginning.) */
        bs.wo = dr::mulsign(bs.wo, cos_theta_i);

        bs.pdf = pdf(ctx, si, bs.wo, active);
        active &= bs.pdf > 0.0f;
        Spectrum result = eval(ctx, si, bs.wo, active);
        return {bs, result / bs.pdf & active};
    }

    Spectrum eval(const BSDFContext &, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override
    {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        // Ignore perfectly grazing configurations
        active &= dr::neq(cos_theta_i, 0.0f);

        if (unlikely(dr::none_or<false>(active)))
            return 0.0f;

        // Store the weights.
        Float anisotropic =
                  m_has_anisotropic ? m_anisotropic->eval_1(si, active) : 0.0f,
              roughness = m_roughness->eval_1(si, active),
              flatness = m_has_flatness ? m_flatness->eval_1(si, active) : 0.0f,
              spec_trans =
                  m_has_spec_trans ? m_spec_trans->eval_1(si, active) : 0.0f,
              eta_t = m_eta_thin->eval_1(si, active),
              // The range of diff_trans parameter is 0 to 2. It is made 0
              // to 1 here.
            diff_trans = m_has_diff_trans
                             ? m_diff_trans->eval_1(si, active) / 2.0f
                             : 0.0f;
        UnpolarizedSpectrum base_color = m_base_color->eval(si, active);

        // Changing the signs in a way that we are always at the front side.
        // Thin BSDF is symmetric!
        Vector3f wi = dr::mulsign(si.wi, cos_theta_i);
        Vector3f wo_t = dr::mulsign(wo, cos_theta_i);
        cos_theta_i = dr::abs(cos_theta_i);
        Float cos_theta_o = Frame3f::cos_theta(wo_t);

        Mask reflect = cos_theta_o > 0.0f;
        Mask refract = cos_theta_o < 0.0f;

        // Halfway vector calculation
        Vector3f wo_r = wo_t;
        wo_r.z() = dr::abs(wo_r.z());
        Vector3f wh = dr::normalize(wi + wo_r);

        /* Masks for controlling the micro-macro surface incompatibilities
           and correct sides. */
        Mask spec_reflect_active =
            active && (spec_trans > 0.0f) && reflect &&
            mac_mic_compatibility(wh, wi, wo_t, wi.z(), true);
        Mask spec_trans_active =
            active && refract && (spec_trans > 0.0f) &&
            mac_mic_compatibility(wh, wi, wo_t, wi.z(), false);
        Mask diffuse_reflect_active =
            active && reflect && (spec_trans < 1.0f) && (diff_trans < 1.0f);
        Mask diffuse_trans_active =
            active && refract && (spec_trans < 1) && (diff_trans > 0.0f);

        // Calculation of eval function starts.
        UnpolarizedSpectrum value = 0.0f;

        // Specular lobes (transmission and reflection)
        if (m_has_spec_trans)
        {
            // Dielectric Fresnel
            Float F_dielectric = std::get<0>(fresnel(dr::dot(wi, wh), eta_t));

            // Specular reflection lobe
            if (dr::any_or<true>(spec_reflect_active))
            {
                // Specular reflection distribution
                auto [ax, ay] =
                    calc_dist_params(anisotropic, roughness, m_has_anisotropic);
                MicrofacetDistribution spec_reflect_distr(MicrofacetType::GGX,
                                                          ax, ay);

                // No need to calculate luminance if there is no color tint.
                Float lum = m_has_spec_tint
                                ? mitsuba::luminance(base_color, si.wavelengths)
                                : 1.0f;
                Float spec_tint =
                    m_has_spec_tint ? m_spec_tint->eval_1(si, active) : 0.0f;

                UnpolarizedSpectrum F_thin =
                    thin_fresnel(F_dielectric, spec_tint, base_color, lum,
                                 dr::dot(wi, wh), eta_t, m_has_spec_tint);

                // Evaluate the microfacet normal distribution
                Float D = spec_reflect_distr.eval(wh);

                // Smith's shadow-masking function
                Float G = spec_reflect_distr.G(wi, wo_t, wh);

                // Calculate the specular reflection component.
                dr::masked(value, spec_reflect_active) +=
                    spec_trans * F_thin * D * G / (4.0f * cos_theta_i);
            }
            // Specular Transmission lobe
            if (dr::any_or<true>(spec_trans_active))
            {
                // Defining the scaled distribution for thin specular
                // reflection Scale roughness based on IOR. (Burley 2015,
                // Figure 15).
                Float roughness_scaled = (0.65f * eta_t - 0.35f) * roughness;
                auto [ax_scaled, ay_scaled] = calc_dist_params(
                    anisotropic, roughness_scaled, m_has_anisotropic);
                MicrofacetDistribution spec_trans_distr(MicrofacetType::GGX,
                                                        ax_scaled, ay_scaled);

                // Evaluate the microfacet normal distribution
                Float D = spec_trans_distr.eval(wh);

                // Smith's shadow-masking function
                Float G = spec_trans_distr.G(wi, wo_t, wh);

                // Calculate the specular transmission component.
                dr::masked(value, spec_trans_active) +=
                    spec_trans * base_color * (1.0f - F_dielectric) * D * G /
                    (4.0f * cos_theta_i);
            }
        }
        // Diffuse, retro reflection, sheen and fake-subsurface evaluation
        if (dr::any_or<true>(diffuse_reflect_active))
        {
            Float Fo = schlick_weight(dr::abs(cos_theta_o)),
                  Fi = schlick_weight(cos_theta_i);

            // Diffuse response
            Float f_diff = (1.0f - 0.5f * Fi) * (1.0f - 0.5f * Fo);

            // Retro response
            Float cos_theta_d = dr::dot(wh, wo_t);
            Float Rr = 2.0f * roughness * dr::sqr(cos_theta_d);
            Float f_retro = Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0f));

            /* Fake subsurface implementation based on Hanrahan-Krueger
               Fss90 used to "flatten" retro reflection based on
               roughness. */
            if (m_has_flatness)
            {
                Float Fss90 = Rr / 2.f;
                Float Fss = dr::lerp(1.f, Fss90, Fo) * dr::lerp(1.f, Fss90, Fi);
                Float f_ss = 1.25f * (Fss * (1.f / (dr::abs(cos_theta_o) +
                                                    dr::abs(cos_theta_i)) -
                                             0.5f) +
                                      0.5f);

                // Adding diffuse, retro and fake subsurface components.
                dr::masked(value, diffuse_reflect_active) +=
                    (1.0f - spec_trans) * cos_theta_o * base_color *
                    dr::InvPi<Float> * (1.0f - diff_trans) *
                    dr::lerp(f_diff + f_retro, f_ss, flatness);
            }
            else
                // Adding diffuse and retro components. (no fake subsurface)
                dr::masked(value, diffuse_reflect_active) +=
                    (1.0f - spec_trans) * cos_theta_o * base_color *
                    dr::InvPi<Float> * (1.0f - diff_trans) * (f_diff + f_retro);

            // Sheen evaluation
            Float sheen =
                m_has_sheen ? m_sheen->eval_1(si, active) : Float(0.0f);
            if (m_has_sheen && dr::any_or<true>(sheen > 0.0f))
            {

                Float Fd = schlick_weight(dr::abs(cos_theta_d));

                if (m_has_sheen_tint)
                { // Tints the sheen evaluation to the
                    // base_color.
                    Float sheen_tint = m_sheen_tint->eval_1(si, active);

                    // Calculation of luminance of base_color.
                    Float lum = mitsuba::luminance(base_color, si.wavelengths);

                    // Normalize color with luminance and apply tint.
                    UnpolarizedSpectrum c_tint =
                        dr::select(lum > 0.0f, base_color / lum, 1.0f);
                    UnpolarizedSpectrum c_sheen =
                        dr::lerp(1.0f, c_tint, sheen_tint);

                    // Adding the sheen component with tint.
                    dr::masked(value, diffuse_reflect_active) +=
                        sheen * (1.0f - spec_trans) * Fd * c_sheen *
                        (1.0f - diff_trans) * dr::abs(cos_theta_o);
                }
                else
                    // Adding the sheen component without tint.
                    dr::masked(value, diffuse_reflect_active) +=
                        sheen * (1.0f - spec_trans) * Fd * (1.0f - diff_trans) *
                        dr::abs(cos_theta_o);
            }
        }
        // Adding diffuse Lambertian transmission component.
        if (m_has_diff_trans && dr::any_or<true>(diffuse_trans_active))
        {
            dr::masked(value, diffuse_trans_active) +=
                (1.0f - spec_trans) * diff_trans * base_color *
                dr::InvPi<Float> * dr::abs(cos_theta_o);
        }
        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override
    {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        // Ignore perfectly grazing configurations.
        active &= dr::neq(cos_theta_i, 0.0f);

        if (unlikely(dr::none_or<false>(active)))
            return 0.0f;

        // Store the weights.
        Float anisotropic =
                  m_has_anisotropic ? m_anisotropic->eval_1(si, active) : 0.0f,
              roughness = m_roughness->eval_1(si, active),
              spec_trans =
                  m_has_spec_trans ? m_spec_trans->eval_1(si, active) : 0.0f,
              eta_t = m_eta_thin->eval_1(si, active),
              // The range of diff_trans parameter is 0 to 2. It is made 0
              // to 1 here
            diff_trans = m_has_diff_trans
                             ? m_diff_trans->eval_1(si, active) / 2.0f
                             : 0.0f;

        // Changing the signs in a way that we are always at the front side.
        // Thin BSDF is symmetric !!
        Vector3f wi = dr::mulsign(si.wi, cos_theta_i);
        // wo_t stands for thin wo.
        Vector3f wo_t = dr::mulsign(wo, cos_theta_i);
        Float cos_theta_o = Frame3f::cos_theta(wo_t);

        Mask reflect = cos_theta_o > 0.0f;
        Mask refract = cos_theta_o < 0.0f;

        // Probability definitions
        Float prob_spec_reflect =
            m_has_spec_trans ? spec_trans * m_spec_refl_srate / 2.0f : 0.0f;
        Float prob_spec_trans =
            m_has_spec_trans ? spec_trans * m_spec_trans_srate / 2.0f : 0.0f;
        Float prob_coshemi_reflect =
            m_diff_refl_srate * (1.0f - spec_trans) * (1.0f - diff_trans);
        Float prob_coshemi_trans =
            m_has_diff_trans
                ? m_diff_trans_srate * (1.0f - spec_trans) * (diff_trans)
                : 0.0f;

        // Normalizing the probabilities
        Float rcp_total_prob =
            dr::rcp(prob_spec_reflect + prob_spec_trans + prob_coshemi_reflect +
                    prob_coshemi_trans);
        prob_spec_reflect *= rcp_total_prob;
        prob_spec_trans *= rcp_total_prob;
        prob_coshemi_reflect *= rcp_total_prob;
        prob_coshemi_trans *= rcp_total_prob;

        // Initializing the final pdf value.
        Float pdf(0.0f);

        // Specular lobes' pdf evaluations
        if (m_has_spec_trans)
        {
            /* Halfway vector calculation. Absolute value is taken since for
             * specular transmission, we first apply microfacet reflection
             * and invert to the other side. */
            Vector3f wo_r = wo_t;
            wo_r.z() = dr::abs(wo_r.z());
            Vector3f wh = dr::normalize(wi + wo_r);

            // Macro-micro surface compatibility masks
            Mask mfacet_reflect_macmic =
                mac_mic_compatibility(wh, wi, wo_t, wi.z(), true) && reflect;
            Mask mfacet_trans_macmic =
                mac_mic_compatibility(wh, wi, wo_t, wi.z(), false) && refract;

            // d(wh) / d(wo) calculation. Inverted wo is used (wo_r) !
            Float dot_wor_wh = dr::dot(wo_r, wh);
            Float dwh_dwo_abs = dr::abs(dr::rcp(4.0f * dot_wor_wh));

            // Specular reflection distribution.
            auto [ax, ay] =
                calc_dist_params(anisotropic, roughness, m_has_anisotropic);
            MicrofacetDistribution spec_reflect_distr(MicrofacetType::GGX, ax,
                                                      ay);
            // Defining the scaled distribution for thin specular reflection
            // Scale roughness based on IOR (Burley 2015, Figure 15).
            Float roughness_scaled = (0.65f * eta_t - 0.35f) * roughness;
            auto [ax_scaled, ay_scaled] = calc_dist_params(
                anisotropic, roughness_scaled, m_has_anisotropic);
            MicrofacetDistribution spec_trans_distr(MicrofacetType::GGX,
                                                    ax_scaled, ay_scaled);
            // Adding specular lobes' pdfs
            dr::masked(pdf, mfacet_reflect_macmic) +=
                prob_spec_reflect * spec_reflect_distr.pdf(wi, wh) *
                dwh_dwo_abs;
            dr::masked(pdf, mfacet_trans_macmic) +=
                prob_spec_trans * spec_trans_distr.pdf(wi, wh) * dwh_dwo_abs;
        }
        // Adding cosine hemisphere reflection pdf
        dr::masked(pdf, reflect) +=
            prob_coshemi_reflect * warp::square_to_cosine_hemisphere_pdf(wo_t);

        // Adding cosine hemisphere transmission pdf
        if (m_has_diff_trans)
        {
            dr::masked(pdf, refract) +=
                prob_coshemi_trans *
                warp::square_to_cosine_hemisphere_pdf(-wo_t);
        }
        return pdf;
    }

    Spectrum eval_diffuse_reflectance(const SurfaceInteraction3f &si,
                                      Mask active) const override
    {
        return m_base_color->eval(si, active);
    }

    std::string to_string() const override
    {
        std::ostringstream oss;
        oss << "CustomFabirc[" << std::endl;
        oss << "  eta = " << m_eta << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ScalarFloat m_eta;
    ref<Texture> m_specular_reflectance;
    ref<Texture> m_specular_transmittance;
    ref<Texture> m_base_color;
    ref<Texture> m_roughness;
    ref<Texture> m_anisotropic;
    ref<Texture> m_sheen;
    ref<Texture> m_sheen_tint;
    ref<Texture> m_spec_trans;
    ref<Texture> m_flatness;
    ref<Texture> m_spec_tint;
    ref<Texture> m_diff_trans;
    ref<Texture> m_eta_thin;
    ref<Texture> m_fiber_density;
    ref<Texture> m_fiber_radius;
    ref<Texture> m_fiber_variation;

    /// Sampling rates
    ScalarFloat m_spec_refl_srate;
    ScalarFloat m_spec_trans_srate;
    ScalarFloat m_diff_trans_srate;
    ScalarFloat m_diff_refl_srate;

    /** Whether the lobes are active or not.*/
    bool m_has_sheen;
    bool m_has_diff_trans;
    bool m_has_spec_trans;
    bool m_has_spec_tint;
    bool m_has_sheen_tint;
    bool m_has_anisotropic;
    bool m_has_flatness;
};

MI_IMPLEMENT_CLASS_VARIANT(Fabric, BSDF)
MI_EXPORT_PLUGIN(Fabric, "Fabric Micro-Appearance Models")
NAMESPACE_END(mitsuba)
