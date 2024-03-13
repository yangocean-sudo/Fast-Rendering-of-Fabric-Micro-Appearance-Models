#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Fabric final : public BSDF<Float, Spectrum>
{
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    Fabric(const Properties &props) : Base(props)
    {
        // Specifies the internal index of refraction at the interface
        ScalarFloat int_ior = lookup_ior(props, "theta", "bk7");

        // Specifies the external index of refraction at the interface
        ScalarFloat ext_ior = lookup_ior(props, "gamma", "air");

        if (int_ior < 0 || ext_ior < 0)
            Throw("The interior and exterior indices of refraction must"
                  " be positive!");

        m_eta = int_ior / ext_ior;

        if (props.has_property("specular_reflectance"))
            m_specular_reflectance =
                props.texture<Texture>("specular_reflectance", 1.f);
        if (props.has_property("specular_transmittance"))
            m_specular_transmittance =
                props.texture<Texture>("specular_transmittance", 1.f);

        m_components.push_back(BSDFFlags::DeltaReflection |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);
        m_components.push_back(BSDFFlags::DeltaTransmission |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide |
                               BSDFFlags::NonSymmetric);

        m_flags = m_components[0] | m_components[1];
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

    void traverse(TraversalCallback *callback) override
    {
        callback->put_parameter("eta", m_eta, +ParamFlags::NonDifferentiable);
        if (m_specular_reflectance)
            callback->put_object("specular_reflectance",
                                 m_specular_reflectance.get(),
                                 +ParamFlags::Differentiable);
        if (m_specular_transmittance)
            callback->put_object("specular_transmittance",
                                 m_specular_transmittance.get(),
                                 +ParamFlags::Differentiable);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f & /* sample2 */,
                                             Mask active) const override
    {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_reflection = ctx.is_enabled(BSDFFlags::DeltaReflection, 0),
             has_transmission = ctx.is_enabled(BSDFFlags::DeltaTransmission, 1);

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(si.wi);

        auto [r_i, cos_theta_t, eta_it, eta_ti] =
            fresnel(cos_theta_i, Float(m_eta));
        Float t_i = 1.f - r_i;

        // Lobe selection
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Mask selected_r;
        if (likely(has_reflection && has_transmission))
        {
            selected_r = sample1 <= r_i && active;
            bs.pdf = dr::detach(dr::select(selected_r, r_i, t_i));
        }
        else
        {
            if (has_reflection || has_transmission)
            {
                selected_r = Mask(has_reflection) && active;
                bs.pdf = 1.f;
            }
            else
            {
                return {bs, 0.f};
            }
        }
        Mask selected_t = !selected_r && active;

        bs.sampled_component = dr::select(selected_r, UInt32(0), UInt32(1));
        bs.sampled_type =
            dr::select(selected_r, UInt32(+BSDFFlags::DeltaReflection),
                       UInt32(+BSDFFlags::DeltaTransmission));

        bs.wo = dr::select(selected_r, reflect(si.wi),
                           refract(si.wi, cos_theta_t, eta_ti));

        bs.eta = dr::select(selected_r, Float(1.f), eta_it);

        UnpolarizedSpectrum reflectance = 1.f, transmittance = 1.f;
        if (m_specular_reflectance)
            reflectance = m_specular_reflectance->eval(si, selected_r);
        if (m_specular_transmittance)
            transmittance = m_specular_transmittance->eval(si, selected_t);

        Spectrum weight(0.f);
        if constexpr (is_polarized_v<Spectrum>)
        {

            Vector3f wo_hat =
                         ctx.mode == TransportMode::Radiance ? bs.wo : si.wi,
                     wi_hat =
                         ctx.mode == TransportMode::Radiance ? si.wi : bs.wo;

            /* BSDF weights are Mueller matrices now. */
            Float cos_theta_o_hat = Frame3f::cos_theta(wo_hat);
            Spectrum R = mueller::specular_reflection(
                         UnpolarizedSpectrum(cos_theta_o_hat),
                         UnpolarizedSpectrum(m_eta)),
                     T = mueller::specular_transmission(
                         UnpolarizedSpectrum(cos_theta_o_hat),
                         UnpolarizedSpectrum(m_eta));

            if (likely(has_reflection && has_transmission))
            {
                weight = dr::select(selected_r, R, T) / bs.pdf;
            }
            else if (has_reflection || has_transmission)
            {
                weight = has_reflection ? R : T;
                bs.pdf = 1.f;
            }

            /* The Stokes reference frame vector of this matrix lies
               perpendicular to the plane of reflection. */
            Vector3f n(0, 0, 1);
            Vector3f s_axis_in = dr::cross(n, -wo_hat);
            Vector3f s_axis_out = dr::cross(n, wi_hat);

            // Singularity when the input & output are collinear with the normal
            Mask collinear = dr::all(dr::eq(s_axis_in, Vector3f(0)));
            s_axis_in = dr::select(collinear, Vector3f(1, 0, 0),
                                   dr::normalize(s_axis_in));
            s_axis_out = dr::select(collinear, Vector3f(1, 0, 0),
                                    dr::normalize(s_axis_out));

            /* Rotate in/out reference vector of `weight` s.t. it aligns with
               the implicit Stokes bases of -wo_hat & wi_hat. */
            weight = mueller::rotate_mueller_basis(
                weight, -wo_hat, s_axis_in, mueller::stokes_basis(-wo_hat),
                wi_hat, s_axis_out, mueller::stokes_basis(wi_hat));

            if (dr::any_or<true>(selected_r))
                weight[selected_r] *= mueller::absorber(reflectance);

            if (dr::any_or<true>(selected_t))
                weight[selected_t] *= mueller::absorber(transmittance);
        }
        else
        {
            if (likely(has_reflection && has_transmission))
            {
                weight = 1.f;
                /* For differentiable variants, lobe choice has to be detached
                   to avoid bias. Sampling weights should be computed
                   accordingly. */
                if constexpr (dr::is_diff_v<Float>)
                {
                    if (dr::grad_enabled(r_i))
                    {
                        Float r_diff =
                            dr::replace_grad(Float(1.f), r_i / dr::detach(r_i));
                        Float t_diff =
                            dr::replace_grad(Float(1.f), t_i / dr::detach(t_i));
                        weight = dr::select(selected_r, r_diff, t_diff);
                    }
                }
            }
            else if (has_reflection || has_transmission)
            {
                weight = has_reflection ? r_i : t_i;
            }

            if (dr::any_or<true>(selected_r))
                weight[selected_r] *= reflectance;

            if (dr::any_or<true>(selected_t))
                weight[selected_t] *= transmittance;
        }

        if (dr::any_or<true>(selected_t))
        {
            /* For transmission, radiance must be scaled to account for the
               solid angle compression that occurs when crossing the interface.
             */
            Float factor =
                (ctx.mode == TransportMode::Radiance) ? eta_ti : Float(1.f);
            weight[selected_t] *= dr::sqr(factor);
        }

        return {bs, weight & active};
    }

    Spectrum eval(const BSDFContext & /* ctx */,
                  const SurfaceInteraction3f & /* si */,
                  const Vector3f & /* wo */, Mask /* active */) const override
    {
        return 0.f;
    }

    Float pdf(const BSDFContext & /* ctx */,
              const SurfaceInteraction3f & /* si */, const Vector3f & /* wo */,
              Mask /* active */) const override
    {
        return 0.f;
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
};

MI_IMPLEMENT_CLASS_VARIANT(Fabric, BSDF)
MI_EXPORT_PLUGIN(Fabric, "Fabric Micro-Appearance Models")
NAMESPACE_END(mitsuba)
