//! Run-time feature detection on OpenBSD

mod auxvec;

cfg_select! {
    target_arch = "aarch64" => {
        mod aarch64;
        pub(crate) use self::aarch64::detect_features;
    }
    target_arch = "powerpc64" => {
        mod powerpc;
        pub(crate) use self::powerpc::detect_features;
    }
    _ => {
        use crate::detect::cache;
        /// Performs run-time feature detection.
        pub(crate) fn detect_features() -> cache::Initializer {
            cache::Initializer::default()
        }
    }
}
