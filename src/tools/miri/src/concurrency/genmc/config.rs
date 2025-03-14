use super::GenmcParams;

/// Configuration for GenMC mode.
/// The `params` field is shared with the C++ side.
/// The remaining options are kept on the Rust side.
#[derive(Debug, Default, Clone)]
pub struct GenmcConfig {
    pub(super) params: GenmcParams,
    do_estimation: bool,
    // FIXME(GenMC): add remaining options.
}

impl GenmcConfig {
    /// Function for parsing command line options for GenMC mode.
    ///
    /// All GenMC arguments start with the string "-Zmiri-genmc".
    /// Passing any GenMC argument will enable GenMC mode.
    ///
    /// `trimmed_arg` should be the argument to be parsed, with the suffix "-Zmiri-genmc" removed.
    pub fn parse_arg(genmc_config: &mut Option<GenmcConfig>, trimmed_arg: &str) {
        // FIXME(genmc,macos): Add `target_os = "macos"` once `https://github.com/dtolnay/cxx/issues/1535` is fixed.
        if !cfg!(all(
            feature = "genmc",
            any(target_os = "linux", target_os = "macos"),
            target_pointer_width = "64",
            target_endian = "little"
        )) {
            unimplemented!(
                "GenMC mode is not supported on this platform, cannot handle argument: \"-Zmiri-genmc{trimmed_arg}\""
            );
        }
        if genmc_config.is_none() {
            *genmc_config = Some(Default::default());
        }
        if trimmed_arg.is_empty() {
            return; // this corresponds to "-Zmiri-genmc"
        }
        // FIXME(GenMC): implement remaining parameters.
        todo!();
    }
}
