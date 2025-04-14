use genmc_sys::{GenmcParams, LogLevel};

/// Configuration for GenMC mode.
/// The `params` field is shared with the C++ side.
/// The remaining options are kept on the Rust side.
#[derive(Debug, Default, Clone)]
pub struct GenmcConfig {
    /// Parameters sent to the C++ side to create a new handle to the GenMC model checker.
    pub(super) params: GenmcParams,
    /// The log level for GenMC.
    pub(super) log_level: LogLevel,
}

impl GenmcConfig {
    /// Function for parsing command line options for GenMC mode.
    ///
    /// All GenMC arguments start with the string "-Zmiri-genmc".
    /// Passing any GenMC argument will enable GenMC mode.
    ///
    /// `trimmed_arg` should be the argument to be parsed, with the suffix "-Zmiri-genmc" removed.
    pub fn parse_arg(
        genmc_config: &mut Option<GenmcConfig>,
        trimmed_arg: &str,
    ) -> Result<(), String> {
        // FIXME(genmc): Ensure host == target somewhere.

        if genmc_config.is_none() {
            *genmc_config = Some(Default::default());
        }
        if trimmed_arg.is_empty() {
            return Ok(()); // this corresponds to "-Zmiri-genmc"
        }
        let genmc_config = genmc_config.as_mut().unwrap();
        let Some(trimmed_arg) = trimmed_arg.strip_prefix("-") else {
            return Err(format!("Invalid GenMC argument \"-Zmiri-genmc{trimmed_arg}\""));
        };
        if let Some(log_level) = trimmed_arg.strip_prefix("log=") {
            genmc_config.log_level = log_level.parse()?;
        } else {
            return Err(format!("Invalid GenMC argument: \"-Zmiri-genmc-{trimmed_arg}\""));
        }
        Ok(())
    }
}
