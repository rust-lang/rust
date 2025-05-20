use crate::MiriConfig;

#[derive(Debug, Default, Clone)]
pub struct GenmcConfig {
    // TODO: add fields
}

impl GenmcConfig {
    /// Function for parsing command line options for GenMC mode.
    /// All GenMC arguments start with the string "-Zmiri-genmc".
    ///
    /// `trimmed_arg` should be the argument to be parsed, with the suffix "-Zmiri-genmc" removed
    pub fn parse_arg(genmc_config: &mut Option<GenmcConfig>, trimmed_arg: &str) {
        if genmc_config.is_none() {
            *genmc_config = Some(Default::default());
        }
        todo!("implement parsing of GenMC options")
    }
}
