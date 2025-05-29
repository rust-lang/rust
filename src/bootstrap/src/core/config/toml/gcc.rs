use serde::{Deserialize, Deserializer};

use crate::core::config::GccCiMode;
use crate::core::config::toml::{Merge, ReplaceOpt};
use crate::{Config, HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how the GCC build is configured.
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
    }
}

impl Config {
    pub fn apply_gcc_config(&mut self, toml_gcc: Option<Gcc>) {
        if let Some(gcc) = toml_gcc {
            self.gcc_ci_mode = match gcc.download_ci_gcc {
                Some(value) => match value {
                    true => GccCiMode::DownloadFromCi,
                    false => GccCiMode::BuildLocally,
                },
                None => GccCiMode::default(),
            };
        }
    }
}
