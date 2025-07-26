//! This module defines the `Gcc` struct, which represents the `[gcc]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[gcc]` table contains options specifically related to building or
//! acquiring the GCC compiler for use within the Rust build process.

use serde::{Deserialize, Deserializer};

use crate::core::config::toml::ReplaceOpt;
use crate::core::config::{GccCiMode, Merge};
use crate::{Config, HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how the GCC build is configured.
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
    }
}

impl Config {
    /// Applies GCC-related configuration from the `TomlGcc` struct to the
    /// global `Config` structure.
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
