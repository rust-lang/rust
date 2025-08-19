//! This module defines the `Gcc` struct, which represents the `[gcc]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[gcc]` table contains options specifically related to building or
//! acquiring the GCC compiler for use within the Rust build process.

use serde::{Deserialize, Deserializer};

use crate::core::config::Merge;
use crate::core::config::toml::ReplaceOpt;
use crate::{HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how the GCC build is configured.
    #[derive(Default)]
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
    }
}
