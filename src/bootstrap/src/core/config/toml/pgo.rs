//! This module defines the `Pgo` struct, which represents the `[pgo]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[pgo]` table contains options related PGO (Profile-Guided Optimization) of various
//! components built by bootstrap.

use serde::{Deserialize, Deserializer};

use crate::core::config::Merge;
use crate::core::config::toml::ReplaceOpt;
use crate::{HashSet, PathBuf, define_config, exit};

#[derive(Clone, Default, Debug, serde_derive::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PgoConfig {
    /// Use the given PGO profile to optimize a component.
    #[serde(default, rename = "use")]
    pub use_profile: Option<PathBuf>,
    /// Build a component with PGO instrumentation. Once executed, the profiles will be stored
    /// into this path.
    #[serde(default, rename = "generate")]
    pub generate_profile: Option<PathBuf>,
}

define_config! {
    #[derive(Default)]
    struct Pgo {
        rustc: Option<PgoConfig> = "rustc",
        rustdoc: Option<PgoConfig> = "rustdoc",
        llvm: Option<PgoConfig> = "llvm",
    }
}
