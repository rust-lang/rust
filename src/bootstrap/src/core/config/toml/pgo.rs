//! This module defines the `Pgo` struct, which represents the `[pgo]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[pgo]` table contains options related PGO (Profile-Guided Optimization) of various
//! components built by bootstrap.

use std::path::PathBuf;

use crate::core::config::macros::define_config;

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
        cargo: Option<PgoConfig> = "cargo",
        llvm: Option<PgoConfig> = "llvm",
    }
}
