//! This module defines the `Tpde` struct, which represents the `[tpde]` table
//! in the `bootstrap.toml` configuration file.

use crate::core::config::macros::define_config;

define_config! {
    #[derive(Default)]
    struct Tpde {
        optimize: Option<bool> = "optimize",
        release_debuginfo: Option<bool> = "release-debuginfo",
        assertions: Option<bool> = "assertions",
    }
}
