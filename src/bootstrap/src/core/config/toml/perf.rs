//! This module defines the `Perf` struct, which represents the `[perf]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[perf]` table contains options specific to
//! benchmarking using the `./x perf` command.

use std::path::PathBuf;

use crate::core::config::macros::define_config;

define_config! {
    /// TOML representation of how rustc-perf is configured
    #[derive(Default)]
    struct Perf {
        rustc_perf_path: Option<PathBuf> = "rustc-perf-path",
        prepare_jobs: Option<usize> = "prepare-jobs",
    }
}
