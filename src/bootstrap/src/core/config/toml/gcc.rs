//! This module defines the `Gcc` struct, which represents the `[gcc]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[gcc]` table contains options specifically related to building or
//! acquiring the GCC compiler for use within the Rust build process.

use std::path::PathBuf;

use crate::core::config::macros::define_config;

define_config! {
    /// TOML representation of how the GCC build is configured.
    #[derive(Default)]
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
        libgccjit_libs_dir: Option<PathBuf> = "libgccjit-libs-dir",
    }
}
