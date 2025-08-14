//! This module defines the `Install` struct, which represents the `[install]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[install]` table contains options that specify the installation paths
//! for various components of the Rust toolchain. These paths determine where
//! executables, libraries, documentation, and other files will be placed
//! during the `install` stage of the build.

use serde::{Deserialize, Deserializer};

use crate::core::config::Merge;
use crate::core::config::toml::ReplaceOpt;
use crate::{HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of various global install decisions.
    #[derive(Default)]
    struct Install {
        prefix: Option<String> = "prefix",
        sysconfdir: Option<String> = "sysconfdir",
        docdir: Option<String> = "docdir",
        bindir: Option<String> = "bindir",
        libdir: Option<String> = "libdir",
        mandir: Option<String> = "mandir",
        datadir: Option<String> = "datadir",
    }
}
