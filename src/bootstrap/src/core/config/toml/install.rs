//! This module defines the `Install` struct, which represents the `[install]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[install]` table contains options that specify the installation paths
//! for various components of the Rust toolchain. These paths determine where
//! executables, libraries, documentation, and other files will be placed
//! during the `install` stage of the build.

use serde::{Deserialize, Deserializer};

use crate::core::config::toml::ReplaceOpt;
use crate::core::config::{Merge, set};
use crate::{Config, HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of various global install decisions.
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

impl Config {
    /// Applies installation-related configuration from the `Install` struct
    /// to the global `Config` structure.
    pub fn apply_install_config(&mut self, toml_install: Option<Install>) {
        if let Some(install) = toml_install {
            let Install { prefix, sysconfdir, docdir, bindir, libdir, mandir, datadir } = install;
            self.prefix = prefix.map(PathBuf::from);
            self.sysconfdir = sysconfdir.map(PathBuf::from);
            self.datadir = datadir.map(PathBuf::from);
            self.docdir = docdir.map(PathBuf::from);
            set(&mut self.bindir, bindir.map(PathBuf::from));
            self.libdir = libdir.map(PathBuf::from);
            self.mandir = mandir.map(PathBuf::from);
        }
    }
}
