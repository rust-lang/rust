//! This module defines the `Dist` struct, which represents the `[dist]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[dist]` table contains options related to the distribution process,
//! including signing, uploading artifacts, source tarballs, compression settings,
//! and inclusion of specific tools.

use serde::{Deserialize, Deserializer};

use crate::core::config::toml::ReplaceOpt;
use crate::core::config::{Merge, set};
use crate::{Config, HashSet, PathBuf, define_config, exit};

define_config! {
    struct Dist {
        sign_folder: Option<String> = "sign-folder",
        upload_addr: Option<String> = "upload-addr",
        src_tarball: Option<bool> = "src-tarball",
        compression_formats: Option<Vec<String>> = "compression-formats",
        compression_profile: Option<String> = "compression-profile",
        include_mingw_linker: Option<bool> = "include-mingw-linker",
        vendor: Option<bool> = "vendor",
    }
}

impl Config {
    /// Applies distribution-related configuration from the `Dist` struct
    /// to the global `Config` structure.
    pub fn apply_dist_config(&mut self, toml_dist: Option<Dist>) {
        if let Some(dist) = toml_dist {
            let Dist {
                sign_folder,
                upload_addr,
                src_tarball,
                compression_formats,
                compression_profile,
                include_mingw_linker,
                vendor,
            } = dist;
            self.dist_sign_folder = sign_folder.map(PathBuf::from);
            self.dist_upload_addr = upload_addr;
            self.dist_compression_formats = compression_formats;
            set(&mut self.dist_compression_profile, compression_profile);
            set(&mut self.rust_dist_src, src_tarball);
            set(&mut self.dist_include_mingw_linker, include_mingw_linker);
            self.dist_vendor = vendor.unwrap_or_else(|| {
                // If we're building from git or tarball sources, enable it by default.
                self.rust_info.is_managed_git_subrepository() || self.rust_info.is_from_tarball()
            });
        }
    }
}
