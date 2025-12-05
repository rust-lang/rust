//! This module defines the `Dist` struct, which represents the `[dist]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[dist]` table contains options related to the distribution process,
//! including signing, uploading artifacts, source tarballs, compression settings,
//! and inclusion of specific tools.

use serde::{Deserialize, Deserializer};

use crate::core::config::Merge;
use crate::core::config::toml::ReplaceOpt;
use crate::{HashSet, PathBuf, define_config, exit};

define_config! {
    #[derive(Default)]
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
