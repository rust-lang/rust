pub mod rustc_cfg;
pub mod target_data_layout;
pub mod target_triple;

use std::path::Path;

use crate::{ManifestPath, Sysroot};

#[derive(Copy, Clone)]
pub enum QueryConfig<'a> {
    /// Directly invoke `rustc` to query the desired information.
    Rustc(&'a Sysroot, &'a Path),
    /// Attempt to use cargo to query the desired information, honoring cargo configurations.
    /// If this fails, falls back to invoking `rustc` directly.
    Cargo(&'a Sysroot, &'a ManifestPath),
}
