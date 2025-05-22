use crate::utils::{FileUpdater, UpdateStatus, Version, parse_cargo_package};
use std::fmt::Write;

static CARGO_TOML_FILES: &[&str] = &[
    "clippy_config/Cargo.toml",
    "clippy_lints/Cargo.toml",
    "clippy_utils/Cargo.toml",
    "Cargo.toml",
];

pub fn bump_version(mut version: Version) {
    version.minor += 1;

    let mut updater = FileUpdater::default();
    for file in CARGO_TOML_FILES {
        updater.update_file(file, &mut |_, src, dst| {
            let package = parse_cargo_package(src);
            if package.version_range.is_empty() {
                dst.push_str(src);
                UpdateStatus::Unchanged
            } else {
                dst.push_str(&src[..package.version_range.start]);
                write!(dst, "\"{}\"", version.toml_display()).unwrap();
                dst.push_str(&src[package.version_range.end..]);
                UpdateStatus::from_changed(src.get(package.version_range.clone()) != dst.get(package.version_range))
            }
        });
    }
}
