use std::path::PathBuf;
use tidy::{t, walk};

pub const VERSION_PLACEHOLDER: &str = "CURRENT_RUSTC_VERSION";

fn main() {
    let root_path: PathBuf = std::env::args_os().nth(1).expect("need path to root of repo").into();
    let version_path = root_path.join("src").join("version");
    let version_str = t!(std::fs::read_to_string(&version_path), version_path);
    let version_str = version_str.trim();
    walk::walk(
        &root_path,
        |path| {
            walk::filter_dirs(path)
                // We exempt these as they require the placeholder
                // for their operation
                || path.ends_with("compiler/rustc_attr/src/builtin.rs")
                || path.ends_with("src/tools/tidy/src/features/version.rs")
                || path.ends_with("src/tools/replace-version-placeholder")
        },
        &mut |entry, contents| {
            if !contents.contains(VERSION_PLACEHOLDER) {
                return;
            }
            let new_contents = contents.replace(VERSION_PLACEHOLDER, version_str);
            let path = entry.path();
            t!(std::fs::write(&path, new_contents), path);
        },
    );
}
