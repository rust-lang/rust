use std::env::consts::EXE_SUFFIX;
use std::env::current_dir;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use super::verify_inside_clippy_dir;

pub fn create(force: bool, release: bool, name: &str) {
    if !verify_inside_clippy_dir() {
        return;
    }

    let rustup_home = std::env::var("RUSTUP_HOME").unwrap();
    let toolchain = std::env::var("RUSTUP_TOOLCHAIN").unwrap();

    let src = PathBuf::from_iter([&rustup_home, "toolchains", &toolchain]);
    let dest = PathBuf::from_iter([&rustup_home, "toolchains", name]);

    if dest.exists() {
        if force {
            fs::remove_dir_all(&dest).unwrap();
        } else {
            println!("{} already exists, pass `--force` to override it", dest.display());
            return;
        }
    }

    for entry in WalkDir::new(&src) {
        let entry = entry.unwrap();
        let relative = entry.path().strip_prefix(&src).unwrap();

        if relative.starts_with("bin")
            && matches!(
                relative.file_stem().and_then(OsStr::to_str),
                Some("cargo-clippy" | "clippy-driver")
            )
        {
            continue;
        }

        let target = dest.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir(&target).unwrap();
        } else {
            fs::hard_link(entry.path(), target).unwrap();
        }
    }

    symlink_bin("cargo-clippy", &dest, release);
    symlink_bin("clippy-driver", &dest, release);

    println!("Created toolchain {name}, use it in other projects with e.g. `cargo +{name} clippy`");
    println!("Note: This will need to be re-run whenever the Clippy `rust-toolchain` changes");
}

fn symlink_bin(bin: &str, dest: &Path, release: bool) {
    #[cfg(windows)]
    use std::os::windows::fs::symlink_file as symlink;

    #[cfg(not(windows))]
    use std::os::unix::fs::symlink;

    let profile = if release { "release" } else { "debug" };
    let file_name = format!("{bin}{EXE_SUFFIX}");

    let mut src = current_dir().unwrap();
    src.extend(["target", profile, &file_name]);

    let mut dest = dest.to_path_buf();
    dest.extend(["bin", &file_name]);

    symlink(src, dest).unwrap();
}
