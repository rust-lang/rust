//! Tidy check to ensure that there are no binaries checked into the source tree
//! by accident.
//!
//! In the past we've accidentally checked in test binaries and such which add a
//! huge amount of bloat to the Git history, so it's good to just ensure we
//! don't do that again.

pub use os_impl::*;

// All files are executable on Windows, so just check on Unix.
#[cfg(windows)]
mod os_impl {
    use std::path::Path;

    pub fn check_filesystem_support(_sources: &[&Path], _output: &Path) -> bool {
        return false;
    }

    pub fn check(_path: &Path, _bad: &mut bool) {}
}

#[cfg(unix)]
mod os_impl {
    use crate::walk::{filter_dirs, walk_no_read};
    use std::fs;
    use std::os::unix::prelude::*;
    use std::path::Path;
    use std::process::{Command, Stdio};

    enum FilesystemSupport {
        Supported,
        Unsupported,
        ReadOnlyFs,
    }

    use FilesystemSupport::*;

    fn is_executable(path: &Path) -> std::io::Result<bool> {
        Ok(path.metadata()?.mode() & 0o111 != 0)
    }

    pub fn check_filesystem_support(sources: &[&Path], output: &Path) -> bool {
        // We want to avoid false positives on filesystems that do not support the
        // executable bit. This occurs on some versions of Window's linux subsystem,
        // for example.
        //
        // We try to create the temporary file first in the src directory, which is
        // the preferred location as it's most likely to be on the same filesystem,
        // and then in the output (`build`) directory if that fails. Sometimes we
        // see the source directory mounted as read-only which means we can't
        // readily create a file there to test.
        //
        // See #36706 and #74753 for context.

        fn check_dir(dir: &Path) -> FilesystemSupport {
            let path = dir.join("tidy-test-file");
            match fs::File::create(&path) {
                Ok(file) => {
                    let exec = is_executable(&path).unwrap_or(false);
                    drop(file);
                    fs::remove_file(&path).expect("Deleted temp file");
                    // If the file is executable, then we assume that this
                    // filesystem does not track executability, so skip this check.
                    return if exec { Unsupported } else { Supported };
                }
                Err(e) => {
                    // If the directory is read-only or we otherwise don't have rights,
                    // just don't run this check.
                    //
                    // 30 is the "Read-only filesystem" code at least in one CI
                    //    environment.
                    if e.raw_os_error() == Some(30) {
                        eprintln!("tidy: Skipping binary file check, read-only filesystem");
                        return ReadOnlyFs;
                    }

                    panic!("unable to create temporary file `{:?}`: {:?}", path, e);
                }
            };
        }

        for &source_dir in sources {
            match check_dir(source_dir) {
                Unsupported => return false,
                ReadOnlyFs => {
                    return match check_dir(output) {
                        Supported => true,
                        _ => false,
                    };
                }
                _ => {}
            }
        }

        return true;
    }

    // FIXME: check when rust-installer test sh files will be removed,
    // and then remove them from exclude list
    const RI_EXCLUSION_LIST: &[&str] = &[
        "src/tools/rust-installer/test/image1/bin/program",
        "src/tools/rust-installer/test/image1/bin/program2",
        "src/tools/rust-installer/test/image1/bin/bad-bin",
        "src/tools/rust-installer/test/image2/bin/oldprogram",
        "src/tools/rust-installer/test/image3/bin/cargo",
    ];

    fn filter_rust_installer_no_so_bins(path: &Path) -> bool {
        RI_EXCLUSION_LIST.iter().any(|p| path.ends_with(p))
    }

    #[cfg(unix)]
    pub fn check(path: &Path, bad: &mut bool) {
        use std::ffi::OsStr;

        const ALLOWED: &[&str] = &["configure", "x"];

        for p in RI_EXCLUSION_LIST {
            if !path.join(Path::new(p)).exists() {
                tidy_error!(bad, "rust-installer test bins missed: {p}");
            }
        }

        // FIXME: we don't need to look at all binaries, only files that have been modified in this branch
        // (e.g. using `git ls-files`).
        walk_no_read(
            &[path],
            |path, _is_dir| {
                filter_dirs(path)
                    || path.ends_with("src/etc")
                    || filter_rust_installer_no_so_bins(path)
            },
            &mut |entry| {
                let file = entry.path();
                let extension = file.extension();
                let scripts = ["py", "sh", "ps1"];
                if scripts.into_iter().any(|e| extension == Some(OsStr::new(e))) {
                    return;
                }

                if t!(is_executable(&file), file) {
                    let rel_path = file.strip_prefix(path).unwrap();
                    let git_friendly_path = rel_path.to_str().unwrap().replace("\\", "/");

                    if ALLOWED.contains(&git_friendly_path.as_str()) {
                        return;
                    }

                    let output = Command::new("git")
                        .arg("ls-files")
                        .arg(&git_friendly_path)
                        .current_dir(path)
                        .stderr(Stdio::null())
                        .output()
                        .unwrap_or_else(|e| {
                            panic!("could not run git ls-files: {e}");
                        });
                    let path_bytes = rel_path.as_os_str().as_bytes();
                    if output.status.success() && output.stdout.starts_with(path_bytes) {
                        tidy_error!(bad, "binary checked into source: {}", file.display());
                    }
                }
            },
        )
    }
}
