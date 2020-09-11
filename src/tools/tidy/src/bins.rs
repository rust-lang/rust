//! Tidy check to ensure that there are no binaries checked into the source tree
//! by accident.
//!
//! In the past we've accidentally checked in test binaries and such which add a
//! huge amount of bloat to the Git history, so it's good to just ensure we
//! don't do that again.

use std::path::Path;

// All files are executable on Windows, so just check on Unix.
#[cfg(windows)]
pub fn check(_path: &Path, _output: &Path, _bad: &mut bool) {}

#[cfg(unix)]
pub fn check(path: &Path, output: &Path, bad: &mut bool) {
    use std::fs;
    use std::os::unix::prelude::*;
    use std::process::{Command, Stdio};

    fn is_executable(path: &Path) -> std::io::Result<bool> {
        Ok(path.metadata()?.mode() & 0o111 != 0)
    }

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
    let mut temp_path = path.join("tidy-test-file");
    match fs::File::create(&temp_path).or_else(|_| {
        temp_path = output.join("tidy-test-file");
        fs::File::create(&temp_path)
    }) {
        Ok(file) => {
            let exec = is_executable(&temp_path).unwrap_or(false);
            std::mem::drop(file);
            std::fs::remove_file(&temp_path).expect("Deleted temp file");
            if exec {
                // If the file is executable, then we assume that this
                // filesystem does not track executability, so skip this check.
                return;
            }
        }
        Err(e) => {
            // If the directory is read-only or we otherwise don't have rights,
            // just don't run this check.
            //
            // 30 is the "Read-only filesystem" code at least in one CI
            //    environment.
            if e.raw_os_error() == Some(30) {
                eprintln!("tidy: Skipping binary file check, read-only filesystem");
                return;
            }

            panic!("unable to create temporary file `{:?}`: {:?}", temp_path, e);
        }
    }

    super::walk_no_read(
        path,
        &mut |path| super::filter_dirs(path) || path.ends_with("src/etc"),
        &mut |entry| {
            let file = entry.path();
            let filename = file.file_name().unwrap().to_string_lossy();
            let extensions = [".py", ".sh"];
            if extensions.iter().any(|e| filename.ends_with(e)) {
                return;
            }

            if t!(is_executable(&file), file) {
                let rel_path = file.strip_prefix(path).unwrap();
                let git_friendly_path = rel_path.to_str().unwrap().replace("\\", "/");
                let output = Command::new("git")
                    .arg("ls-files")
                    .arg(&git_friendly_path)
                    .current_dir(path)
                    .stderr(Stdio::null())
                    .output()
                    .unwrap_or_else(|e| {
                        panic!("could not run git ls-files: {}", e);
                    });
                let path_bytes = rel_path.as_os_str().as_bytes();
                if output.status.success() && output.stdout.starts_with(path_bytes) {
                    tidy_error!(bad, "binary checked into source: {}", file.display());
                }
            }
        },
    )
}
