//! Tidy check to ensure that there are no binaries checked into the source tree
//! by accident.
//!
//! In the past we've accidentally checked in test binaries and such which add a
//! huge amount of bloat to the Git history, so it's good to just ensure we
//! don't do that again.

use std::path::Path;

// All files are executable on Windows, so just check on Unix.
#[cfg(windows)]
pub fn check(_path: &Path, _bad: &mut bool) {}

#[cfg(unix)]
pub fn check(path: &Path, bad: &mut bool) {
    use std::fs;
    use std::process::{Command, Stdio};
    use std::os::unix::prelude::*;

    if let Ok(contents) = fs::read_to_string("/proc/version") {
        // Probably on Windows Linux Subsystem or Docker via VirtualBox,
        // all files will be marked as executable, so skip checking.
        if contents.contains("Microsoft") || contents.contains("boot2docker") {
            return;
        }
    }

    super::walk_no_read(path,
                &mut |path| super::filter_dirs(path) || path.ends_with("src/etc"),
                &mut |entry| {
        let file = entry.path();
        let filename = file.file_name().unwrap().to_string_lossy();
        let extensions = [".py", ".sh"];
        if extensions.iter().any(|e| filename.ends_with(e)) {
            return;
        }

        let metadata = t!(entry.metadata(), file);
        if metadata.mode() & 0o111 != 0 {
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
    })
}
