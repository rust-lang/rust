use std::{
    fs, io,
    sync::atomic::{AtomicUsize, Ordering},
};

use paths::{Utf8Path, Utf8PathBuf};

pub(crate) struct TestDir {
    path: Utf8PathBuf,
    keep: bool,
}

impl TestDir {
    pub(crate) fn new() -> TestDir {
        TestDir::new_dir(false)
    }

    pub(crate) fn new_symlink() -> TestDir {
        TestDir::new_dir(true)
    }

    fn new_dir(symlink: bool) -> TestDir {
        let temp_dir = std::env::temp_dir();
        // On MacOS builders on GitHub actions, the temp dir is a symlink, and
        // that causes problems down the line. Specifically:
        // * Cargo may emit different PackageId depending on the working directory
        // * rust-analyzer may fail to map LSP URIs to correct paths.
        //
        // Work-around this by canonicalizing. Note that we don't want to do this
        // on *every* OS, as on windows `canonicalize` itself creates problems.
        #[cfg(target_os = "macos")]
        let temp_dir = temp_dir.canonicalize().unwrap();

        let base = temp_dir.join("testdir");
        let pid = std::process::id();

        static CNT: AtomicUsize = AtomicUsize::new(0);
        for _ in 0..100 {
            let cnt = CNT.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{pid}_{cnt}"));
            if path.is_dir() {
                continue;
            }
            fs::create_dir_all(&path).unwrap();

            #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
            if symlink {
                let symlink_path = base.join(format!("{pid}_{cnt}_symlink"));
                #[cfg(any(target_os = "macos", target_os = "linux"))]
                std::os::unix::fs::symlink(path, &symlink_path).unwrap();

                #[cfg(target_os = "windows")]
                std::os::windows::fs::symlink_dir(path, &symlink_path).unwrap();

                return TestDir {
                    path: Utf8PathBuf::from_path_buf(symlink_path).unwrap(),
                    keep: false,
                };
            }

            return TestDir { path: Utf8PathBuf::from_path_buf(path).unwrap(), keep: false };
        }
        panic!("Failed to create a temporary directory")
    }

    #[allow(unused)]
    pub(crate) fn keep(mut self) -> TestDir {
        self.keep = true;
        self
    }
    pub(crate) fn path(&self) -> &Utf8Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        if self.keep {
            return;
        }

        let filetype = fs::symlink_metadata(&self.path).unwrap().file_type();
        let actual_path = filetype.is_symlink().then(|| fs::read_link(&self.path).unwrap());

        if let Some(actual_path) = actual_path {
            remove_dir_all(Utf8Path::from_path(&actual_path).unwrap()).unwrap_or_else(|err| {
                panic!(
                    "failed to remove temporary link to directory {}: {err}",
                    actual_path.display()
                )
            })
        }

        remove_dir_all(&self.path).unwrap_or_else(|err| {
            panic!("failed to remove temporary directory {}: {err}", self.path)
        });
    }
}

#[cfg(not(windows))]
fn remove_dir_all(path: &Utf8Path) -> io::Result<()> {
    fs::remove_dir_all(path)
}

#[cfg(windows)]
fn remove_dir_all(path: &Utf8Path) -> io::Result<()> {
    for _ in 0..99 {
        if fs::remove_dir_all(path).is_ok() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(10))
    }
    fs::remove_dir_all(path)
}
