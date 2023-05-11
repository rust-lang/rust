use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

pub(crate) struct TestDir {
    path: PathBuf,
    keep: bool,
}

impl TestDir {
    pub(crate) fn new() -> TestDir {
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
            return TestDir { path, keep: false };
        }
        panic!("Failed to create a temporary directory")
    }
    #[allow(unused)]
    pub(crate) fn keep(mut self) -> TestDir {
        self.keep = true;
        self
    }
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        remove_dir_all(&self.path).unwrap_or_else(|err| {
            panic!("failed to remove temporary directory {}: {err}", self.path.display())
        })
    }
}

#[cfg(not(windows))]
fn remove_dir_all(path: &Path) -> io::Result<()> {
    fs::remove_dir_all(path)
}

#[cfg(windows)]
fn remove_dir_all(path: &Path) -> io::Result<()> {
    for _ in 0..99 {
        if fs::remove_dir_all(path).is_ok() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(10))
    }
    fs::remove_dir_all(path)
}
