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
        let base = std::env::temp_dir().join("testdir");
        let pid = std::process::id();

        static CNT: AtomicUsize = AtomicUsize::new(0);
        for _ in 0..100 {
            let cnt = CNT.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{}_{}", pid, cnt));
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
        remove_dir_all(&self.path).unwrap()
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
