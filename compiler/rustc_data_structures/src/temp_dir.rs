use std::mem::ManuallyDrop;
use std::path::Path;
use tempfile::TempDir;

/// This is used to avoid TempDir being dropped on error paths unintentionally.
#[derive(Debug)]
pub struct MaybeTempDir {
    dir: ManuallyDrop<TempDir>,
    // Whether the TempDir should be deleted on drop.
    keep: bool,
}

impl Drop for MaybeTempDir {
    fn drop(&mut self) {
        // SAFETY: We are in the destructor, and no further access will
        // occur.
        let dir = unsafe { ManuallyDrop::take(&mut self.dir) };
        if self.keep {
            let _ = dir.into_path();
        }
    }
}

impl AsRef<Path> for MaybeTempDir {
    fn as_ref(&self) -> &Path {
        self.dir.path()
    }
}

impl MaybeTempDir {
    pub fn new(dir: TempDir, keep_on_drop: bool) -> MaybeTempDir {
        MaybeTempDir { dir: ManuallyDrop::new(dir), keep: keep_on_drop }
    }
}
