//! Abstract-ish representation of paths for VFS.
use std::fmt;

use paths::{AbsPath, AbsPathBuf};

/// Long-term, we want to support files which do not reside in the file-system,
/// so we treat VfsPaths as opaque identifiers.
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VfsPath(VfsPathRepr);

impl VfsPath {
    pub fn as_path(&self) -> Option<&AbsPath> {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => Some(it.as_path()),
        }
    }
    pub fn join(&self, path: &str) -> VfsPath {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => {
                let res = it.join(path).normalize();
                VfsPath(VfsPathRepr::PathBuf(res))
            }
        }
    }
    pub fn pop(&mut self) -> bool {
        match &mut self.0 {
            VfsPathRepr::PathBuf(it) => it.pop(),
        }
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
enum VfsPathRepr {
    PathBuf(AbsPathBuf),
}

impl From<AbsPathBuf> for VfsPath {
    fn from(v: AbsPathBuf) -> Self {
        VfsPath(VfsPathRepr::PathBuf(v))
    }
}

impl fmt::Display for VfsPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => fmt::Display::fmt(&it.display(), f),
        }
    }
}
