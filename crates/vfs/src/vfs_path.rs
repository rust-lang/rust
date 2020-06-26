//! Abstract-ish representation of paths for VFS.
use std::fmt;

use paths::{AbsPath, AbsPathBuf};

/// Long-term, we want to support files which do not reside in the file-system,
/// so we treat VfsPaths as opaque identifiers.
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VfsPath(VfsPathRepr);

impl VfsPath {
    /// Creates an "in-memory" path from `/`-separates string.
    /// This is most useful for testing, to avoid windows/linux differences
    pub fn new_virtual_path(path: String) -> VfsPath {
        assert!(path.starts_with('/'));
        VfsPath(VfsPathRepr::VirtualPath(VirtualPath(path)))
    }

    pub fn as_path(&self) -> Option<&AbsPath> {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => Some(it.as_path()),
            VfsPathRepr::VirtualPath(_) => None,
        }
    }
    pub fn join(&self, path: &str) -> Option<VfsPath> {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => {
                let res = it.join(path).normalize();
                Some(VfsPath(VfsPathRepr::PathBuf(res)))
            }
            VfsPathRepr::VirtualPath(it) => {
                let res = it.join(path)?;
                Some(VfsPath(VfsPathRepr::VirtualPath(res)))
            }
        }
    }
    pub fn pop(&mut self) -> bool {
        match &mut self.0 {
            VfsPathRepr::PathBuf(it) => it.pop(),
            VfsPathRepr::VirtualPath(it) => it.pop(),
        }
    }
    pub fn starts_with(&self, other: &VfsPath) -> bool {
        match (&self.0, &other.0) {
            (VfsPathRepr::PathBuf(lhs), VfsPathRepr::PathBuf(rhs)) => lhs.starts_with(rhs),
            (VfsPathRepr::PathBuf(_), _) => false,
            (VfsPathRepr::VirtualPath(lhs), VfsPathRepr::VirtualPath(rhs)) => lhs.starts_with(rhs),
            (VfsPathRepr::VirtualPath(_), _) => false,
        }
    }
}

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
enum VfsPathRepr {
    PathBuf(AbsPathBuf),
    VirtualPath(VirtualPath),
}

impl From<AbsPathBuf> for VfsPath {
    fn from(v: AbsPathBuf) -> Self {
        VfsPath(VfsPathRepr::PathBuf(v.normalize()))
    }
}

impl fmt::Display for VfsPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => fmt::Display::fmt(&it.display(), f),
            VfsPathRepr::VirtualPath(VirtualPath(it)) => fmt::Display::fmt(it, f),
        }
    }
}

impl fmt::Debug for VfsPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Debug for VfsPathRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            VfsPathRepr::PathBuf(it) => fmt::Debug::fmt(&it.display(), f),
            VfsPathRepr::VirtualPath(VirtualPath(it)) => fmt::Debug::fmt(&it, f),
        }
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct VirtualPath(String);

impl VirtualPath {
    fn starts_with(&self, other: &VirtualPath) -> bool {
        self.0.starts_with(&other.0)
    }
    fn pop(&mut self) -> bool {
        let pos = match self.0.rfind('/') {
            Some(pos) => pos,
            None => return false,
        };
        self.0 = self.0[..pos].to_string();
        true
    }
    fn join(&self, mut path: &str) -> Option<VirtualPath> {
        let mut res = self.clone();
        while path.starts_with("../") {
            if !res.pop() {
                return None;
            }
            path = &path["../".len()..]
        }
        res.0 = format!("{}/{}", res.0, path);
        Some(res)
    }
}
