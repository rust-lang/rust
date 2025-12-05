//! See [`ManifestPath`].
use std::{borrow::Borrow, fmt, ops};

use paths::{AbsPath, AbsPathBuf, Utf8Path};

/// More or less [`AbsPathBuf`] with non-None parent.
///
/// We use it to store path to Cargo.toml, as we frequently use the parent dir
/// as a working directory to spawn various commands, and its nice to not have
/// to `.unwrap()` everywhere.
///
/// This could have been named `AbsNonRootPathBuf`, as we don't enforce that
/// this stores manifest files in particular, but we only use this for manifests
/// at the moment in practice.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ManifestPath {
    file: AbsPathBuf,
}

impl TryFrom<AbsPathBuf> for ManifestPath {
    type Error = AbsPathBuf;

    fn try_from(file: AbsPathBuf) -> Result<Self, Self::Error> {
        if file.parent().is_none() { Err(file) } else { Ok(ManifestPath { file }) }
    }
}

impl From<ManifestPath> for AbsPathBuf {
    fn from(it: ManifestPath) -> Self {
        it.file
    }
}

impl ManifestPath {
    // Shadow `parent` from `Deref`.
    pub fn parent(&self) -> &AbsPath {
        self.file.parent().unwrap()
    }

    pub fn canonicalize(&self) -> ! {
        (**self).canonicalize()
    }

    pub fn is_rust_manifest(&self) -> bool {
        self.file.extension() == Some("rs")
    }
}

impl fmt::Display for ManifestPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.file, f)
    }
}

impl ops::Deref for ManifestPath {
    type Target = AbsPath;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

impl AsRef<AbsPath> for ManifestPath {
    fn as_ref(&self) -> &AbsPath {
        self.file.as_ref()
    }
}

impl AsRef<std::path::Path> for ManifestPath {
    fn as_ref(&self) -> &std::path::Path {
        self.file.as_ref()
    }
}

impl AsRef<std::ffi::OsStr> for ManifestPath {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.file.as_ref()
    }
}

impl AsRef<Utf8Path> for ManifestPath {
    fn as_ref(&self) -> &Utf8Path {
        self.file.as_ref()
    }
}

impl Borrow<AbsPath> for ManifestPath {
    fn borrow(&self) -> &AbsPath {
        self.file.borrow()
    }
}
