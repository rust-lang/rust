//! See [`ManifestPath`].
use std::{ops, path::Path};

use paths::{AbsPath, AbsPathBuf};

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
        if file.parent().is_none() {
            Err(file)
        } else {
            Ok(ManifestPath { file })
        }
    }
}

impl ManifestPath {
    // Shadow `parent` from `Deref`.
    pub fn parent(&self) -> &AbsPath {
        self.file.parent().unwrap()
    }

    pub fn canonicalize(&self) -> ! {
        (&**self).canonicalize()
    }
}

impl ops::Deref for ManifestPath {
    type Target = AbsPath;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

impl AsRef<Path> for ManifestPath {
    fn as_ref(&self) -> &Path {
        self.file.as_ref()
    }
}
