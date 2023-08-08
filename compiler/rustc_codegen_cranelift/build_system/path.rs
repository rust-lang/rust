use std::fs;
use std::path::PathBuf;

use super::utils::remove_dir_if_exists;

#[derive(Debug, Clone)]
pub(crate) struct Dirs {
    pub(crate) source_dir: PathBuf,
    pub(crate) download_dir: PathBuf,
    pub(crate) build_dir: PathBuf,
    pub(crate) dist_dir: PathBuf,
    pub(crate) frozen: bool,
}

#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub(crate) enum PathBase {
    Source,
    Download,
    Build,
    Dist,
}

impl PathBase {
    fn to_path(self, dirs: &Dirs) -> PathBuf {
        match self {
            PathBase::Source => dirs.source_dir.clone(),
            PathBase::Download => dirs.download_dir.clone(),
            PathBase::Build => dirs.build_dir.clone(),
            PathBase::Dist => dirs.dist_dir.clone(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum RelPath {
    Base(PathBase),
    Join(&'static RelPath, &'static str),
}

impl RelPath {
    pub(crate) const SOURCE: RelPath = RelPath::Base(PathBase::Source);
    pub(crate) const DOWNLOAD: RelPath = RelPath::Base(PathBase::Download);
    pub(crate) const BUILD: RelPath = RelPath::Base(PathBase::Build);
    pub(crate) const DIST: RelPath = RelPath::Base(PathBase::Dist);

    pub(crate) const SCRIPTS: RelPath = RelPath::SOURCE.join("scripts");
    pub(crate) const PATCHES: RelPath = RelPath::SOURCE.join("patches");

    pub(crate) const fn join(&'static self, suffix: &'static str) -> RelPath {
        RelPath::Join(self, suffix)
    }

    pub(crate) fn to_path(&self, dirs: &Dirs) -> PathBuf {
        match self {
            RelPath::Base(base) => base.to_path(dirs),
            RelPath::Join(base, suffix) => base.to_path(dirs).join(suffix),
        }
    }

    pub(crate) fn ensure_exists(&self, dirs: &Dirs) {
        fs::create_dir_all(self.to_path(dirs)).unwrap();
    }

    pub(crate) fn ensure_fresh(&self, dirs: &Dirs) {
        let path = self.to_path(dirs);
        remove_dir_if_exists(&path);
        fs::create_dir_all(path).unwrap();
    }
}
