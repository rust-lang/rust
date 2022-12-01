use std::fs;
use std::path::PathBuf;

/*pub(crate) struct Paths {
    source_dir: PathBuf,
    download_dir: PathBuf,
    build_dir: PathBuf,
    dist_dir: PathBuf,
}*/

#[doc(hidden)]
#[derive(Debug, Copy, Clone)]
pub(crate) enum PathBase {
    Source,
    Download,
    Build,
    Dist,
}

impl PathBase {
    fn to_path(self) -> PathBuf {
        // FIXME pass in all paths instead
        let current_dir = std::env::current_dir().unwrap();
        match self {
            PathBase::Source => current_dir,
            PathBase::Download => current_dir.join("download"),
            PathBase::Build => current_dir.join("build"),
            PathBase::Dist => current_dir.join("dist"),
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
    pub(crate) const BUILD_SYSROOT: RelPath = RelPath::SOURCE.join("build_sysroot");
    pub(crate) const PATCHES: RelPath = RelPath::SOURCE.join("patches");

    pub(crate) const fn join(&'static self, suffix: &'static str) -> RelPath {
        RelPath::Join(self, suffix)
    }

    pub(crate) fn to_path(&self) -> PathBuf {
        match self {
            RelPath::Base(base) => base.to_path(),
            RelPath::Join(base, suffix) => base.to_path().join(suffix),
        }
    }

    pub(crate) fn ensure_exists(&self) {
        fs::create_dir_all(self.to_path()).unwrap();
    }

    pub(crate) fn ensure_fresh(&self) {
        let path = self.to_path();
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        fs::create_dir_all(path).unwrap();
    }
}
