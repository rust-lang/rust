use std::path::PathBuf;

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
    Build,
}

impl PathBase {
    fn to_path(self, dirs: &Dirs) -> PathBuf {
        match self {
            PathBase::Source => dirs.source_dir.clone(),
            PathBase::Build => dirs.build_dir.clone(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct RelPath {
    base: PathBase,
    suffix: &'static str,
}

impl RelPath {
    pub(crate) const fn source(suffix: &'static str) -> RelPath {
        RelPath { base: PathBase::Source, suffix }
    }

    pub(crate) const fn build(suffix: &'static str) -> RelPath {
        RelPath { base: PathBase::Build, suffix }
    }

    pub(crate) fn to_path(&self, dirs: &Dirs) -> PathBuf {
        self.base.to_path(dirs).join(self.suffix)
    }
}
