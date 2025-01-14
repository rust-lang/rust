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
enum PathBase {
    Source,
    Build,
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
        match self.base {
            PathBase::Source => dirs.source_dir.join(self.suffix),
            PathBase::Build => dirs.build_dir.join(self.suffix),
        }
    }
}
