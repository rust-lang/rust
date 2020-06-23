//! See `Include`.

use std::convert::TryFrom;

use globset::{Glob, GlobSet, GlobSetBuilder};
use paths::{RelPath, RelPathBuf};

/// `Include` is the opposite of .gitignore.
///
/// It describes the set of files inside some directory.
///
/// The current implementation is very limited, it allows white-listing file
/// globs and black-listing directories.
#[derive(Debug, Clone)]
pub(crate) struct Include {
    include_files: GlobSet,
    exclude_dirs: Vec<RelPathBuf>,
}

impl Include {
    pub(crate) fn new(include: Vec<String>) -> Include {
        let mut include_files = GlobSetBuilder::new();
        let mut exclude_dirs = Vec::new();

        for glob in include {
            if glob.starts_with("!/") {
                if let Ok(path) = RelPathBuf::try_from(&glob["!/".len()..]) {
                    exclude_dirs.push(path)
                }
            } else {
                include_files.add(Glob::new(&glob).unwrap());
            }
        }
        let include_files = include_files.build().unwrap();
        Include { include_files, exclude_dirs }
    }
    pub(crate) fn include_file(&self, path: &RelPath) -> bool {
        self.include_files.is_match(path)
    }
    pub(crate) fn exclude_dir(&self, path: &RelPath) -> bool {
        self.exclude_dirs.iter().any(|excluded| path.starts_with(excluded))
    }
}
