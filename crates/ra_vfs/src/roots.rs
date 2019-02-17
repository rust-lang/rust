use std::{
    sync::Arc,
    path::{Path, PathBuf},
};

use relative_path::RelativePathBuf;
use ra_arena::{impl_arena_id, Arena, RawId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VfsRoot(pub RawId);
impl_arena_id!(VfsRoot);

/// Describes the contents of a single source root.
///
/// `RootConfig` can be thought of as a glob pattern like `src/**.rs` which
/// specifies the source root or as a function which takes a `PathBuf` and
/// returns `true` iff path belongs to the source root
pub(crate) struct RootConfig {
    pub(crate) root: PathBuf,
    // result of `root.canonicalize()` if that differs from `root`; `None` otherwise.
    canonical_root: Option<PathBuf>,
    excluded_dirs: Vec<PathBuf>,
}

pub(crate) struct Roots {
    roots: Arena<VfsRoot, Arc<RootConfig>>,
}

impl std::ops::Deref for Roots {
    type Target = Arena<VfsRoot, Arc<RootConfig>>;
    fn deref(&self) -> &Self::Target {
        &self.roots
    }
}

impl RootConfig {
    fn new(root: PathBuf, excluded_dirs: Vec<PathBuf>) -> RootConfig {
        let mut canonical_root = root.canonicalize().ok();
        if Some(&root) == canonical_root.as_ref() {
            canonical_root = None;
        }
        RootConfig { root, canonical_root, excluded_dirs }
    }
    /// Checks if root contains a path and returns a root-relative path.
    pub(crate) fn contains(&self, path: &Path) -> Option<RelativePathBuf> {
        // First, check excluded dirs
        if self.excluded_dirs.iter().any(|it| path.starts_with(it)) {
            return None;
        }
        let rel_path = path
            .strip_prefix(&self.root)
            .or_else(|err_payload| {
                self.canonical_root
                    .as_ref()
                    .map_or(Err(err_payload), |canonical_root| path.strip_prefix(canonical_root))
            })
            .ok()?;
        let rel_path = RelativePathBuf::from_path(rel_path).ok()?;

        // Ignore some common directories.
        //
        // FIXME: don't hard-code, specify at source-root creation time using
        // gitignore
        for (i, c) in rel_path.components().enumerate() {
            if let relative_path::Component::Normal(c) = c {
                if (i == 0 && c == "target") || c == ".git" || c == "node_modules" {
                    return None;
                }
            }
        }

        if path.is_file() && rel_path.extension() != Some("rs") {
            return None;
        }

        Some(rel_path)
    }
}

impl Roots {
    pub(crate) fn new(mut paths: Vec<PathBuf>) -> Roots {
        let mut roots = Arena::default();
        // A hack to make nesting work.
        paths.sort_by_key(|it| std::cmp::Reverse(it.as_os_str().len()));
        paths.dedup();
        for (i, path) in paths.iter().enumerate() {
            let nested_roots = paths[..i]
                .iter()
                .filter(|it| it.starts_with(path))
                .map(|it| it.clone())
                .collect::<Vec<_>>();

            let config = Arc::new(RootConfig::new(path.clone(), nested_roots));

            roots.alloc(config.clone());
        }
        Roots { roots }
    }
    pub(crate) fn find(&self, path: &Path) -> Option<(VfsRoot, RelativePathBuf)> {
        self.roots.iter().find_map(|(root, data)| data.contains(path).map(|it| (root, it)))
    }
}
