use std::{
    iter,
    sync::Arc,
    path::{Path, PathBuf},
};

use relative_path::{ RelativePath, RelativePathBuf};
use ra_arena::{impl_arena_id, Arena, RawId};

/// VfsRoot identifies a watched directory on the file system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VfsRoot(pub RawId);
impl_arena_id!(VfsRoot);

/// Describes the contents of a single source root.
///
/// `RootConfig` can be thought of as a glob pattern like `src/**.rs` which
/// specifies the source root or as a function which takes a `PathBuf` and
/// returns `true` iff path belongs to the source root
struct RootData {
    path: PathBuf,
    // result of `root.canonicalize()` if that differs from `root`; `None` otherwise.
    canonical_path: Option<PathBuf>,
    excluded_dirs: Vec<RelativePathBuf>,
}

pub(crate) struct Roots {
    roots: Arena<VfsRoot, Arc<RootData>>,
}

impl Roots {
    pub(crate) fn new(mut paths: Vec<PathBuf>) -> Roots {
        let mut roots = Arena::default();
        // A hack to make nesting work.
        paths.sort_by_key(|it| std::cmp::Reverse(it.as_os_str().len()));
        paths.dedup();
        for (i, path) in paths.iter().enumerate() {
            let nested_roots =
                paths[..i].iter().filter_map(|it| rel_path(path, it)).collect::<Vec<_>>();

            let config = Arc::new(RootData::new(path.clone(), nested_roots));

            roots.alloc(config.clone());
        }
        Roots { roots }
    }
    pub(crate) fn find(&self, path: &Path) -> Option<(VfsRoot, RelativePathBuf)> {
        self.iter().find_map(|root| {
            let rel_path = self.contains(root, path)?;
            Some((root, rel_path))
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.roots.len()
    }
    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = VfsRoot> + 'a {
        self.roots.iter().map(|(id, _)| id)
    }
    pub(crate) fn path(&self, root: VfsRoot) -> &Path {
        self.roots[root].path.as_path()
    }
    /// Checks if root contains a path and returns a root-relative path.
    pub(crate) fn contains(&self, root: VfsRoot, path: &Path) -> Option<RelativePathBuf> {
        let data = &self.roots[root];
        iter::once(&data.path)
            .chain(data.canonical_path.as_ref().into_iter())
            .find_map(|base| rel_path(base, path))
            .filter(|path| !data.excluded_dirs.contains(path))
            .filter(|path| !data.is_excluded(path))
    }
}

impl RootData {
    fn new(path: PathBuf, excluded_dirs: Vec<RelativePathBuf>) -> RootData {
        let mut canonical_path = path.canonicalize().ok();
        if Some(&path) == canonical_path.as_ref() {
            canonical_path = None;
        }
        RootData { path, canonical_path, excluded_dirs }
    }

    fn is_excluded(&self, path: &RelativePath) -> bool {
        if self.excluded_dirs.iter().any(|it| it == path) {
            return true;
        }
        // Ignore some common directories.
        //
        // FIXME: don't hard-code, specify at source-root creation time using
        // gitignore
        for (i, c) in path.components().enumerate() {
            if let relative_path::Component::Normal(c) = c {
                if (i == 0 && c == "target") || c == ".git" || c == "node_modules" {
                    return true;
                }
            }
        }

        match path.extension() {
            None | Some("rs") => false,
            _ => true,
        }
    }
}

fn rel_path(base: &Path, path: &Path) -> Option<RelativePathBuf> {
    let path = path.strip_prefix(base).ok()?;
    let path = RelativePathBuf::from_path(path).unwrap();
    Some(path)
}
