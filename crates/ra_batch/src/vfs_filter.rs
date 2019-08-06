use ra_project_model::PackageRoot;
use ra_vfs::{Filter, RelativePath, RootEntry};
use std::path::PathBuf;

/// `IncludeRustFiles` is used to convert
/// from `PackageRoot` to `RootEntry` for VFS
pub struct IncludeRustFiles {
    root: PackageRoot,
}

impl IncludeRustFiles {
    pub fn from_roots<R>(roots: R) -> impl Iterator<Item = RootEntry>
    where
        R: IntoIterator<Item = PackageRoot>,
    {
        roots.into_iter().map(IncludeRustFiles::from_root)
    }

    pub fn from_root(root: PackageRoot) -> RootEntry {
        IncludeRustFiles::from(root).into()
    }

    #[allow(unused)]
    pub fn external(path: PathBuf) -> RootEntry {
        IncludeRustFiles::from_root(PackageRoot::new(path, false))
    }

    pub fn member(path: PathBuf) -> RootEntry {
        IncludeRustFiles::from_root(PackageRoot::new(path, true))
    }
}

impl Filter for IncludeRustFiles {
    fn include_dir(&self, dir_path: &RelativePath) -> bool {
        self.root.include_dir(dir_path)
    }

    fn include_file(&self, file_path: &RelativePath) -> bool {
        self.root.include_file(file_path)
    }
}

impl From<PackageRoot> for IncludeRustFiles {
    fn from(v: PackageRoot) -> IncludeRustFiles {
        IncludeRustFiles { root: v }
    }
}

impl From<IncludeRustFiles> for RootEntry {
    fn from(v: IncludeRustFiles) -> RootEntry {
        let path = v.root.path().clone();
        RootEntry::new(path, Box::new(v))
    }
}
