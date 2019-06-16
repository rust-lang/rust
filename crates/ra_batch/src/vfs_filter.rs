use std::path::PathBuf;
use ra_project_model::ProjectRoot;
use ra_vfs::{RootEntry, Filter, RelativePath};

/// `IncludeRustFiles` is used to convert
/// from `ProjectRoot` to `RootEntry` for VFS
pub struct IncludeRustFiles {
    root: ProjectRoot,
}

impl IncludeRustFiles {
    pub fn from_roots<R>(roots: R) -> impl Iterator<Item = RootEntry>
    where
        R: IntoIterator<Item = ProjectRoot>,
    {
        roots.into_iter().map(IncludeRustFiles::from_root)
    }

    pub fn from_root(root: ProjectRoot) -> RootEntry {
        IncludeRustFiles::from(root).into()
    }

    #[allow(unused)]
    pub fn external(path: PathBuf) -> RootEntry {
        IncludeRustFiles::from_root(ProjectRoot::new(path, false))
    }

    pub fn member(path: PathBuf) -> RootEntry {
        IncludeRustFiles::from_root(ProjectRoot::new(path, true))
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

impl From<ProjectRoot> for IncludeRustFiles {
    fn from(v: ProjectRoot) -> IncludeRustFiles {
        IncludeRustFiles { root: v }
    }
}

impl From<IncludeRustFiles> for RootEntry {
    fn from(v: IncludeRustFiles) -> RootEntry {
        let path = v.root.path().clone();
        RootEntry::new(path, Box::new(v))
    }
}
