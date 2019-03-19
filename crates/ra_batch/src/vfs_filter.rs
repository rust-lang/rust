use std::path::PathBuf;
use ra_project_model::ProjectRoot;
use ra_vfs::{RootEntry, Filter, RelativePath};

pub struct IncludeRustFiles {
    /// Is a member of the current workspace
    is_member: bool,
}

impl IncludeRustFiles {
    pub fn from_roots<R>(roots: R) -> impl Iterator<Item = RootEntry>
    where
        R: IntoIterator<Item = ProjectRoot>,
    {
        roots.into_iter().map(IncludeRustFiles::from_root)
    }

    pub fn from_root(root: ProjectRoot) -> RootEntry {
        let is_member = root.is_member();
        IncludeRustFiles::into_entry(root.into_path(), is_member)
    }

    #[allow(unused)]
    pub fn external(path: PathBuf) -> RootEntry {
        IncludeRustFiles::into_entry(path, false)
    }

    pub fn member(path: PathBuf) -> RootEntry {
        IncludeRustFiles::into_entry(path, true)
    }

    fn into_entry(path: PathBuf, is_member: bool) -> RootEntry {
        RootEntry::new(path, Box::new(Self { is_member }))
    }
}

impl Filter for IncludeRustFiles {
    fn include_dir(&self, dir_path: &RelativePath) -> bool {
        const COMMON_IGNORED_DIRS: &[&str] = &["node_modules", "target", ".git"];
        const EXTERNAL_IGNORED_DIRS: &[&str] = &["examples", "tests", "benches"];

        let is_ignored = if self.is_member {
            dir_path.components().any(|c| COMMON_IGNORED_DIRS.contains(&c.as_str()))
        } else {
            dir_path.components().any(|c| {
                let path = c.as_str();
                COMMON_IGNORED_DIRS.contains(&path) || EXTERNAL_IGNORED_DIRS.contains(&path)
            })
        };

        let hidden = dir_path.components().any(|c| c.as_str().starts_with("."));

        !is_ignored && !hidden
    }

    fn include_file(&self, file_path: &RelativePath) -> bool {
        file_path.extension() == Some("rs")
    }
}
