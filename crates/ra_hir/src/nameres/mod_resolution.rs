//! This module resolves `mod foo;` declaration to file.
use std::borrow::Cow;

use ra_db::FileId;
use ra_syntax::SmolStr;
use relative_path::{RelativePath, RelativePathBuf};

use crate::{db::DefDatabase, HirFileId, Name};

#[derive(Clone, Debug)]
pub(super) struct ModDir {
    /// `.` for `mod.rs`, `lib.rs`
    /// `./foo` for `foo.rs`
    /// `./foo/bar` for `mod bar { mod x; }` nested in `foo.rs`
    path: RelativePathBuf,
    /// inside `./foo.rs`, mods with `#[path]` should *not* be relative to `./foo/`
    root_non_dir_owner: bool,
}

impl ModDir {
    pub(super) fn root() -> ModDir {
        ModDir { path: RelativePathBuf::default(), root_non_dir_owner: false }
    }

    pub(super) fn descend_into_definition(
        &self,
        name: &Name,
        attr_path: Option<&SmolStr>,
    ) -> ModDir {
        let mut path = self.path.clone();
        match attr_path {
            None => path.push(&name.to_string()),
            Some(attr_path) => {
                if self.root_non_dir_owner {
                    path = path
                        .parent()
                        .map(|it| it.to_relative_path_buf())
                        .unwrap_or_else(RelativePathBuf::new);
                }
                let attr_path = &*normalize_attribute_path(attr_path);
                path.push(RelativePath::new(attr_path));
            }
        }
        ModDir { path, root_non_dir_owner: false }
    }

    pub(super) fn resolve_submodule(
        &self,
        db: &impl DefDatabase,
        file_id: HirFileId,
        name: &Name,
        attr_path: Option<&SmolStr>,
    ) -> Result<(FileId, ModDir), RelativePathBuf> {
        let empty_path = RelativePathBuf::default();
        let file_id = file_id.original_file(db);
        let base_dir = {
            let path = db.file_relative_path(file_id);
            path.parent().unwrap_or(&empty_path).join(&self.path)
        };

        let mut candidate_files = Vec::new();
        match attr_path {
            Some(attr) => {
                let base = if self.root_non_dir_owner {
                    base_dir.parent().unwrap_or(&empty_path)
                } else {
                    &base_dir
                };
                candidate_files.push(base.join(&*normalize_attribute_path(attr)))
            }
            None => {
                candidate_files.push(base_dir.join(&format!("{}.rs", name)));
                candidate_files.push(base_dir.join(&format!("{}/mod.rs", name)));
            }
        };

        let source_root_id = db.file_source_root(file_id);
        let source_root = db.source_root(source_root_id);
        for candidate in candidate_files.iter() {
            let candidate = candidate.normalize();
            if let Some(file_id) = source_root.file_by_relative_path(&candidate) {
                let mut root_non_dir_owner = false;
                let mut mod_path = RelativePathBuf::new();
                if !(candidate.ends_with("mod.rs") || attr_path.is_some()) {
                    root_non_dir_owner = true;
                    mod_path.push(&name.to_string());
                }
                return Ok((file_id, ModDir { path: mod_path, root_non_dir_owner }));
            }
        }
        let suggestion = candidate_files.first().unwrap();
        Err(base_dir.join(suggestion))
    }
}

fn normalize_attribute_path(file_path: &str) -> Cow<str> {
    let current_dir = "./";
    let windows_path_separator = r#"\"#;
    let current_dir_normalize = if file_path.starts_with(current_dir) {
        &file_path[current_dir.len()..]
    } else {
        file_path
    };
    if current_dir_normalize.contains(windows_path_separator) {
        Cow::Owned(current_dir_normalize.replace(windows_path_separator, "/"))
    } else {
        Cow::Borrowed(current_dir_normalize)
    }
}
