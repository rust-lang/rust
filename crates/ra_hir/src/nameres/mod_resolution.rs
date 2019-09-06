//! This module resolves `mod foo;` declaration to file.

use std::{borrow::Cow, sync::Arc};

use ra_db::{FileId, SourceRoot};
use ra_syntax::SmolStr;
use relative_path::RelativePathBuf;

use crate::{DefDatabase, HirFileId, Name};

#[derive(Clone, Copy)]
pub(super) struct ParentModule<'a> {
    pub(super) name: &'a Name,
    pub(super) attr_path: Option<&'a SmolStr>,
}

impl<'a> ParentModule<'a> {
    fn attribute_path(&self) -> Option<&SmolStr> {
        self.attr_path.filter(|p| !p.is_empty())
    }
}

pub(super) fn resolve_submodule(
    db: &impl DefDatabase,
    file_id: HirFileId,
    name: &Name,
    is_root: bool,
    attr_path: Option<&SmolStr>,
    parent_module: Option<ParentModule<'_>>,
) -> Result<FileId, RelativePathBuf> {
    let file_id = file_id.original_file(db);
    let source_root_id = db.file_source_root(file_id);
    let path = db.file_relative_path(file_id);
    let root = RelativePathBuf::default();
    let dir_path = path.parent().unwrap_or(&root);
    let mod_name = path.file_stem().unwrap_or("unknown");

    let resolve_mode = match (attr_path.filter(|p| !p.is_empty()), parent_module) {
        (Some(file_path), Some(parent_module)) => {
            let file_path = normalize_attribute_path(file_path);
            match parent_module.attribute_path() {
                Some(parent_module_attr_path) => {
                    let path = dir_path
                        .join(format!(
                            "{}/{}",
                            normalize_attribute_path(parent_module_attr_path),
                            file_path
                        ))
                        .normalize();
                    ResolutionMode::InlineModuleWithAttributePath(
                        InsideInlineModuleMode::WithAttributePath(path),
                    )
                }
                None => {
                    let path =
                        dir_path.join(format!("{}/{}", parent_module.name, file_path)).normalize();
                    ResolutionMode::InsideInlineModule(InsideInlineModuleMode::WithAttributePath(
                        path,
                    ))
                }
            }
        }
        (None, Some(parent_module)) => match parent_module.attribute_path() {
            Some(parent_module_attr_path) => {
                let path = dir_path.join(format!(
                    "{}/{}.rs",
                    normalize_attribute_path(parent_module_attr_path),
                    name
                ));
                ResolutionMode::InlineModuleWithAttributePath(InsideInlineModuleMode::File(path))
            }
            None => {
                let path = dir_path.join(format!("{}/{}.rs", parent_module.name, name));
                ResolutionMode::InsideInlineModule(InsideInlineModuleMode::File(path))
            }
        },
        (Some(file_path), None) => {
            let file_path = normalize_attribute_path(file_path);
            let path = dir_path.join(file_path.as_ref()).normalize();
            ResolutionMode::OutOfLine(OutOfLineMode::WithAttributePath(path))
        }
        (None, None) => {
            let is_dir_owner = is_root || mod_name == "mod";
            if is_dir_owner {
                let file_mod = dir_path.join(format!("{}.rs", name));
                let dir_mod = dir_path.join(format!("{}/mod.rs", name));
                ResolutionMode::OutOfLine(OutOfLineMode::RootOrModRs {
                    file: file_mod,
                    directory: dir_mod,
                })
            } else {
                let path = dir_path.join(format!("{}/{}.rs", mod_name, name));
                ResolutionMode::OutOfLine(OutOfLineMode::FileInDirectory(path))
            }
        }
    };

    resolve_mode.resolve(db.source_root(source_root_id))
}

fn normalize_attribute_path(file_path: &SmolStr) -> Cow<str> {
    let current_dir = "./";
    let windows_path_separator = r#"\"#;
    let current_dir_normalize = if file_path.starts_with(current_dir) {
        &file_path[current_dir.len()..]
    } else {
        file_path.as_str()
    };
    if current_dir_normalize.contains(windows_path_separator) {
        Cow::Owned(current_dir_normalize.replace(windows_path_separator, "/"))
    } else {
        Cow::Borrowed(current_dir_normalize)
    }
}

enum OutOfLineMode {
    RootOrModRs { file: RelativePathBuf, directory: RelativePathBuf },
    FileInDirectory(RelativePathBuf),
    WithAttributePath(RelativePathBuf),
}

impl OutOfLineMode {
    pub fn resolve(&self, source_root: Arc<SourceRoot>) -> Result<FileId, RelativePathBuf> {
        match self {
            OutOfLineMode::RootOrModRs { file, directory } => match source_root.files.get(file) {
                None => resolve_simple_path(source_root, directory).map_err(|_| file.clone()),
                file_id => resolve_find_result(file_id, file),
            },
            OutOfLineMode::FileInDirectory(path) => resolve_simple_path(source_root, path),
            OutOfLineMode::WithAttributePath(path) => resolve_simple_path(source_root, path),
        }
    }
}

enum InsideInlineModuleMode {
    File(RelativePathBuf),
    WithAttributePath(RelativePathBuf),
}

impl InsideInlineModuleMode {
    pub fn resolve(&self, source_root: Arc<SourceRoot>) -> Result<FileId, RelativePathBuf> {
        match self {
            InsideInlineModuleMode::File(path) => resolve_simple_path(source_root, path),
            InsideInlineModuleMode::WithAttributePath(path) => {
                resolve_simple_path(source_root, path)
            }
        }
    }
}

enum ResolutionMode {
    OutOfLine(OutOfLineMode),
    InsideInlineModule(InsideInlineModuleMode),
    InlineModuleWithAttributePath(InsideInlineModuleMode),
}

impl ResolutionMode {
    pub fn resolve(&self, source_root: Arc<SourceRoot>) -> Result<FileId, RelativePathBuf> {
        use self::ResolutionMode::*;

        match self {
            OutOfLine(mode) => mode.resolve(source_root),
            InsideInlineModule(mode) => mode.resolve(source_root),
            InlineModuleWithAttributePath(mode) => mode.resolve(source_root),
        }
    }
}

fn resolve_simple_path(
    source_root: Arc<SourceRoot>,
    path: &RelativePathBuf,
) -> Result<FileId, RelativePathBuf> {
    resolve_find_result(source_root.files.get(path), path)
}

fn resolve_find_result(
    file_id: Option<&FileId>,
    path: &RelativePathBuf,
) -> Result<FileId, RelativePathBuf> {
    match file_id {
        Some(file_id) => Ok(file_id.clone()),
        None => Err(path.clone()),
    }
}
