use relative_path::RelativePathBuf;

use hir::{
    self, ModuleSource, source_binder::module_from_declaration,
};
use ra_syntax::{
    algo::find_node_at_offset,
    ast,
    AstNode,
    SyntaxNode
};

use crate::{
    db::RootDatabase,
    FilePosition,
    FileSystemEdit,
    SourceChange,
    SourceFileEdit,
};
use ra_db::{FilesDatabase, SyntaxDatabase};
use relative_path::RelativePath;

pub(crate) fn rename(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let source_file = db.source_file(position.file_id);
    let syntax = source_file.syntax();

    if let Some((ast_name, ast_module)) = find_name_and_module_at_offset(syntax, position) {
        rename_mod(db, ast_name, ast_module, position, new_name)
    } else {
        rename_reference(db, position, new_name)
    }
}

fn find_name_and_module_at_offset(
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<(&ast::Name, &ast::Module)> {
    let ast_name = find_node_at_offset::<ast::Name>(syntax, position.offset);
    let ast_name_parent = ast_name
        .and_then(|n| n.syntax().parent())
        .and_then(|p| ast::Module::cast(p));

    if let (Some(ast_module), Some(name)) = (ast_name_parent, ast_name) {
        return Some((name, ast_module));
    }
    None
}

fn rename_mod(
    db: &RootDatabase,
    ast_name: &ast::Name,
    ast_module: &ast::Module,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let mut source_file_edits = Vec::new();
    let mut file_system_edits = Vec::new();

    if let Some(module) = module_from_declaration(db, position.file_id, &ast_module) {
        let (file_id, module_source) = module.definition_source(db);
        match module_source {
            ModuleSource::SourceFile(..) => {
                let mod_path: RelativePathBuf = db.file_relative_path(file_id);
                // mod is defined in path/to/dir/mod.rs
                let dst_path = if mod_path.file_stem() == Some("mod") {
                    mod_path
                        .parent()
                        .and_then(|p| p.parent())
                        .or_else(|| Some(RelativePath::new("")))
                        .map(|p| p.join(new_name).join("mod.rs"))
                } else {
                    Some(mod_path.with_file_name(new_name).with_extension("rs"))
                };
                if let Some(path) = dst_path {
                    let move_file = FileSystemEdit::MoveFile {
                        src: file_id,
                        dst_source_root: db.file_source_root(position.file_id),
                        dst_path: path,
                    };
                    file_system_edits.push(move_file);
                }
            }
            ModuleSource::Module(..) => {}
        }
    }

    let edit = SourceFileEdit {
        file_id: position.file_id,
        edit: {
            let mut builder = ra_text_edit::TextEditBuilder::default();
            builder.replace(ast_name.syntax().range(), new_name.into());
            builder.finish()
        },
    };
    source_file_edits.push(edit);

    return Some(SourceChange {
        label: "rename".to_string(),
        source_file_edits,
        file_system_edits,
        cursor_position: None,
    });
}

fn rename_reference(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let edit = db
        .find_all_refs(position)
        .iter()
        .map(|(file_id, text_range)| SourceFileEdit {
            file_id: *file_id,
            edit: {
                let mut builder = ra_text_edit::TextEditBuilder::default();
                builder.replace(*text_range, new_name.into());
                builder.finish()
            },
        })
        .collect::<Vec<_>>();
    if edit.is_empty() {
        return None;
    }

    return Some(SourceChange {
        label: "rename".to_string(),
        source_file_edits: edit,
        file_system_edits: Vec::new(),
        cursor_position: None,
    });
}
