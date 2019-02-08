use relative_path::{RelativePath, RelativePathBuf};
use hir::{ModuleSource, source_binder};
use ra_db::{FileId, SourceDatabase};
use ra_syntax::{
    AstNode, SyntaxNode, TextRange, SourceFile,
    ast::{self, NameOwner},
    algo::find_node_at_offset,
};

use crate::{
    db::RootDatabase,
    FilePosition,
    FileSystemEdit,
    SourceChange,
    SourceFileEdit,
};

pub(crate) fn find_all_refs(db: &RootDatabase, position: FilePosition) -> Vec<(FileId, TextRange)> {
    let file = db.parse(position.file_id);
    // Find the binding associated with the offset
    let (binding, descr) = match find_binding(db, &file, position) {
        None => return Vec::new(),
        Some(it) => it,
    };

    let mut ret = binding
        .name()
        .into_iter()
        .map(|name| (position.file_id, name.syntax().range()))
        .collect::<Vec<_>>();
    ret.extend(
        descr
            .scopes(db)
            .find_all_refs(binding)
            .into_iter()
            .map(|ref_desc| (position.file_id, ref_desc.range)),
    );

    return ret;

    fn find_binding<'a>(
        db: &RootDatabase,
        source_file: &'a SourceFile,
        position: FilePosition,
    ) -> Option<(&'a ast::BindPat, hir::Function)> {
        let syntax = source_file.syntax();
        if let Some(binding) = find_node_at_offset::<ast::BindPat>(syntax, position.offset) {
            let descr =
                source_binder::function_from_child_node(db, position.file_id, binding.syntax())?;
            return Some((binding, descr));
        };
        let name_ref = find_node_at_offset::<ast::NameRef>(syntax, position.offset)?;
        let descr =
            source_binder::function_from_child_node(db, position.file_id, name_ref.syntax())?;
        let scope = descr.scopes(db);
        let resolved = scope.resolve_local_name(name_ref)?;
        let resolved = resolved.ptr().to_node(source_file);
        let binding = find_node_at_offset::<ast::BindPat>(syntax, resolved.range().end())?;
        Some((binding, descr))
    }
}

pub(crate) fn rename(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let source_file = db.parse(position.file_id);
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
    let ast_name_parent = ast::Module::cast(ast_name?.syntax().parent()?);

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
    if let Some(module) = source_binder::module_from_declaration(db, position.file_id, &ast_module)
    {
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

    Some(SourceChange {
        label: "rename".to_string(),
        source_file_edits,
        file_system_edits,
        cursor_position: None,
    })
}

fn rename_reference(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let edit = find_all_refs(db, position)
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

    Some(SourceChange {
        label: "rename".to_string(),
        source_file_edits: edit,
        file_system_edits: Vec::new(),
        cursor_position: None,
    })
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;
    use test_utils::assert_eq_text;
    use crate::{
        mock_analysis::single_file_with_position,
        mock_analysis::analysis_and_position,
        FileId
};

    #[test]
    fn test_rename_for_local() {
        test_rename(
            r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#,
            "k",
            r#"
    fn main() {
        let mut k = 1;
        let j = 1;
        k = k + j;

        {
            k = 0;
        }

        k = 5;
    }"#,
        );
    }

    #[test]
    fn test_rename_for_param_inside() {
        test_rename(
            r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#,
            "j",
            r#"
    fn foo(j : u32) -> u32 {
        j
    }"#,
        );
    }

    #[test]
    fn test_rename_refs_for_fn_param() {
        test_rename(
            r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#,
            "new_name",
            r#"
    fn foo(new_name : u32) -> u32 {
        new_name
    }"#,
        );
    }

    #[test]
    fn test_rename_for_mut_param() {
        test_rename(
            r#"
    fn foo(mut i<|> : u32) -> u32 {
        i
    }"#,
            "new_name",
            r#"
    fn foo(mut new_name : u32) -> u32 {
        new_name
    }"#,
        );
    }

    #[test]
    fn test_rename_mod() {
        let (analysis, position) = analysis_and_position(
            "
            //- /lib.rs
            mod bar;

            //- /bar.rs
            mod foo<|>;

            //- /bar/foo.rs
            // emtpy
            ",
        );
        let new_name = "foo2";
        let source_change = analysis.rename(position, new_name).unwrap();
        assert_debug_snapshot_matches!("rename_mod", &source_change);
    }

    #[test]
    fn test_rename_mod_in_dir() {
        let (analysis, position) = analysis_and_position(
            "
            //- /lib.rs
            mod fo<|>o;
            //- /foo/mod.rs
            // emtpy
            ",
        );
        let new_name = "foo2";
        let source_change = analysis.rename(position, new_name).unwrap();
        assert_debug_snapshot_matches!("rename_mod_in_dir", &source_change);
    }

    fn test_rename(text: &str, new_name: &str, expected: &str) {
        let (analysis, position) = single_file_with_position(text);
        let source_change = analysis.rename(position, new_name).unwrap();
        let mut text_edit_bulder = ra_text_edit::TextEditBuilder::default();
        let mut file_id: Option<FileId> = None;
        if let Some(change) = source_change {
            for edit in change.source_file_edits {
                file_id = Some(edit.file_id);
                for atom in edit.edit.as_atoms() {
                    text_edit_bulder.replace(atom.delete, atom.insert.clone());
                }
            }
        }
        let result = text_edit_bulder
            .finish()
            .apply(&*analysis.file_text(file_id.unwrap()));
        assert_eq_text!(expected, &*result);
    }
}
