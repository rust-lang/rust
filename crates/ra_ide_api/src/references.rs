use hir::{source_binder, Either, ModuleSource};
use ra_db::SourceDatabase;
use ra_syntax::{algo::find_node_at_offset, ast, AstNode, SourceFile, SyntaxNode};
use relative_path::{RelativePath, RelativePathBuf};

use crate::{
    db::RootDatabase, FileId, FilePosition, FileRange, FileSystemEdit, NavigationTarget,
    SourceChange, SourceFileEdit, TextRange,
};

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    declaration: NavigationTarget,
    references: Vec<FileRange>,
}

impl ReferenceSearchResult {
    pub fn declaration(&self) -> &NavigationTarget {
        &self.declaration
    }

    pub fn references(&self) -> &[FileRange] {
        &self.references
    }

    /// Total number of references
    /// At least 1 since all valid references should
    /// Have a declaration
    pub fn len(&self) -> usize {
        self.references.len() + 1
    }
}

// allow turning ReferenceSearchResult into an iterator
// over FileRanges
impl IntoIterator for ReferenceSearchResult {
    type Item = FileRange;
    type IntoIter = std::vec::IntoIter<FileRange>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut v = Vec::with_capacity(self.len());
        v.push(FileRange { file_id: self.declaration.file_id(), range: self.declaration.range() });
        v.append(&mut self.references);
        v.into_iter()
    }
}

pub(crate) fn find_all_refs(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<ReferenceSearchResult> {
    let parse = db.parse(position.file_id);
    let (binding, analyzer) = find_binding(db, &parse.tree(), position)?;
    let declaration = NavigationTarget::from_bind_pat(position.file_id, &binding);

    let references = analyzer
        .find_all_refs(&binding)
        .into_iter()
        .map(move |ref_desc| FileRange { file_id: position.file_id, range: ref_desc.range })
        .collect::<Vec<_>>();

    return Some(ReferenceSearchResult { declaration, references });

    fn find_binding<'a>(
        db: &RootDatabase,
        source_file: &SourceFile,
        position: FilePosition,
    ) -> Option<(ast::BindPat, hir::SourceAnalyzer)> {
        let syntax = source_file.syntax();
        if let Some(binding) = find_node_at_offset::<ast::BindPat>(syntax, position.offset) {
            let analyzer = hir::SourceAnalyzer::new(db, position.file_id, binding.syntax(), None);
            return Some((binding, analyzer));
        };
        let name_ref = find_node_at_offset::<ast::NameRef>(syntax, position.offset)?;
        let analyzer = hir::SourceAnalyzer::new(db, position.file_id, name_ref.syntax(), None);
        let resolved = analyzer.resolve_local_name(&name_ref)?;
        if let Either::A(ptr) = resolved.ptr() {
            if let ast::PatKind::BindPat(binding) = ptr.to_node(source_file.syntax()).kind() {
                return Some((binding, analyzer));
            }
        }
        None
    }
}

pub(crate) fn rename(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let parse = db.parse(position.file_id);
    if let Some((ast_name, ast_module)) =
        find_name_and_module_at_offset(parse.tree().syntax(), position)
    {
        rename_mod(db, &ast_name, &ast_module, position, new_name)
    } else {
        rename_reference(db, position, new_name)
    }
}

fn find_name_and_module_at_offset(
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<(ast::Name, ast::Module)> {
    let ast_name = find_node_at_offset::<ast::Name>(syntax, position.offset)?;
    let ast_module = ast::Module::cast(ast_name.syntax().parent()?)?;
    Some((ast_name, ast_module))
}

fn source_edit_from_fileid_range(
    file_id: FileId,
    range: TextRange,
    new_name: &str,
) -> SourceFileEdit {
    SourceFileEdit {
        file_id,
        edit: {
            let mut builder = ra_text_edit::TextEditBuilder::default();
            builder.replace(range, new_name.into());
            builder.finish()
        },
    }
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
    if let Some(module) =
        source_binder::module_from_declaration(db, position.file_id, ast_module.clone())
    {
        let src = module.definition_source(db);
        let file_id = src.file_id.as_original_file();
        match src.ast {
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
            builder.replace(ast_name.syntax().text_range(), new_name.into());
            builder.finish()
        },
    };
    source_file_edits.push(edit);

    Some(SourceChange::from_edits("rename", source_file_edits, file_system_edits))
}

fn rename_reference(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let refs = find_all_refs(db, position)?;

    let edit = refs
        .into_iter()
        .map(|range| source_edit_from_fileid_range(range.file_id, range.range, new_name))
        .collect::<Vec<_>>();

    if edit.is_empty() {
        return None;
    }

    Some(SourceChange::source_file_edits("rename", edit))
}

#[cfg(test)]
mod tests {
    use crate::{
        mock_analysis::analysis_and_position, mock_analysis::single_file_with_position, FileId,
        ReferenceSearchResult,
    };
    use insta::assert_debug_snapshot_matches;
    use test_utils::assert_eq_text;

    #[test]
    fn test_find_all_refs_for_local() {
        let code = r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 5);
    }

    #[test]
    fn test_find_all_refs_for_param_inside() {
        let code = r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_find_all_refs_for_fn_param() {
        let code = r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 2);
    }

    fn get_all_refs(text: &str) -> ReferenceSearchResult {
        let (analysis, position) = single_file_with_position(text);
        analysis.find_all_refs(position).unwrap().unwrap()
    }

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
        assert_debug_snapshot_matches!(&source_change,
@r#"Some(
    SourceChange {
        label: "rename",
        source_file_edits: [
            SourceFileEdit {
                file_id: FileId(
                    2,
                ),
                edit: TextEdit {
                    atoms: [
                        AtomTextEdit {
                            delete: [4; 7),
                            insert: "foo2",
                        },
                    ],
                },
            },
        ],
        file_system_edits: [
            MoveFile {
                src: FileId(
                    3,
                ),
                dst_source_root: SourceRootId(
                    0,
                ),
                dst_path: "bar/foo2.rs",
            },
        ],
        cursor_position: None,
    },
)"#);
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
        assert_debug_snapshot_matches!(&source_change,
        @r###"Some(
    SourceChange {
        label: "rename",
        source_file_edits: [
            SourceFileEdit {
                file_id: FileId(
                    1,
                ),
                edit: TextEdit {
                    atoms: [
                        AtomTextEdit {
                            delete: [4; 7),
                            insert: "foo2",
                        },
                    ],
                },
            },
        ],
        file_system_edits: [
            MoveFile {
                src: FileId(
                    2,
                ),
                dst_source_root: SourceRootId(
                    0,
                ),
                dst_path: "foo2/mod.rs",
            },
        ],
        cursor_position: None,
    },
)"###
               );
    }

    fn test_rename(text: &str, new_name: &str, expected: &str) {
        let (analysis, position) = single_file_with_position(text);
        let source_change = analysis.rename(position, new_name).unwrap();
        let mut text_edit_builder = ra_text_edit::TextEditBuilder::default();
        let mut file_id: Option<FileId> = None;
        if let Some(change) = source_change {
            for edit in change.source_file_edits {
                file_id = Some(edit.file_id);
                for atom in edit.edit.as_atoms() {
                    text_edit_builder.replace(atom.delete, atom.insert.clone());
                }
            }
        }
        let result =
            text_edit_builder.finish().apply(&*analysis.file_text(file_id.unwrap()).unwrap());
        assert_eq_text!(expected, &*result);
    }
}
