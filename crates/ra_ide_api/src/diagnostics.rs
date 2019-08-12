use std::cell::RefCell;

use hir::{
    diagnostics::{AstDiagnostic, Diagnostic as _, DiagnosticSink},
    source_binder,
};
use itertools::Itertools;
use ra_assists::ast_editor::{AstBuilder, AstEditor};
use ra_db::SourceDatabase;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, NamedField},
    Location, SyntaxNode, TextRange, T,
};
use ra_text_edit::{TextEdit, TextEditBuilder};

use crate::{db::RootDatabase, Diagnostic, FileId, FileSystemEdit, SourceChange, SourceFileEdit};

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    Error,
    WeakWarning,
}

pub(crate) fn diagnostics(db: &RootDatabase, file_id: FileId) -> Vec<Diagnostic> {
    let _p = profile("diagnostics");
    let parse = db.parse(file_id);
    let mut res = Vec::new();

    res.extend(parse.errors().iter().map(|err| Diagnostic {
        range: location_to_range(err.location()),
        message: format!("Syntax Error: {}", err),
        severity: Severity::Error,
        fix: None,
    }));

    for node in parse.tree().syntax().descendants() {
        check_unnecessary_braces_in_use_statement(&mut res, file_id, &node);
        check_struct_shorthand_initialization(&mut res, file_id, &node);
    }
    let res = RefCell::new(res);
    let mut sink = DiagnosticSink::new(|d| {
        res.borrow_mut().push(Diagnostic {
            message: d.message(),
            range: d.highlight_range(),
            severity: Severity::Error,
            fix: None,
        })
    })
    .on::<hir::diagnostics::UnresolvedModule, _>(|d| {
        let source_root = db.file_source_root(d.source().file_id.original_file(db));
        let create_file = FileSystemEdit::CreateFile { source_root, path: d.candidate.clone() };
        let fix = SourceChange::file_system_edit("create module", create_file);
        res.borrow_mut().push(Diagnostic {
            range: d.highlight_range(),
            message: d.message(),
            severity: Severity::Error,
            fix: Some(fix),
        })
    })
    .on::<hir::diagnostics::MissingFields, _>(|d| {
        let node = d.ast(db);
        let mut ast_editor = AstEditor::new(node);
        for f in d.missed_fields.iter() {
            ast_editor.append_field(&AstBuilder::<NamedField>::from_name(f));
        }

        let mut builder = TextEditBuilder::default();
        ast_editor.into_text_edit(&mut builder);
        let fix =
            SourceChange::source_file_edit_from("fill struct fields", file_id, builder.finish());
        res.borrow_mut().push(Diagnostic {
            range: d.highlight_range(),
            message: d.message(),
            severity: Severity::Error,
            fix: Some(fix),
        })
    });
    if let Some(m) = source_binder::module_from_file_id(db, file_id) {
        m.diagnostics(db, &mut sink);
    };
    drop(sink);
    res.into_inner()
}
fn location_to_range(location: Location) -> TextRange {
    match location {
        Location::Offset(offset) => TextRange::offset_len(offset, 1.into()),
        Location::Range(range) => range,
    }
}

fn check_unnecessary_braces_in_use_statement(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let use_tree_list = ast::UseTreeList::cast(node.clone())?;
    if let Some((single_use_tree,)) = use_tree_list.use_trees().collect_tuple() {
        let range = use_tree_list.syntax().text_range();
        let edit =
            text_edit_for_remove_unnecessary_braces_with_self_in_use_statement(&single_use_tree)
                .unwrap_or_else(|| {
                    let to_replace = single_use_tree.syntax().text().to_string();
                    let mut edit_builder = TextEditBuilder::default();
                    edit_builder.delete(range);
                    edit_builder.insert(range.start(), to_replace);
                    edit_builder.finish()
                });

        acc.push(Diagnostic {
            range,
            message: "Unnecessary braces in use statement".to_string(),
            severity: Severity::WeakWarning,
            fix: Some(SourceChange::source_file_edit(
                "Remove unnecessary braces",
                SourceFileEdit { file_id, edit },
            )),
        });
    }

    Some(())
}

fn text_edit_for_remove_unnecessary_braces_with_self_in_use_statement(
    single_use_tree: &ast::UseTree,
) -> Option<TextEdit> {
    let use_tree_list_node = single_use_tree.syntax().parent()?;
    if single_use_tree.path()?.segment()?.syntax().first_child_or_token()?.kind() == T![self] {
        let start = use_tree_list_node.prev_sibling_or_token()?.text_range().start();
        let end = use_tree_list_node.text_range().end();
        let range = TextRange::from_to(start, end);
        let mut edit_builder = TextEditBuilder::default();
        edit_builder.delete(range);
        return Some(edit_builder.finish());
    }
    None
}

fn check_struct_shorthand_initialization(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let struct_lit = ast::StructLit::cast(node.clone())?;
    let named_field_list = struct_lit.named_field_list()?;
    for named_field in named_field_list.fields() {
        if let (Some(name_ref), Some(expr)) = (named_field.name_ref(), named_field.expr()) {
            let field_name = name_ref.syntax().text().to_string();
            let field_expr = expr.syntax().text().to_string();
            if field_name == field_expr {
                let mut edit_builder = TextEditBuilder::default();
                edit_builder.delete(named_field.syntax().text_range());
                edit_builder.insert(named_field.syntax().text_range().start(), field_name);
                let edit = edit_builder.finish();

                acc.push(Diagnostic {
                    range: named_field.syntax().text_range(),
                    message: "Shorthand struct initialization".to_string(),
                    severity: Severity::WeakWarning,
                    fix: Some(SourceChange::source_file_edit(
                        "use struct shorthand initialization",
                        SourceFileEdit { file_id, edit },
                    )),
                });
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;
    use ra_syntax::SourceFile;
    use test_utils::assert_eq_text;

    use crate::mock_analysis::single_file;

    use super::*;

    type DiagnosticChecker = fn(&mut Vec<Diagnostic>, FileId, &SyntaxNode) -> Option<()>;

    fn check_not_applicable(code: &str, func: DiagnosticChecker) {
        let parse = SourceFile::parse(code);
        let mut diagnostics = Vec::new();
        for node in parse.tree().syntax().descendants() {
            func(&mut diagnostics, FileId(0), &node);
        }
        assert!(diagnostics.is_empty());
    }

    fn check_apply(before: &str, after: &str, func: DiagnosticChecker) {
        let parse = SourceFile::parse(before);
        let mut diagnostics = Vec::new();
        for node in parse.tree().syntax().descendants() {
            func(&mut diagnostics, FileId(0), &node);
        }
        let diagnostic =
            diagnostics.pop().unwrap_or_else(|| panic!("no diagnostics for:\n{}\n", before));
        let mut fix = diagnostic.fix.unwrap();
        let edit = fix.source_file_edits.pop().unwrap().edit;
        let actual = edit.apply(&before);
        assert_eq_text!(after, &actual);
    }

    fn check_apply_diagnostic_fix(before: &str, after: &str) {
        let (analysis, file_id) = single_file(before);
        let diagnostic = analysis.diagnostics(file_id).unwrap().pop().unwrap();
        let mut fix = diagnostic.fix.unwrap();
        let edit = fix.source_file_edits.pop().unwrap().edit;
        let actual = edit.apply(&before);
        assert_eq_text!(after, &actual);
    }

    fn check_no_diagnostic(content: &str) {
        let (analysis, file_id) = single_file(content);
        let diagnostics = analysis.diagnostics(file_id).unwrap();
        assert_eq!(diagnostics.len(), 0);
    }

    #[test]
    fn test_fill_struct_fields_empty() {
        let before = r"
            struct TestStruct {
                one: i32,
                two: i64,
            }

            fn test_fn() {
                let s = TestStruct{};
            }
        ";
        let after = r"
            struct TestStruct {
                one: i32,
                two: i64,
            }

            fn test_fn() {
                let s = TestStruct{ one: (), two: ()};
            }
        ";
        check_apply_diagnostic_fix(before, after);
    }

    #[test]
    fn test_fill_struct_fields_partial() {
        let before = r"
            struct TestStruct {
                one: i32,
                two: i64,
            }

            fn test_fn() {
                let s = TestStruct{ two: 2 };
            }
        ";
        let after = r"
            struct TestStruct {
                one: i32,
                two: i64,
            }

            fn test_fn() {
                let s = TestStruct{ two: 2, one: () };
            }
        ";
        check_apply_diagnostic_fix(before, after);
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic() {
        let content = r"
            struct TestStruct {
                one: i32,
                two: i64,
            }

            fn test_fn() {
                let one = 1;
                let s = TestStruct{ one, two: 2 };
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic_on_spread() {
        let content = r"
            struct TestStruct {
                one: i32,
                two: i64,
            }

            fn test_fn() {
                let one = 1;
                let s = TestStruct{ ..a };
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn test_unresolved_module_diagnostic() {
        let (analysis, file_id) = single_file("mod foo;");
        let diagnostics = analysis.diagnostics(file_id).unwrap();
        assert_debug_snapshot_matches!(diagnostics, @r###"
       ⋮[
       ⋮    Diagnostic {
       ⋮        message: "unresolved module",
       ⋮        range: [0; 8),
       ⋮        fix: Some(
       ⋮            SourceChange {
       ⋮                label: "create module",
       ⋮                source_file_edits: [],
       ⋮                file_system_edits: [
       ⋮                    CreateFile {
       ⋮                        source_root: SourceRootId(
       ⋮                            0,
       ⋮                        ),
       ⋮                        path: "foo.rs",
       ⋮                    },
       ⋮                ],
       ⋮                cursor_position: None,
       ⋮            },
       ⋮        ),
       ⋮        severity: Error,
       ⋮    },
       ⋮]
        "###);
    }

    #[test]
    fn test_check_unnecessary_braces_in_use_statement() {
        check_not_applicable(
            "
            use a;
            use a::{c, d::e};
        ",
            check_unnecessary_braces_in_use_statement,
        );
        check_apply("use {b};", "use b;", check_unnecessary_braces_in_use_statement);
        check_apply("use a::{c};", "use a::c;", check_unnecessary_braces_in_use_statement);
        check_apply("use a::{self};", "use a;", check_unnecessary_braces_in_use_statement);
        check_apply(
            "use a::{c, d::{e}};",
            "use a::{c, d::e};",
            check_unnecessary_braces_in_use_statement,
        );
    }

    #[test]
    fn test_check_struct_shorthand_initialization() {
        check_not_applicable(
            r#"
            struct A {
                a: &'static str
            }

            fn main() {
                A {
                    a: "hello"
                }
            }
        "#,
            check_struct_shorthand_initialization,
        );

        check_apply(
            r#"
struct A {
    a: &'static str
}

fn main() {
    let a = "haha";
    A {
        a: a
    }
}
        "#,
            r#"
struct A {
    a: &'static str
}

fn main() {
    let a = "haha";
    A {
        a
    }
}
        "#,
            check_struct_shorthand_initialization,
        );

        check_apply(
            r#"
struct A {
    a: &'static str,
    b: &'static str
}

fn main() {
    let a = "haha";
    let b = "bb";
    A {
        a: a,
        b
    }
}
        "#,
            r#"
struct A {
    a: &'static str,
    b: &'static str
}

fn main() {
    let a = "haha";
    let b = "bb";
    A {
        a,
        b
    }
}
        "#,
            check_struct_shorthand_initialization,
        );
    }
}
