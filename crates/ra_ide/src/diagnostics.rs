//! Collects diagnostics & fixits  for a single file.
//!
//! The tricky bit here is that diagnostics are produced by hir in terms of
//! macro-expanded files, but we need to present them to the users in terms of
//! original files. So we need to map the ranges.

use std::cell::RefCell;

use hir::{
    diagnostics::{AstDiagnostic, Diagnostic as _, DiagnosticSink},
    HasSource, HirDisplay, Semantics, VariantDef,
};
use itertools::Itertools;
use ra_db::SourceDatabase;
use ra_ide_db::RootDatabase;
use ra_prof::profile;
use ra_syntax::{
    algo,
    ast::{self, edit::IndentLevel, make, AstNode},
    SyntaxNode, TextRange, T,
};
use ra_text_edit::{TextEdit, TextEditBuilder};

use crate::{Diagnostic, FileId, FileSystemEdit, Fix, SourceFileEdit};

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    Error,
    WeakWarning,
}

pub(crate) fn diagnostics(db: &RootDatabase, file_id: FileId) -> Vec<Diagnostic> {
    let _p = profile("diagnostics");
    let sema = Semantics::new(db);
    let parse = db.parse(file_id);
    let mut res = Vec::new();

    res.extend(parse.errors().iter().map(|err| Diagnostic {
        range: err.range(),
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
            range: sema.diagnostics_range(d).range,
            severity: Severity::Error,
            fix: None,
        })
    })
    .on::<hir::diagnostics::UnresolvedModule, _>(|d| {
        let original_file = d.source().file_id.original_file(db);
        let fix = Fix::new(
            "Create module",
            FileSystemEdit::CreateFile { anchor: original_file, dst: d.candidate.clone() }.into(),
        );
        res.borrow_mut().push(Diagnostic {
            range: sema.diagnostics_range(d).range,
            message: d.message(),
            severity: Severity::Error,
            fix: Some(fix),
        })
    })
    .on::<hir::diagnostics::MissingFields, _>(|d| {
        // Note that although we could add a diagnostics to
        // fill the missing tuple field, e.g :
        // `struct A(usize);`
        // `let a = A { 0: () }`
        // but it is uncommon usage and it should not be encouraged.
        let fix = if d.missed_fields.iter().any(|it| it.as_tuple_index().is_some()) {
            None
        } else {
            let mut field_list = d.ast(db);
            for f in d.missed_fields.iter() {
                let field =
                    make::record_field(make::name_ref(&f.to_string()), Some(make::expr_unit()));
                field_list = field_list.append_field(&field);
            }

            let edit = {
                let mut builder = TextEditBuilder::default();
                algo::diff(&d.ast(db).syntax(), &field_list.syntax()).into_text_edit(&mut builder);
                builder.finish()
            };
            Some(Fix::new("Fill struct fields", SourceFileEdit { file_id, edit }.into()))
        };

        res.borrow_mut().push(Diagnostic {
            range: sema.diagnostics_range(d).range,
            message: d.message(),
            severity: Severity::Error,
            fix,
        })
    })
    .on::<hir::diagnostics::MissingMatchArms, _>(|d| {
        res.borrow_mut().push(Diagnostic {
            range: sema.diagnostics_range(d).range,
            message: d.message(),
            severity: Severity::Error,
            fix: None,
        })
    })
    .on::<hir::diagnostics::MissingOkInTailExpr, _>(|d| {
        let node = d.ast(db);
        let replacement = format!("Ok({})", node.syntax());
        let edit = TextEdit::replace(node.syntax().text_range(), replacement);
        let source_change = SourceFileEdit { file_id, edit }.into();
        let fix = Fix::new("Wrap with ok", source_change);
        res.borrow_mut().push(Diagnostic {
            range: sema.diagnostics_range(d).range,
            message: d.message(),
            severity: Severity::Error,
            fix: Some(fix),
        })
    })
    .on::<hir::diagnostics::NoSuchField, _>(|d| {
        res.borrow_mut().push(Diagnostic {
            range: sema.diagnostics_range(d).range,
            message: d.message(),
            severity: Severity::Error,
            fix: missing_struct_field_fix(&sema, file_id, d),
        })
    });

    if let Some(m) = sema.to_module_def(file_id) {
        m.diagnostics(db, &mut sink);
    };
    drop(sink);
    res.into_inner()
}

fn missing_struct_field_fix(
    sema: &Semantics<RootDatabase>,
    file_id: FileId,
    d: &hir::diagnostics::NoSuchField,
) -> Option<Fix> {
    let record_expr = sema.ast(d);

    let record_lit = ast::RecordLit::cast(record_expr.syntax().parent()?.parent()?)?;
    let def_id = sema.resolve_variant(record_lit)?;
    let module;
    let record_fields = match VariantDef::from(def_id) {
        VariantDef::Struct(s) => {
            module = s.module(sema.db);
            let source = s.source(sema.db);
            let fields = source.value.field_def_list()?;
            record_field_def_list(fields)?
        }
        VariantDef::Union(u) => {
            module = u.module(sema.db);
            let source = u.source(sema.db);
            source.value.record_field_def_list()?
        }
        VariantDef::EnumVariant(e) => {
            module = e.module(sema.db);
            let source = e.source(sema.db);
            let fields = source.value.field_def_list()?;
            record_field_def_list(fields)?
        }
    };

    let new_field_type = sema.type_of_expr(&record_expr.expr()?)?;
    if new_field_type.is_unknown() {
        return None;
    }
    let new_field = make::record_field_def(
        record_expr.field_name()?,
        make::type_ref(&new_field_type.display_source_code(sema.db, module.into()).ok()?),
    );

    let last_field = record_fields.fields().last()?;
    let last_field_syntax = last_field.syntax();
    let indent = IndentLevel::from_node(last_field_syntax);

    let mut new_field = format!("\n{}{}", indent, new_field);

    let needs_comma = !last_field_syntax.to_string().ends_with(",");
    if needs_comma {
        new_field = format!(",{}", new_field);
    }

    let source_change = SourceFileEdit {
        file_id,
        edit: TextEdit::insert(last_field_syntax.text_range().end(), new_field),
    };
    let fix = Fix::new("Create field", source_change.into());
    return Some(fix);

    fn record_field_def_list(field_def_list: ast::FieldDefList) -> Option<ast::RecordFieldDefList> {
        match field_def_list {
            ast::FieldDefList::RecordFieldDefList(it) => Some(it),
            ast::FieldDefList::TupleFieldDefList(_) => None,
        }
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
            fix: Some(Fix::new(
                "Remove unnecessary braces",
                SourceFileEdit { file_id, edit }.into(),
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
        let range = TextRange::new(start, end);
        return Some(TextEdit::delete(range));
    }
    None
}

fn check_struct_shorthand_initialization(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let record_lit = ast::RecordLit::cast(node.clone())?;
    let record_field_list = record_lit.record_field_list()?;
    for record_field in record_field_list.fields() {
        if let (Some(name_ref), Some(expr)) = (record_field.name_ref(), record_field.expr()) {
            let field_name = name_ref.syntax().text().to_string();
            let field_expr = expr.syntax().text().to_string();
            let field_name_is_tup_index = name_ref.as_tuple_field().is_some();
            if field_name == field_expr && !field_name_is_tup_index {
                let mut edit_builder = TextEditBuilder::default();
                edit_builder.delete(record_field.syntax().text_range());
                edit_builder.insert(record_field.syntax().text_range().start(), field_name);
                let edit = edit_builder.finish();

                acc.push(Diagnostic {
                    range: record_field.syntax().text_range(),
                    message: "Shorthand struct initialization".to_string(),
                    severity: Severity::WeakWarning,
                    fix: Some(Fix::new(
                        "Use struct shorthand initialization",
                        SourceFileEdit { file_id, edit }.into(),
                    )),
                });
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use stdx::trim_indent;
    use test_utils::assert_eq_text;

    use crate::mock_analysis::{analysis_and_position, single_file, MockAnalysis};
    use expect::{expect, Expect};

    /// Takes a multi-file input fixture with annotated cursor positions,
    /// and checks that:
    ///  * a diagnostic is produced
    ///  * this diagnostic touches the input cursor position
    ///  * that the contents of the file containing the cursor match `after` after the diagnostic fix is applied
    fn check_fix(ra_fixture_before: &str, ra_fixture_after: &str) {
        let after = trim_indent(ra_fixture_after);

        let (analysis, file_position) = analysis_and_position(ra_fixture_before);
        let diagnostic = analysis.diagnostics(file_position.file_id).unwrap().pop().unwrap();
        let mut fix = diagnostic.fix.unwrap();
        let edit = fix.source_change.source_file_edits.pop().unwrap().edit;
        let target_file_contents = analysis.file_text(file_position.file_id).unwrap();
        let actual = {
            let mut actual = target_file_contents.to_string();
            edit.apply(&mut actual);
            actual
        };

        assert_eq_text!(&after, &actual);
        assert!(
            diagnostic.range.start() <= file_position.offset
                && diagnostic.range.end() >= file_position.offset,
            "diagnostic range {:?} does not touch cursor position {:?}",
            diagnostic.range,
            file_position.offset
        );
    }

    /// Takes a multi-file input fixture with annotated cursor position and checks that no diagnostics
    /// apply to the file containing the cursor.
    fn check_no_diagnostics(ra_fixture: &str) {
        let mock = MockAnalysis::with_files(ra_fixture);
        let files = mock.files().map(|(it, _)| it).collect::<Vec<_>>();
        let analysis = mock.analysis();
        let diagnostics = files
            .into_iter()
            .flat_map(|file_id| analysis.diagnostics(file_id).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 0, "unexpected diagnostics:\n{:#?}", diagnostics);
    }

    fn check_expect(ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = single_file(ra_fixture);
        let diagnostics = analysis.diagnostics(file_id).unwrap();
        expect.assert_debug_eq(&diagnostics)
    }

    #[test]
    fn test_wrap_return_type() {
        check_fix(
            r#"
//- /main.rs
use core::result::Result::{self, Ok, Err};

fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    x / y<|>
}
//- /core/lib.rs
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
"#,
            r#"
use core::result::Result::{self, Ok, Err};

fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_generic_functions() {
        check_fix(
            r#"
//- /main.rs
use core::result::Result::{self, Ok, Err};

fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    <|>x
}
//- /core/lib.rs
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
"#,
            r#"
use core::result::Result::{self, Ok, Err};

fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    Ok(x)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_type_aliases() {
        check_fix(
            r#"
//- /main.rs
use core::result::Result::{self, Ok, Err};

type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    x <|>/ y
}
//- /core/lib.rs
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
"#,
            r#"
use core::result::Result::{self, Ok, Err};

type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_expr_type_does_not_match_ok_type() {
        check_no_diagnostics(
            r#"
//- /main.rs
use core::result::Result::{self, Ok, Err};

fn foo() -> Result<(), i32> { 0 }

//- /core/lib.rs
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_return_type_is_not_result() {
        check_no_diagnostics(
            r#"
//- /main.rs
use core::result::Result::{self, Ok, Err};

enum SomeOtherEnum { Ok(i32), Err(String) }

fn foo() -> SomeOtherEnum { 0 }

//- /core/lib.rs
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_empty() {
        check_fix(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct {<|>};
}
"#,
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct { one: (), two: ()};
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_self() {
        check_fix(
            r#"
struct TestStruct { one: i32 }

impl TestStruct {
    fn test_fn() { let s = Self {<|>}; }
}
"#,
            r#"
struct TestStruct { one: i32 }

impl TestStruct {
    fn test_fn() { let s = Self { one: ()}; }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_enum() {
        check_fix(
            r#"
enum Expr {
    Bin { lhs: Box<Expr>, rhs: Box<Expr> }
}

impl Expr {
    fn new_bin(lhs: Box<Expr>, rhs: Box<Expr>) -> Expr {
        Expr::Bin {<|> }
    }
}
"#,
            r#"
enum Expr {
    Bin { lhs: Box<Expr>, rhs: Box<Expr> }
}

impl Expr {
    fn new_bin(lhs: Box<Expr>, rhs: Box<Expr>) -> Expr {
        Expr::Bin { lhs: (), rhs: () }
    }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_partial() {
        check_fix(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct{ two: 2<|> };
}
"#,
            r"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct{ two: 2, one: () };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic() {
        check_no_diagnostics(
            r"
            struct TestStruct { one: i32, two: i64 }

            fn test_fn() {
                let one = 1;
                let s = TestStruct{ one, two: 2 };
            }
        ",
        );
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic_on_spread() {
        check_no_diagnostics(
            r"
            struct TestStruct { one: i32, two: i64 }

            fn test_fn() {
                let one = 1;
                let s = TestStruct{ ..a };
            }
        ",
        );
    }

    #[test]
    fn test_unresolved_module_diagnostic() {
        check_expect(
            r#"mod foo;"#,
            expect![[r#"
                [
                    Diagnostic {
                        message: "unresolved module",
                        range: 0..8,
                        severity: Error,
                        fix: Some(
                            Fix {
                                label: "Create module",
                                source_change: SourceChange {
                                    source_file_edits: [],
                                    file_system_edits: [
                                        CreateFile {
                                            anchor: FileId(
                                                1,
                                            ),
                                            dst: "foo.rs",
                                        },
                                    ],
                                    is_snippet: false,
                                },
                            },
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn range_mapping_out_of_macros() {
        // FIXME: this is very wrong, but somewhat tricky to fix.
        check_fix(
            r#"
fn some() {}
fn items() {}
fn here() {}

macro_rules! id { ($($tt:tt)*) => { $($tt)*}; }

fn main() {
    let _x = id![Foo { a: <|>42 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
            r#"
fn {a:42, b: ()} {}
fn items() {}
fn here() {}

macro_rules! id { ($($tt:tt)*) => { $($tt)*}; }

fn main() {
    let _x = id![Foo { a: 42 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
        );
    }

    #[test]
    fn test_check_unnecessary_braces_in_use_statement() {
        check_no_diagnostics(
            r#"
use a;
use a::{c, d::e};
"#,
        );
        check_fix(r#"use {<|>b};"#, r#"use b;"#);
        check_fix(r#"use {b<|>};"#, r#"use b;"#);
        check_fix(r#"use a::{c<|>};"#, r#"use a::c;"#);
        check_fix(r#"use a::{self<|>};"#, r#"use a;"#);
        check_fix(r#"use a::{c, d::{e<|>}};"#, r#"use a::{c, d::e};"#);
    }

    #[test]
    fn test_check_struct_shorthand_initialization() {
        check_no_diagnostics(
            r#"
struct A { a: &'static str }
fn main() { A { a: "hello" } }
"#,
        );
        check_no_diagnostics(
            r#"
struct A(usize);
fn main() { A { 0: 0 } }
"#,
        );

        check_fix(
            r#"
struct A { a: &'static str }
fn main() {
    let a = "haha";
    A { a<|>: a }
}
"#,
            r#"
struct A { a: &'static str }
fn main() {
    let a = "haha";
    A { a }
}
"#,
        );

        check_fix(
            r#"
struct A { a: &'static str, b: &'static str }
fn main() {
    let a = "haha";
    let b = "bb";
    A { a<|>: a, b }
}
"#,
            r#"
struct A { a: &'static str, b: &'static str }
fn main() {
    let a = "haha";
    let b = "bb";
    A { a, b }
}
"#,
        );
    }

    #[test]
    fn test_add_field_from_usage() {
        check_fix(
            r"
fn main() {
    Foo { bar: 3, baz<|>: false};
}
struct Foo {
    bar: i32
}
",
            r"
fn main() {
    Foo { bar: 3, baz: false};
}
struct Foo {
    bar: i32,
    baz: bool
}
",
        )
    }
}
