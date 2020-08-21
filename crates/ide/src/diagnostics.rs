//! Collects diagnostics & fixits  for a single file.
//!
//! The tricky bit here is that diagnostics are produced by hir in terms of
//! macro-expanded files, but we need to present them to the users in terms of
//! original files. So we need to map the ranges.

mod fixes;

use std::cell::RefCell;

use base_db::SourceDatabase;
use hir::{diagnostics::DiagnosticSinkBuilder, Semantics};
use ide_db::RootDatabase;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, AstNode},
    SyntaxNode, TextRange, T,
};
use text_edit::TextEdit;

use crate::{FileId, Label, SourceChange, SourceFileEdit};

use self::fixes::DiagnosticWithFix;

#[derive(Debug)]
pub struct Diagnostic {
    // pub name: Option<String>,
    pub message: String,
    pub range: TextRange,
    pub severity: Severity,
    pub fix: Option<Fix>,
}

#[derive(Debug)]
pub struct Fix {
    pub label: Label,
    pub source_change: SourceChange,
    /// Allows to trigger the fix only when the caret is in the range given
    pub fix_trigger_range: TextRange,
}

impl Fix {
    fn new(label: &str, source_change: SourceChange, fix_trigger_range: TextRange) -> Self {
        let label = Label::new(label);
        Self { label, source_change, fix_trigger_range }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    Error,
    WeakWarning,
}

#[derive(Default, Debug, Clone)]
pub struct DiagnosticsConfig {
    pub disable_experimental: bool,
    pub disabled: FxHashSet<String>,
}

pub(crate) fn diagnostics(
    db: &RootDatabase,
    config: &DiagnosticsConfig,
    file_id: FileId,
) -> Vec<Diagnostic> {
    let _p = profile::span("diagnostics");
    let sema = Semantics::new(db);
    let parse = db.parse(file_id);
    let mut res = Vec::new();

    // [#34344] Only take first 128 errors to prevent slowing down editor/ide, the number 128 is chosen arbitrarily.
    res.extend(parse.errors().iter().take(128).map(|err| Diagnostic {
        // name: None,
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
    let sink_builder = DiagnosticSinkBuilder::new()
        .on::<hir::diagnostics::UnresolvedModule, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::MissingFields, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::MissingOkInTailExpr, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::NoSuchField, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        // Only collect experimental diagnostics when they're enabled.
        .filter(|diag| !(diag.is_experimental() && config.disable_experimental))
        .filter(|diag| !config.disabled.contains(diag.code().as_str()));

    // Finalize the `DiagnosticSink` building process.
    let mut sink = sink_builder
        // Diagnostics not handled above get no fix and default treatment.
        .build(|d| {
            res.borrow_mut().push(Diagnostic {
                // name: Some(d.name().into()),
                message: d.message(),
                range: sema.diagnostics_display_range(d).range,
                severity: Severity::Error,
                fix: None,
            })
        });

    if let Some(m) = sema.to_module_def(file_id) {
        m.diagnostics(db, &mut sink);
    };
    drop(sink);
    res.into_inner()
}

fn diagnostic_with_fix<D: DiagnosticWithFix>(d: &D, sema: &Semantics<RootDatabase>) -> Diagnostic {
    Diagnostic {
        // name: Some(d.name().into()),
        range: sema.diagnostics_display_range(d).range,
        message: d.message(),
        severity: Severity::Error,
        fix: d.fix(&sema),
    }
}

fn check_unnecessary_braces_in_use_statement(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let use_tree_list = ast::UseTreeList::cast(node.clone())?;
    if let Some((single_use_tree,)) = use_tree_list.use_trees().collect_tuple() {
        let use_range = use_tree_list.syntax().text_range();
        let edit =
            text_edit_for_remove_unnecessary_braces_with_self_in_use_statement(&single_use_tree)
                .unwrap_or_else(|| {
                    let to_replace = single_use_tree.syntax().text().to_string();
                    let mut edit_builder = TextEdit::builder();
                    edit_builder.delete(use_range);
                    edit_builder.insert(use_range.start(), to_replace);
                    edit_builder.finish()
                });

        acc.push(Diagnostic {
            // name: None,
            range: use_range,
            message: "Unnecessary braces in use statement".to_string(),
            severity: Severity::WeakWarning,
            fix: Some(Fix::new(
                "Remove unnecessary braces",
                SourceFileEdit { file_id, edit }.into(),
                use_range,
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
        return Some(TextEdit::delete(TextRange::new(start, end)));
    }
    None
}

fn check_struct_shorthand_initialization(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let record_lit = ast::RecordExpr::cast(node.clone())?;
    let record_field_list = record_lit.record_expr_field_list()?;
    for record_field in record_field_list.fields() {
        if let (Some(name_ref), Some(expr)) = (record_field.name_ref(), record_field.expr()) {
            let field_name = name_ref.syntax().text().to_string();
            let field_expr = expr.syntax().text().to_string();
            let field_name_is_tup_index = name_ref.as_tuple_field().is_some();
            if field_name == field_expr && !field_name_is_tup_index {
                let mut edit_builder = TextEdit::builder();
                edit_builder.delete(record_field.syntax().text_range());
                edit_builder.insert(record_field.syntax().text_range().start(), field_name);
                let edit = edit_builder.finish();

                let field_range = record_field.syntax().text_range();
                acc.push(Diagnostic {
                    // name: None,
                    range: field_range,
                    message: "Shorthand struct initialization".to_string(),
                    severity: Severity::WeakWarning,
                    fix: Some(Fix::new(
                        "Use struct shorthand initialization",
                        SourceFileEdit { file_id, edit }.into(),
                        field_range,
                    )),
                });
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use stdx::trim_indent;
    use test_utils::assert_eq_text;

    use crate::{
        mock_analysis::{analysis_and_position, single_file, MockAnalysis},
        DiagnosticsConfig,
    };

    /// Takes a multi-file input fixture with annotated cursor positions,
    /// and checks that:
    ///  * a diagnostic is produced
    ///  * this diagnostic fix trigger range touches the input cursor position
    ///  * that the contents of the file containing the cursor match `after` after the diagnostic fix is applied
    fn check_fix(ra_fixture_before: &str, ra_fixture_after: &str) {
        let after = trim_indent(ra_fixture_after);

        let (analysis, file_position) = analysis_and_position(ra_fixture_before);
        let diagnostic = analysis
            .diagnostics(&DiagnosticsConfig::default(), file_position.file_id)
            .unwrap()
            .pop()
            .unwrap();
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
            fix.fix_trigger_range.start() <= file_position.offset
                && fix.fix_trigger_range.end() >= file_position.offset,
            "diagnostic fix range {:?} does not touch cursor position {:?}",
            fix.fix_trigger_range,
            file_position.offset
        );
    }

    /// Checks that a diagnostic applies to the file containing the `<|>` cursor marker
    /// which has a fix that can apply to other files.
    fn check_apply_diagnostic_fix_in_other_file(ra_fixture_before: &str, ra_fixture_after: &str) {
        let ra_fixture_after = &trim_indent(ra_fixture_after);
        let (analysis, file_pos) = analysis_and_position(ra_fixture_before);
        let current_file_id = file_pos.file_id;
        let diagnostic = analysis
            .diagnostics(&DiagnosticsConfig::default(), current_file_id)
            .unwrap()
            .pop()
            .unwrap();
        let mut fix = diagnostic.fix.unwrap();
        let edit = fix.source_change.source_file_edits.pop().unwrap();
        let changed_file_id = edit.file_id;
        let before = analysis.file_text(changed_file_id).unwrap();
        let actual = {
            let mut actual = before.to_string();
            edit.edit.apply(&mut actual);
            actual
        };
        assert_eq_text!(ra_fixture_after, &actual);
    }

    /// Takes a multi-file input fixture with annotated cursor position and checks that no diagnostics
    /// apply to the file containing the cursor.
    fn check_no_diagnostics(ra_fixture: &str) {
        let mock = MockAnalysis::with_files(ra_fixture);
        let files = mock.files().map(|(it, _)| it).collect::<Vec<_>>();
        let analysis = mock.analysis();
        let diagnostics = files
            .into_iter()
            .flat_map(|file_id| {
                analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 0, "unexpected diagnostics:\n{:#?}", diagnostics);
    }

    fn check_expect(ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = single_file(ra_fixture);
        let diagnostics = analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap();
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
                                fix_trigger_range: 0..8,
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

    #[test]
    fn test_add_field_in_other_file_from_usage() {
        check_apply_diagnostic_fix_in_other_file(
            r"
            //- /main.rs
            mod foo;

            fn main() {
                <|>foo::Foo { bar: 3, baz: false};
            }
            //- /foo.rs
            struct Foo {
                bar: i32
            }
            ",
            r"
            struct Foo {
                bar: i32,
                pub(crate) baz: bool
            }
            ",
        )
    }

    #[test]
    fn test_disabled_diagnostics() {
        let mut config = DiagnosticsConfig::default();
        config.disabled.insert("unresolved-module".into());

        let (analysis, file_id) = single_file(r#"mod foo;"#);

        let diagnostics = analysis.diagnostics(&config, file_id).unwrap();
        assert!(diagnostics.is_empty());

        let diagnostics = analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap();
        assert!(!diagnostics.is_empty());
    }
}
