//! Collects diagnostics & fixits  for a single file.
//!
//! The tricky bit here is that diagnostics are produced by hir in terms of
//! macro-expanded files, but we need to present them to the users in terms of
//! original files. So we need to map the ranges.

mod fixes;
mod field_shorthand;
mod unlinked_file;

use std::cell::RefCell;

use hir::{
    db::AstDatabase,
    diagnostics::{Diagnostic as _, DiagnosticCode, DiagnosticSinkBuilder},
    InFile, Semantics,
};
use ide_db::{base_db::SourceDatabase, RootDatabase};
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, AstNode},
    SyntaxNode, SyntaxNodePtr, TextRange, TextSize,
};
use text_edit::TextEdit;
use unlinked_file::UnlinkedFile;

use crate::{Assist, AssistId, AssistKind, FileId, Label, SourceChange};

use self::fixes::DiagnosticWithFix;

#[derive(Debug)]
pub struct Diagnostic {
    // pub name: Option<String>,
    pub message: String,
    pub range: TextRange,
    pub severity: Severity,
    pub fix: Option<Assist>,
    pub unused: bool,
    pub code: Option<DiagnosticCode>,
}

impl Diagnostic {
    fn error(range: TextRange, message: String) -> Self {
        Self { message, range, severity: Severity::Error, fix: None, unused: false, code: None }
    }

    fn hint(range: TextRange, message: String) -> Self {
        Self {
            message,
            range,
            severity: Severity::WeakWarning,
            fix: None,
            unused: false,
            code: None,
        }
    }

    fn with_fix(self, fix: Option<Assist>) -> Self {
        Self { fix, ..self }
    }

    fn with_unused(self, unused: bool) -> Self {
        Self { unused, ..self }
    }

    fn with_code(self, code: Option<DiagnosticCode>) -> Self {
        Self { code, ..self }
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
    res.extend(
        parse
            .errors()
            .iter()
            .take(128)
            .map(|err| Diagnostic::error(err.range(), format!("Syntax Error: {}", err))),
    );

    for node in parse.tree().syntax().descendants() {
        check_unnecessary_braces_in_use_statement(&mut res, file_id, &node);
        field_shorthand::check(&mut res, file_id, &node);
    }
    let res = RefCell::new(res);
    let sink_builder = DiagnosticSinkBuilder::new()
        .on::<hir::diagnostics::UnresolvedModule, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::MissingFields, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::MissingOkOrSomeInTailExpr, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::NoSuchField, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::RemoveThisSemicolon, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::IncorrectCase, _>(|d| {
            res.borrow_mut().push(warning_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::ReplaceFilterMapNextWithFindMap, _>(|d| {
            res.borrow_mut().push(warning_with_fix(d, &sema));
        })
        .on::<hir::diagnostics::InactiveCode, _>(|d| {
            // If there's inactive code somewhere in a macro, don't propagate to the call-site.
            if d.display_source().file_id.expansion_info(db).is_some() {
                return;
            }

            // Override severity and mark as unused.
            res.borrow_mut().push(
                Diagnostic::hint(
                    sema.diagnostics_display_range(d.display_source()).range,
                    d.message(),
                )
                .with_unused(true)
                .with_code(Some(d.code())),
            );
        })
        .on::<UnlinkedFile, _>(|d| {
            // Limit diagnostic to the first few characters in the file. This matches how VS Code
            // renders it with the full span, but on other editors, and is less invasive.
            let range = sema.diagnostics_display_range(d.display_source()).range;
            let range = range.intersect(TextRange::up_to(TextSize::of("..."))).unwrap_or(range);

            // Override severity and mark as unused.
            res.borrow_mut().push(
                Diagnostic::hint(range, d.message())
                    .with_fix(d.fix(&sema))
                    .with_code(Some(d.code())),
            );
        })
        .on::<hir::diagnostics::UnresolvedProcMacro, _>(|d| {
            // Use more accurate position if available.
            let display_range = d
                .precise_location
                .unwrap_or_else(|| sema.diagnostics_display_range(d.display_source()).range);

            // FIXME: it would be nice to tell the user whether proc macros are currently disabled
            res.borrow_mut()
                .push(Diagnostic::hint(display_range, d.message()).with_code(Some(d.code())));
        })
        .on::<hir::diagnostics::UnresolvedMacroCall, _>(|d| {
            let last_path_segment = sema.db.parse_or_expand(d.file).and_then(|root| {
                d.node
                    .to_node(&root)
                    .path()
                    .and_then(|it| it.segment())
                    .and_then(|it| it.name_ref())
                    .map(|it| InFile::new(d.file, SyntaxNodePtr::new(it.syntax())))
            });
            let diagnostics = last_path_segment.unwrap_or_else(|| d.display_source());
            let display_range = sema.diagnostics_display_range(diagnostics).range;
            res.borrow_mut()
                .push(Diagnostic::error(display_range, d.message()).with_code(Some(d.code())));
        })
        // Only collect experimental diagnostics when they're enabled.
        .filter(|diag| !(diag.is_experimental() && config.disable_experimental))
        .filter(|diag| !config.disabled.contains(diag.code().as_str()));

    // Finalize the `DiagnosticSink` building process.
    let mut sink = sink_builder
        // Diagnostics not handled above get no fix and default treatment.
        .build(|d| {
            res.borrow_mut().push(
                Diagnostic::error(
                    sema.diagnostics_display_range(d.display_source()).range,
                    d.message(),
                )
                .with_code(Some(d.code())),
            );
        });

    match sema.to_module_def(file_id) {
        Some(m) => m.diagnostics(db, &mut sink),
        None => {
            sink.push(UnlinkedFile { file_id, node: SyntaxNodePtr::new(&parse.tree().syntax()) });
        }
    }

    drop(sink);
    res.into_inner()
}

fn diagnostic_with_fix<D: DiagnosticWithFix>(d: &D, sema: &Semantics<RootDatabase>) -> Diagnostic {
    Diagnostic::error(sema.diagnostics_display_range(d.display_source()).range, d.message())
        .with_fix(d.fix(&sema))
        .with_code(Some(d.code()))
}

fn warning_with_fix<D: DiagnosticWithFix>(d: &D, sema: &Semantics<RootDatabase>) -> Diagnostic {
    Diagnostic::hint(sema.diagnostics_display_range(d.display_source()).range, d.message())
        .with_fix(d.fix(&sema))
        .with_code(Some(d.code()))
}

fn check_unnecessary_braces_in_use_statement(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let use_tree_list = ast::UseTreeList::cast(node.clone())?;
    if let Some((single_use_tree,)) = use_tree_list.use_trees().collect_tuple() {
        // If there is a comment inside the bracketed `use`,
        // assume it is a commented out module path and don't show diagnostic.
        if use_tree_list.has_inner_comment() {
            return Some(());
        }

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

        acc.push(
            Diagnostic::hint(use_range, "Unnecessary braces in use statement".to_string())
                .with_fix(Some(fix(
                    "remove_braces",
                    "Remove unnecessary braces",
                    SourceChange::from_text_edit(file_id, edit),
                    use_range,
                ))),
        );
    }

    Some(())
}

fn text_edit_for_remove_unnecessary_braces_with_self_in_use_statement(
    single_use_tree: &ast::UseTree,
) -> Option<TextEdit> {
    let use_tree_list_node = single_use_tree.syntax().parent()?;
    if single_use_tree.path()?.segment()?.self_token().is_some() {
        let start = use_tree_list_node.prev_sibling_or_token()?.text_range().start();
        let end = use_tree_list_node.text_range().end();
        return Some(TextEdit::delete(TextRange::new(start, end)));
    }
    None
}

fn fix(id: &'static str, label: &str, source_change: SourceChange, target: TextRange) -> Assist {
    assert!(!id.contains(' '));
    Assist {
        id: AssistId(id, AssistKind::QuickFix),
        label: Label::new(label),
        group: None,
        target,
        source_change: Some(source_change),
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use stdx::trim_indent;
    use test_utils::assert_eq_text;

    use crate::{fixture, DiagnosticsConfig};

    /// Takes a multi-file input fixture with annotated cursor positions,
    /// and checks that:
    ///  * a diagnostic is produced
    ///  * this diagnostic fix trigger range touches the input cursor position
    ///  * that the contents of the file containing the cursor match `after` after the diagnostic fix is applied
    pub(crate) fn check_fix(ra_fixture_before: &str, ra_fixture_after: &str) {
        let after = trim_indent(ra_fixture_after);

        let (analysis, file_position) = fixture::position(ra_fixture_before);
        let diagnostic = analysis
            .diagnostics(&DiagnosticsConfig::default(), file_position.file_id)
            .unwrap()
            .pop()
            .unwrap();
        let fix = diagnostic.fix.unwrap();
        let actual = {
            let source_change = fix.source_change.unwrap();
            let file_id = *source_change.source_file_edits.keys().next().unwrap();
            let mut actual = analysis.file_text(file_id).unwrap().to_string();

            for edit in source_change.source_file_edits.values() {
                edit.apply(&mut actual);
            }
            actual
        };

        assert_eq_text!(&after, &actual);
        assert!(
            fix.target.contains_inclusive(file_position.offset),
            "diagnostic fix range {:?} does not touch cursor position {:?}",
            fix.target,
            file_position.offset
        );
    }

    /// Checks that there's a diagnostic *without* fix at `$0`.
    fn check_no_fix(ra_fixture: &str) {
        let (analysis, file_position) = fixture::position(ra_fixture);
        let diagnostic = analysis
            .diagnostics(&DiagnosticsConfig::default(), file_position.file_id)
            .unwrap()
            .pop()
            .unwrap();
        assert!(diagnostic.fix.is_none(), "got a fix when none was expected: {:?}", diagnostic);
    }

    /// Takes a multi-file input fixture with annotated cursor position and checks that no diagnostics
    /// apply to the file containing the cursor.
    pub(crate) fn check_no_diagnostics(ra_fixture: &str) {
        let (analysis, files) = fixture::files(ra_fixture);
        let diagnostics = files
            .into_iter()
            .flat_map(|file_id| {
                analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 0, "unexpected diagnostics:\n{:#?}", diagnostics);
    }

    fn check_expect(ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let diagnostics = analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap();
        expect.assert_debug_eq(&diagnostics)
    }

    #[test]
    fn test_wrap_return_type_option() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::option::Option::{self, Some, None};

fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    x / y$0
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
            r#"
use core::option::Option::{self, Some, None};

fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    Some(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    x / y$0
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
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
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    $0x
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
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
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    x $0/ y
}
//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
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
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

fn foo() -> Result<(), i32> { 0 }

//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_return_type_is_not_result_or_option() {
        check_no_diagnostics(
            r#"
//- /main.rs crate:main deps:core
use core::result::Result::{self, Ok, Err};

enum SomeOtherEnum { Ok(i32), Err(String) }

fn foo() -> SomeOtherEnum { 0 }

//- /core/lib.rs crate:core
pub mod result {
    pub enum Result<T, E> { Ok(T), Err(E) }
}
pub mod option {
    pub enum Option<T> { Some(T), None }
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
    let s = TestStruct {$0};
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
    fn test_fn() { let s = Self {$0}; }
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
        Expr::Bin {$0 }
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
    let s = TestStruct{ two: 2$0 };
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
                            Assist {
                                id: AssistId(
                                    "create_module",
                                    QuickFix,
                                ),
                                label: "Create module",
                                group: None,
                                target: 0..8,
                                source_change: Some(
                                    SourceChange {
                                        source_file_edits: {},
                                        file_system_edits: [
                                            CreateFile {
                                                dst: AnchoredPathBuf {
                                                    anchor: FileId(
                                                        0,
                                                    ),
                                                    path: "foo.rs",
                                                },
                                                initial_contents: "",
                                            },
                                        ],
                                        is_snippet: false,
                                    },
                                ),
                            },
                        ),
                        unused: false,
                        code: Some(
                            DiagnosticCode(
                                "unresolved-module",
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_unresolved_macro_range() {
        check_expect(
            r#"foo::bar!(92);"#,
            expect![[r#"
                [
                    Diagnostic {
                        message: "unresolved macro call",
                        range: 5..8,
                        severity: Error,
                        fix: None,
                        unused: false,
                        code: Some(
                            DiagnosticCode(
                                "unresolved-macro-call",
                            ),
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
    let _x = id![Foo { a: $042 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
            r#"
fn some(, b: ()) {}
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

mod a {
    mod c {}
    mod d {
        mod e {}
    }
}
"#,
        );
        check_no_diagnostics(
            r#"
use a;
use a::{
    c,
    // d::e
};

mod a {
    mod c {}
    mod d {
        mod e {}
    }
}
"#,
        );
        check_fix(
            r"
            mod b {}
            use {$0b};
            ",
            r"
            mod b {}
            use b;
            ",
        );
        check_fix(
            r"
            mod b {}
            use {b$0};
            ",
            r"
            mod b {}
            use b;
            ",
        );
        check_fix(
            r"
            mod a { mod c {} }
            use a::{c$0};
            ",
            r"
            mod a { mod c {} }
            use a::c;
            ",
        );
        check_fix(
            r"
            mod a {}
            use a::{self$0};
            ",
            r"
            mod a {}
            use a;
            ",
        );
        check_fix(
            r"
            mod a { mod c {} mod d { mod e {} } }
            use a::{c, d::{e$0}};
            ",
            r"
            mod a { mod c {} mod d { mod e {} } }
            use a::{c, d::e};
            ",
        );
    }

    #[test]
    fn test_add_field_from_usage() {
        check_fix(
            r"
fn main() {
    Foo { bar: 3, baz$0: false};
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
        check_fix(
            r#"
//- /main.rs
mod foo;

fn main() {
    foo::Foo { bar: 3, $0baz: false};
}
//- /foo.rs
struct Foo {
    bar: i32
}
"#,
            r#"
struct Foo {
    bar: i32,
    pub(crate) baz: bool
}
"#,
        )
    }

    #[test]
    fn test_disabled_diagnostics() {
        let mut config = DiagnosticsConfig::default();
        config.disabled.insert("unresolved-module".into());

        let (analysis, file_id) = fixture::file(r#"mod foo;"#);

        let diagnostics = analysis.diagnostics(&config, file_id).unwrap();
        assert!(diagnostics.is_empty());

        let diagnostics = analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap();
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn test_rename_incorrect_case() {
        check_fix(
            r#"
pub struct test_struct$0 { one: i32 }

pub fn some_fn(val: test_struct) -> test_struct {
    test_struct { one: val.one + 1 }
}
"#,
            r#"
pub struct TestStruct { one: i32 }

pub fn some_fn(val: TestStruct) -> TestStruct {
    TestStruct { one: val.one + 1 }
}
"#,
        );

        check_fix(
            r#"
pub fn some_fn(NonSnakeCase$0: u8) -> u8 {
    NonSnakeCase
}
"#,
            r#"
pub fn some_fn(non_snake_case: u8) -> u8 {
    non_snake_case
}
"#,
        );

        check_fix(
            r#"
pub fn SomeFn$0(val: u8) -> u8 {
    if val != 0 { SomeFn(val - 1) } else { val }
}
"#,
            r#"
pub fn some_fn(val: u8) -> u8 {
    if val != 0 { some_fn(val - 1) } else { val }
}
"#,
        );

        check_fix(
            r#"
fn some_fn() {
    let whatAWeird_Formatting$0 = 10;
    another_func(whatAWeird_Formatting);
}
"#,
            r#"
fn some_fn() {
    let what_a_weird_formatting = 10;
    another_func(what_a_weird_formatting);
}
"#,
        );
    }

    #[test]
    fn test_uppercase_const_no_diagnostics() {
        check_no_diagnostics(
            r#"
fn foo() {
    const ANOTHER_ITEM$0: &str = "some_item";
}
"#,
        );
    }

    #[test]
    fn test_rename_incorrect_case_struct_method() {
        check_fix(
            r#"
pub struct TestStruct;

impl TestStruct {
    pub fn SomeFn$0() -> TestStruct {
        TestStruct
    }
}
"#,
            r#"
pub struct TestStruct;

impl TestStruct {
    pub fn some_fn() -> TestStruct {
        TestStruct
    }
}
"#,
        );
    }

    #[test]
    fn test_single_incorrect_case_diagnostic_in_function_name_issue_6970() {
        let input = r#"fn FOO$0() {}"#;
        let expected = r#"fn foo() {}"#;

        let (analysis, file_position) = fixture::position(input);
        let diagnostics =
            analysis.diagnostics(&DiagnosticsConfig::default(), file_position.file_id).unwrap();
        assert_eq!(diagnostics.len(), 1);

        check_fix(input, expected);
    }

    #[test]
    fn unlinked_file_prepend_first_item() {
        cov_mark::check!(unlinked_file_prepend_before_first_item);
        check_fix(
            r#"
//- /main.rs
fn f() {}
//- /foo.rs
$0
"#,
            r#"
mod foo;

fn f() {}
"#,
        );
    }

    #[test]
    fn unlinked_file_append_mod() {
        cov_mark::check!(unlinked_file_append_to_existing_mods);
        check_fix(
            r#"
//- /main.rs
//! Comment on top

mod preexisting;

mod preexisting2;

struct S;

mod preexisting_bottom;)
//- /foo.rs
$0
"#,
            r#"
//! Comment on top

mod preexisting;

mod preexisting2;
mod foo;

struct S;

mod preexisting_bottom;)
"#,
        );
    }

    #[test]
    fn unlinked_file_insert_in_empty_file() {
        cov_mark::check!(unlinked_file_empty_file);
        check_fix(
            r#"
//- /main.rs
//- /foo.rs
$0
"#,
            r#"
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_old_style_modrs() {
        check_fix(
            r#"
//- /main.rs
mod submod;
//- /submod/mod.rs
// in mod.rs
//- /submod/foo.rs
$0
"#,
            r#"
// in mod.rs
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_new_style_mod() {
        check_fix(
            r#"
//- /main.rs
mod submod;
//- /submod.rs
//- /submod/foo.rs
$0
"#,
            r#"
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_with_cfg_off() {
        cov_mark::check!(unlinked_file_skip_fix_when_mod_already_exists);
        check_no_fix(
            r#"
//- /main.rs
#[cfg(never)]
mod foo;

//- /foo.rs
$0
"#,
        );
    }

    #[test]
    fn unlinked_file_with_cfg_on() {
        check_no_diagnostics(
            r#"
//- /main.rs
#[cfg(not(never))]
mod foo;

//- /foo.rs
"#,
        );
    }
}
