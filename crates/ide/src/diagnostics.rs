//! Collects diagnostics & fixits  for a single file.
//!
//! The tricky bit here is that diagnostics are produced by hir in terms of
//! macro-expanded files, but we need to present them to the users in terms of
//! original files. So we need to map the ranges.

mod unresolved_module;
mod missing_fields;

mod fixes;
mod field_shorthand;
mod unlinked_file;

use std::cell::RefCell;

use hir::{
    db::AstDatabase,
    diagnostics::{AnyDiagnostic, Diagnostic as _, DiagnosticCode, DiagnosticSinkBuilder},
    InFile, Semantics,
};
use ide_assists::AssistResolveStrategy;
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

use self::fixes::DiagnosticWithFixes;

#[derive(Debug)]
pub struct Diagnostic {
    // pub name: Option<String>,
    pub message: String,
    pub range: TextRange,
    pub severity: Severity,
    pub fixes: Option<Vec<Assist>>,
    pub unused: bool,
    pub code: Option<DiagnosticCode>,
}

impl Diagnostic {
    fn new(code: &'static str, message: impl Into<String>, range: TextRange) -> Diagnostic {
        let message = message.into();
        let code = Some(DiagnosticCode(code));
        Self { message, range, severity: Severity::Error, fixes: None, unused: false, code }
    }

    fn error(range: TextRange, message: String) -> Self {
        Self { message, range, severity: Severity::Error, fixes: None, unused: false, code: None }
    }

    fn hint(range: TextRange, message: String) -> Self {
        Self {
            message,
            range,
            severity: Severity::WeakWarning,
            fixes: None,
            unused: false,
            code: None,
        }
    }

    fn with_fixes(self, fixes: Option<Vec<Assist>>) -> Self {
        Self { fixes, ..self }
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

struct DiagnosticsContext<'a> {
    config: &'a DiagnosticsConfig,
    sema: Semantics<'a, RootDatabase>,
    #[allow(unused)]
    resolve: &'a AssistResolveStrategy,
}

pub(crate) fn diagnostics(
    db: &RootDatabase,
    config: &DiagnosticsConfig,
    resolve: &AssistResolveStrategy,
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
        .on::<hir::diagnostics::MissingOkOrSomeInTailExpr, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema, resolve));
        })
        .on::<hir::diagnostics::NoSuchField, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema, resolve));
        })
        .on::<hir::diagnostics::RemoveThisSemicolon, _>(|d| {
            res.borrow_mut().push(diagnostic_with_fix(d, &sema, resolve));
        })
        .on::<hir::diagnostics::IncorrectCase, _>(|d| {
            res.borrow_mut().push(warning_with_fix(d, &sema, resolve));
        })
        .on::<hir::diagnostics::ReplaceFilterMapNextWithFindMap, _>(|d| {
            res.borrow_mut().push(warning_with_fix(d, &sema, resolve));
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
                    .with_fixes(d.fixes(&sema, resolve))
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
        .on::<hir::diagnostics::UnimplementedBuiltinMacro, _>(|d| {
            let display_range = sema.diagnostics_display_range(d.display_source()).range;
            res.borrow_mut()
                .push(Diagnostic::hint(display_range, d.message()).with_code(Some(d.code())));
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

    let mut diags = Vec::new();
    let internal_diagnostics = cfg!(test);
    match sema.to_module_def(file_id) {
        Some(m) => diags = m.diagnostics(db, &mut sink, internal_diagnostics),
        None => {
            sink.push(UnlinkedFile { file_id, node: SyntaxNodePtr::new(parse.tree().syntax()) });
        }
    }

    drop(sink);

    let mut res = res.into_inner();

    let ctx = DiagnosticsContext { config, sema, resolve };
    for diag in diags {
        let d = match diag {
            AnyDiagnostic::UnresolvedModule(d) => unresolved_module::unresolved_module(&ctx, &d),
            AnyDiagnostic::MissingFields(d) => missing_fields::missing_fields(&ctx, &d),
        };
        if let Some(code) = d.code {
            if ctx.config.disabled.contains(code.as_str()) {
                continue;
            }
        }
        res.push(d)
    }

    res
}

fn diagnostic_with_fix<D: DiagnosticWithFixes>(
    d: &D,
    sema: &Semantics<RootDatabase>,
    resolve: &AssistResolveStrategy,
) -> Diagnostic {
    Diagnostic::error(sema.diagnostics_display_range(d.display_source()).range, d.message())
        .with_fixes(d.fixes(sema, resolve))
        .with_code(Some(d.code()))
}

fn warning_with_fix<D: DiagnosticWithFixes>(
    d: &D,
    sema: &Semantics<RootDatabase>,
    resolve: &AssistResolveStrategy,
) -> Diagnostic {
    Diagnostic::hint(sema.diagnostics_display_range(d.display_source()).range, d.message())
        .with_fixes(d.fixes(sema, resolve))
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
                .with_fixes(Some(vec![fix(
                    "remove_braces",
                    "Remove unnecessary braces",
                    SourceChange::from_text_edit(file_id, edit),
                    use_range,
                )])),
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
    let mut res = unresolved_fix(id, label, target);
    res.source_change = Some(source_change);
    res
}

fn unresolved_fix(id: &'static str, label: &str, target: TextRange) -> Assist {
    assert!(!id.contains(' '));
    Assist {
        id: AssistId(id, AssistKind::QuickFix),
        label: Label::new(label),
        group: None,
        target,
        source_change: None,
    }
}

#[cfg(test)]
mod tests {
    use expect_test::Expect;
    use hir::diagnostics::DiagnosticCode;
    use ide_assists::AssistResolveStrategy;
    use stdx::trim_indent;
    use test_utils::{assert_eq_text, extract_annotations};

    use crate::{fixture, DiagnosticsConfig};

    /// Takes a multi-file input fixture with annotated cursor positions,
    /// and checks that:
    ///  * a diagnostic is produced
    ///  * the first diagnostic fix trigger range touches the input cursor position
    ///  * that the contents of the file containing the cursor match `after` after the diagnostic fix is applied
    #[track_caller]
    pub(crate) fn check_fix(ra_fixture_before: &str, ra_fixture_after: &str) {
        check_nth_fix(0, ra_fixture_before, ra_fixture_after);
    }
    /// Takes a multi-file input fixture with annotated cursor positions,
    /// and checks that:
    ///  * a diagnostic is produced
    ///  * every diagnostic fixes trigger range touches the input cursor position
    ///  * that the contents of the file containing the cursor match `after` after each diagnostic fix is applied
    pub(crate) fn check_fixes(ra_fixture_before: &str, ra_fixtures_after: Vec<&str>) {
        for (i, ra_fixture_after) in ra_fixtures_after.iter().enumerate() {
            check_nth_fix(i, ra_fixture_before, ra_fixture_after)
        }
    }

    #[track_caller]
    fn check_nth_fix(nth: usize, ra_fixture_before: &str, ra_fixture_after: &str) {
        let after = trim_indent(ra_fixture_after);

        let (analysis, file_position) = fixture::position(ra_fixture_before);
        let diagnostic = analysis
            .diagnostics(
                &DiagnosticsConfig::default(),
                AssistResolveStrategy::All,
                file_position.file_id,
            )
            .unwrap()
            .pop()
            .unwrap();
        let fix = &diagnostic.fixes.unwrap()[nth];
        let actual = {
            let source_change = fix.source_change.as_ref().unwrap();
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
            .diagnostics(
                &DiagnosticsConfig::default(),
                AssistResolveStrategy::All,
                file_position.file_id,
            )
            .unwrap()
            .pop()
            .unwrap();
        assert!(diagnostic.fixes.is_none(), "got a fix when none was expected: {:?}", diagnostic);
    }

    /// Takes a multi-file input fixture with annotated cursor position and checks that no diagnostics
    /// apply to the file containing the cursor.
    pub(crate) fn check_no_diagnostics(ra_fixture: &str) {
        let (analysis, files) = fixture::files(ra_fixture);
        let diagnostics = files
            .into_iter()
            .flat_map(|file_id| {
                analysis
                    .diagnostics(&DiagnosticsConfig::default(), AssistResolveStrategy::All, file_id)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 0, "unexpected diagnostics:\n{:#?}", diagnostics);
    }

    pub(crate) fn check_expect(ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let diagnostics = analysis
            .diagnostics(&DiagnosticsConfig::default(), AssistResolveStrategy::All, file_id)
            .unwrap();
        expect.assert_debug_eq(&diagnostics)
    }

    pub(crate) fn check_diagnostics(ra_fixture: &str) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let diagnostics = analysis
            .diagnostics(&DiagnosticsConfig::default(), AssistResolveStrategy::All, file_id)
            .unwrap();

        let expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
        let mut actual = diagnostics
            .into_iter()
            .filter(|d| d.code != Some(DiagnosticCode("inactive-code")))
            .map(|d| (d.range, d.message))
            .collect::<Vec<_>>();
        actual.sort_by_key(|(range, _)| range.start());
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_unresolved_macro_range() {
        check_diagnostics(
            r#"
foo::bar!(92);
   //^^^ unresolved macro `foo::bar!`
"#,
        );
    }

    #[test]
    fn unresolved_import_in_use_tree() {
        // Only the relevant part of a nested `use` item should be highlighted.
        check_diagnostics(
            r#"
use does_exist::{Exists, DoesntExist};
                       //^^^^^^^^^^^ unresolved import

use {does_not_exist::*, does_exist};
   //^^^^^^^^^^^^^^^^^ unresolved import

use does_not_exist::{
    a,
  //^ unresolved import
    b,
  //^ unresolved import
    c,
  //^ unresolved import
};

mod does_exist {
    pub struct Exists;
}
"#,
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
fn some(, b: () ) {}
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
    fn test_disabled_diagnostics() {
        let mut config = DiagnosticsConfig::default();
        config.disabled.insert("unresolved-module".into());

        let (analysis, file_id) = fixture::file(r#"mod foo;"#);

        let diagnostics =
            analysis.diagnostics(&config, AssistResolveStrategy::All, file_id).unwrap();
        assert!(diagnostics.is_empty());

        let diagnostics = analysis
            .diagnostics(&DiagnosticsConfig::default(), AssistResolveStrategy::All, file_id)
            .unwrap();
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn unlinked_file_prepend_first_item() {
        cov_mark::check!(unlinked_file_prepend_before_first_item);
        // Only tests the first one for `pub mod` since the rest are the same
        check_fixes(
            r#"
//- /main.rs
fn f() {}
//- /foo.rs
$0
"#,
            vec![
                r#"
mod foo;

fn f() {}
"#,
                r#"
pub mod foo;

fn f() {}
"#,
            ],
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

    #[test]
    fn break_outside_of_loop() {
        check_diagnostics(
            r#"
fn foo() { break; }
         //^^^^^ break outside of loop
"#,
        );
    }

    #[test]
    fn no_such_field_diagnostics() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
impl S {
    fn new() -> S {
        S {
      //^ Missing structure fields:
      //|    - bar
            foo: 92,
            baz: 62,
          //^^^^^^^ no such field
        }
    }
}
"#,
        );
    }
    #[test]
    fn no_such_field_with_feature_flag_diagnostics() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
struct MyStruct {
    my_val: usize,
    #[cfg(feature = "foo")]
    bar: bool,
}

impl MyStruct {
    #[cfg(feature = "foo")]
    pub(crate) fn new(my_val: usize, bar: bool) -> Self {
        Self { my_val, bar }
    }
    #[cfg(not(feature = "foo"))]
    pub(crate) fn new(my_val: usize, _bar: bool) -> Self {
        Self { my_val }
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_enum_with_feature_flag_diagnostics() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
enum Foo {
    #[cfg(not(feature = "foo"))]
    Buz,
    #[cfg(feature = "foo")]
    Bar,
    Baz
}

fn test_fn(f: Foo) {
    match f {
        Foo::Bar => {},
        Foo::Baz => {},
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_with_feature_flag_diagnostics_on_struct_lit() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
struct S {
    #[cfg(feature = "foo")]
    foo: u32,
    #[cfg(not(feature = "foo"))]
    bar: u32,
}

impl S {
    #[cfg(feature = "foo")]
    fn new(foo: u32) -> Self {
        Self { foo }
    }
    #[cfg(not(feature = "foo"))]
    fn new(bar: u32) -> Self {
        Self { bar }
    }
    fn new2(bar: u32) -> Self {
        #[cfg(feature = "foo")]
        { Self { foo: bar } }
        #[cfg(not(feature = "foo"))]
        { Self { bar } }
    }
    fn new2(val: u32) -> Self {
        Self {
            #[cfg(feature = "foo")]
            foo: val,
            #[cfg(not(feature = "foo"))]
            bar: val,
        }
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_with_type_macro() {
        check_diagnostics(
            r#"
macro_rules! Type { () => { u32 }; }
struct Foo { bar: Type![] }

impl Foo {
    fn new() -> Self {
        Foo { bar: 0 }
    }
}
"#,
        );
    }

    #[test]
    fn missing_unsafe_diagnostic_with_raw_ptr() {
        check_diagnostics(
            r#"
fn main() {
    let x = &5 as *const usize;
    unsafe { let y = *x; }
    let z = *x;
}         //^^ This operation is unsafe and requires an unsafe function or block
"#,
        )
    }

    #[test]
    fn missing_unsafe_diagnostic_with_unsafe_call() {
        check_diagnostics(
            r#"
struct HasUnsafe;

impl HasUnsafe {
    unsafe fn unsafe_fn(&self) {
        let x = &5 as *const usize;
        let y = *x;
    }
}

unsafe fn unsafe_fn() {
    let x = &5 as *const usize;
    let y = *x;
}

fn main() {
    unsafe_fn();
  //^^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
    HasUnsafe.unsafe_fn();
  //^^^^^^^^^^^^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
    unsafe {
        unsafe_fn();
        HasUnsafe.unsafe_fn();
    }
}
"#,
        );
    }

    #[test]
    fn missing_unsafe_diagnostic_with_static_mut() {
        check_diagnostics(
            r#"
struct Ty {
    a: u8,
}

static mut STATIC_MUT: Ty = Ty { a: 0 };

fn main() {
    let x = STATIC_MUT.a;
          //^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
    unsafe {
        let x = STATIC_MUT.a;
    }
}
"#,
        );
    }

    #[test]
    fn no_missing_unsafe_diagnostic_with_safe_intrinsic() {
        check_diagnostics(
            r#"
extern "rust-intrinsic" {
    pub fn bitreverse(x: u32) -> u32; // Safe intrinsic
    pub fn floorf32(x: f32) -> f32; // Unsafe intrinsic
}

fn main() {
    let _ = bitreverse(12);
    let _ = floorf32(12.0);
          //^^^^^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
}
"#,
        );
    }

    // Register the required standard library types to make the tests work
    fn add_filter_map_with_find_next_boilerplate(body: &str) -> String {
        let prefix = r#"
        //- /main.rs crate:main deps:core
        use core::iter::Iterator;
        use core::option::Option::{self, Some, None};
        "#;
        let suffix = r#"
        //- /core/lib.rs crate:core
        pub mod option {
            pub enum Option<T> { Some(T), None }
        }
        pub mod iter {
            pub trait Iterator {
                type Item;
                fn filter_map<B, F>(self, f: F) -> FilterMap where F: FnMut(Self::Item) -> Option<B> { FilterMap }
                fn next(&mut self) -> Option<Self::Item>;
            }
            pub struct FilterMap {}
            impl Iterator for FilterMap {
                type Item = i32;
                fn next(&mut self) -> i32 { 7 }
            }
        }
        "#;
        format!("{}{}{}", prefix, body, suffix)
    }

    #[test]
    fn replace_filter_map_next_with_find_map2() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3].iter().filter_map(|x| if *x == 2 { Some (4) } else { None }).next();
                      //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ replace filter_map(..).next() with find_map(..)
            }
        "#,
        ));
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_without_next() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3]
                    .iter()
                    .filter_map(|x| if *x == 2 { Some (4) } else { None })
                    .len();
            }
            "#,
        ));
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_with_intervening_methods() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3]
                    .iter()
                    .filter_map(|x| if *x == 2 { Some (4) } else { None })
                    .map(|x| x + 2)
                    .len();
            }
            "#,
        ));
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_if_not_in_chain() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3]
                    .iter()
                    .filter_map(|x| if *x == 2 { Some (4) } else { None });
                let n = m.next();
            }
            "#,
        ));
    }

    #[test]
    fn missing_record_pat_field_diagnostic() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
fn baz(s: S) {
    let S { foo: _ } = s;
      //^ Missing structure fields:
      //| - bar
}
"#,
        );
    }

    #[test]
    fn missing_record_pat_field_no_diagnostic_if_not_exhaustive() {
        check_diagnostics(
            r"
struct S { foo: i32, bar: () }
fn baz(s: S) -> i32 {
    match s {
        S { foo, .. } => foo,
    }
}
",
        )
    }

    #[test]
    fn missing_record_pat_field_box() {
        check_diagnostics(
            r"
struct S { s: Box<u32> }
fn x(a: S) {
    let S { box s } = a;
}
",
        )
    }

    #[test]
    fn missing_record_pat_field_ref() {
        check_diagnostics(
            r"
struct S { s: u32 }
fn x(a: S) {
    let S { ref s } = a;
}
",
        )
    }

    #[test]
    fn simple_free_fn_zero() {
        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(1); }
       //^^^^^^^ Expected 0 arguments, found 1
"#,
        );

        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(); }
"#,
        );
    }

    #[test]
    fn simple_free_fn_one() {
        check_diagnostics(
            r#"
fn one(arg: u8) {}
fn f() { one(); }
       //^^^^^ Expected 1 argument, found 0
"#,
        );

        check_diagnostics(
            r#"
fn one(arg: u8) {}
fn f() { one(1); }
"#,
        );
    }

    #[test]
    fn method_as_fn() {
        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self) {} }

fn f() {
    S::method();
} //^^^^^^^^^^^ Expected 1 argument, found 0
"#,
        );

        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self) {} }

fn f() {
    S::method(&S);
    S.method();
}
"#,
        );
    }

    #[test]
    fn method_with_arg() {
        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self, arg: u8) {} }

            fn f() {
                S.method();
            } //^^^^^^^^^^ Expected 1 argument, found 0
            "#,
        );

        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self, arg: u8) {} }

fn f() {
    S::method(&S, 0);
    S.method(1);
}
"#,
        );
    }

    #[test]
    fn method_unknown_receiver() {
        // note: this is incorrect code, so there might be errors on this in the
        // future, but we shouldn't emit an argument count diagnostic here
        check_diagnostics(
            r#"
trait Foo { fn method(&self, arg: usize) {} }

fn f() {
    let x;
    x.method();
}
"#,
        );
    }

    #[test]
    fn tuple_struct() {
        check_diagnostics(
            r#"
struct Tup(u8, u16);
fn f() {
    Tup(0);
} //^^^^^^ Expected 2 arguments, found 1
"#,
        )
    }

    #[test]
    fn enum_variant() {
        check_diagnostics(
            r#"
enum En { Variant(u8, u16), }
fn f() {
    En::Variant(0);
} //^^^^^^^^^^^^^^ Expected 2 arguments, found 1
"#,
        )
    }

    #[test]
    fn enum_variant_type_macro() {
        check_diagnostics(
            r#"
macro_rules! Type {
    () => { u32 };
}
enum Foo {
    Bar(Type![])
}
impl Foo {
    fn new() {
        Foo::Bar(0);
        Foo::Bar(0, 1);
      //^^^^^^^^^^^^^^ Expected 1 argument, found 2
        Foo::Bar();
      //^^^^^^^^^^ Expected 1 argument, found 0
    }
}
        "#,
        );
    }

    #[test]
    fn varargs() {
        check_diagnostics(
            r#"
extern "C" {
    fn fixed(fixed: u8);
    fn varargs(fixed: u8, ...);
    fn varargs2(...);
}

fn f() {
    unsafe {
        fixed(0);
        fixed(0, 1);
      //^^^^^^^^^^^ Expected 1 argument, found 2
        varargs(0);
        varargs(0, 1);
        varargs2();
        varargs2(0);
        varargs2(0, 1);
    }
}
        "#,
        )
    }

    #[test]
    fn arg_count_lambda() {
        check_diagnostics(
            r#"
fn main() {
    let f = |()| ();
    f();
  //^^^ Expected 1 argument, found 0
    f(());
    f((), ());
  //^^^^^^^^^ Expected 1 argument, found 2
}
"#,
        )
    }

    #[test]
    fn cfgd_out_call_arguments() {
        check_diagnostics(
            r#"
struct C(#[cfg(FALSE)] ());
impl C {
    fn new() -> Self {
        Self(
            #[cfg(FALSE)]
            (),
        )
    }

    fn method(&self) {}
}

fn main() {
    C::new().method(#[cfg(FALSE)] 0);
}
            "#,
        );
    }

    #[test]
    fn cfgd_out_fn_params() {
        check_diagnostics(
            r#"
fn foo(#[cfg(NEVER)] x: ()) {}

struct S;

impl S {
    fn method(#[cfg(NEVER)] self) {}
    fn method2(#[cfg(NEVER)] self, arg: u8) {}
    fn method3(self, #[cfg(NEVER)] arg: u8) {}
}

extern "C" {
    fn fixed(fixed: u8, #[cfg(NEVER)] ...);
    fn varargs(#[cfg(not(NEVER))] ...);
}

fn main() {
    foo();
    S::method();
    S::method2(0);
    S::method3(S);
    S.method3();
    unsafe {
        fixed(0);
        varargs(1, 2, 3);
    }
}
            "#,
        )
    }

    #[test]
    fn missing_semicolon() {
        check_diagnostics(
            r#"
                fn test() -> i32 { 123; }
                                 //^^^ Remove this semicolon
            "#,
        );
    }

    #[test]
    fn import_extern_crate_clash_with_inner_item() {
        // This is more of a resolver test, but doesn't really work with the hir_def testsuite.

        check_diagnostics(
            r#"
//- /lib.rs crate:lib deps:jwt
mod permissions;

use permissions::jwt;

fn f() {
    fn inner() {}
    jwt::Claims {}; // should resolve to the local one with 0 fields, and not get a diagnostic
}

//- /permissions.rs
pub mod jwt  {
    pub struct Claims {}
}

//- /jwt/lib.rs crate:jwt
pub struct Claims {
    field: u8,
}
        "#,
        );
    }
}

#[cfg(test)]
pub(super) mod match_check_tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn empty_tuple() {
        check_diagnostics(
            r#"
fn main() {
    match () { }
        //^^ Missing match arm
    match (()) { }
        //^^^^ Missing match arm

    match () { _ => (), }
    match () { () => (), }
    match (()) { (()) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_two_empty_tuple() {
        check_diagnostics(
            r#"
fn main() {
    match ((), ()) { }
        //^^^^^^^^ Missing match arm

    match ((), ()) { ((), ()) => (), }
}
"#,
        );
    }

    #[test]
    fn boolean() {
        check_diagnostics(
            r#"
fn test_main() {
    match false { }
        //^^^^^ Missing match arm
    match false { true => (), }
        //^^^^^ Missing match arm
    match (false, true) {}
        //^^^^^^^^^^^^^ Missing match arm
    match (false, true) { (true, true) => (), }
        //^^^^^^^^^^^^^ Missing match arm
    match (false, true) {
        //^^^^^^^^^^^^^ Missing match arm
        (false, true) => (),
        (false, false) => (),
        (true, false) => (),
    }
    match (false, true) { (true, _x) => (), }
        //^^^^^^^^^^^^^ Missing match arm

    match false { true => (), false => (), }
    match (false, true) {
        (false, _) => (),
        (true, false) => (),
        (_, true) => (),
    }
    match (false, true) {
        (true, true) => (),
        (true, false) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true) {
        (true, _x) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true, false) {
        (false, ..) => (),
        (true, ..) => (),
    }
    match (false, true, false) {
        (.., false) => (),
        (.., true) => (),
    }
    match (false, true, false) { (..) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_tuple_and_bools() {
        check_diagnostics(
            r#"
fn main() {
    match (false, ((), false)) {}
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
    match (false, ((), false)) { (true, ((), true)) => (), }
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
    match (false, ((), false)) { (true, _) => (), }
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm

    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, ((), true)) => (),
        (false, ((), false)) => (),
    }
    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn enums() {
        check_diagnostics(
            r#"
enum Either { A, B, }

fn main() {
    match Either::A { }
        //^^^^^^^^^ Missing match arm
    match Either::B { Either::A => (), }
        //^^^^^^^^^ Missing match arm

    match &Either::B {
        //^^^^^^^^^^ Missing match arm
        Either::A => (),
    }

    match Either::B {
        Either::A => (), Either::B => (),
    }
    match &Either::B {
        Either::A => (), Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_containing_bool() {
        check_diagnostics(
            r#"
enum Either { A(bool), B }

fn main() {
    match Either::B { }
        //^^^^^^^^^ Missing match arm
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true) => (), Either::B => ()
    }

    match Either::B {
        Either::A(true) => (),
        Either::A(false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::B => (),
        _ => (),
    }
    match Either::B {
        Either::A(_) => (),
        Either::B => (),
    }

}
        "#,
        );
    }

    #[test]
    fn enum_different_sizes() {
        check_diagnostics(
            r#"
enum Either { A(bool), B(bool, bool) }

fn main() {
    match Either::A(false) {
        //^^^^^^^^^^^^^^^^ Missing match arm
        Either::A(_) => (),
        Either::B(false, _) => (),
    }

    match Either::A(false) {
        Either::A(_) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
    match Either::A(false) {
        Either::A(true) | Either::A(false) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_enum_no_diagnostic() {
        check_diagnostics(
            r#"
enum Either { A(bool), B(bool, bool) }
enum Either2 { C, D }

fn main() {
    match (Either::A(false), Either2::C) {
        (Either::A(true), _) | (Either::A(false), _) => (),
        (Either::B(true, _), Either2::C) => (),
        (Either::B(false, _), Either2::C) => (),
        (Either::B(_, _), Either2::D) => (),
    }
}
"#,
        );
    }

    #[test]
    fn or_pattern_no_diagnostic() {
        check_diagnostics(
            r#"
enum Either {A, B}

fn main() {
    match (Either::A, Either::B) {
        (Either::A | Either::B, _) => (),
    }
}"#,
        )
    }

    #[test]
    fn mismatched_types() {
        // Match statements with arms that don't match the
        // expression pattern do not fire this diagnostic.
        check_diagnostics(
            r#"
enum Either { A, B }
enum Either2 { C, D }

fn main() {
    match Either::A {
        Either2::C => (),
    //  ^^^^^^^^^^ Internal: match check bailed out
        Either2::D => (),
    }
    match (true, false) {
        (true, false, true) => (),
    //  ^^^^^^^^^^^^^^^^^^^ Internal: match check bailed out
        (true) => (),
    }
    match (true, false) { (true,) => {} }
    //                    ^^^^^^^ Internal: match check bailed out
    match (0) { () => () }
            //  ^^ Internal: match check bailed out
    match Unresolved::Bar { Unresolved::Baz => () }
}
        "#,
        );
    }

    #[test]
    fn mismatched_types_in_or_patterns() {
        check_diagnostics(
            r#"
fn main() {
    match false { true | () => {} }
    //            ^^^^^^^^^ Internal: match check bailed out
    match (false,) { (true | (),) => {} }
    //               ^^^^^^^^^^^^ Internal: match check bailed out
}
"#,
        );
    }

    #[test]
    fn malformed_match_arm_tuple_enum_missing_pattern() {
        // We are testing to be sure we don't panic here when the match
        // arm `Either::B` is missing its pattern.
        check_diagnostics(
            r#"
enum Either { A, B(u32) }

fn main() {
    match Either::A {
        Either::A => (),
        Either::B() => (),
    }
}
"#,
        );
    }

    #[test]
    fn malformed_match_arm_extra_fields() {
        check_diagnostics(
            r#"
enum A { B(isize, isize), C }
fn main() {
    match A::B(1, 2) {
        A::B(_, _, _) => (),
    //  ^^^^^^^^^^^^^ Internal: match check bailed out
    }
    match A::B(1, 2) {
        A::C(_) => (),
    //  ^^^^^^^ Internal: match check bailed out
    }
}
"#,
        );
    }

    #[test]
    fn expr_diverges() {
        check_diagnostics(
            r#"
enum Either { A, B }

fn main() {
    match loop {} {
        Either::A => (),
    //  ^^^^^^^^^ Internal: match check bailed out
        Either::B => (),
    }
    match loop {} {
        Either::A => (),
    //  ^^^^^^^^^ Internal: match check bailed out
    }
    match loop { break Foo::A } {
        //^^^^^^^^^^^^^^^^^^^^^ Missing match arm
        Either::A => (),
    }
    match loop { break Foo::A } {
        Either::A => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn expr_partially_diverges() {
        check_diagnostics(
            r#"
enum Either<T> { A(T), B }

fn foo() -> Either<!> { Either::B }
fn main() -> u32 {
    match foo() {
        Either::A(val) => val,
        Either::B => 0,
    }
}
"#,
        );
    }

    #[test]
    fn enum_record() {
        check_diagnostics(
            r#"
enum Either { A { foo: bool }, B }

fn main() {
    let a = Either::A { foo: true };
    match a { }
        //^ Missing match arm
    match a { Either::A { foo: true } => () }
        //^ Missing match arm
    match a {
        Either::A { } => (),
      //^^^^^^^^^ Missing structure fields:
      //        | - foo
        Either::B => (),
    }
    match a {
        //^ Missing match arm
        Either::A { } => (),
    } //^^^^^^^^^ Missing structure fields:
      //        | - foo

    match a {
        Either::A { foo: true } => (),
        Either::A { foo: false } => (),
        Either::B => (),
    }
    match a {
        Either::A { foo: _ } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_fields_out_of_order() {
        check_diagnostics(
            r#"
enum Either {
    A { foo: bool, bar: () },
    B,
}

fn main() {
    let a = Either::A { foo: true, bar: () };
    match a {
        //^ Missing match arm
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
    }

    match a {
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_ellipsis() {
        check_diagnostics(
            r#"
enum Either {
    A { foo: bool, bar: bool },
    B,
}

fn main() {
    let a = Either::B;
    match a {
        //^ Missing match arm
        Either::A { foo: true, .. } => (),
        Either::B => (),
    }
    match a {
        //^ Missing match arm
        Either::A { .. } => (),
    }

    match a {
        Either::A { foo: true, .. } => (),
        Either::A { foo: false, .. } => (),
        Either::B => (),
    }

    match a {
        Either::A { .. } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_tuple_partial_ellipsis() {
        check_diagnostics(
            r#"
enum Either {
    A(bool, bool, bool, bool),
    B,
}

fn main() {
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::B => (),
    }

    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., true) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::A(.., false) => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn never() {
        check_diagnostics(
            r#"
enum Never {}

fn enum_(never: Never) {
    match never {}
}
fn enum_ref(never: &Never) {
    match never {}
        //^^^^^ Missing match arm
}
fn bang(never: !) {
    match never {}
}
"#,
        );
    }

    #[test]
    fn unknown_type() {
        check_diagnostics(
            r#"
enum Option<T> { Some(T), None }

fn main() {
    // `Never` is deliberately not defined so that it's an uninferred type.
    match Option::<Never>::None {
        None => (),
        Some(never) => match never {},
    //  ^^^^^^^^^^^ Internal: match check bailed out
    }
    match Option::<Never>::None {
        //^^^^^^^^^^^^^^^^^^^^^ Missing match arm
        Option::Some(_never) => {},
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_end_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (false, ..) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_beginning_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (.., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_in_middle_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (true, .., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn record_struct() {
        check_diagnostics(
            r#"struct Foo { a: bool }
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo { a: true } => () }
        //^ Missing match arm
    match &f { Foo { a: true } => () }
        //^^ Missing match arm
    match f { Foo { a: _ } => () }
    match f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
    match &f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_struct() {
        check_diagnostics(
            r#"struct Foo(bool);
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo(true) => () }
        //^ Missing match arm
    match f {
        Foo(true) => (),
        Foo(false) => (),
    }
}
"#,
        );
    }

    #[test]
    fn unit_struct() {
        check_diagnostics(
            r#"struct Foo;
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo => () }
}
"#,
        );
    }

    #[test]
    fn record_struct_ellipsis() {
        check_diagnostics(
            r#"struct Foo { foo: bool, bar: bool }
fn main(f: Foo) {
    match f { Foo { foo: true, .. } => () }
        //^ Missing match arm
    match f {
        //^ Missing match arm
        Foo { foo: true, .. } => (),
        Foo { bar: false, .. } => ()
    }
    match f { Foo { .. } => () }
    match f {
        Foo { foo: true, .. } => (),
        Foo { foo: false, .. } => ()
    }
}
"#,
        );
    }

    #[test]
    fn internal_or() {
        check_diagnostics(
            r#"
fn main() {
    enum Either { A(bool), B }
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true | false) => (),
    }
}
"#,
        );
    }

    #[test]
    fn no_panic_at_unimplemented_subpattern_type() {
        check_diagnostics(
            r#"
struct S { a: char}
fn main(v: S) {
    match v { S{ a }      => {} }
    match v { S{ a: _x }  => {} }
    match v { S{ a: 'a' } => {} }
            //^^^^^^^^^^^ Internal: match check bailed out
    match v { S{..}       => {} }
    match v { _           => {} }
    match v { }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn binding() {
        check_diagnostics(
            r#"
fn main() {
    match true {
        _x @ true => {}
        false     => {}
    }
    match true { _x @ true => {} }
        //^^^^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn binding_ref_has_correct_type() {
        // Asserts `PatKind::Binding(ref _x): bool`, not &bool.
        // If that's not true match checking will panic with "incompatible constructors"
        // FIXME: make facilities to test this directly like `tests::check_infer(..)`
        check_diagnostics(
            r#"
enum Foo { A }
fn main() {
    // FIXME: this should not bail out but current behavior is such as the old algorithm.
    // ExprValidator::validate_match(..) checks types of top level patterns incorrecly.
    match Foo::A {
        ref _x => {}
    //  ^^^^^^ Internal: match check bailed out
        Foo::A => {}
    }
    match (true,) {
        (ref _x,) => {}
        (true,) => {}
    }
}
"#,
        );
    }

    #[test]
    fn enum_non_exhaustive() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib
#[non_exhaustive]
pub enum E { A, B }
fn _local() {
    match E::A { _ => {} }
    match E::A {
        E::A => {}
        E::B => {}
    }
    match E::A {
        E::A | E::B => {}
    }
}

//- /main.rs crate:main deps:lib
use lib::E;
fn main() {
    match E::A { _ => {} }
    match E::A {
        //^^^^ Missing match arm
        E::A => {}
        E::B => {}
    }
    match E::A {
        //^^^^ Missing match arm
        E::A | E::B => {}
    }
}
"#,
        );
    }

    #[test]
    fn match_guard() {
        check_diagnostics(
            r#"
fn main() {
    match true {
        true if false => {}
        true          => {}
        false         => {}
    }
    match true {
        //^^^^ Missing match arm
        true if false => {}
        false         => {}
    }
}
"#,
        );
    }

    #[test]
    fn pattern_type_is_of_substitution() {
        cov_mark::check!(match_check_wildcard_expanded_to_substitutions);
        check_diagnostics(
            r#"
struct Foo<T>(T);
struct Bar;
fn main() {
    match Foo(Bar) {
        _ | Foo(Bar) => {}
    }
}
"#,
        );
    }

    #[test]
    fn record_struct_no_such_field() {
        check_diagnostics(
            r#"
struct Foo { }
fn main(f: Foo) {
    match f { Foo { bar } => () }
    //        ^^^^^^^^^^^ Internal: match check bailed out
}
"#,
        );
    }

    #[test]
    fn match_ergonomics_issue_9095() {
        check_diagnostics(
            r#"
enum Foo<T> { A(T) }
fn main() {
    match &Foo::A(true) {
        _ => {}
        Foo::A(_) => {}
    }
}
"#,
        );
    }

    mod false_negatives {
        //! The implementation of match checking here is a work in progress. As we roll this out, we
        //! prefer false negatives to false positives (ideally there would be no false positives). This
        //! test module should document known false negatives. Eventually we will have a complete
        //! implementation of match checking and this module will be empty.
        //!
        //! The reasons for documenting known false negatives:
        //!
        //!   1. It acts as a backlog of work that can be done to improve the behavior of the system.
        //!   2. It ensures the code doesn't panic when handling these cases.
        use super::*;

        #[test]
        fn integers() {
            // We don't currently check integer exhaustiveness.
            check_diagnostics(
                r#"
fn main() {
    match 5 {
        10 => (),
    //  ^^ Internal: match check bailed out
        11..20 => (),
    }
}
"#,
            );
        }

        #[test]
        fn reference_patterns_at_top_level() {
            check_diagnostics(
                r#"
fn main() {
    match &false {
        &true => {}
    //  ^^^^^ Internal: match check bailed out
    }
}
            "#,
            );
        }

        #[test]
        fn reference_patterns_in_fields() {
            check_diagnostics(
                r#"
fn main() {
    match (&false,) {
        (true,) => {}
    //  ^^^^^^^ Internal: match check bailed out
    }
    match (&false,) {
        (&true,) => {}
    //  ^^^^^^^^ Internal: match check bailed out
    }
}
            "#,
            );
        }
    }
}

#[cfg(test)]
mod decl_check_tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn incorrect_function_name() {
        check_diagnostics(
            r#"
fn NonSnakeCaseName() {}
// ^^^^^^^^^^^^^^^^ Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
"#,
        );
    }

    #[test]
    fn incorrect_function_params() {
        check_diagnostics(
            r#"
fn foo(SomeParam: u8) {}
    // ^^^^^^^^^ Parameter `SomeParam` should have snake_case name, e.g. `some_param`

fn foo2(ok_param: &str, CAPS_PARAM: u8) {}
                     // ^^^^^^^^^^ Parameter `CAPS_PARAM` should have snake_case name, e.g. `caps_param`
"#,
        );
    }

    #[test]
    fn incorrect_variable_names() {
        check_diagnostics(
            r#"
fn foo() {
    let SOME_VALUE = 10;
     // ^^^^^^^^^^ Variable `SOME_VALUE` should have snake_case name, e.g. `some_value`
    let AnotherValue = 20;
     // ^^^^^^^^^^^^ Variable `AnotherValue` should have snake_case name, e.g. `another_value`
}
"#,
        );
    }

    #[test]
    fn incorrect_struct_names() {
        check_diagnostics(
            r#"
struct non_camel_case_name {}
    // ^^^^^^^^^^^^^^^^^^^ Structure `non_camel_case_name` should have CamelCase name, e.g. `NonCamelCaseName`

struct SCREAMING_CASE {}
    // ^^^^^^^^^^^^^^ Structure `SCREAMING_CASE` should have CamelCase name, e.g. `ScreamingCase`
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_camel_cased_acronyms_in_struct_name() {
        check_diagnostics(
            r#"
struct AABB {}
"#,
        );
    }

    #[test]
    fn incorrect_struct_field() {
        check_diagnostics(
            r#"
struct SomeStruct { SomeField: u8 }
                 // ^^^^^^^^^ Field `SomeField` should have snake_case name, e.g. `some_field`
"#,
        );
    }

    #[test]
    fn incorrect_enum_names() {
        check_diagnostics(
            r#"
enum some_enum { Val(u8) }
  // ^^^^^^^^^ Enum `some_enum` should have CamelCase name, e.g. `SomeEnum`

enum SOME_ENUM {}
  // ^^^^^^^^^ Enum `SOME_ENUM` should have CamelCase name, e.g. `SomeEnum`
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_camel_cased_acronyms_in_enum_name() {
        check_diagnostics(
            r#"
enum AABB {}
"#,
        );
    }

    #[test]
    fn incorrect_enum_variant_name() {
        check_diagnostics(
            r#"
enum SomeEnum { SOME_VARIANT(u8) }
             // ^^^^^^^^^^^^ Variant `SOME_VARIANT` should have CamelCase name, e.g. `SomeVariant`
"#,
        );
    }

    #[test]
    fn incorrect_const_name() {
        check_diagnostics(
            r#"
const some_weird_const: u8 = 10;
   // ^^^^^^^^^^^^^^^^ Constant `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`
"#,
        );
    }

    #[test]
    fn incorrect_static_name() {
        check_diagnostics(
            r#"
static some_weird_const: u8 = 10;
    // ^^^^^^^^^^^^^^^^ Static variable `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`
"#,
        );
    }

    #[test]
    fn fn_inside_impl_struct() {
        check_diagnostics(
            r#"
struct someStruct;
    // ^^^^^^^^^^ Structure `someStruct` should have CamelCase name, e.g. `SomeStruct`

impl someStruct {
    fn SomeFunc(&self) {
    // ^^^^^^^^ Function `SomeFunc` should have snake_case name, e.g. `some_func`
        let WHY_VAR_IS_CAPS = 10;
         // ^^^^^^^^^^^^^^^ Variable `WHY_VAR_IS_CAPS` should have snake_case name, e.g. `why_var_is_caps`
    }
}
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_enum_varinats() {
        check_diagnostics(
            r#"
enum Option { Some, None }

fn main() {
    match Option::None {
        None => (),
        Some => (),
    }
}
"#,
        );
    }

    #[test]
    fn non_let_bind() {
        check_diagnostics(
            r#"
enum Option { Some, None }

fn main() {
    match Option::None {
        SOME_VAR @ None => (),
     // ^^^^^^^^ Variable `SOME_VAR` should have snake_case name, e.g. `some_var`
        Some => (),
    }
}
"#,
        );
    }

    #[test]
    fn allow_attributes() {
        check_diagnostics(
            r#"
#[allow(non_snake_case)]
fn NonSnakeCaseName(SOME_VAR: u8) -> u8{
    // cov_flags generated output from elsewhere in this file
    extern "C" {
        #[no_mangle]
        static lower_case: u8;
    }

    let OtherVar = SOME_VAR + 1;
    OtherVar
}

#[allow(nonstandard_style)]
mod CheckNonstandardStyle {
    fn HiImABadFnName() {}
}

#[allow(bad_style)]
mod CheckBadStyle {
    fn HiImABadFnName() {}
}

mod F {
    #![allow(non_snake_case)]
    fn CheckItWorksWithModAttr(BAD_NAME_HI: u8) {}
}

#[allow(non_snake_case, non_camel_case_types)]
pub struct some_type {
    SOME_FIELD: u8,
    SomeField: u16,
}

#[allow(non_upper_case_globals)]
pub const some_const: u8 = 10;

#[allow(non_upper_case_globals)]
pub static SomeStatic: u8 = 10;
    "#,
        );
    }

    #[test]
    fn allow_attributes_crate_attr() {
        check_diagnostics(
            r#"
#![allow(non_snake_case)]

mod F {
    fn CheckItWorksWithCrateAttr(BAD_NAME_HI: u8) {}
}
    "#,
        );
    }

    #[test]
    #[ignore]
    fn bug_trait_inside_fn() {
        // FIXME:
        // This is broken, and in fact, should not even be looked at by this
        // lint in the first place. There's weird stuff going on in the
        // collection phase.
        // It's currently being brought in by:
        // * validate_func on `a` recursing into modules
        // * then it finds the trait and then the function while iterating
        //   through modules
        // * then validate_func is called on Dirty
        // * ... which then proceeds to look at some unknown module taking no
        //   attrs from either the impl or the fn a, and then finally to the root
        //   module
        //
        // It should find the attribute on the trait, but it *doesn't even see
        // the trait* as far as I can tell.

        check_diagnostics(
            r#"
trait T { fn a(); }
struct U {}
impl T for U {
    fn a() {
        // this comes out of bitflags, mostly
        #[allow(non_snake_case)]
        trait __BitFlags {
            const HiImAlsoBad: u8 = 2;
            #[inline]
            fn Dirty(&self) -> bool {
                false
            }
        }

    }
}
    "#,
        );
    }

    #[test]
    #[ignore]
    fn bug_traits_arent_checked() {
        // FIXME: Traits and functions in traits aren't currently checked by
        // r-a, even though rustc will complain about them.
        check_diagnostics(
            r#"
trait BAD_TRAIT {
    // ^^^^^^^^^ Trait `BAD_TRAIT` should have CamelCase name, e.g. `BadTrait`
    fn BAD_FUNCTION();
    // ^^^^^^^^^^^^ Function `BAD_FUNCTION` should have snake_case name, e.g. `bad_function`
    fn BadFunction();
    // ^^^^^^^^^^^^ Function `BadFunction` should have snake_case name, e.g. `bad_function`
}
    "#,
        );
    }

    #[test]
    fn ignores_extern_items() {
        cov_mark::check!(extern_func_incorrect_case_ignored);
        cov_mark::check!(extern_static_incorrect_case_ignored);
        check_diagnostics(
            r#"
extern {
    fn NonSnakeCaseName(SOME_VAR: u8) -> u8;
    pub static SomeStatic: u8 = 10;
}
            "#,
        );
    }

    #[test]
    fn infinite_loop_inner_items() {
        check_diagnostics(
            r#"
fn qualify() {
    mod foo {
        use super::*;
    }
}
            "#,
        )
    }

    #[test] // Issue #8809.
    fn parenthesized_parameter() {
        check_diagnostics(r#"fn f((O): _) {}"#)
    }
}
