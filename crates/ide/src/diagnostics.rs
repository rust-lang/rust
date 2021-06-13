//! Collects diagnostics & fixits  for a single file.
//!
//! The tricky bit here is that diagnostics are produced by hir in terms of
//! macro-expanded files, but we need to present them to the users in terms of
//! original files. So we need to map the ranges.

mod break_outside_of_loop;
mod inactive_code;
mod incorrect_case;
mod macro_error;
mod mismatched_arg_count;
mod missing_fields;
mod missing_ok_or_some_in_tail_expr;
mod missing_unsafe;
mod no_such_field;
mod remove_this_semicolon;
mod replace_filter_map_next_with_find_map;
mod unimplemented_builtin_macro;
mod unlinked_file;
mod unresolved_extern_crate;
mod unresolved_import;
mod unresolved_macro_call;
mod unresolved_module;
mod unresolved_proc_macro;

mod field_shorthand;

use std::cell::RefCell;

use hir::{
    diagnostics::{AnyDiagnostic, DiagnosticCode, DiagnosticSinkBuilder},
    Semantics,
};
use ide_assists::AssistResolveStrategy;
use ide_db::{base_db::SourceDatabase, RootDatabase};
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, AstNode},
    SyntaxNode, TextRange,
};
use text_edit::TextEdit;
use unlinked_file::UnlinkedFile;

use crate::{Assist, AssistId, AssistKind, FileId, Label, SourceChange};

#[derive(Debug)]
pub struct Diagnostic {
    // pub name: Option<String>,
    pub message: String,
    pub range: TextRange,
    pub severity: Severity,
    pub fixes: Option<Vec<Assist>>,
    pub unused: bool,
    pub code: Option<DiagnosticCode>,
    pub experimental: bool,
}

impl Diagnostic {
    fn new(code: &'static str, message: impl Into<String>, range: TextRange) -> Diagnostic {
        let message = message.into();
        let code = Some(DiagnosticCode(code));
        Self {
            message,
            range,
            severity: Severity::Error,
            fixes: None,
            unused: false,
            code,
            experimental: false,
        }
    }

    fn experimental(mut self) -> Diagnostic {
        self.experimental = true;
        self
    }

    fn severity(mut self, severity: Severity) -> Diagnostic {
        self.severity = severity;
        self
    }

    fn error(range: TextRange, message: String) -> Self {
        Self {
            message,
            range,
            severity: Severity::Error,
            fixes: None,
            unused: false,
            code: None,
            experimental: false,
        }
    }

    fn hint(range: TextRange, message: String) -> Self {
        Self {
            message,
            range,
            severity: Severity::WeakWarning,
            fixes: None,
            unused: false,
            code: None,
            experimental: false,
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
    let module = sema.to_module_def(file_id);
    if let Some(m) = module {
        diags = m.diagnostics(db, &mut sink, internal_diagnostics)
    }

    drop(sink);

    let mut res = res.into_inner();

    let ctx = DiagnosticsContext { config, sema, resolve };
    if module.is_none() {
        let d = UnlinkedFile { file: file_id };
        let d = unlinked_file::unlinked_file(&ctx, &d);
        res.push(d)
    }

    for diag in diags {
        #[rustfmt::skip]
        let d = match diag {
            AnyDiagnostic::BreakOutsideOfLoop(d) => break_outside_of_loop::break_outside_of_loop(&ctx, &d),
            AnyDiagnostic::IncorrectCase(d) => incorrect_case::incorrect_case(&ctx, &d),
            AnyDiagnostic::MacroError(d) => macro_error::macro_error(&ctx, &d),
            AnyDiagnostic::MismatchedArgCount(d) => mismatched_arg_count::mismatched_arg_count(&ctx, &d),
            AnyDiagnostic::MissingFields(d) => missing_fields::missing_fields(&ctx, &d),
            AnyDiagnostic::MissingOkOrSomeInTailExpr(d) => missing_ok_or_some_in_tail_expr::missing_ok_or_some_in_tail_expr(&ctx, &d),
            AnyDiagnostic::MissingUnsafe(d) => missing_unsafe::missing_unsafe(&ctx, &d),
            AnyDiagnostic::NoSuchField(d) => no_such_field::no_such_field(&ctx, &d),
            AnyDiagnostic::RemoveThisSemicolon(d) => remove_this_semicolon::remove_this_semicolon(&ctx, &d),
            AnyDiagnostic::ReplaceFilterMapNextWithFindMap(d) => replace_filter_map_next_with_find_map::replace_filter_map_next_with_find_map(&ctx, &d),
            AnyDiagnostic::UnimplementedBuiltinMacro(d) => unimplemented_builtin_macro::unimplemented_builtin_macro(&ctx, &d),
            AnyDiagnostic::UnresolvedExternCrate(d) => unresolved_extern_crate::unresolved_extern_crate(&ctx, &d),
            AnyDiagnostic::UnresolvedImport(d) => unresolved_import::unresolved_import(&ctx, &d),
            AnyDiagnostic::UnresolvedMacroCall(d) => unresolved_macro_call::unresolved_macro_call(&ctx, &d),
            AnyDiagnostic::UnresolvedModule(d) => unresolved_module::unresolved_module(&ctx, &d),
            AnyDiagnostic::UnresolvedProcMacro(d) => unresolved_proc_macro::unresolved_proc_macro(&ctx, &d),

            AnyDiagnostic::InactiveCode(d) => match inactive_code::inactive_code(&ctx, &d) {
                Some(it) => it,
                None => continue,
            }
        };
        res.push(d)
    }

    res.retain(|d| {
        if let Some(code) = d.code {
            if ctx.config.disabled.contains(code.as_str()) {
                return false;
            }
        }
        if ctx.config.disable_experimental && d.experimental {
            return false;
        }
        true
    });

    res
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
    pub(crate) fn check_no_fix(ra_fixture: &str) {
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

    pub(crate) fn check_expect(ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let diagnostics = analysis
            .diagnostics(&DiagnosticsConfig::default(), AssistResolveStrategy::All, file_id)
            .unwrap();
        expect.assert_debug_eq(&diagnostics)
    }

    #[track_caller]
    pub(crate) fn check_diagnostics(ra_fixture: &str) {
        let mut config = DiagnosticsConfig::default();
        config.disabled.insert("inactive-code".to_string());
        check_diagnostics_with_config(config, ra_fixture)
    }

    #[track_caller]
    pub(crate) fn check_diagnostics_with_config(config: DiagnosticsConfig, ra_fixture: &str) {
        let (analysis, files) = fixture::files(ra_fixture);
        for file_id in files {
            let diagnostics =
                analysis.diagnostics(&config, AssistResolveStrategy::All, file_id).unwrap();

            let expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
            let mut actual =
                diagnostics.into_iter().map(|d| (d.range, d.message)).collect::<Vec<_>>();
            actual.sort_by_key(|(range, _)| range.start());
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_check_unnecessary_braces_in_use_statement() {
        check_diagnostics(
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
        check_diagnostics(
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
