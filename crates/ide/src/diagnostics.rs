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
mod missing_match_arms;
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
    let module = sema.to_module_def(file_id);
    if let Some(m) = module {
        diags = m.diagnostics(db, &mut sink)
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
            AnyDiagnostic::MissingMatchArms(d) => missing_match_arms::missing_match_arms(&ctx, &d),
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
