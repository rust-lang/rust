//! Diagnostics rendering and fixits.
//!
//! Most of the diagnostics originate from the dark depth of the compiler, and
//! are originally expressed in term of IR. When we emit the diagnostic, we are
//! usually not in the position to decide how to best "render" it in terms of
//! user-authored source code. We are especially not in the position to offer
//! fixits, as the compiler completely lacks the infrastructure to edit the
//! source code.
//!
//! Instead, we "bubble up" raw, structured diagnostics until the `hir` crate,
//! where we "cook" them so that each diagnostic is formulated in terms of `hir`
//! types. Well, at least that's the aspiration, the "cooking" is somewhat
//! ad-hoc at the moment. Anyways, we get a bunch of ide-friendly diagnostic
//! structs from hir, and we want to render them to unified serializable
//! representation (span, level, message) here. If we can, we also provide
//! fixits. By the way, that's why we want to keep diagnostics structured
//! internally -- so that we have all the info to make fixes.
//!
//! We have one "handler" module per diagnostic code. Such a module contains
//! rendering, optional fixes and tests. It's OK if some low-level compiler
//! functionality ends up being tested via a diagnostic.
//!
//! There are also a couple of ad-hoc diagnostics implemented directly here, we
//! don't yet have a great pattern for how to do them properly.

mod handlers {
    pub(crate) mod break_outside_of_loop;
    pub(crate) mod inactive_code;
    pub(crate) mod incorrect_case;
    pub(crate) mod macro_error;
    pub(crate) mod mismatched_arg_count;
    pub(crate) mod missing_fields;
    pub(crate) mod missing_match_arms;
    pub(crate) mod missing_ok_or_some_in_tail_expr;
    pub(crate) mod missing_unsafe;
    pub(crate) mod no_such_field;
    pub(crate) mod remove_this_semicolon;
    pub(crate) mod replace_filter_map_next_with_find_map;
    pub(crate) mod unimplemented_builtin_macro;
    pub(crate) mod unresolved_extern_crate;
    pub(crate) mod unresolved_import;
    pub(crate) mod unresolved_macro_call;
    pub(crate) mod unresolved_module;
    pub(crate) mod unresolved_proc_macro;

    // The handlers bellow are unusual, the implement the diagnostics as well.
    pub(crate) mod field_shorthand;
    pub(crate) mod useless_braces;
    pub(crate) mod unlinked_file;
}

use hir::{diagnostics::AnyDiagnostic, Semantics};
use ide_db::{
    assists::{Assist, AssistId, AssistKind, AssistResolveStrategy},
    base_db::{FileId, SourceDatabase},
    label::Label,
    source_change::SourceChange,
    RootDatabase,
};
use rustc_hash::FxHashSet;
use syntax::{ast::AstNode, TextRange};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagnosticCode(pub &'static str);

impl DiagnosticCode {
    pub fn as_str(&self) -> &str {
        self.0
    }
}

#[derive(Debug)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    pub range: TextRange,
    pub severity: Severity,
    pub unused: bool,
    pub experimental: bool,
    pub fixes: Option<Vec<Assist>>,
}

impl Diagnostic {
    fn new(code: &'static str, message: impl Into<String>, range: TextRange) -> Diagnostic {
        let message = message.into();
        Diagnostic {
            code: DiagnosticCode(code),
            message,
            range,
            severity: Severity::Error,
            unused: false,
            experimental: false,
            fixes: None,
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

    fn with_fixes(mut self, fixes: Option<Vec<Assist>>) -> Diagnostic {
        self.fixes = fixes;
        self
    }

    fn with_unused(mut self, unused: bool) -> Diagnostic {
        self.unused = unused;
        self
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    Error,
    // We don't actually emit this one yet, but we should at some point.
    // Warning,
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

pub fn diagnostics(
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
        parse.errors().iter().take(128).map(|err| {
            Diagnostic::new("syntax-error", format!("Syntax Error: {}", err), err.range())
        }),
    );

    for node in parse.tree().syntax().descendants() {
        handlers::useless_braces::useless_braces(&mut res, file_id, &node);
        handlers::field_shorthand::field_shorthand(&mut res, file_id, &node);
    }

    let module = sema.to_module_def(file_id);

    let ctx = DiagnosticsContext { config, sema, resolve };
    if module.is_none() {
        handlers::unlinked_file::unlinked_file(&ctx, &mut res, file_id);
    }

    let mut diags = Vec::new();
    if let Some(m) = module {
        m.diagnostics(db, &mut diags)
    }

    for diag in diags {
        #[rustfmt::skip]
        let d = match diag {
            AnyDiagnostic::BreakOutsideOfLoop(d) => handlers::break_outside_of_loop::break_outside_of_loop(&ctx, &d),
            AnyDiagnostic::IncorrectCase(d) => handlers::incorrect_case::incorrect_case(&ctx, &d),
            AnyDiagnostic::MacroError(d) => handlers::macro_error::macro_error(&ctx, &d),
            AnyDiagnostic::MismatchedArgCount(d) => handlers::mismatched_arg_count::mismatched_arg_count(&ctx, &d),
            AnyDiagnostic::MissingFields(d) => handlers::missing_fields::missing_fields(&ctx, &d),
            AnyDiagnostic::MissingMatchArms(d) => handlers::missing_match_arms::missing_match_arms(&ctx, &d),
            AnyDiagnostic::MissingOkOrSomeInTailExpr(d) => handlers::missing_ok_or_some_in_tail_expr::missing_ok_or_some_in_tail_expr(&ctx, &d),
            AnyDiagnostic::MissingUnsafe(d) => handlers::missing_unsafe::missing_unsafe(&ctx, &d),
            AnyDiagnostic::NoSuchField(d) => handlers::no_such_field::no_such_field(&ctx, &d),
            AnyDiagnostic::RemoveThisSemicolon(d) => handlers::remove_this_semicolon::remove_this_semicolon(&ctx, &d),
            AnyDiagnostic::ReplaceFilterMapNextWithFindMap(d) => handlers::replace_filter_map_next_with_find_map::replace_filter_map_next_with_find_map(&ctx, &d),
            AnyDiagnostic::UnimplementedBuiltinMacro(d) => handlers::unimplemented_builtin_macro::unimplemented_builtin_macro(&ctx, &d),
            AnyDiagnostic::UnresolvedExternCrate(d) => handlers::unresolved_extern_crate::unresolved_extern_crate(&ctx, &d),
            AnyDiagnostic::UnresolvedImport(d) => handlers::unresolved_import::unresolved_import(&ctx, &d),
            AnyDiagnostic::UnresolvedMacroCall(d) => handlers::unresolved_macro_call::unresolved_macro_call(&ctx, &d),
            AnyDiagnostic::UnresolvedModule(d) => handlers::unresolved_module::unresolved_module(&ctx, &d),
            AnyDiagnostic::UnresolvedProcMacro(d) => handlers::unresolved_proc_macro::unresolved_proc_macro(&ctx, &d),

            AnyDiagnostic::InactiveCode(d) => match handlers::inactive_code::inactive_code(&ctx, &d) {
                Some(it) => it,
                None => continue,
            }
        };
        res.push(d)
    }

    res.retain(|d| {
        !ctx.config.disabled.contains(d.code.as_str())
            && !(ctx.config.disable_experimental && d.experimental)
    });

    res
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
    use ide_db::{
        assists::AssistResolveStrategy,
        base_db::{fixture::WithFixture, SourceDatabaseExt},
        RootDatabase,
    };
    use stdx::trim_indent;
    use test_utils::{assert_eq_text, extract_annotations};

    use crate::{DiagnosticsConfig, Severity};

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

        let (db, file_position) = RootDatabase::with_position(ra_fixture_before);
        let diagnostic = super::diagnostics(
            &db,
            &DiagnosticsConfig::default(),
            &AssistResolveStrategy::All,
            file_position.file_id,
        )
        .pop()
        .expect("no diagnostics");
        let fix = &diagnostic.fixes.expect("diagnostic misses fixes")[nth];
        let actual = {
            let source_change = fix.source_change.as_ref().unwrap();
            let file_id = *source_change.source_file_edits.keys().next().unwrap();
            let mut actual = db.file_text(file_id).to_string();

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
        let (db, file_position) = RootDatabase::with_position(ra_fixture);
        let diagnostic = super::diagnostics(
            &db,
            &DiagnosticsConfig::default(),
            &AssistResolveStrategy::All,
            file_position.file_id,
        )
        .pop()
        .unwrap();
        assert!(diagnostic.fixes.is_none(), "got a fix when none was expected: {:?}", diagnostic);
    }

    pub(crate) fn check_expect(ra_fixture: &str, expect: Expect) {
        let (db, file_id) = RootDatabase::with_single_file(ra_fixture);
        let diagnostics = super::diagnostics(
            &db,
            &DiagnosticsConfig::default(),
            &AssistResolveStrategy::All,
            file_id,
        );
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
        let (db, files) = RootDatabase::with_many_files(ra_fixture);
        for file_id in files {
            let diagnostics =
                super::diagnostics(&db, &config, &AssistResolveStrategy::All, file_id);

            let expected = extract_annotations(&*db.file_text(file_id));
            let mut actual = diagnostics
                .into_iter()
                .map(|d| {
                    let mut annotation = String::new();
                    if let Some(fixes) = &d.fixes {
                        assert!(!fixes.is_empty());
                        annotation.push_str("💡 ")
                    }
                    annotation.push_str(match d.severity {
                        Severity::Error => "error",
                        Severity::WeakWarning => "weak",
                    });
                    annotation.push_str(": ");
                    annotation.push_str(&d.message);
                    (d.range, annotation)
                })
                .collect::<Vec<_>>();
            actual.sort_by_key(|(range, _)| range.start());
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_disabled_diagnostics() {
        let mut config = DiagnosticsConfig::default();
        config.disabled.insert("unresolved-module".into());

        let (db, file_id) = RootDatabase::with_single_file(r#"mod foo;"#);

        let diagnostics = super::diagnostics(&db, &config, &AssistResolveStrategy::All, file_id);
        assert!(diagnostics.is_empty());

        let diagnostics = super::diagnostics(
            &db,
            &DiagnosticsConfig::default(),
            &AssistResolveStrategy::All,
            file_id,
        );
        assert!(!diagnostics.is_empty());
    }
}
