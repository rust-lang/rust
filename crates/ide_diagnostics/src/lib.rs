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
    pub(crate) mod add_reference_here;
    pub(crate) mod break_outside_of_loop;
    pub(crate) mod inactive_code;
    pub(crate) mod incorrect_case;
    pub(crate) mod invalid_derive_target;
    pub(crate) mod macro_error;
    pub(crate) mod malformed_derive;
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

    // The handlers below are unusual, the implement the diagnostics as well.
    pub(crate) mod field_shorthand;
    pub(crate) mod useless_braces;
    pub(crate) mod unlinked_file;
}

#[cfg(test)]
mod tests;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprFillDefaultMode {
    Todo,
    Default,
}
impl Default for ExprFillDefaultMode {
    fn default() -> Self {
        Self::Todo
    }
}

#[derive(Default, Debug, Clone)]
pub struct DiagnosticsConfig {
    pub disable_experimental: bool,
    pub disabled: FxHashSet<String>,
    pub expr_fill_default: ExprFillDefaultMode,
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
            AnyDiagnostic::AddReferenceHere(d) => handlers::add_reference_here::add_reference_here(&ctx, &d),
            AnyDiagnostic::BreakOutsideOfLoop(d) => handlers::break_outside_of_loop::break_outside_of_loop(&ctx, &d),
            AnyDiagnostic::IncorrectCase(d) => handlers::incorrect_case::incorrect_case(&ctx, &d),
            AnyDiagnostic::MacroError(d) => handlers::macro_error::macro_error(&ctx, &d),
            AnyDiagnostic::MalformedDerive(d) => handlers::malformed_derive::malformed_derive(&ctx, &d),
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
            AnyDiagnostic::InvalidDeriveTarget(d) => handlers::invalid_derive_target::invalid_derive_target(&ctx, &d),

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
        label: Label::new(label.to_string()),
        group: None,
        target,
        source_change: None,
    }
}
