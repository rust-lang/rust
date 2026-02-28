use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level, Subdiagnostic, msg,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::mir::AssertKind;
use rustc_middle::query::QueryKey;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{self, Lint};
use rustc_span::def_id::DefId;
use rustc_span::{Ident, Span, Symbol};

/// Emit diagnostic for calls to `#[inline(always)]`-annotated functions with a
/// `#[target_feature]` attribute where the caller enables a different set of target features.
pub(crate) fn emit_inline_always_target_feature_diagnostic<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    call_span: Span,
    callee_def_id: DefId,
    caller_def_id: DefId,
    callee_only: &[&'a str],
) {
    tcx.node_span_lint(
        lint::builtin::INLINE_ALWAYS_MISMATCHING_TARGET_FEATURES,
        tcx.local_def_id_to_hir_id(caller_def_id.as_local().unwrap()),
        call_span,
        |lint| {
            let callee = tcx.def_path_str(callee_def_id);
            let caller = tcx.def_path_str(caller_def_id);

            lint.primary_message(format!(
                "call to `#[inline(always)]`-annotated `{callee}` \
                requires the same target features to be inlined"
            ));
            lint.note("function will not be inlined");

            lint.note(format!(
                "the following target features are on `{callee}` but missing from `{caller}`: {}",
                callee_only.join(", ")
            ));
            lint.span_note(callee_def_id.default_span(tcx), format!("`{callee}` is defined here"));

            let feats = callee_only.join(",");
            lint.span_suggestion(
                tcx.def_span(caller_def_id).shrink_to_lo(),
                format!("add `#[target_feature]` attribute to `{caller}`"),
                format!("#[target_feature(enable = \"{feats}\")]\n"),
                lint::Applicability::MaybeIncorrect,
            );
        },
    );
}

#[derive(Diagnostic)]
#[diag("function cannot return without recursing")]
#[help("a `loop` may express intention better if this is on purpose")]
pub(crate) struct UnconditionalRecursion {
    #[label("cannot return without recursing")]
    pub(crate) span: Span,
    #[label("recursive call site")]
    pub(crate) call_sites: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("`{$callee}` is incompatible with `#[rustc_force_inline]`")]
#[note("incompatible due to: {$reason}")]
pub(crate) struct InvalidForceInline {
    #[primary_span]
    pub attr_span: Span,
    #[label("`{$callee}` defined here")]
    pub callee_span: Span,
    pub callee: String,
    pub reason: &'static str,
}

#[derive(Diagnostic)]
pub(crate) enum ConstMutate {
    #[diag("attempting to modify a `const` item")]
    #[note(
        "each usage of a `const` item creates a new temporary; the original `const` item will not be modified"
    )]
    Modify {
        #[note("`const` item defined here")]
        konst: Span,
    },
    #[diag("taking a mutable reference to a `const` item")]
    #[note("each usage of a `const` item creates a new temporary")]
    #[note("the mutable reference will refer to this temporary, not the original `const` item")]
    MutBorrow {
        #[note("mutable reference created due to call to this method")]
        method_call: Option<Span>,
        #[note("`const` item defined here")]
        konst: Span,
    },
}

#[derive(Diagnostic)]
#[diag("reference to field of packed {$ty_descr} is unaligned", code = E0793)]
#[note(
    "this {$ty_descr} is {$align ->
        [one] {\"\"}
        *[other] {\"at most \"}
    }{$align}-byte aligned, but the type of this field may require higher alignment"
)]
#[note(
    "creating a misaligned reference is undefined behavior (even if that reference is never dereferenced)"
)]
#[help(
    "copy the field contents to a local variable, or replace the reference with a raw pointer and use `read_unaligned`/`write_unaligned` (loads and stores via `*p` must be properly aligned even when using raw pointers)"
)]
pub(crate) struct UnalignedPackedRef {
    #[primary_span]
    pub span: Span,
    pub ty_descr: &'static str,
    pub align: u64,
}

#[derive(Diagnostic)]
#[diag("MIR pass `{$name}` is unknown and will be ignored")]
pub(crate) struct UnknownPassName<'a> {
    pub(crate) name: &'a str,
}

pub(crate) struct AssertLint<P> {
    pub span: Span,
    pub assert_kind: AssertKind<P>,
    pub lint_kind: AssertLintKind,
}

pub(crate) enum AssertLintKind {
    ArithmeticOverflow,
    UnconditionalPanic,
}

impl<'a, P: std::fmt::Debug> Diagnostic<'a, ()> for AssertLint<P> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        let mut diag = Diag::new(
            dcx,
            level,
            match self.lint_kind {
                AssertLintKind::ArithmeticOverflow => {
                    msg!("this arithmetic operation will overflow")
                }
                AssertLintKind::UnconditionalPanic => {
                    msg!("this operation will panic at runtime")
                }
            },
        );
        let label = self.assert_kind.diagnostic_message();
        self.assert_kind.add_args(&mut |name, value| {
            diag.arg(name, value);
        });
        diag.span_label(self.span, label);
        diag
    }
}

impl AssertLintKind {
    pub(crate) fn lint(&self) -> &'static Lint {
        match self {
            AssertLintKind::ArithmeticOverflow => lint::builtin::ARITHMETIC_OVERFLOW,
            AssertLintKind::UnconditionalPanic => lint::builtin::UNCONDITIONAL_PANIC,
        }
    }
}

#[derive(Diagnostic)]
#[diag("call to inline assembly that may unwind")]
pub(crate) struct AsmUnwindCall {
    #[label("call to inline assembly that may unwind")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "call to {$foreign ->
        [true] foreign function
        *[false] function pointer
    } with FFI-unwind ABI"
)]
pub(crate) struct FfiUnwindCall {
    #[label(
        "call to {$foreign ->
            [true] foreign function
            *[false] function pointer
        } with FFI-unwind ABI"
    )]
    pub span: Span,
    pub foreign: bool,
}

#[derive(Diagnostic)]
#[diag("taking a reference to a function item does not give a function pointer")]
pub(crate) struct FnItemRef {
    #[suggestion(
        "cast `{$ident}` to obtain a function pointer",
        code = "{sugg}",
        applicability = "unspecified"
    )]
    pub span: Span,
    pub sugg: String,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag("value captured by `{$name}` is never read")]
#[help("did you mean to capture by reference instead?")]
pub(crate) struct UnusedCaptureMaybeCaptureRef {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("variable `{$name}` is assigned to, but never used")]
#[note("consider using `_{$name}` instead")]
pub(crate) struct UnusedVarAssignedOnly {
    pub name: Symbol,
    #[subdiagnostic]
    pub typo: Option<PatternTypo>,
}

#[derive(Diagnostic)]
#[diag("value assigned to `{$name}` is never read")]
pub(crate) struct UnusedAssign {
    pub name: Symbol,
    #[subdiagnostic]
    pub suggestion: Option<UnusedAssignSuggestion>,
    #[help("maybe it is overwritten before being read?")]
    pub help: bool,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "you might have meant to mutate the pointed at value being passed in, instead of changing the reference in the local binding",
    applicability = "maybe-incorrect"
)]
pub(crate) struct UnusedAssignSuggestion {
    pub pre: &'static str,
    #[suggestion_part(code = "{pre}mut ")]
    pub ty_span: Option<Span>,
    #[suggestion_part(code = "")]
    pub ty_ref_span: Span,
    #[suggestion_part(code = "*")]
    pub pre_lhs_span: Span,
    #[suggestion_part(code = "")]
    pub rhs_borrow_span: Span,
}

#[derive(Diagnostic)]
#[diag("value passed to `{$name}` is never read")]
#[help("maybe it is overwritten before being read?")]
pub(crate) struct UnusedAssignPassed {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("unused variable: `{$name}`")]
pub(crate) struct UnusedVariable {
    pub name: Symbol,
    #[subdiagnostic]
    pub string_interp: Vec<UnusedVariableStringInterp>,
    #[subdiagnostic]
    pub sugg: UnusedVariableSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedVariableSugg {
    #[multipart_suggestion("try ignoring the field", applicability = "machine-applicable")]
    TryIgnore {
        #[suggestion_part(code = "{name}: _")]
        shorthands: Vec<Span>,
        #[suggestion_part(code = "_")]
        non_shorthands: Vec<Span>,
        name: Symbol,
    },

    #[multipart_suggestion(
        "if this is intentional, prefix it with an underscore",
        applicability = "machine-applicable"
    )]
    TryPrefix {
        #[suggestion_part(code = "_{name}")]
        spans: Vec<Span>,
        name: Symbol,
        #[subdiagnostic]
        typo: Option<PatternTypo>,
    },

    #[help("`{$name}` is captured in macro and introduced a unused variable")]
    NoSugg {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
}

pub(crate) struct UnusedVariableStringInterp {
    pub lit: Span,
}

impl Subdiagnostic for UnusedVariableStringInterp {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_label(
            self.lit,
            msg!("you might have meant to use string interpolation in this string literal"),
        );
        diag.multipart_suggestion(
            msg!("string interpolation only works in `format!` invocations"),
            vec![
                (self.lit.shrink_to_lo(), String::from("format!(")),
                (self.lit.shrink_to_hi(), String::from(")")),
            ],
            Applicability::MachineApplicable,
        );
    }
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "you might have meant to pattern match on the similarly named {$kind} `{$item_name}`",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct PatternTypo {
    #[suggestion_part(code = "{code}")]
    pub span: Span,
    pub code: String,
    pub item_name: Symbol,
    pub kind: &'static str,
}

pub(crate) struct MustNotSupend<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub yield_sp: Span,
    pub reason: Option<MustNotSuspendReason>,
    pub src_sp: Span,
    pub pre: &'a str,
    pub def_id: DefId,
    pub post: &'a str,
}

// Needed for def_path_str
impl<'a> Diagnostic<'a, ()> for MustNotSupend<'_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        let mut diag = Diag::new(
            dcx,
            level,
            msg!("{$pre}`{$def_path}`{$post} held across a suspend point, but should not be"),
        );
        diag.span_label(self.yield_sp, msg!("the value is held across this suspend point"));
        if let Some(reason) = self.reason {
            diag.subdiagnostic(reason);
        }
        diag.span_help(self.src_sp, msg!("consider using a block (`{\"{ ... }\"}`) to shrink the value's scope, ending before the suspend point"));
        diag.arg("pre", self.pre);
        diag.arg("def_path", self.tcx.def_path_str(self.def_id));
        diag.arg("post", self.post);
        diag
    }
}

#[derive(Subdiagnostic)]
#[note("{$reason}")]
pub(crate) struct MustNotSuspendReason {
    #[primary_span]
    pub span: Span,
    pub reason: Symbol,
}

#[derive(Diagnostic)]
#[diag("`{$callee}` could not be inlined into `{$caller}` but is required to be inlined")]
#[note("could not be inlined due to: {$reason}")]
pub(crate) struct ForceInlineFailure {
    #[label("within `{$caller}`...")]
    pub caller_span: Span,
    #[label("`{$callee}` defined here")]
    pub callee_span: Span,
    #[label("annotation here")]
    pub attr_span: Span,
    #[primary_span]
    #[label("...`{$callee}` called here")]
    pub call_span: Span,
    pub callee: String,
    pub caller: String,
    pub reason: &'static str,
    #[subdiagnostic]
    pub justification: Option<ForceInlineJustification>,
}

#[derive(Subdiagnostic)]
#[note("`{$callee}` is required to be inlined to: {$sym}")]
pub(crate) struct ForceInlineJustification {
    pub sym: Symbol,
    pub callee: String,
}
