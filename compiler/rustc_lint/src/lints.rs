// ignore-tidy-filelength

use std::num::NonZero;

use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagMessage, DiagStyledString, ElidedLifetimeInPathSubdiag,
    EmissionGuarantee, LintDiagnostic, MultiSpan, Subdiagnostic, SuggestionStyle, msg,
};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::VisitorExt;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::inhabitedness::InhabitedPredicate;
use rustc_middle::ty::{Clause, PolyExistentialTraitRef, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::{Ident, Span, Symbol, sym};

use crate::LateContext;
use crate::builtin::{InitError, ShorthandAssocTyCollector, TypeAliasBounds};
use crate::errors::{OverruledAttributeSub, RequestedLevel};
use crate::lifetime_syntax::LifetimeSyntaxCategories;

// array_into_iter.rs
#[derive(LintDiagnostic)]
#[diag(
    "this method call resolves to `<&{$target} as IntoIterator>::into_iter` (due to backwards compatibility), but will resolve to `<{$target} as IntoIterator>::into_iter` in Rust {$edition}"
)]
pub(crate) struct ShadowedIntoIterDiag {
    pub target: &'static str,
    pub edition: &'static str,
    #[suggestion(
        "use `.iter()` instead of `.into_iter()` to avoid ambiguity",
        code = "iter",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
    #[subdiagnostic]
    pub sub: Option<ShadowedIntoIterDiagSub>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ShadowedIntoIterDiagSub {
    #[suggestion(
        "or remove `.into_iter()` to iterate by value",
        code = "",
        applicability = "maybe-incorrect"
    )]
    RemoveIntoIter {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "or use `IntoIterator::into_iter(..)` instead of `.into_iter()` to explicitly iterate by value",
        applicability = "maybe-incorrect"
    )]
    UseExplicitIntoIter {
        #[suggestion_part(code = "IntoIterator::into_iter(")]
        start_span: Span,
        #[suggestion_part(code = ")")]
        end_span: Span,
    },
}

// autorefs.rs
#[derive(LintDiagnostic)]
#[diag("implicit autoref creates a reference to the dereference of a raw pointer")]
#[note(
    "creating a reference requires the pointer target to be valid and imposes aliasing requirements"
)]
pub(crate) struct ImplicitUnsafeAutorefsDiag<'a> {
    #[label("this raw pointer has type `{$raw_ptr_ty}`")]
    pub raw_ptr_span: Span,
    pub raw_ptr_ty: Ty<'a>,
    #[subdiagnostic]
    pub origin: ImplicitUnsafeAutorefsOrigin<'a>,
    #[subdiagnostic]
    pub method: Option<ImplicitUnsafeAutorefsMethodNote>,
    #[subdiagnostic]
    pub suggestion: ImplicitUnsafeAutorefsSuggestion,
}

#[derive(Subdiagnostic)]
pub(crate) enum ImplicitUnsafeAutorefsOrigin<'a> {
    #[note("autoref is being applied to this expression, resulting in: `{$autoref_ty}`")]
    Autoref {
        #[primary_span]
        autoref_span: Span,
        autoref_ty: Ty<'a>,
    },
    #[note(
        "references are created through calls to explicit `Deref(Mut)::deref(_mut)` implementations"
    )]
    OverloadedDeref,
}

#[derive(Subdiagnostic)]
#[note("method calls to `{$method_name}` require a reference")]
pub(crate) struct ImplicitUnsafeAutorefsMethodNote {
    #[primary_span]
    pub def_span: Span,
    pub method_name: Symbol,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "try using a raw pointer method instead; or if this reference is intentional, make it explicit",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ImplicitUnsafeAutorefsSuggestion {
    pub mutbl: &'static str,
    pub deref: &'static str,
    #[suggestion_part(code = "({mutbl}{deref}")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

// builtin.rs
#[derive(LintDiagnostic)]
#[diag("denote infinite loops with `loop {\"{\"} ... {\"}\"}`")]
pub(crate) struct BuiltinWhileTrue {
    #[suggestion(
        "use `loop`",
        style = "short",
        code = "{replace}",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
    pub replace: String,
}

#[derive(LintDiagnostic)]
#[diag("the `{$ident}:` in this pattern is redundant")]
pub(crate) struct BuiltinNonShorthandFieldPatterns {
    pub ident: Ident,
    #[suggestion(
        "use shorthand field pattern",
        code = "{prefix}{ident}",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
    pub prefix: &'static str,
}

#[derive(LintDiagnostic)]
pub(crate) enum BuiltinUnsafe {
    #[diag(
        "`allow_internal_unsafe` allows defining macros using unsafe without triggering the `unsafe_code` lint at their call site"
    )]
    AllowInternalUnsafe,
    #[diag("usage of an `unsafe` block")]
    UnsafeBlock,
    #[diag("usage of an `unsafe extern` block")]
    UnsafeExternBlock,
    #[diag("declaration of an `unsafe` trait")]
    UnsafeTrait,
    #[diag("implementation of an `unsafe` trait")]
    UnsafeImpl,
    #[diag("declaration of a `no_mangle` function")]
    #[note(
        "the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them"
    )]
    NoMangleFn,
    #[diag("declaration of a function with `export_name`")]
    #[note(
        "the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them"
    )]
    ExportNameFn,
    #[diag("declaration of a function with `link_section`")]
    #[note(
        "the program's behavior with overridden link sections on items is unpredictable and Rust cannot provide guarantees when you manually override them"
    )]
    LinkSectionFn,
    #[diag("declaration of a `no_mangle` static")]
    #[note(
        "the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them"
    )]
    NoMangleStatic,
    #[diag("declaration of a static with `export_name`")]
    #[note(
        "the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them"
    )]
    ExportNameStatic,
    #[diag("declaration of a static with `link_section`")]
    #[note(
        "the program's behavior with overridden link sections on items is unpredictable and Rust cannot provide guarantees when you manually override them"
    )]
    LinkSectionStatic,
    #[diag("declaration of a `no_mangle` method")]
    #[note(
        "the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them"
    )]
    NoMangleMethod,
    #[diag("declaration of a method with `export_name`")]
    #[note(
        "the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them"
    )]
    ExportNameMethod,
    #[diag("declaration of an `unsafe` function")]
    DeclUnsafeFn,
    #[diag("declaration of an `unsafe` method")]
    DeclUnsafeMethod,
    #[diag("implementation of an `unsafe` method")]
    ImplUnsafeMethod,
    #[diag("usage of `core::arch::global_asm`")]
    #[note("using this macro is unsafe even though it does not need an `unsafe` block")]
    GlobalAsm,
}

#[derive(LintDiagnostic)]
#[diag("missing documentation for {$article} {$desc}")]
pub(crate) struct BuiltinMissingDoc<'a> {
    pub article: &'a str,
    pub desc: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("type could implement `Copy`; consider adding `impl Copy`")]
pub(crate) struct BuiltinMissingCopyImpl;

pub(crate) struct BuiltinMissingDebugImpl<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> LintDiagnostic<'a, ()> for BuiltinMissingDebugImpl<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.primary_message(msg!("type does not implement `{$debug}`; consider adding `#[derive(Debug)]` or a manual implementation"));
        diag.arg("debug", self.tcx.def_path_str(self.def_id));
    }
}

#[derive(LintDiagnostic)]
#[diag("anonymous parameters are deprecated and will be removed in the next edition")]
pub(crate) struct BuiltinAnonymousParams<'a> {
    #[suggestion("try naming the parameter or explicitly ignoring it", code = "_: {ty_snip}")]
    pub suggestion: (Span, Applicability),
    pub ty_snip: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("unused doc comment")]
pub(crate) struct BuiltinUnusedDocComment<'a> {
    pub kind: &'a str,
    #[label("rustdoc does not generate documentation for {$kind}")]
    pub label: Span,
    #[subdiagnostic]
    pub sub: BuiltinUnusedDocCommentSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum BuiltinUnusedDocCommentSub {
    #[help("use `//` for a plain comment")]
    PlainHelp,
    #[help("use `/* */` for a plain comment")]
    BlockHelp,
}

#[derive(LintDiagnostic)]
#[diag("functions generic over types or consts must be mangled")]
pub(crate) struct BuiltinNoMangleGeneric {
    // Use of `#[no_mangle]` suggests FFI intent; correct
    // fix may be to monomorphize source by hand
    #[suggestion(
        "remove this attribute",
        style = "short",
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag("const items should never be `#[no_mangle]`")]
pub(crate) struct BuiltinConstNoMangle {
    #[suggestion("try a static value", code = "pub static ", applicability = "machine-applicable")]
    pub suggestion: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(
    "transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell"
)]
pub(crate) struct BuiltinMutablesTransmutes;

#[derive(LintDiagnostic)]
#[diag("use of an unstable feature")]
pub(crate) struct BuiltinUnstableFeatures;

// lint_ungated_async_fn_track_caller
pub(crate) struct BuiltinUngatedAsyncFnTrackCaller<'a> {
    pub label: Span,
    pub session: &'a Session,
}

impl<'a> LintDiagnostic<'a, ()> for BuiltinUngatedAsyncFnTrackCaller<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!("`#[track_caller]` on async functions is a no-op"));
        diag.span_label(self.label, msg!("this function will not propagate the caller location"));
        rustc_session::parse::add_feature_diagnostics(
            diag,
            self.session,
            sym::async_fn_track_caller,
        );
    }
}

#[derive(LintDiagnostic)]
#[diag("unreachable `pub` {$what}")]
pub(crate) struct BuiltinUnreachablePub<'a> {
    pub what: &'a str,
    pub new_vis: &'a str,
    #[suggestion("consider restricting its visibility", code = "{new_vis}")]
    pub suggestion: (Span, Applicability),
    #[help("or consider exporting it for use by other crates")]
    pub help: bool,
}

#[derive(LintDiagnostic)]
#[diag("the `expr` fragment specifier will accept more expressions in the 2024 edition")]
pub(crate) struct MacroExprFragment2024 {
    #[suggestion(
        "to keep the existing behavior, use the `expr_2021` fragment specifier",
        code = "expr_2021",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

pub(crate) struct BuiltinTypeAliasBounds<'hir> {
    pub in_where_clause: bool,
    pub label: Span,
    pub enable_feat_help: bool,
    pub suggestions: Vec<(Span, String)>,
    pub preds: &'hir [hir::WherePredicate<'hir>],
    pub ty: Option<&'hir hir::Ty<'hir>>,
}

impl<'a> LintDiagnostic<'a, ()> for BuiltinTypeAliasBounds<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(if self.in_where_clause {
            msg!("where clauses on type aliases are not enforced")
        } else {
            msg!("bounds on generic parameters in type aliases are not enforced")
        });
        diag.span_label(self.label, msg!("will not be checked at usage sites of the type alias"));
        diag.note(msg!(
            "this is a known limitation of the type checker that may be lifted in a future edition.
            see issue #112792 <https://github.com/rust-lang/rust/issues/112792> for more information"
        ));
        if self.enable_feat_help {
            diag.help(msg!("add `#![feature(lazy_type_alias)]` to the crate attributes to enable the desired semantics"));
        }

        // We perform the walk in here instead of in `<TypeAliasBounds as LateLintPass>` to
        // avoid doing throwaway work in case the lint ends up getting suppressed.
        let mut collector = ShorthandAssocTyCollector { qselves: Vec::new() };
        if let Some(ty) = self.ty {
            collector.visit_ty_unambig(ty);
        }

        let affect_object_lifetime_defaults = self
            .preds
            .iter()
            .filter(|pred| pred.kind.in_where_clause() == self.in_where_clause)
            .any(|pred| TypeAliasBounds::affects_object_lifetime_defaults(pred));

        // If there are any shorthand assoc tys, then the bounds can't be removed automatically.
        // The user first needs to fully qualify the assoc tys.
        let applicability = if !collector.qselves.is_empty() || affect_object_lifetime_defaults {
            Applicability::MaybeIncorrect
        } else {
            Applicability::MachineApplicable
        };

        diag.arg("count", self.suggestions.len());
        diag.multipart_suggestion(
            if self.in_where_clause {
                msg!("remove this where clause")
            } else {
                msg!(
                    "remove {$count ->
                        [one] this bound
                        *[other] these bounds
                    }"
                )
            },
            self.suggestions,
            applicability,
        );

        // Suggest fully qualifying paths of the form `T::Assoc` with `T` type param via
        // `<T as /* Trait */>::Assoc` to remove their reliance on any type param bounds.
        //
        // Instead of attempting to figure out the necessary trait ref, just use a
        // placeholder. Since we don't record type-dependent resolutions for non-body
        // items like type aliases, we can't simply deduce the corresp. trait from
        // the HIR path alone without rerunning parts of HIR ty lowering here
        // (namely `probe_single_ty_param_bound_for_assoc_ty`) which is infeasible.
        //
        // (We could employ some simple heuristics but that's likely not worth it).
        for qself in collector.qselves {
            diag.multipart_suggestion(
                msg!("fully qualify this associated type"),
                vec![
                    (qself.shrink_to_lo(), "<".into()),
                    (qself.shrink_to_hi(), " as /* Trait */>".into()),
                ],
                Applicability::HasPlaceholders,
            );
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(
    "{$predicate_kind_name} bound {$predicate} does not depend on any type or lifetime parameters"
)]
pub(crate) struct BuiltinTrivialBounds<'a> {
    pub predicate_kind_name: &'a str,
    pub predicate: Clause<'a>,
}

#[derive(LintDiagnostic)]
#[diag("use of a double negation")]
#[note(
    "the prefix `--` could be misinterpreted as a decrement operator which exists in other languages"
)]
#[note("use `-= 1` if you meant to decrement the value")]
pub(crate) struct BuiltinDoubleNegations {
    #[subdiagnostic]
    pub add_parens: BuiltinDoubleNegationsAddParens,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("add parentheses for clarity", applicability = "maybe-incorrect")]
pub(crate) struct BuiltinDoubleNegationsAddParens {
    #[suggestion_part(code = "(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

#[derive(LintDiagnostic)]
pub(crate) enum BuiltinEllipsisInclusiveRangePatternsLint {
    #[diag("`...` range patterns are deprecated")]
    Parenthesise {
        #[suggestion(
            "use `..=` for an inclusive range",
            code = "{replace}",
            applicability = "machine-applicable"
        )]
        suggestion: Span,
        replace: String,
    },
    #[diag("`...` range patterns are deprecated")]
    NonParenthesise {
        #[suggestion(
            "use `..=` for an inclusive range",
            style = "short",
            code = "..=",
            applicability = "machine-applicable"
        )]
        suggestion: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag("`{$kw}` is a keyword in the {$next} edition")]
pub(crate) struct BuiltinKeywordIdents {
    pub kw: Ident,
    pub next: Edition,
    #[suggestion(
        "you can use a raw identifier to stay compatible",
        code = "{prefix}r#{kw}",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
    pub prefix: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("outlives requirements can be inferred")]
pub(crate) struct BuiltinExplicitOutlives {
    pub count: usize,
    #[subdiagnostic]
    pub suggestion: BuiltinExplicitOutlivesSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "remove {$count ->
        [one] this bound
        *[other] these bounds
    }"
)]
pub(crate) struct BuiltinExplicitOutlivesSuggestion {
    #[suggestion_part(code = "")]
    pub spans: Vec<Span>,
    #[applicability]
    pub applicability: Applicability,
}

#[derive(LintDiagnostic)]
#[diag(
    "the feature `{$name}` is incomplete and may not be safe to use and/or cause compiler crashes"
)]
pub(crate) struct BuiltinIncompleteFeatures {
    pub name: Symbol,
    #[subdiagnostic]
    pub note: Option<BuiltinFeatureIssueNote>,
    #[subdiagnostic]
    pub help: Option<BuiltinIncompleteFeaturesHelp>,
}

#[derive(LintDiagnostic)]
#[diag("the feature `{$name}` is internal to the compiler or standard library")]
#[note("using it is strongly discouraged")]
pub(crate) struct BuiltinInternalFeatures {
    pub name: Symbol,
}

#[derive(Subdiagnostic)]
#[help("consider using `min_{$name}` instead, which is more stable and complete")]
pub(crate) struct BuiltinIncompleteFeaturesHelp;

#[derive(Subdiagnostic)]
#[note("see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information")]
pub(crate) struct BuiltinFeatureIssueNote {
    pub n: NonZero<u32>,
}

pub(crate) struct BuiltinUnpermittedTypeInit<'a> {
    pub msg: DiagMessage,
    pub ty: Ty<'a>,
    pub label: Span,
    pub sub: BuiltinUnpermittedTypeInitSub,
    pub tcx: TyCtxt<'a>,
}

impl<'a> LintDiagnostic<'a, ()> for BuiltinUnpermittedTypeInit<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(self.msg);
        diag.arg("ty", self.ty);
        diag.span_label(self.label, msg!("this code causes undefined behavior when executed"));
        if let InhabitedPredicate::True = self.ty.inhabited_predicate(self.tcx) {
            // Only suggest late `MaybeUninit::assume_init` initialization if the type is inhabited.
            diag.span_label(
                self.label,
                msg!("help: use `MaybeUninit<T>` instead, and only call `assume_init` after initialization is done"),
            );
        }
        self.sub.add_to_diag(diag);
    }
}

// FIXME(davidtwco): make translatable
pub(crate) struct BuiltinUnpermittedTypeInitSub {
    pub err: InitError,
}

impl Subdiagnostic for BuiltinUnpermittedTypeInitSub {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut err = self.err;
        loop {
            if let Some(span) = err.span {
                diag.span_note(span, err.message);
            } else {
                diag.note(err.message);
            }
            if let Some(e) = err.nested {
                err = *e;
            } else {
                break;
            }
        }
    }
}

#[derive(LintDiagnostic)]
pub(crate) enum BuiltinClashingExtern<'a> {
    #[diag("`{$this}` redeclared with a different signature")]
    SameName {
        this: Symbol,
        orig: Symbol,
        #[label("`{$orig}` previously declared here")]
        previous_decl_label: Span,
        #[label("this signature doesn't match the previous declaration")]
        mismatch_label: Span,
        #[subdiagnostic]
        sub: BuiltinClashingExternSub<'a>,
    },
    #[diag("`{$this}` redeclares `{$orig}` with a different signature")]
    DiffName {
        this: Symbol,
        orig: Symbol,
        #[label("`{$orig}` previously declared here")]
        previous_decl_label: Span,
        #[label("this signature doesn't match the previous declaration")]
        mismatch_label: Span,
        #[subdiagnostic]
        sub: BuiltinClashingExternSub<'a>,
    },
}

// FIXME(davidtwco): translatable expected/found
pub(crate) struct BuiltinClashingExternSub<'a> {
    pub tcx: TyCtxt<'a>,
    pub expected: Ty<'a>,
    pub found: Ty<'a>,
}

impl Subdiagnostic for BuiltinClashingExternSub<'_> {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut expected_str = DiagStyledString::new();
        expected_str.push(self.expected.fn_sig(self.tcx).to_string(), false);
        let mut found_str = DiagStyledString::new();
        found_str.push(self.found.fn_sig(self.tcx).to_string(), true);
        diag.note_expected_found("", expected_str, "", found_str);
    }
}

#[derive(LintDiagnostic)]
#[diag("dereferencing a null pointer")]
pub(crate) struct BuiltinDerefNullptr {
    #[label("this code causes undefined behavior when executed")]
    pub label: Span,
}

// FIXME: migrate fluent::lint::builtin_asm_labels

#[derive(LintDiagnostic)]
pub(crate) enum BuiltinSpecialModuleNameUsed {
    #[diag("found module declaration for lib.rs")]
    #[note("lib.rs is the root of this crate's library target")]
    #[help("to refer to it from other targets, use the library's name as the path")]
    Lib,
    #[diag("found module declaration for main.rs")]
    #[note("a binary crate cannot be used as library")]
    Main,
}

// deref_into_dyn_supertrait.rs
#[derive(LintDiagnostic)]
#[diag("this `Deref` implementation is covered by an implicit supertrait coercion")]
pub(crate) struct SupertraitAsDerefTarget<'a> {
    pub self_ty: Ty<'a>,
    pub supertrait_principal: PolyExistentialTraitRef<'a>,
    pub target_principal: PolyExistentialTraitRef<'a>,
    #[label(
        "`{$self_ty}` implements `Deref<Target = dyn {$target_principal}>` which conflicts with supertrait `{$supertrait_principal}`"
    )]
    pub label: Span,
    #[subdiagnostic]
    pub label2: Option<SupertraitAsDerefTargetLabel>,
}

#[derive(Subdiagnostic)]
#[label("target type is a supertrait of `{$self_ty}`")]
pub(crate) struct SupertraitAsDerefTargetLabel {
    #[primary_span]
    pub label: Span,
}

// enum_intrinsics_non_enums.rs
#[derive(LintDiagnostic)]
#[diag("the return value of `mem::discriminant` is unspecified when called with a non-enum type")]
pub(crate) struct EnumIntrinsicsMemDiscriminate<'a> {
    pub ty_param: Ty<'a>,
    #[note(
        "the argument to `discriminant` should be a reference to an enum, but it was passed a reference to a `{$ty_param}`, which is not an enum"
    )]
    pub note: Span,
}

#[derive(LintDiagnostic)]
#[diag("the return value of `mem::variant_count` is unspecified when called with a non-enum type")]
#[note(
    "the type parameter of `variant_count` should be an enum, but it was instantiated with the type `{$ty_param}`, which is not an enum"
)]
pub(crate) struct EnumIntrinsicsMemVariant<'a> {
    pub ty_param: Ty<'a>,
}

// expect.rs
#[derive(LintDiagnostic)]
#[diag("this lint expectation is unfulfilled")]
pub(crate) struct Expectation {
    #[subdiagnostic]
    pub rationale: Option<ExpectationNote>,
    #[note(
        "the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message"
    )]
    pub note: bool,
}

#[derive(Subdiagnostic)]
#[note("{$rationale}")]
pub(crate) struct ExpectationNote {
    pub rationale: Symbol,
}

// ptr_nulls.rs
#[derive(LintDiagnostic)]
pub(crate) enum UselessPtrNullChecksDiag<'a> {
    #[diag(
        "function pointers are not nullable, so checking them for null will always return false"
    )]
    #[help(
        "wrap the function pointer inside an `Option` and use `Option::is_none` to check for null pointer value"
    )]
    FnPtr {
        orig_ty: Ty<'a>,
        #[label("expression has type `{$orig_ty}`")]
        label: Span,
    },
    #[diag("references are not nullable, so checking them for null will always return false")]
    Ref {
        orig_ty: Ty<'a>,
        #[label("expression has type `{$orig_ty}`")]
        label: Span,
    },
    #[diag(
        "returned pointer of `{$fn_name}` call is never null, so checking it for null will always return false"
    )]
    FnRet { fn_name: Ident },
}

#[derive(LintDiagnostic)]
pub(crate) enum InvalidNullArgumentsDiag {
    #[diag(
        "calling this function with a null pointer is undefined behavior, even if the result of the function is unused"
    )]
    #[help(
        "for more information, visit <https://doc.rust-lang.org/std/ptr/index.html> and <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>"
    )]
    NullPtrInline {
        #[label("null pointer originates from here")]
        null_span: Span,
    },
    #[diag(
        "calling this function with a null pointer is undefined behavior, even if the result of the function is unused"
    )]
    #[help(
        "for more information, visit <https://doc.rust-lang.org/std/ptr/index.html> and <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>"
    )]
    NullPtrThroughBinding {
        #[note("null pointer originates from here")]
        null_span: Span,
    },
}

// for_loops_over_fallibles.rs
#[derive(LintDiagnostic)]
#[diag(
    "for loop over {$article} `{$ref_prefix}{$ty}`. This is more readably written as an `if let` statement"
)]
pub(crate) struct ForLoopsOverFalliblesDiag<'a> {
    pub article: &'static str,
    pub ref_prefix: &'static str,
    pub ty: &'static str,
    #[subdiagnostic]
    pub sub: ForLoopsOverFalliblesLoopSub<'a>,
    #[subdiagnostic]
    pub question_mark: Option<ForLoopsOverFalliblesQuestionMark>,
    #[subdiagnostic]
    pub suggestion: ForLoopsOverFalliblesSuggestion<'a>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ForLoopsOverFalliblesLoopSub<'a> {
    #[suggestion(
        "to iterate over `{$recv_snip}` remove the call to `next`",
        code = ".by_ref()",
        applicability = "maybe-incorrect"
    )]
    RemoveNext {
        #[primary_span]
        suggestion: Span,
        recv_snip: String,
    },
    #[multipart_suggestion(
        "to check pattern in a loop use `while let`",
        applicability = "maybe-incorrect"
    )]
    UseWhileLet {
        #[suggestion_part(code = "while let {var}(")]
        start_span: Span,
        #[suggestion_part(code = ") = ")]
        end_span: Span,
        var: &'a str,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(
    "consider unwrapping the `Result` with `?` to iterate over its contents",
    code = "?",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ForLoopsOverFalliblesQuestionMark {
    #[primary_span]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider using `if let` to clear intent",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ForLoopsOverFalliblesSuggestion<'a> {
    pub var: &'a str,
    #[suggestion_part(code = "if let {var}(")]
    pub start_span: Span,
    #[suggestion_part(code = ") = ")]
    pub end_span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum UseLetUnderscoreIgnoreSuggestion {
    #[note("use `let _ = ...` to ignore the expression or result")]
    Note,
    #[multipart_suggestion(
        "use `let _ = ...` to ignore the expression or result",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    Suggestion {
        #[suggestion_part(code = "let _ = ")]
        start_span: Span,
        #[suggestion_part(code = "")]
        end_span: Span,
    },
}

// drop_forget_useless.rs
#[derive(LintDiagnostic)]
#[diag("calls to `std::mem::drop` with a reference instead of an owned value does nothing")]
pub(crate) struct DropRefDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label("argument has type `{$arg_ty}`")]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag("calls to `std::mem::drop` with a value that implements `Copy` does nothing")]
pub(crate) struct DropCopyDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label("argument has type `{$arg_ty}`")]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag("calls to `std::mem::forget` with a reference instead of an owned value does nothing")]
pub(crate) struct ForgetRefDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label("argument has type `{$arg_ty}`")]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag("calls to `std::mem::forget` with a value that implements `Copy` does nothing")]
pub(crate) struct ForgetCopyDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label("argument has type `{$arg_ty}`")]
    pub label: Span,
    #[subdiagnostic]
    pub sugg: UseLetUnderscoreIgnoreSuggestion,
}

#[derive(LintDiagnostic)]
#[diag(
    "calls to `std::mem::drop` with `std::mem::ManuallyDrop` instead of the inner value does nothing"
)]
pub(crate) struct UndroppedManuallyDropsDiag<'a> {
    pub arg_ty: Ty<'a>,
    #[label("argument has type `{$arg_ty}`")]
    pub label: Span,
    #[subdiagnostic]
    pub suggestion: UndroppedManuallyDropsSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use `std::mem::ManuallyDrop::into_inner` to get the inner value",
    applicability = "machine-applicable"
)]
pub(crate) struct UndroppedManuallyDropsSuggestion {
    #[suggestion_part(code = "std::mem::ManuallyDrop::into_inner(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

// invalid_from_utf8.rs
#[derive(LintDiagnostic)]
pub(crate) enum InvalidFromUtf8Diag {
    #[diag("calls to `{$method}` with an invalid literal are undefined behavior")]
    Unchecked {
        method: String,
        valid_up_to: usize,
        #[label("the literal was valid UTF-8 up to the {$valid_up_to} bytes")]
        label: Span,
    },
    #[diag("calls to `{$method}` with an invalid literal always return an error")]
    Checked {
        method: String,
        valid_up_to: usize,
        #[label("the literal was valid UTF-8 up to the {$valid_up_to} bytes")]
        label: Span,
    },
}

// interior_mutable_consts.rs
#[derive(LintDiagnostic)]
#[diag("mutation of an interior mutable `const` item with call to `{$method_name}`")]
#[note("each usage of a `const` item creates a new temporary")]
#[note("only the temporaries and never the original `const {$const_name}` will be modified")]
#[help(
    "for more details on interior mutability see <https://doc.rust-lang.org/reference/interior-mutability.html>"
)]
pub(crate) struct ConstItemInteriorMutationsDiag<'tcx> {
    pub method_name: Ident,
    pub const_name: Ident,
    pub const_ty: Ty<'tcx>,
    #[label("`{$const_name}` is a interior mutable `const` item of type `{$const_ty}`")]
    pub receiver_span: Span,
    #[subdiagnostic]
    pub sugg_static: Option<ConstItemInteriorMutationsSuggestionStatic>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ConstItemInteriorMutationsSuggestionStatic {
    #[suggestion(
        "for a shared instance of `{$const_name}`, consider making it a `static` item instead",
        code = "{before}static ",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    Spanful {
        #[primary_span]
        const_: Span,
        before: &'static str,
    },
    #[help("for a shared instance of `{$const_name}`, consider making it a `static` item instead")]
    Spanless,
}

// reference_casting.rs
#[derive(LintDiagnostic)]
pub(crate) enum InvalidReferenceCastingDiag<'tcx> {
    #[diag(
        "casting `&T` to `&mut T` is undefined behavior, even if the reference is unused, consider instead using an `UnsafeCell`"
    )]
    #[note(
        "for more information, visit <https://doc.rust-lang.org/book/ch15-05-interior-mutability.html>"
    )]
    BorrowAsMut {
        #[label("casting happened here")]
        orig_cast: Option<Span>,
        #[note(
            "even for types with interior mutability, the only legal way to obtain a mutable pointer from a shared reference is through `UnsafeCell::get`"
        )]
        ty_has_interior_mutability: bool,
    },
    #[diag("assigning to `&T` is undefined behavior, consider using an `UnsafeCell`")]
    #[note(
        "for more information, visit <https://doc.rust-lang.org/book/ch15-05-interior-mutability.html>"
    )]
    AssignToRef {
        #[label("casting happened here")]
        orig_cast: Option<Span>,
        #[note(
            "even for types with interior mutability, the only legal way to obtain a mutable pointer from a shared reference is through `UnsafeCell::get`"
        )]
        ty_has_interior_mutability: bool,
    },
    #[diag(
        "casting references to a bigger memory layout than the backing allocation is undefined behavior, even if the reference is unused"
    )]
    #[note("casting from `{$from_ty}` ({$from_size} bytes) to `{$to_ty}` ({$to_size} bytes)")]
    BiggerLayout {
        #[label("casting happened here")]
        orig_cast: Option<Span>,
        #[label("backing allocation comes from here")]
        alloc: Span,
        from_ty: Ty<'tcx>,
        from_size: u64,
        to_ty: Ty<'tcx>,
        to_size: u64,
    },
}

// map_unit_fn.rs
#[derive(LintDiagnostic)]
#[diag("`Iterator::map` call that discard the iterator's values")]
#[note(
    "`Iterator::map`, like many of the methods on `Iterator`, gets executed lazily, meaning that its effects won't be visible until it is iterated"
)]
pub(crate) struct MappingToUnit {
    #[label("this function returns `()`, which is likely not what you wanted")]
    pub function_label: Span,
    #[label("called `Iterator::map` with callable that returns `()`")]
    pub argument_label: Span,
    #[label(
        "after this call to map, the resulting iterator is `impl Iterator<Item = ()>`, which means the only information carried by the iterator is the number of items"
    )]
    pub map_label: Span,
    #[suggestion(
        "you might have meant to use `Iterator::for_each`",
        style = "verbose",
        code = "for_each",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Span,
}

// internal.rs
#[derive(LintDiagnostic)]
#[diag("prefer `{$preferred}` over `{$used}`, it has better performance")]
#[note("a `use rustc_data_structures::fx::{$preferred}` may be necessary")]
pub(crate) struct DefaultHashTypesDiag<'a> {
    pub preferred: &'a str,
    pub used: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("using `{$query}` can result in unstable query results")]
#[note(
    "if you believe this case to be fine, allow this lint and add a comment explaining your rationale"
)]
pub(crate) struct QueryInstability {
    pub query: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("`{$method}` accesses information that is not tracked by the query system")]
#[note(
    "if you believe this case to be fine, allow this lint and add a comment explaining your rationale"
)]
pub(crate) struct QueryUntracked {
    pub method: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("use `.eq_ctxt()` instead of `.ctxt() == .ctxt()`")]
pub(crate) struct SpanUseEqCtxtDiag;

#[derive(LintDiagnostic)]
#[diag("using `Symbol::intern` on a string literal")]
#[help("consider adding the symbol to `compiler/rustc_span/src/symbol.rs`")]
pub(crate) struct SymbolInternStringLiteralDiag;

#[derive(LintDiagnostic)]
#[diag("usage of `ty::TyKind::<kind>`")]
pub(crate) struct TykindKind {
    #[suggestion(
        "try using `ty::<kind>` directly",
        code = "ty",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag("usage of `ty::TyKind`")]
#[help("try using `Ty` instead")]
pub(crate) struct TykindDiag;

#[derive(LintDiagnostic)]
#[diag("usage of qualified `ty::{$ty}`")]
pub(crate) struct TyQualified {
    pub ty: String,
    #[suggestion(
        "try importing it and using it unqualified",
        code = "{ty}",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag("do not use `rustc_type_ir::inherent` unless you're inside of the trait solver")]
#[note(
    "the method or struct you're looking for is likely defined somewhere else downstream in the compiler"
)]
pub(crate) struct TypeIrInherentUsage;

#[derive(LintDiagnostic)]
#[diag(
    "do not use `rustc_type_ir::Interner` or `rustc_type_ir::InferCtxtLike` unless you're inside of the trait solver"
)]
#[note(
    "the method or struct you're looking for is likely defined somewhere else downstream in the compiler"
)]
pub(crate) struct TypeIrTraitUsage;

#[derive(LintDiagnostic)]
#[diag("do not use `rustc_type_ir` unless you are implementing type system internals")]
#[note("use `rustc_middle::ty` instead")]
pub(crate) struct TypeIrDirectUse;

#[derive(LintDiagnostic)]
#[diag("non-glob import of `rustc_type_ir::inherent`")]
pub(crate) struct NonGlobImportTypeIrInherent {
    #[suggestion(
        "try using a glob import instead",
        code = "{snippet}",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Option<Span>,
    pub snippet: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("implementing `LintPass` by hand")]
#[help("try using `declare_lint_pass!` or `impl_lint_pass!` instead")]
pub(crate) struct LintPassByHand;

#[derive(LintDiagnostic)]
#[diag("{$msg}")]
pub(crate) struct BadOptAccessDiag<'a> {
    pub msg: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(
    "dangerous use of `extern crate {$name}` which is not guaranteed to exist exactly once in the sysroot"
)]
#[help(
    "try using a cargo dependency or using a re-export of the dependency provided by a rustc_* crate"
)]
pub(crate) struct ImplicitSysrootCrateImportDiag<'a> {
    pub name: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("use of `AttributeKind` in `find_attr!(...)` invocation")]
#[note("`find_attr!(...)` already imports `AttributeKind::*`")]
#[help("remove `AttributeKind`")]
pub(crate) struct AttributeKindInFindAttr {}

// let_underscore.rs
#[derive(LintDiagnostic)]
pub(crate) enum NonBindingLet {
    #[diag("non-binding let on a synchronization lock")]
    SyncLock {
        #[label("this lock is not assigned to a binding and is immediately dropped")]
        pat: Span,
        #[subdiagnostic]
        sub: NonBindingLetSub,
    },
    #[diag("non-binding let on a type that has a destructor")]
    DropType {
        #[subdiagnostic]
        sub: NonBindingLetSub,
    },
}

pub(crate) struct NonBindingLetSub {
    pub suggestion: Span,
    pub drop_fn_start_end: Option<(Span, Span)>,
    pub is_assign_desugar: bool,
}

impl Subdiagnostic for NonBindingLetSub {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let can_suggest_binding = self.drop_fn_start_end.is_some() || !self.is_assign_desugar;

        if can_suggest_binding {
            let prefix = if self.is_assign_desugar { "let " } else { "" };
            diag.span_suggestion_verbose(
                self.suggestion,
                msg!(
                    "consider binding to an unused variable to avoid immediately dropping the value"
                ),
                format!("{prefix}_unused"),
                Applicability::MachineApplicable,
            );
        } else {
            diag.span_help(
                self.suggestion,
                msg!(
                    "consider binding to an unused variable to avoid immediately dropping the value"
                ),
            );
        }
        if let Some(drop_fn_start_end) = self.drop_fn_start_end {
            diag.multipart_suggestion(
                msg!("consider immediately dropping the value"),
                vec![
                    (drop_fn_start_end.0, "drop(".to_string()),
                    (drop_fn_start_end.1, ")".to_string()),
                ],
                Applicability::MachineApplicable,
            );
        } else {
            diag.help(msg!(
                "consider immediately dropping the value using `drop(..)` after the `let` statement"
            ));
        }
    }
}

// levels.rs
#[derive(LintDiagnostic)]
#[diag("{$lint_level}({$lint_source}) incompatible with previous forbid")]
pub(crate) struct OverruledAttributeLint<'a> {
    #[label("overruled by previous forbid")]
    pub overruled: Span,
    pub lint_level: &'a str,
    pub lint_source: Symbol,
    #[subdiagnostic]
    pub sub: OverruledAttributeSub,
}

#[derive(LintDiagnostic)]
#[diag("lint name `{$name}` is deprecated and may not have an effect in the future")]
pub(crate) struct DeprecatedLintName<'a> {
    pub name: String,
    #[suggestion("change it to", code = "{replace}", applicability = "machine-applicable")]
    pub suggestion: Span,
    pub replace: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("lint name `{$name}` is deprecated and may not have an effect in the future")]
#[help("change it to {$replace}")]
pub(crate) struct DeprecatedLintNameFromCommandLine<'a> {
    pub name: String,
    pub replace: &'a str,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag("lint `{$name}` has been renamed to `{$replace}`")]
pub(crate) struct RenamedLint<'a> {
    pub name: &'a str,
    pub replace: &'a str,
    #[subdiagnostic]
    pub suggestion: RenamedLintSuggestion<'a>,
}

#[derive(Subdiagnostic)]
pub(crate) enum RenamedLintSuggestion<'a> {
    #[suggestion("use the new name", code = "{replace}", applicability = "machine-applicable")]
    WithSpan {
        #[primary_span]
        suggestion: Span,
        replace: &'a str,
    },
    #[help("use the new name `{$replace}`")]
    WithoutSpan { replace: &'a str },
}

#[derive(LintDiagnostic)]
#[diag("lint `{$name}` has been renamed to `{$replace}`")]
pub(crate) struct RenamedLintFromCommandLine<'a> {
    pub name: &'a str,
    pub replace: &'a str,
    #[subdiagnostic]
    pub suggestion: RenamedLintSuggestion<'a>,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag("lint `{$name}` has been removed: {$reason}")]
pub(crate) struct RemovedLint<'a> {
    pub name: &'a str,
    pub reason: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("lint `{$name}` has been removed: {$reason}")]
pub(crate) struct RemovedLintFromCommandLine<'a> {
    pub name: &'a str,
    pub reason: &'a str,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag("unknown lint: `{$name}`")]
pub(crate) struct UnknownLint {
    pub name: String,
    #[subdiagnostic]
    pub suggestion: Option<UnknownLintSuggestion>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnknownLintSuggestion {
    #[suggestion(
        "{$from_rustc ->
            [true] a lint with a similar name exists in `rustc` lints
            *[false] did you mean
        }",
        code = "{replace}",
        applicability = "maybe-incorrect"
    )]
    WithSpan {
        #[primary_span]
        suggestion: Span,
        replace: Symbol,
        from_rustc: bool,
    },
    #[help(
        "{$from_rustc ->
            [true] a lint with a similar name exists in `rustc` lints: `{$replace}`
            *[false] did you mean: `{$replace}`
        }"
    )]
    WithoutSpan { replace: Symbol, from_rustc: bool },
}

#[derive(LintDiagnostic)]
#[diag("unknown lint: `{$name}`", code = E0602)]
pub(crate) struct UnknownLintFromCommandLine<'a> {
    pub name: String,
    #[subdiagnostic]
    pub suggestion: Option<UnknownLintSuggestion>,
    #[subdiagnostic]
    pub requested_level: RequestedLevel<'a>,
}

#[derive(LintDiagnostic)]
#[diag("{$level}({$name}) is ignored unless specified at crate level")]
pub(crate) struct IgnoredUnlessCrateSpecified<'a> {
    pub level: &'a str,
    pub name: Symbol,
}

// dangling.rs
#[derive(LintDiagnostic)]
#[diag("this creates a dangling pointer because temporary `{$ty}` is dropped at end of statement")]
#[help("bind the `{$ty}` to a variable such that it outlives the pointer returned by `{$callee}`")]
#[note("a dangling pointer is safe, but dereferencing one is undefined behavior")]
#[note("returning a pointer to a local variable will always result in a dangling pointer")]
#[note("for more information, see <https://doc.rust-lang.org/reference/destructors.html>")]
// FIXME: put #[primary_span] on `ptr_span` once it does not cause conflicts
pub(crate) struct DanglingPointersFromTemporaries<'tcx> {
    pub callee: Ident,
    pub ty: Ty<'tcx>,
    #[label("pointer created here")]
    pub ptr_span: Span,
    #[label("this `{$ty}` is dropped at end of statement")]
    pub temporary_span: Span,
}

#[derive(LintDiagnostic)]
#[diag("{$fn_kind} returns a dangling pointer to dropped local variable `{$local_var_name}`")]
#[note("a dangling pointer is safe, but dereferencing one is undefined behavior")]
#[note("for more information, see <https://doc.rust-lang.org/reference/destructors.html>")]
pub(crate) struct DanglingPointersFromLocals<'tcx> {
    pub ret_ty: Ty<'tcx>,
    #[label("return type is `{$ret_ty}`")]
    pub ret_ty_span: Span,
    pub fn_kind: &'static str,
    #[label("local variable `{$local_var_name}` is dropped at the end of the {$fn_kind}")]
    pub local_var: Span,
    pub local_var_name: Ident,
    pub local_var_ty: Ty<'tcx>,
    #[label("dangling pointer created here")]
    pub created_at: Option<Span>,
}

// multiple_supertrait_upcastable.rs
#[derive(LintDiagnostic)]
#[diag("`{$ident}` is dyn-compatible and has multiple supertraits")]
pub(crate) struct MultipleSupertraitUpcastable {
    pub ident: Ident,
}

// non_ascii_idents.rs
#[derive(LintDiagnostic)]
#[diag("identifier contains non-ASCII characters")]
pub(crate) struct IdentifierNonAsciiChar;

#[derive(LintDiagnostic)]
#[diag(
    "identifier contains {$codepoints_len ->
        [one] { $identifier_type ->
            [Exclusion] a character from an archaic script
            [Technical] a character that is for non-linguistic, specialized usage
            [Limited_Use] a character from a script in limited use
            [Not_NFKC] a non normalized (NFKC) character
            *[other] an uncommon character
        }
        *[other] { $identifier_type ->
            [Exclusion] {$codepoints_len} characters from archaic scripts
            [Technical] {$codepoints_len} characters that are for non-linguistic, specialized usage
            [Limited_Use] {$codepoints_len} characters from scripts in limited use
            [Not_NFKC] {$codepoints_len} non normalized (NFKC) characters
            *[other] uncommon characters
        }
    }: {$codepoints}"
)]
#[note(
    r#"{$codepoints_len ->
        [one] this character is
        *[other] these characters are
    } included in the{$identifier_type ->
        [Restricted] {""}
        *[other] {" "}{$identifier_type}
    } Unicode general security profile"#
)]
pub(crate) struct IdentifierUncommonCodepoints {
    pub codepoints: Vec<char>,
    pub codepoints_len: usize,
    pub identifier_type: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("found both `{$existing_sym}` and `{$sym}` as identifiers, which look alike")]
pub(crate) struct ConfusableIdentifierPair {
    pub existing_sym: Symbol,
    pub sym: Symbol,
    #[label("other identifier used here")]
    pub label: Span,
    #[label("this identifier can be confused with `{$existing_sym}`")]
    pub main_label: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "the usage of Script Group `{$set}` in this crate consists solely of mixed script confusables"
)]
#[note("the usage includes {$includes}")]
#[note("please recheck to make sure their usages are indeed what you want")]
pub(crate) struct MixedScriptConfusables {
    pub set: String,
    pub includes: String,
}

// non_fmt_panic.rs
pub(crate) struct NonFmtPanicUnused {
    pub count: usize,
    pub suggestion: Option<Span>,
}

// Used because of two suggestions based on one Option<Span>
impl<'a> LintDiagnostic<'a, ()> for NonFmtPanicUnused {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!(
            "panic message contains {$count ->
                [one] an unused
                *[other] unused
            } formatting {$count ->
                [one] placeholder
                *[other] placeholders
            }"
        ));
        diag.arg("count", self.count);
        diag.note(msg!("this message is not used as a format string when given without arguments, but will be in Rust 2021"));
        if let Some(span) = self.suggestion {
            diag.span_suggestion(
                span.shrink_to_hi(),
                msg!(
                    "add the missing {$count ->
                        [one] argument
                        *[other] arguments
                    }"
                ),
                ", ...",
                Applicability::HasPlaceholders,
            );
            diag.span_suggestion(
                span.shrink_to_lo(),
                msg!(r#"or add a "{"{"}{"}"}" format string to use the message literally"#),
                "\"{}\", ",
                Applicability::MachineApplicable,
            );
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(
    "panic message contains {$count ->
        [one] a brace
        *[other] braces
    }"
)]
#[note("this message is not used as a format string, but will be in Rust 2021")]
pub(crate) struct NonFmtPanicBraces {
    pub count: usize,
    #[suggestion(
        "add a \"{\"{\"}{\"}\"}\" format string to use the message literally",
        code = "\"{{}}\", ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Option<Span>,
}

// nonstandard_style.rs
#[derive(LintDiagnostic)]
#[diag("{$sort} `{$name}` should have an upper camel case name")]
pub(crate) struct NonCamelCaseType<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonCamelCaseTypeSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum NonCamelCaseTypeSub {
    #[label("should have an UpperCamelCase name")]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        "convert the identifier to upper camel case",
        code = "{replace}",
        applicability = "maybe-incorrect"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        replace: String,
    },
}

#[derive(LintDiagnostic)]
#[diag("{$sort} `{$name}` should have a snake case name")]
pub(crate) struct NonSnakeCaseDiag<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    pub sc: String,
    #[subdiagnostic]
    pub sub: NonSnakeCaseDiagSub,
}

pub(crate) enum NonSnakeCaseDiagSub {
    Label { span: Span },
    Help,
    RenameOrConvertSuggestion { span: Span, suggestion: Ident },
    ConvertSuggestion { span: Span, suggestion: String },
    SuggestionAndNote { span: Span },
}

impl Subdiagnostic for NonSnakeCaseDiagSub {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            NonSnakeCaseDiagSub::Label { span } => {
                diag.span_label(span, msg!("should have a snake_case name"));
            }
            NonSnakeCaseDiagSub::Help => {
                diag.help(msg!("convert the identifier to snake case: `{$sc}`"));
            }
            NonSnakeCaseDiagSub::ConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    msg!("convert the identifier to snake case"),
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::RenameOrConvertSuggestion { span, suggestion } => {
                diag.span_suggestion(
                    span,
                    msg!("rename the identifier or convert it to a snake case raw identifier"),
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            NonSnakeCaseDiagSub::SuggestionAndNote { span } => {
                diag.note(msg!("`{$sc}` cannot be used as a raw identifier"));
                diag.span_suggestion(
                    span,
                    msg!("rename the identifier"),
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag("{$sort} `{$name}` should have an upper case name")]
pub(crate) struct NonUpperCaseGlobal<'a> {
    pub sort: &'a str,
    pub name: &'a str,
    #[subdiagnostic]
    pub sub: NonUpperCaseGlobalSub,
    #[subdiagnostic]
    pub usages: Vec<NonUpperCaseGlobalSubTool>,
}

#[derive(Subdiagnostic)]
pub(crate) enum NonUpperCaseGlobalSub {
    #[label("should have an UPPER_CASE name")]
    Label {
        #[primary_span]
        span: Span,
    },
    #[suggestion("convert the identifier to upper case", code = "{replace}")]
    Suggestion {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        replace: String,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(
    "convert the identifier to upper case",
    code = "{replace}",
    applicability = "machine-applicable",
    style = "tool-only"
)]
pub(crate) struct NonUpperCaseGlobalSubTool {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) replace: String,
}

// noop_method_call.rs
#[derive(LintDiagnostic)]
#[diag("call to `.{$method}()` on a reference in this situation does nothing")]
#[note(
    "the type `{$orig_ty}` does not implement `{$trait_}`, so calling `{$method}` on `&{$orig_ty}` copies the reference, which does not do anything and can be removed"
)]
pub(crate) struct NoopMethodCallDiag<'a> {
    pub method: Ident,
    pub orig_ty: Ty<'a>,
    pub trait_: Symbol,
    #[suggestion("remove this redundant call", code = "", applicability = "machine-applicable")]
    pub label: Span,
    #[suggestion(
        "if you meant to clone `{$orig_ty}`, implement `Clone` for it",
        code = "#[derive(Clone)]\n",
        applicability = "maybe-incorrect"
    )]
    pub suggest_derive: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(
    "using `.deref()` on a double reference, which returns `{$ty}` instead of dereferencing the inner type"
)]
pub(crate) struct SuspiciousDoubleRefDerefDiag<'a> {
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(
    "using `.clone()` on a double reference, which returns `{$ty}` instead of cloning the inner type"
)]
pub(crate) struct SuspiciousDoubleRefCloneDiag<'a> {
    pub ty: Ty<'a>,
}

// non_local_defs.rs
pub(crate) enum NonLocalDefinitionsDiag {
    Impl {
        depth: u32,
        body_kind_descr: &'static str,
        body_name: String,
        cargo_update: Option<NonLocalDefinitionsCargoUpdateNote>,
        const_anon: Option<Option<Span>>,
        doctest: bool,
        macro_to_change: Option<(String, &'static str)>,
    },
    MacroRules {
        depth: u32,
        body_kind_descr: &'static str,
        body_name: String,
        doctest: bool,
        cargo_update: Option<NonLocalDefinitionsCargoUpdateNote>,
    },
}

impl<'a> LintDiagnostic<'a, ()> for NonLocalDefinitionsDiag {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        match self {
            NonLocalDefinitionsDiag::Impl {
                depth,
                body_kind_descr,
                body_name,
                cargo_update,
                const_anon,
                doctest,
                macro_to_change,
            } => {
                diag.primary_message(msg!("non-local `impl` definition, `impl` blocks should be written at the same level as their item"));
                diag.arg("depth", depth);
                diag.arg("body_kind_descr", body_kind_descr);
                diag.arg("body_name", body_name);

                if let Some((macro_to_change, macro_kind)) = macro_to_change {
                    diag.arg("macro_to_change", macro_to_change);
                    diag.arg("macro_kind", macro_kind);
                    diag.note(msg!("the {$macro_kind} `{$macro_to_change}` defines the non-local `impl`, and may need to be changed"));
                }
                if let Some(cargo_update) = cargo_update {
                    diag.subdiagnostic(cargo_update);
                }

                diag.note(msg!("an `impl` is never scoped, even when it is nested inside an item, as it may impact type checking outside of that item, which can be the case if neither the trait or the self type are at the same nesting level as the `impl`"));

                if doctest {
                    diag.help(msg!("make this doc-test a standalone test with its own `fn main() {\"{\"} ... {\"}\"}`"));
                }

                if let Some(const_anon) = const_anon {
                    diag.note(msg!("items in an anonymous const item (`const _: () = {\"{\"} ... {\"}\"}`) are treated as in the same scope as the anonymous const's declaration for the purpose of this lint"));
                    if let Some(const_anon) = const_anon {
                        diag.span_suggestion(
                            const_anon,
                            msg!("use a const-anon item to suppress this lint"),
                            "_",
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
            NonLocalDefinitionsDiag::MacroRules {
                depth,
                body_kind_descr,
                body_name,
                doctest,
                cargo_update,
            } => {
                diag.primary_message(msg!("non-local `macro_rules!` definition, `#[macro_export]` macro should be written at top level module"));
                diag.arg("depth", depth);
                diag.arg("body_kind_descr", body_kind_descr);
                diag.arg("body_name", body_name);

                if doctest {
                    diag.help(msg!(r#"remove the `#[macro_export]` or make this doc-test a standalone test with its own `fn main() {"{"} ... {"}"}`"#));
                } else {
                    diag.help(msg!(
                        "remove the `#[macro_export]` or move this `macro_rules!` outside the of the current {$body_kind_descr} {$depth ->
                            [one] `{$body_name}`
                            *[other] `{$body_name}` and up {$depth} bodies
                        }"
                    ));
                }

                diag.note(msg!("a `macro_rules!` definition is non-local if it is nested inside an item and has a `#[macro_export]` attribute"));

                if let Some(cargo_update) = cargo_update {
                    diag.subdiagnostic(cargo_update);
                }
            }
        }
    }
}

#[derive(Subdiagnostic)]
#[note(
    "the {$macro_kind} `{$macro_name}` may come from an old version of the `{$crate_name}` crate, try updating your dependency with `cargo update -p {$crate_name}`"
)]
pub(crate) struct NonLocalDefinitionsCargoUpdateNote {
    pub macro_kind: &'static str,
    pub macro_name: Symbol,
    pub crate_name: Symbol,
}

// precedence.rs
#[derive(LintDiagnostic)]
#[diag("`-` has lower precedence than method calls, which might be unexpected")]
#[note("e.g. `-4.abs()` equals `-4`; while `(-4).abs()` equals `4`")]
pub(crate) struct AmbiguousNegativeLiteralsDiag {
    #[subdiagnostic]
    pub negative_literal: AmbiguousNegativeLiteralsNegativeLiteralSuggestion,
    #[subdiagnostic]
    pub current_behavior: AmbiguousNegativeLiteralsCurrentBehaviorSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "add parentheses around the `-` and the literal to call the method on a negative literal",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AmbiguousNegativeLiteralsNegativeLiteralSuggestion {
    #[suggestion_part(code = "(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "add parentheses around the literal and the method call to keep the current behavior",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AmbiguousNegativeLiteralsCurrentBehaviorSuggestion {
    #[suggestion_part(code = "(")]
    pub start_span: Span,
    #[suggestion_part(code = ")")]
    pub end_span: Span,
}

// pass_by_value.rs
#[derive(LintDiagnostic)]
#[diag("passing `{$ty}` by reference")]
pub(crate) struct PassByValueDiag {
    pub ty: String,
    #[suggestion("try passing by value", code = "{ty}", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

// redundant_semicolon.rs
#[derive(LintDiagnostic)]
#[diag(
    "unnecessary trailing {$multiple ->
        [true] semicolons
        *[false] semicolon
    }"
)]
pub(crate) struct RedundantSemicolonsDiag {
    pub multiple: bool,
    #[subdiagnostic]
    pub suggestion: Option<RedundantSemicolonsSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "remove {$multiple_semicolons ->
        [true] these semicolons
        *[false] this semicolon
    }",
    code = "",
    applicability = "maybe-incorrect"
)]
pub(crate) struct RedundantSemicolonsSuggestion {
    pub multiple_semicolons: bool,
    #[primary_span]
    pub span: Span,
}

// traits.rs
pub(crate) struct DropTraitConstraintsDiag<'a> {
    pub predicate: Clause<'a>,
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> LintDiagnostic<'a, ()> for DropTraitConstraintsDiag<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!("bounds on `{$predicate}` are most likely incorrect, consider instead using `{$needs_drop}` to detect whether a type can be trivially dropped"));
        diag.arg("predicate", self.predicate);
        diag.arg("needs_drop", self.tcx.def_path_str(self.def_id));
    }
}

pub(crate) struct DropGlue<'a> {
    pub tcx: TyCtxt<'a>,
    pub def_id: DefId,
}

// Needed for def_path_str
impl<'a> LintDiagnostic<'a, ()> for DropGlue<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!("types that do not implement `Drop` can still have drop glue, consider instead using `{$needs_drop}` to detect whether a type is trivially dropped"));
        diag.arg("needs_drop", self.tcx.def_path_str(self.def_id));
    }
}

// transmute.rs
#[derive(LintDiagnostic)]
#[diag("transmuting an integer to a pointer creates a pointer without provenance")]
#[note("this is dangerous because dereferencing the resulting pointer is undefined behavior")]
#[note(
    "exposed provenance semantics can be used to create a pointer based on some previously exposed provenance"
)]
#[help(
    "if you truly mean to create a pointer without provenance, use `std::ptr::without_provenance_mut`"
)]
#[help(
    "for more information about transmute, see <https://doc.rust-lang.org/std/mem/fn.transmute.html#transmutation-between-pointers-and-integers>"
)]
#[help(
    "for more information about exposed provenance, see <https://doc.rust-lang.org/std/ptr/index.html#exposed-provenance>"
)]
pub(crate) struct IntegerToPtrTransmutes<'tcx> {
    #[subdiagnostic]
    pub suggestion: Option<IntegerToPtrTransmutesSuggestion<'tcx>>,
}

#[derive(Subdiagnostic)]
pub(crate) enum IntegerToPtrTransmutesSuggestion<'tcx> {
    #[multipart_suggestion(
        "use `std::ptr::with_exposed_provenance{$suffix}` instead to use a previously exposed provenance",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    ToPtr {
        dst: Ty<'tcx>,
        suffix: &'static str,
        #[suggestion_part(code = "std::ptr::with_exposed_provenance{suffix}::<{dst}>(")]
        start_call: Span,
    },
    #[multipart_suggestion(
        "use `std::ptr::with_exposed_provenance{$suffix}` instead to use a previously exposed provenance",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    ToRef {
        dst: Ty<'tcx>,
        suffix: &'static str,
        ref_mutbl: &'static str,
        #[suggestion_part(
            code = "&{ref_mutbl}*std::ptr::with_exposed_provenance{suffix}::<{dst}>("
        )]
        start_call: Span,
    },
}

// types.rs
#[derive(LintDiagnostic)]
#[diag("range endpoint is out of range for `{$ty}`")]
pub(crate) struct RangeEndpointOutOfRange<'a> {
    pub ty: &'a str,
    #[subdiagnostic]
    pub sub: UseInclusiveRange<'a>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UseInclusiveRange<'a> {
    #[suggestion(
        "use an inclusive range instead",
        code = "{start}..={literal}{suffix}",
        applicability = "machine-applicable"
    )]
    WithoutParen {
        #[primary_span]
        sugg: Span,
        start: String,
        literal: u128,
        suffix: &'a str,
    },
    #[multipart_suggestion("use an inclusive range instead", applicability = "machine-applicable")]
    WithParen {
        #[suggestion_part(code = "=")]
        eq_sugg: Span,
        #[suggestion_part(code = "{literal}{suffix}")]
        lit_sugg: Span,
        literal: u128,
        suffix: &'a str,
    },
}

#[derive(LintDiagnostic)]
#[diag("literal out of range for `{$ty}`")]
pub(crate) struct OverflowingBinHex<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub dec: u128,
    pub actually: String,
    #[subdiagnostic]
    pub sign: OverflowingBinHexSign,
    #[subdiagnostic]
    pub sub: Option<OverflowingBinHexSub<'a>>,
    #[subdiagnostic]
    pub sign_bit_sub: Option<OverflowingBinHexSignBitSub<'a>>,
}

#[derive(Subdiagnostic)]
pub(crate) enum OverflowingBinHexSign {
    #[note(
        "the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}` and will become `{$actually}{$ty}`"
    )]
    Positive,
    #[note("the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}`")]
    #[note("and the value `-{$lit}` will become `{$actually}{$ty}`")]
    Negative,
}

#[derive(Subdiagnostic)]
pub(crate) enum OverflowingBinHexSub<'a> {
    #[suggestion(
        "consider using the type `{$suggestion_ty}` instead",
        code = "{sans_suffix}{suggestion_ty}",
        applicability = "machine-applicable"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion_ty: &'a str,
        sans_suffix: &'a str,
    },
    #[help("consider using the type `{$suggestion_ty}` instead")]
    Help { suggestion_ty: &'a str },
}

#[derive(Subdiagnostic)]
#[suggestion(
    "to use as a negative number (decimal `{$negative_val}`), consider using the type `{$uint_ty}` for the literal and cast it to `{$int_ty}`",
    code = "{lit_no_suffix}{uint_ty} as {int_ty}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct OverflowingBinHexSignBitSub<'a> {
    #[primary_span]
    pub span: Span,
    pub lit_no_suffix: &'a str,
    pub negative_val: String,
    pub uint_ty: &'a str,
    pub int_ty: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("literal out of range for `{$ty}`")]
#[note("the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`")]
pub(crate) struct OverflowingInt<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub min: i128,
    pub max: u128,
    #[subdiagnostic]
    pub help: Option<OverflowingIntHelp<'a>>,
}

#[derive(Subdiagnostic)]
#[help("consider using the type `{$suggestion_ty}` instead")]
pub(crate) struct OverflowingIntHelp<'a> {
    pub suggestion_ty: &'a str,
}

#[derive(LintDiagnostic)]
#[diag("only `u8` can be cast into `char`")]
pub(crate) struct OnlyCastu8ToChar {
    #[suggestion(
        "use a `char` literal instead",
        code = "'\\u{{{literal:X}}}'",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub literal: u128,
}

#[derive(LintDiagnostic)]
#[diag("literal out of range for `{$ty}`")]
#[note("the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`")]
pub(crate) struct OverflowingUInt<'a> {
    pub ty: &'a str,
    pub lit: String,
    pub min: u128,
    pub max: u128,
}

#[derive(LintDiagnostic)]
#[diag("literal out of range for `{$ty}`")]
#[note(
    "the literal `{$lit}` does not fit into the type `{$ty}` and will be converted to `{$ty}::INFINITY`"
)]
pub(crate) struct OverflowingLiteral<'a> {
    pub ty: &'a str,
    pub lit: String,
}

#[derive(LintDiagnostic)]
#[diag("surrogate values are not valid for `char`")]
#[note("`0xD800..=0xDFFF` are reserved for Unicode surrogates and are not valid `char` values")]
pub(crate) struct SurrogateCharCast {
    pub literal: u128,
}

#[derive(LintDiagnostic)]
#[diag("value exceeds maximum `char` value")]
#[note("maximum valid `char` value is `0x10FFFF`")]
pub(crate) struct TooLargeCharCast {
    pub literal: u128,
}

#[derive(LintDiagnostic)]
#[diag(
    "repr(C) does not follow the power alignment rule. This may affect platform C ABI compatibility for this type"
)]
pub(crate) struct UsesPowerAlignment;

#[derive(LintDiagnostic)]
#[diag("comparison is useless due to type limits")]
pub(crate) struct UnusedComparisons;

#[derive(LintDiagnostic)]
pub(crate) enum InvalidNanComparisons {
    #[diag("incorrect NaN comparison, NaN cannot be directly compared to itself")]
    EqNe {
        #[subdiagnostic]
        suggestion: InvalidNanComparisonsSuggestion,
    },
    #[diag("incorrect NaN comparison, NaN is not orderable")]
    LtLeGtGe,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidNanComparisonsSuggestion {
    #[multipart_suggestion(
        "use `f32::is_nan()` or `f64::is_nan()` instead",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    Spanful {
        #[suggestion_part(code = "!")]
        neg: Option<Span>,
        #[suggestion_part(code = ".is_nan()")]
        float: Span,
        #[suggestion_part(code = "")]
        nan_plus_binop: Span,
    },
    #[help("use `f32::is_nan()` or `f64::is_nan()` instead")]
    Spanless,
}

#[derive(LintDiagnostic)]
pub(crate) enum AmbiguousWidePointerComparisons<'a> {
    #[diag(
        "ambiguous wide pointer comparison, the comparison includes metadata which may not be expected"
    )]
    SpanfulEq {
        #[subdiagnostic]
        addr_suggestion: AmbiguousWidePointerComparisonsAddrSuggestion<'a>,
        #[subdiagnostic]
        addr_metadata_suggestion: Option<AmbiguousWidePointerComparisonsAddrMetadataSuggestion<'a>>,
    },
    #[diag(
        "ambiguous wide pointer comparison, the comparison includes metadata which may not be expected"
    )]
    SpanfulCmp {
        #[subdiagnostic]
        cast_suggestion: AmbiguousWidePointerComparisonsCastSuggestion<'a>,
        #[subdiagnostic]
        expect_suggestion: AmbiguousWidePointerComparisonsExpectSuggestion<'a>,
    },
    #[diag(
        "ambiguous wide pointer comparison, the comparison includes metadata which may not be expected"
    )]
    #[help("use explicit `std::ptr::eq` method to compare metadata and addresses")]
    #[help("use `std::ptr::addr_eq` or untyped pointers to only compare their addresses")]
    Spanless,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use explicit `std::ptr::eq` method to compare metadata and addresses",
    style = "verbose",
    // FIXME(#53934): make machine-applicable again
    applicability = "maybe-incorrect"
)]
pub(crate) struct AmbiguousWidePointerComparisonsAddrMetadataSuggestion<'a> {
    pub ne: &'a str,
    pub deref_left: &'a str,
    pub deref_right: &'a str,
    pub l_modifiers: &'a str,
    pub r_modifiers: &'a str,
    #[suggestion_part(code = "{ne}std::ptr::eq({deref_left}")]
    pub left: Span,
    #[suggestion_part(code = "{l_modifiers}, {deref_right}")]
    pub middle: Span,
    #[suggestion_part(code = "{r_modifiers})")]
    pub right: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use `std::ptr::addr_eq` or untyped pointers to only compare their addresses",
    style = "verbose",
    // FIXME(#53934): make machine-applicable again
    applicability = "maybe-incorrect"
)]
pub(crate) struct AmbiguousWidePointerComparisonsAddrSuggestion<'a> {
    pub(crate) ne: &'a str,
    pub(crate) deref_left: &'a str,
    pub(crate) deref_right: &'a str,
    pub(crate) l_modifiers: &'a str,
    pub(crate) r_modifiers: &'a str,
    #[suggestion_part(code = "{ne}std::ptr::addr_eq({deref_left}")]
    pub(crate) left: Span,
    #[suggestion_part(code = "{l_modifiers}, {deref_right}")]
    pub(crate) middle: Span,
    #[suggestion_part(code = "{r_modifiers})")]
    pub(crate) right: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use untyped pointers to only compare their addresses",
    style = "verbose",
    // FIXME(#53934): make machine-applicable again
    applicability = "maybe-incorrect"
)]
pub(crate) struct AmbiguousWidePointerComparisonsCastSuggestion<'a> {
    pub(crate) deref_left: &'a str,
    pub(crate) deref_right: &'a str,
    pub(crate) paren_left: &'a str,
    pub(crate) paren_right: &'a str,
    pub(crate) l_modifiers: &'a str,
    pub(crate) r_modifiers: &'a str,
    #[suggestion_part(code = "({deref_left}")]
    pub(crate) left_before: Option<Span>,
    #[suggestion_part(code = "{l_modifiers}{paren_left}.cast::<()>()")]
    pub(crate) left_after: Span,
    #[suggestion_part(code = "({deref_right}")]
    pub(crate) right_before: Option<Span>,
    #[suggestion_part(code = "{r_modifiers}{paren_right}.cast::<()>()")]
    pub(crate) right_after: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "or expect the lint to compare the pointers metadata and addresses",
    style = "verbose",
    // FIXME(#53934): make machine-applicable again
    applicability = "maybe-incorrect"
)]
pub(crate) struct AmbiguousWidePointerComparisonsExpectSuggestion<'a> {
    pub(crate) paren_left: &'a str,
    pub(crate) paren_right: &'a str,
    // FIXME(#127436): Adjust once resolved
    #[suggestion_part(
        code = r#"{{ #[expect(ambiguous_wide_pointer_comparisons, reason = "...")] {paren_left}"#
    )]
    pub(crate) before: Span,
    #[suggestion_part(code = "{paren_right} }}")]
    pub(crate) after: Span,
}

#[derive(LintDiagnostic)]
pub(crate) enum UnpredictableFunctionPointerComparisons<'a, 'tcx> {
    #[diag(
        "function pointer comparisons do not produce meaningful results since their addresses are not guaranteed to be unique"
    )]
    #[note("the address of the same function can vary between different codegen units")]
    #[note(
        "furthermore, different functions could have the same address after being merged together"
    )]
    #[note(
        "for more information visit <https://doc.rust-lang.org/nightly/core/ptr/fn.fn_addr_eq.html>"
    )]
    Suggestion {
        #[subdiagnostic]
        sugg: UnpredictableFunctionPointerComparisonsSuggestion<'a, 'tcx>,
    },
    #[diag(
        "function pointer comparisons do not produce meaningful results since their addresses are not guaranteed to be unique"
    )]
    #[note("the address of the same function can vary between different codegen units")]
    #[note(
        "furthermore, different functions could have the same address after being merged together"
    )]
    #[note(
        "for more information visit <https://doc.rust-lang.org/nightly/core/ptr/fn.fn_addr_eq.html>"
    )]
    Warn,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnpredictableFunctionPointerComparisonsSuggestion<'a, 'tcx> {
    #[multipart_suggestion(
        "refactor your code, or use `std::ptr::fn_addr_eq` to suppress the lint",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    FnAddrEq {
        ne: &'a str,
        deref_left: &'a str,
        deref_right: &'a str,
        #[suggestion_part(code = "{ne}std::ptr::fn_addr_eq({deref_left}")]
        left: Span,
        #[suggestion_part(code = ", {deref_right}")]
        middle: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
    #[multipart_suggestion(
        "refactor your code, or use `std::ptr::fn_addr_eq` to suppress the lint",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    FnAddrEqWithCast {
        ne: &'a str,
        deref_left: &'a str,
        deref_right: &'a str,
        fn_sig: rustc_middle::ty::PolyFnSig<'tcx>,
        #[suggestion_part(code = "{ne}std::ptr::fn_addr_eq({deref_left}")]
        left: Span,
        #[suggestion_part(code = ", {deref_right}")]
        middle: Span,
        #[suggestion_part(code = " as {fn_sig})")]
        right: Span,
    },
}

pub(crate) struct ImproperCTypes<'a> {
    pub ty: Ty<'a>,
    pub desc: &'a str,
    pub label: Span,
    pub help: Option<DiagMessage>,
    pub note: DiagMessage,
    pub span_note: Option<Span>,
}

// Used because of the complexity of Option<DiagMessage>, DiagMessage, and Option<Span>
impl<'a> LintDiagnostic<'a, ()> for ImproperCTypes<'_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!("`extern` {$desc} uses type `{$ty}`, which is not FFI-safe"));
        diag.arg("ty", self.ty);
        diag.arg("desc", self.desc);
        diag.span_label(self.label, msg!("not FFI-safe"));
        if let Some(help) = self.help {
            diag.help(help);
        }
        diag.note(self.note);
        if let Some(note) = self.span_note {
            diag.span_note(note, msg!("the type is defined here"));
        }
    }
}

#[derive(LintDiagnostic)]
#[diag("passing type `{$ty}` to a function with \"gpu-kernel\" ABI may have unexpected behavior")]
#[help("use primitive types and raw pointers to get reliable behavior")]
pub(crate) struct ImproperGpuKernelArg<'a> {
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag("function with the \"gpu-kernel\" ABI has a mangled name")]
#[help("use `unsafe(no_mangle)` or `unsafe(export_name = \"<name>\")`")]
#[note("mangled names make it hard to find the kernel, this is usually not intended")]
pub(crate) struct MissingGpuKernelExportName;

#[derive(LintDiagnostic)]
#[diag("enum variant is more than three times larger ({$largest} bytes) than the next largest")]
pub(crate) struct VariantSizeDifferencesDiag {
    pub largest: u64,
}

#[derive(LintDiagnostic)]
#[diag("atomic loads cannot have `Release` or `AcqRel` ordering")]
#[help("consider using ordering modes `Acquire`, `SeqCst` or `Relaxed`")]
pub(crate) struct AtomicOrderingLoad;

#[derive(LintDiagnostic)]
#[diag("atomic stores cannot have `Acquire` or `AcqRel` ordering")]
#[help("consider using ordering modes `Release`, `SeqCst` or `Relaxed`")]
pub(crate) struct AtomicOrderingStore;

#[derive(LintDiagnostic)]
#[diag("memory fences cannot have `Relaxed` ordering")]
#[help("consider using ordering modes `Acquire`, `Release`, `AcqRel` or `SeqCst`")]
pub(crate) struct AtomicOrderingFence;

#[derive(LintDiagnostic)]
#[diag(
    "`{$method}`'s failure ordering may not be `Release` or `AcqRel`, since a failed `{$method}` does not result in a write"
)]
#[help("consider using `Acquire` or `Relaxed` failure ordering instead")]
pub(crate) struct InvalidAtomicOrderingDiag {
    pub method: Symbol,
    #[label("invalid failure ordering")]
    pub fail_order_arg_span: Span,
}

// unused.rs
#[derive(LintDiagnostic)]
#[diag("unused {$op} that must be used")]
pub(crate) struct UnusedOp<'a> {
    pub op: &'a str,
    #[label("the {$op} produces a value")]
    pub label: Span,
    #[subdiagnostic]
    pub suggestion: UnusedOpSuggestion,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedOpSuggestion {
    #[suggestion(
        "use `let _ = ...` to ignore the resulting value",
        style = "verbose",
        code = "let _ = ",
        applicability = "maybe-incorrect"
    )]
    NormalExpr {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "use `let _ = ...` to ignore the resulting value",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    BlockTailExpr {
        #[suggestion_part(code = "let _ = ")]
        before_span: Span,
        #[suggestion_part(code = ";")]
        after_span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag("unused result of type `{$ty}`")]
pub(crate) struct UnusedResult<'a> {
    pub ty: Ty<'a>,
}

// FIXME(davidtwco): this isn't properly translatable because of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(
    "unused {$pre}{$count ->
        [one] closure
        *[other] closures
    }{$post} that must be used"
)]
#[note("closures are lazy and do nothing unless called")]
pub(crate) struct UnusedClosure<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable because of the
// pre/post strings
#[derive(LintDiagnostic)]
#[diag(
    "unused {$pre}{$count ->
        [one] coroutine
        *[other] coroutine
    }{$post} that must be used"
)]
#[note("coroutines are lazy and do nothing unless resumed")]
pub(crate) struct UnusedCoroutine<'a> {
    pub count: usize,
    pub pre: &'a str,
    pub post: &'a str,
}

// FIXME(davidtwco): this isn't properly translatable because of the pre/post
// strings
pub(crate) struct UnusedDef<'a, 'b> {
    pub pre: &'a str,
    pub post: &'a str,
    pub cx: &'a LateContext<'b>,
    pub def_id: DefId,
    pub note: Option<Symbol>,
    pub suggestion: Option<UnusedDefSuggestion>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedDefSuggestion {
    #[suggestion(
        "use `let _ = ...` to ignore the resulting value",
        style = "verbose",
        code = "let _ = ",
        applicability = "maybe-incorrect"
    )]
    NormalExpr {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "use `let _ = ...` to ignore the resulting value",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    BlockTailExpr {
        #[suggestion_part(code = "let _ = ")]
        before_span: Span,
        #[suggestion_part(code = ";")]
        after_span: Span,
    },
}

// Needed because of def_path_str
impl<'a> LintDiagnostic<'a, ()> for UnusedDef<'_, '_> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!("unused {$pre}`{$def}`{$post} that must be used"));
        diag.arg("pre", self.pre);
        diag.arg("post", self.post);
        diag.arg("def", self.cx.tcx.def_path_str(self.def_id));
        // check for #[must_use = "..."]
        if let Some(note) = self.note {
            diag.note(note.to_string());
        }
        if let Some(sugg) = self.suggestion {
            diag.subdiagnostic(sugg);
        }
    }
}

#[derive(LintDiagnostic)]
#[diag("path statement drops value")]
pub(crate) struct PathStatementDrop {
    #[subdiagnostic]
    pub sub: PathStatementDropSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum PathStatementDropSub {
    #[suggestion(
        "use `drop` to clarify the intent",
        code = "drop({snippet});",
        applicability = "machine-applicable"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[help("use `drop` to clarify the intent")]
    Help {
        #[primary_span]
        span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag("path statement with no effect")]
pub(crate) struct PathStatementNoEffect;

#[derive(LintDiagnostic)]
#[diag("unnecessary {$delim} around {$item}")]
pub(crate) struct UnusedDelim<'a> {
    pub delim: &'static str,
    pub item: &'a str,
    #[subdiagnostic]
    pub suggestion: Option<UnusedDelimSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("remove these {$delim}", applicability = "machine-applicable")]
pub(crate) struct UnusedDelimSuggestion {
    #[suggestion_part(code = "{start_replace}")]
    pub start_span: Span,
    pub start_replace: &'static str,
    #[suggestion_part(code = "{end_replace}")]
    pub end_span: Span,
    pub end_replace: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("braces around {$node} is unnecessary")]
pub(crate) struct UnusedImportBracesDiag {
    pub node: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("unnecessary allocation, use `&` instead")]
pub(crate) struct UnusedAllocationDiag;

#[derive(LintDiagnostic)]
#[diag("unnecessary allocation, use `&mut` instead")]
pub(crate) struct UnusedAllocationMutDiag;

pub(crate) struct AsyncFnInTraitDiag {
    pub sugg: Option<Vec<(Span, String)>>,
}

impl<'a> LintDiagnostic<'a, ()> for AsyncFnInTraitDiag {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(msg!("use of `async fn` in public traits is discouraged as auto trait bounds cannot be specified"));
        diag.note(msg!("you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`"));
        if let Some(sugg) = self.sugg {
            diag.multipart_suggestion(msg!("you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change"), sugg, Applicability::MaybeIncorrect);
        }
    }
}

#[derive(LintDiagnostic)]
#[diag("binding has unit type `()`")]
pub(crate) struct UnitBindingsDiag {
    #[label("this pattern is inferred to be the unit type `()`")]
    pub label: Span,
}

#[derive(LintDiagnostic)]
pub(crate) enum InvalidAsmLabel {
    #[diag("avoid using named labels in inline assembly")]
    #[help("only local labels of the form `<number>:` should be used in inline asm")]
    #[note(
        "see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information"
    )]
    Named {
        #[note("the label may be declared in the expansion of a macro")]
        missing_precise_span: bool,
    },
    #[diag("avoid using named labels in inline assembly")]
    #[help("only local labels of the form `<number>:` should be used in inline asm")]
    #[note("format arguments may expand to a non-numeric value")]
    #[note(
        "see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information"
    )]
    FormatArg {
        #[note("the label may be declared in the expansion of a macro")]
        missing_precise_span: bool,
    },
    #[diag("avoid using labels containing only the digits `0` and `1` in inline assembly")]
    #[help("start numbering with `2` instead")]
    #[note("an LLVM bug makes these labels ambiguous with a binary literal number on x86")]
    #[note("see <https://github.com/llvm/llvm-project/issues/99547> for more information")]
    Binary {
        #[note("the label may be declared in the expansion of a macro")]
        missing_precise_span: bool,
        // hack to get a label on the whole span, must match the emitted span
        #[label("use a different label that doesn't start with `0` or `1`")]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum UnexpectedCfgCargoHelp {
    #[help("consider using a Cargo feature instead")]
    #[help(
        "or consider adding in `Cargo.toml` the `check-cfg` lint config for the lint:{$cargo_toml_lint_cfg}"
    )]
    LintCfg { cargo_toml_lint_cfg: String },
    #[help("consider using a Cargo feature instead")]
    #[help(
        "or consider adding in `Cargo.toml` the `check-cfg` lint config for the lint:{$cargo_toml_lint_cfg}"
    )]
    #[help("or consider adding `{$build_rs_println}` to the top of the `build.rs`")]
    LintCfgAndBuildRs { cargo_toml_lint_cfg: String, build_rs_println: String },
}

impl UnexpectedCfgCargoHelp {
    fn cargo_toml_lint_cfg(unescaped: &str) -> String {
        format!(
            "\n [lints.rust]\n unexpected_cfgs = {{ level = \"warn\", check-cfg = ['{unescaped}'] }}"
        )
    }

    pub(crate) fn lint_cfg(unescaped: &str) -> Self {
        UnexpectedCfgCargoHelp::LintCfg {
            cargo_toml_lint_cfg: Self::cargo_toml_lint_cfg(unescaped),
        }
    }

    pub(crate) fn lint_cfg_and_build_rs(unescaped: &str, escaped: &str) -> Self {
        UnexpectedCfgCargoHelp::LintCfgAndBuildRs {
            cargo_toml_lint_cfg: Self::cargo_toml_lint_cfg(unescaped),
            build_rs_println: format!("println!(\"cargo::rustc-check-cfg={escaped}\");"),
        }
    }
}

#[derive(Subdiagnostic)]
#[help("to expect this configuration use `{$cmdline_arg}`")]
pub(crate) struct UnexpectedCfgRustcHelp {
    pub cmdline_arg: String,
}

impl UnexpectedCfgRustcHelp {
    pub(crate) fn new(unescaped: &str) -> Self {
        Self { cmdline_arg: format!("--check-cfg={unescaped}") }
    }
}

#[derive(Subdiagnostic)]
#[note(
    "using a cfg inside a {$macro_kind} will use the cfgs from the destination crate and not the ones from the defining crate"
)]
#[help("try referring to `{$macro_name}` crate for guidance on how handle this unexpected cfg")]
pub(crate) struct UnexpectedCfgRustcMacroHelp {
    pub macro_kind: &'static str,
    pub macro_name: Symbol,
}

#[derive(Subdiagnostic)]
#[note(
    "using a cfg inside a {$macro_kind} will use the cfgs from the destination crate and not the ones from the defining crate"
)]
#[help("try referring to `{$macro_name}` crate for guidance on how handle this unexpected cfg")]
#[help(
    "the {$macro_kind} `{$macro_name}` may come from an old version of the `{$crate_name}` crate, try updating your dependency with `cargo update -p {$crate_name}`"
)]
pub(crate) struct UnexpectedCfgCargoMacroHelp {
    pub macro_kind: &'static str,
    pub macro_name: Symbol,
    pub crate_name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("unexpected `cfg` condition name: `{$name}`")]
pub(crate) struct UnexpectedCfgName {
    #[subdiagnostic]
    pub code_sugg: unexpected_cfg_name::CodeSuggestion,
    #[subdiagnostic]
    pub invocation_help: unexpected_cfg_name::InvocationHelp,

    pub name: Symbol,
}

pub(crate) mod unexpected_cfg_name {
    use rustc_errors::DiagSymbolList;
    use rustc_macros::Subdiagnostic;
    use rustc_span::{Ident, Span, Symbol};

    #[derive(Subdiagnostic)]
    pub(crate) enum CodeSuggestion {
        #[help("consider defining some features in `Cargo.toml`")]
        DefineFeatures,
        #[multipart_suggestion(
            "there is a similar config predicate: `version(\"..\")`",
            applicability = "machine-applicable"
        )]
        VersionSyntax {
            #[suggestion_part(code = "(")]
            between_name_and_value: Span,
            #[suggestion_part(code = ")")]
            after_value: Span,
        },
        #[suggestion(
            "there is a config with a similar name and value",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameAndValue {
            #[primary_span]
            span: Span,
            code: String,
        },
        #[suggestion(
            "there is a config with a similar name and no value",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameNoValue {
            #[primary_span]
            span: Span,
            code: String,
        },
        #[suggestion(
            "there is a config with a similar name and different values",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarNameDifferentValues {
            #[primary_span]
            span: Span,
            code: String,
            #[subdiagnostic]
            expected: Option<ExpectedValues>,
        },
        #[suggestion(
            "there is a config with a similar name",
            applicability = "maybe-incorrect",
            code = "{code}"
        )]
        SimilarName {
            #[primary_span]
            span: Span,
            code: String,
            #[subdiagnostic]
            expected: Option<ExpectedValues>,
        },
        SimilarValues {
            #[subdiagnostic]
            with_similar_values: Vec<FoundWithSimilarValue>,
            #[subdiagnostic]
            expected_names: Option<ExpectedNames>,
        },
        #[suggestion(
            "you may have meant to use `{$literal}` (notice the capitalization). Doing so makes this predicate evaluate to `{$literal}` unconditionally",
            applicability = "machine-applicable",
            style = "verbose",
            code = "{literal}"
        )]
        BooleanLiteral {
            #[primary_span]
            span: Span,
            literal: bool,
        },
    }

    #[derive(Subdiagnostic)]
    #[help("expected values for `{$best_match}` are: {$possibilities}")]
    pub(crate) struct ExpectedValues {
        pub best_match: Symbol,
        pub possibilities: DiagSymbolList,
    }

    #[derive(Subdiagnostic)]
    #[suggestion(
        "found config with similar value",
        applicability = "maybe-incorrect",
        code = "{code}"
    )]
    pub(crate) struct FoundWithSimilarValue {
        #[primary_span]
        pub span: Span,
        pub code: String,
    }

    #[derive(Subdiagnostic)]
    #[help_once(
        "expected names are: {$possibilities}{$and_more ->
            [0] {\"\"}
            *[other] {\" \"}and {$and_more} more
        }"
    )]
    pub(crate) struct ExpectedNames {
        pub possibilities: DiagSymbolList<Ident>,
        pub and_more: usize,
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum InvocationHelp {
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration"
        )]
        Cargo {
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgCargoMacroHelp>,
            #[subdiagnostic]
            help: Option<super::UnexpectedCfgCargoHelp>,
        },
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg.html> for more information about checking conditional configuration"
        )]
        Rustc {
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgRustcMacroHelp>,
            #[subdiagnostic]
            help: super::UnexpectedCfgRustcHelp,
        },
    }
}

#[derive(LintDiagnostic)]
#[diag(
    "unexpected `cfg` condition value: {$has_value ->
        [true] `{$value}`
        *[false] (none)
    }"
)]
pub(crate) struct UnexpectedCfgValue {
    #[subdiagnostic]
    pub code_sugg: unexpected_cfg_value::CodeSuggestion,
    #[subdiagnostic]
    pub invocation_help: unexpected_cfg_value::InvocationHelp,

    pub has_value: bool,
    pub value: String,
}

pub(crate) mod unexpected_cfg_value {
    use rustc_errors::DiagSymbolList;
    use rustc_macros::Subdiagnostic;
    use rustc_span::{Span, Symbol};

    #[derive(Subdiagnostic)]
    pub(crate) enum CodeSuggestion {
        ChangeValue {
            #[subdiagnostic]
            expected_values: ExpectedValues,
            #[subdiagnostic]
            suggestion: Option<ChangeValueSuggestion>,
        },
        #[note("no expected value for `{$name}`")]
        RemoveValue {
            #[subdiagnostic]
            suggestion: Option<RemoveValueSuggestion>,

            name: Symbol,
        },
        #[note("no expected values for `{$name}`")]
        RemoveCondition {
            #[subdiagnostic]
            suggestion: RemoveConditionSuggestion,

            name: Symbol,
        },
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum ChangeValueSuggestion {
        #[suggestion(
            "there is a expected value with a similar name",
            code = r#""{best_match}""#,
            applicability = "maybe-incorrect"
        )]
        SimilarName {
            #[primary_span]
            span: Span,
            best_match: Symbol,
        },
        #[suggestion(
            "specify a config value",
            code = r#" = "{first_possibility}""#,
            applicability = "maybe-incorrect"
        )]
        SpecifyValue {
            #[primary_span]
            span: Span,
            first_possibility: Symbol,
        },
    }

    #[derive(Subdiagnostic)]
    #[suggestion("remove the value", code = "", applicability = "maybe-incorrect")]
    pub(crate) struct RemoveValueSuggestion {
        #[primary_span]
        pub span: Span,
    }

    #[derive(Subdiagnostic)]
    #[suggestion("remove the condition", code = "", applicability = "maybe-incorrect")]
    pub(crate) struct RemoveConditionSuggestion {
        #[primary_span]
        pub span: Span,
    }

    #[derive(Subdiagnostic)]
    #[note(
        "expected values for `{$name}` are: {$have_none_possibility ->
            [true] {\"(none), \"}
            *[false] {\"\"}
        }{$possibilities}{$and_more ->
            [0] {\"\"}
            *[other] {\" \"}and {$and_more} more
        }"
    )]
    pub(crate) struct ExpectedValues {
        pub name: Symbol,
        pub have_none_possibility: bool,
        pub possibilities: DiagSymbolList,
        pub and_more: usize,
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum InvocationHelp {
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration"
        )]
        Cargo {
            #[subdiagnostic]
            help: Option<CargoHelp>,
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgCargoMacroHelp>,
        },
        #[note(
            "see <https://doc.rust-lang.org/nightly/rustc/check-cfg.html> for more information about checking conditional configuration"
        )]
        Rustc {
            #[subdiagnostic]
            help: Option<super::UnexpectedCfgRustcHelp>,
            #[subdiagnostic]
            macro_help: Option<super::UnexpectedCfgRustcMacroHelp>,
        },
    }

    #[derive(Subdiagnostic)]
    pub(crate) enum CargoHelp {
        #[help("consider adding `{$value}` as a feature in `Cargo.toml`")]
        AddFeature {
            value: Symbol,
        },
        #[help("consider defining some features in `Cargo.toml`")]
        DefineFeatures,
        Other(#[subdiagnostic] super::UnexpectedCfgCargoHelp),
    }
}

#[derive(LintDiagnostic)]
#[diag("extern crate `{$extern_crate}` is unused in crate `{$local_crate}`")]
#[help("remove the dependency or add `use {$extern_crate} as _;` to the crate root")]
pub(crate) struct UnusedCrateDependency {
    pub extern_crate: Symbol,
    pub local_crate: Symbol,
}

// FIXME(jdonszelmann): duplicated in rustc_attr_parsing, should be moved there completely.
#[derive(LintDiagnostic)]
#[diag(
    "{$num_suggestions ->
        [1] attribute must be of the form {$suggestions}
        *[other] valid forms for the attribute are {$suggestions}
    }"
)]
pub(crate) struct IllFormedAttributeInput {
    pub num_suggestions: usize,
    pub suggestions: DiagArgValue,
    #[note("for more information, visit <{$docs}>")]
    pub has_docs: bool,
    pub docs: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("unicode codepoint changing visible direction of text present in comment")]
#[note(
    "these kind of unicode codepoints change the way text flows on applications that support them, but can cause confusion because they change the order of characters on the screen"
)]
pub(crate) struct UnicodeTextFlow {
    #[label(
        "{$num_codepoints ->
            [1] this comment contains an invisible unicode text flow control codepoint
            *[other] this comment contains invisible unicode text flow control codepoints
        }"
    )]
    pub comment_span: Span,
    #[subdiagnostic]
    pub characters: Vec<UnicodeCharNoteSub>,
    #[subdiagnostic]
    pub suggestions: Option<UnicodeTextFlowSuggestion>,

    pub num_codepoints: usize,
}

#[derive(Subdiagnostic)]
#[label("{$c_debug}")]
pub(crate) struct UnicodeCharNoteSub {
    #[primary_span]
    pub span: Span,
    pub c_debug: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "if their presence wasn't intentional, you can remove them",
    applicability = "machine-applicable",
    style = "hidden"
)]
pub(crate) struct UnicodeTextFlowSuggestion {
    #[suggestion_part(code = "")]
    pub spans: Vec<Span>,
}

#[derive(LintDiagnostic)]
#[diag(
    "absolute paths must start with `self`, `super`, `crate`, or an external crate name in the 2018 edition"
)]
pub(crate) struct AbsPathWithModule {
    #[subdiagnostic]
    pub sugg: AbsPathWithModuleSugg,
}

#[derive(Subdiagnostic)]
#[suggestion("use `crate`", code = "{replacement}")]
pub(crate) struct AbsPathWithModuleSugg {
    #[primary_span]
    pub span: Span,
    #[applicability]
    pub applicability: Applicability,
    pub replacement: String,
}

#[derive(LintDiagnostic)]
#[diag("hidden lifetime parameters in types are deprecated")]
pub(crate) struct ElidedLifetimesInPaths {
    #[subdiagnostic]
    pub subdiag: ElidedLifetimeInPathSubdiag,
}

#[derive(LintDiagnostic)]
#[diag(
    "{$num_snippets ->
        [one] unused import: {$span_snippets}
        *[other] unused imports: {$span_snippets}
    }"
)]
pub(crate) struct UnusedImports {
    #[subdiagnostic]
    pub sugg: UnusedImportsSugg,
    #[help("if this is a test module, consider adding a `#[cfg(test)]` to the containing module")]
    pub test_module_span: Option<Span>,

    pub span_snippets: DiagArgValue,
    pub num_snippets: usize,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedImportsSugg {
    #[suggestion(
        "remove the whole `use` item",
        applicability = "machine-applicable",
        code = "",
        style = "tool-only"
    )]
    RemoveWholeUse {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "{$num_to_remove ->
            [one] remove the unused import
            *[other] remove the unused imports
        }",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    RemoveImports {
        #[suggestion_part(code = "")]
        remove_spans: Vec<Span>,
        num_to_remove: usize,
    },
}

#[derive(LintDiagnostic)]
#[diag("the item `{$ident}` is imported redundantly")]
pub(crate) struct RedundantImport {
    #[subdiagnostic]
    pub subs: Vec<RedundantImportSub>,

    pub ident: Ident,
}

#[derive(Subdiagnostic)]
pub(crate) enum RedundantImportSub {
    #[label("the item `{$ident}` is already imported here")]
    ImportedHere(#[primary_span] Span),
    #[label("the item `{$ident}` is already defined here")]
    DefinedHere(#[primary_span] Span),
    #[label("the item `{$ident}` is already imported by the extern prelude")]
    ImportedPrelude(#[primary_span] Span),
    #[label("the item `{$ident}` is already defined by the extern prelude")]
    DefinedPrelude(#[primary_span] Span),
}

#[derive(LintDiagnostic)]
pub(crate) enum PatternsInFnsWithoutBody {
    #[diag("patterns aren't allowed in foreign function declarations")]
    Foreign {
        #[subdiagnostic]
        sub: PatternsInFnsWithoutBodySub,
    },
    #[diag("patterns aren't allowed in functions without bodies")]
    Bodiless {
        #[subdiagnostic]
        sub: PatternsInFnsWithoutBodySub,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(
    "remove `mut` from the parameter",
    code = "{ident}",
    applicability = "machine-applicable"
)]
pub(crate) struct PatternsInFnsWithoutBodySub {
    #[primary_span]
    pub span: Span,

    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag("prefix `{$prefix}` is unknown")]
pub(crate) struct ReservedPrefix {
    #[label("unknown prefix")]
    pub label: Span,
    #[suggestion(
        "insert whitespace here to avoid this being parsed as a prefix in Rust 2021",
        code = " ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,

    pub prefix: String,
}

#[derive(LintDiagnostic)]
#[diag("prefix `'r` is reserved")]
pub(crate) struct RawPrefix {
    #[label("reserved prefix")]
    pub label: Span,
    #[suggestion(
        "insert whitespace here to avoid this being parsed as a prefix in Rust 2021",
        code = " ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "this labeled break expression is easy to confuse with an unlabeled break with a labeled value expression"
)]
pub(crate) struct BreakWithLabelAndLoop {
    #[subdiagnostic]
    pub sub: BreakWithLabelAndLoopSub,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("wrap this expression in parentheses", applicability = "machine-applicable")]
pub(crate) struct BreakWithLabelAndLoopSub {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(LintDiagnostic)]
#[diag("where clause not allowed here")]
#[note("see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information")]
pub(crate) struct DeprecatedWhereClauseLocation {
    #[subdiagnostic]
    pub suggestion: DeprecatedWhereClauseLocationSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum DeprecatedWhereClauseLocationSugg {
    #[multipart_suggestion(
        "move it to the end of the type declaration",
        applicability = "machine-applicable"
    )]
    MoveToEnd {
        #[suggestion_part(code = "")]
        left: Span,
        #[suggestion_part(code = "{sugg}")]
        right: Span,

        sugg: String,
    },
    #[suggestion("remove this `where`", code = "", applicability = "machine-applicable")]
    RemoveWhere {
        #[primary_span]
        span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag("lifetime parameter `{$ident}` only used once")]
pub(crate) struct SingleUseLifetime {
    #[label("this lifetime...")]
    pub param_span: Span,
    #[label("...is used only here")]
    pub use_span: Span,
    #[subdiagnostic]
    pub suggestion: Option<SingleUseLifetimeSugg>,

    pub ident: Ident,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("elide the single-use lifetime", applicability = "machine-applicable")]
pub(crate) struct SingleUseLifetimeSugg {
    #[suggestion_part(code = "")]
    pub deletion_span: Option<Span>,
    #[suggestion_part(code = "{replace_lt}")]
    pub use_span: Span,

    pub replace_lt: String,
}

#[derive(LintDiagnostic)]
#[diag("lifetime parameter `{$ident}` never used")]
pub(crate) struct UnusedLifetime {
    #[suggestion("elide the unused lifetime", code = "", applicability = "machine-applicable")]
    pub deletion_span: Option<Span>,

    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag("named argument `{$named_arg_name}` is not used by name")]
pub(crate) struct NamedArgumentUsedPositionally {
    #[label("this named argument is referred to by position in formatting string")]
    pub named_arg_sp: Span,
    #[label("this formatting argument uses named argument `{$named_arg_name}` by position")]
    pub position_label_sp: Option<Span>,
    #[suggestion(
        "use the named argument by name to avoid ambiguity",
        style = "verbose",
        code = "{name}",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Option<Span>,

    pub name: String,
    pub named_arg_name: String,
}

#[derive(LintDiagnostic)]
#[diag("ambiguous glob re-exports")]
pub(crate) struct AmbiguousGlobReexports {
    #[label("the name `{$name}` in the {$namespace} namespace is first re-exported here")]
    pub first_reexport: Span,
    #[label("but the name `{$name}` in the {$namespace} namespace is also re-exported here")]
    pub duplicate_reexport: Span,

    pub name: String,
    pub namespace: String,
}

#[derive(LintDiagnostic)]
#[diag("private item shadows public glob re-export")]
pub(crate) struct HiddenGlobReexports {
    #[note(
        "the name `{$name}` in the {$namespace} namespace is supposed to be publicly re-exported here"
    )]
    pub glob_reexport: Span,
    #[note("but the private item here shadows it")]
    pub private_item: Span,

    pub name: String,
    pub namespace: String,
}

#[derive(LintDiagnostic)]
#[diag("unnecessary qualification")]
pub(crate) struct UnusedQualifications {
    #[suggestion(
        "remove the unnecessary path segments",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub removal_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "{$elided ->
        [true] `&` without an explicit lifetime name cannot be used here
        *[false] `'_` cannot be used here
    }"
)]
pub(crate) struct AssociatedConstElidedLifetime {
    #[suggestion(
        "use the `'static` lifetime",
        style = "verbose",
        code = "{code}",
        applicability = "machine-applicable"
    )]
    pub span: Span,

    pub code: &'static str,
    pub elided: bool,
    #[note("cannot automatically infer `'static` because of other lifetimes in scope")]
    pub lifetimes_in_scope: MultiSpan,
}

#[derive(LintDiagnostic)]
#[diag("creating a {$shared_label}reference to mutable static")]
pub(crate) struct RefOfMutStatic<'a> {
    #[label("{$shared_label}reference to mutable static")]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<MutRefSugg>,
    pub shared_label: &'a str,
    #[note(
        "shared references to mutable statics are dangerous; it's undefined behavior if the static is mutated or if a mutable reference is created for it while the shared reference lives"
    )]
    pub shared_note: bool,
    #[note(
        "mutable references to mutable statics are dangerous; it's undefined behavior if any other pointer to the static is used or if any other reference is created for the static while the mutable reference lives"
    )]
    pub mut_note: bool,
}

#[derive(Subdiagnostic)]
pub(crate) enum MutRefSugg {
    #[multipart_suggestion(
        "use `&raw const` instead to create a raw pointer",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    Shared {
        #[suggestion_part(code = "&raw const ")]
        span: Span,
    },
    #[multipart_suggestion(
        "use `&raw mut` instead to create a raw pointer",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    Mut {
        #[suggestion_part(code = "&raw mut ")]
        span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag("`use` of a local item without leading `self::`, `super::`, or `crate::`")]
pub(crate) struct UnqualifiedLocalImportsDiag {}

#[derive(LintDiagnostic)]
#[diag("will be parsed as a guarded string in Rust 2024")]
pub(crate) struct ReservedString {
    #[suggestion(
        "insert whitespace here to avoid this being parsed as a guarded string in Rust 2024",
        code = " ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag("reserved token in Rust 2024")]
pub(crate) struct ReservedMultihash {
    #[suggestion(
        "insert whitespace here to avoid this being parsed as a forbidden token in Rust 2024",
        code = " ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(LintDiagnostic)]
#[diag("direct cast of function item into an integer")]
pub(crate) struct FunctionCastsAsIntegerDiag<'tcx> {
    #[subdiagnostic]
    pub(crate) sugg: FunctionCastsAsIntegerSugg<'tcx>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "first cast to a pointer `as *const ()`",
    code = " as *const ()",
    applicability = "machine-applicable",
    style = "verbose"
)]
pub(crate) struct FunctionCastsAsIntegerSugg<'tcx> {
    #[primary_span]
    pub suggestion: Span,
    pub cast_to_ty: Ty<'tcx>,
}

#[derive(Debug)]
pub(crate) struct MismatchedLifetimeSyntaxes {
    pub inputs: LifetimeSyntaxCategories<Vec<Span>>,
    pub outputs: LifetimeSyntaxCategories<Vec<Span>>,

    pub suggestions: Vec<MismatchedLifetimeSyntaxesSuggestion>,
}

impl<'a, G: EmissionGuarantee> LintDiagnostic<'a, G> for MismatchedLifetimeSyntaxes {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, G>) {
        let counts = self.inputs.len() + self.outputs.len();
        let message = match counts {
            LifetimeSyntaxCategories { hidden: 0, elided: 0, named: 0 } => {
                panic!("No lifetime mismatch detected")
            }

            LifetimeSyntaxCategories { hidden: _, elided: _, named: 0 } => {
                msg!("hiding a lifetime that's elided elsewhere is confusing")
            }

            LifetimeSyntaxCategories { hidden: _, elided: 0, named: _ } => {
                msg!("hiding a lifetime that's named elsewhere is confusing")
            }

            LifetimeSyntaxCategories { hidden: 0, elided: _, named: _ } => {
                msg!("eliding a lifetime that's named elsewhere is confusing")
            }

            LifetimeSyntaxCategories { hidden: _, elided: _, named: _ } => {
                msg!("hiding or eliding a lifetime that's named elsewhere is confusing")
            }
        };
        diag.primary_message(message);

        for s in self.inputs.hidden {
            diag.span_label(s, msg!("the lifetime is hidden here"));
        }
        for s in self.inputs.elided {
            diag.span_label(s, msg!("the lifetime is elided here"));
        }
        for s in self.inputs.named {
            diag.span_label(s, msg!("the lifetime is named here"));
        }

        for s in self.outputs.hidden {
            diag.span_label(s, msg!("the same lifetime is hidden here"));
        }
        for s in self.outputs.elided {
            diag.span_label(s, msg!("the same lifetime is elided here"));
        }
        for s in self.outputs.named {
            diag.span_label(s, msg!("the same lifetime is named here"));
        }

        diag.help(msg!(
            "the same lifetime is referred to in inconsistent ways, making the signature confusing"
        ));

        let mut suggestions = self.suggestions.into_iter();
        if let Some(s) = suggestions.next() {
            diag.subdiagnostic(s);

            for mut s in suggestions {
                s.make_optional_alternative();
                diag.subdiagnostic(s);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum MismatchedLifetimeSyntaxesSuggestion {
    Implicit {
        suggestions: Vec<Span>,
        optional_alternative: bool,
    },

    Mixed {
        implicit_suggestions: Vec<Span>,
        explicit_anonymous_suggestions: Vec<(Span, String)>,
        optional_alternative: bool,
    },

    Explicit {
        lifetime_name: String,
        suggestions: Vec<(Span, String)>,
        optional_alternative: bool,
    },
}

impl MismatchedLifetimeSyntaxesSuggestion {
    fn make_optional_alternative(&mut self) {
        use MismatchedLifetimeSyntaxesSuggestion::*;

        let optional_alternative = match self {
            Implicit { optional_alternative, .. }
            | Mixed { optional_alternative, .. }
            | Explicit { optional_alternative, .. } => optional_alternative,
        };

        *optional_alternative = true;
    }
}

impl Subdiagnostic for MismatchedLifetimeSyntaxesSuggestion {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        use MismatchedLifetimeSyntaxesSuggestion::*;

        let style = |optional_alternative| {
            if optional_alternative {
                SuggestionStyle::CompletelyHidden
            } else {
                SuggestionStyle::ShowAlways
            }
        };

        let applicability = |optional_alternative| {
            // `cargo fix` can't handle more than one fix for the same issue,
            // so hide alternative suggestions from it by marking them as maybe-incorrect
            if optional_alternative {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            }
        };

        match self {
            Implicit { suggestions, optional_alternative } => {
                let suggestions = suggestions.into_iter().map(|s| (s, String::new())).collect();
                diag.multipart_suggestion_with_style(
                    msg!("remove the lifetime name from references"),
                    suggestions,
                    applicability(optional_alternative),
                    style(optional_alternative),
                );
            }

            Mixed {
                implicit_suggestions,
                explicit_anonymous_suggestions,
                optional_alternative,
            } => {
                let message = if implicit_suggestions.is_empty() {
                    msg!("use `'_` for type paths")
                } else {
                    msg!("remove the lifetime name from references and use `'_` for type paths")
                };

                let implicit_suggestions =
                    implicit_suggestions.into_iter().map(|s| (s, String::new()));

                let suggestions =
                    implicit_suggestions.chain(explicit_anonymous_suggestions).collect();

                diag.multipart_suggestion_with_style(
                    message,
                    suggestions,
                    applicability(optional_alternative),
                    style(optional_alternative),
                );
            }

            Explicit { lifetime_name, suggestions, optional_alternative } => {
                diag.arg("lifetime_name", lifetime_name);
                let msg = diag.eagerly_translate(msg!("consistently use `{$lifetime_name}`"));
                diag.remove_arg("lifetime_name");
                diag.multipart_suggestion_with_style(
                    msg,
                    suggestions,
                    applicability(optional_alternative),
                    style(optional_alternative),
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag("unused attribute")]
#[note(
    "{$valid_without_list ->
        [true] using `{$attr_path}` with an empty list is equivalent to not using a list at all
        *[other] using `{$attr_path}` with an empty list has no effect
    }"
)]
pub(crate) struct EmptyAttributeList {
    #[suggestion(
        "{$valid_without_list ->
            [true] remove these parentheses
            *[other] remove this attribute
        }",
        code = "",
        applicability = "machine-applicable"
    )]
    pub attr_span: Span,
    pub attr_path: String,
    pub valid_without_list: bool,
}

#[derive(LintDiagnostic)]
#[diag("`#[{$name}]` attribute cannot be used on {$target}")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
#[help("`#[{$name}]` can {$only}be applied to {$applied}")]
pub(crate) struct InvalidTargetLint {
    pub name: String,
    pub target: &'static str,
    pub applied: DiagArgValue,
    pub only: &'static str,
    #[suggestion(
        "remove the attribute",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub attr_span: Span,
}

#[derive(LintDiagnostic)]
#[diag(
    "{$is_used_as_inner ->
        [false] crate-level attribute should be an inner attribute: add an exclamation mark: `#![{$name}]`
        *[other] the `#![{$name}]` attribute can only be used at the crate root
    }"
)]
pub(crate) struct InvalidAttrStyle {
    pub name: String,
    pub is_used_as_inner: bool,
    #[note("this attribute does not have an `!`, which means it is applied to this {$target}")]
    pub target_span: Option<Span>,
    pub target: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("unused attribute")]
pub(crate) struct UnusedDuplicate {
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    #[warning(
        "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
    )]
    pub warning: bool,
}

#[derive(LintDiagnostic)]
#[diag("malformed `doc` attribute input")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct MalformedDoc;

#[derive(LintDiagnostic)]
#[diag("didn't expect any arguments here")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct ExpectedNoArgs;

#[derive(LintDiagnostic)]
#[diag("expected this to be of the form `... = \"...\"`")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct ExpectedNameValue;

#[derive(LintDiagnostic)]
#[diag("unsafe attribute used without unsafe")]
pub(crate) struct UnsafeAttrOutsideUnsafeLint {
    #[label("usage of unsafe attribute")]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: Option<UnsafeAttrOutsideUnsafeSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("wrap the attribute in `unsafe(...)`", applicability = "machine-applicable")]
pub(crate) struct UnsafeAttrOutsideUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(LintDiagnostic)]
#[diag("visibility qualifiers have no effect on `const _` declarations")]
#[note("`const _` does not declare a name, so there is nothing for the qualifier to apply to")]
pub(crate) struct UnusedVisibility {
    #[suggestion(
        "remove the qualifier",
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("doc alias is duplicated")]
pub(crate) struct DocAliasDuplicated {
    #[label("first defined here")]
    pub first_defn: Span,
}

#[derive(LintDiagnostic)]
#[diag("only `hide` or `show` are allowed in `#[doc(auto_cfg(...))]`")]
pub(crate) struct DocAutoCfgExpectsHideOrShow;

#[derive(LintDiagnostic)]
#[diag("there exists a built-in attribute with the same name")]
pub(crate) struct AmbiguousDeriveHelpers;

#[derive(LintDiagnostic)]
#[diag("`#![doc(auto_cfg({$attr_name}(...)))]` only accepts identifiers or key/value items")]
pub(crate) struct DocAutoCfgHideShowUnexpectedItem {
    pub attr_name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("`#![doc(auto_cfg({$attr_name}(...)))]` expects a list of items")]
pub(crate) struct DocAutoCfgHideShowExpectsList {
    pub attr_name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("invalid `doc` attribute")]
pub(crate) struct DocInvalid;

#[derive(LintDiagnostic)]
#[diag("unknown `doc` attribute `include`")]
pub(crate) struct DocUnknownInclude {
    pub inner: &'static str,
    pub value: Symbol,
    #[suggestion(
        "use `doc = include_str!` instead",
        code = "#{inner}[doc = include_str!(\"{value}\")]"
    )]
    pub sugg: (Span, Applicability),
}

#[derive(LintDiagnostic)]
#[diag("unknown `doc` attribute `spotlight`")]
#[note("`doc(spotlight)` was renamed to `doc(notable_trait)`")]
#[note("`doc(spotlight)` is now a no-op")]
pub(crate) struct DocUnknownSpotlight {
    #[suggestion(
        "use `notable_trait` instead",
        style = "short",
        applicability = "machine-applicable",
        code = "notable_trait"
    )]
    pub sugg_span: Span,
}

#[derive(LintDiagnostic)]
#[diag("unknown `doc` attribute `{$name}`")]
#[note(
    "`doc` attribute `{$name}` no longer functions; see issue #44136 <https://github.com/rust-lang/rust/issues/44136>"
)]
#[note("`doc({$name})` is now a no-op")]
pub(crate) struct DocUnknownPasses {
    pub name: Symbol,
    #[label("no longer functions")]
    pub note_span: Span,
}

#[derive(LintDiagnostic)]
#[diag("unknown `doc` attribute `plugins`")]
#[note(
    "`doc` attribute `plugins` no longer functions; see issue #44136 <https://github.com/rust-lang/rust/issues/44136> and CVE-2018-1000622 <https://nvd.nist.gov/vuln/detail/CVE-2018-1000622>"
)]
#[note("`doc(plugins)` is now a no-op")]
pub(crate) struct DocUnknownPlugins {
    #[label("no longer functions")]
    pub label_span: Span,
}

#[derive(LintDiagnostic)]
#[diag("unknown `doc` attribute `{$name}`")]
pub(crate) struct DocUnknownAny {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("expected boolean for `#[doc(auto_cfg = ...)]`")]
pub(crate) struct DocAutoCfgWrongLiteral;

#[derive(LintDiagnostic)]
#[diag("`#[doc(test(...)]` takes a list of attributes")]
pub(crate) struct DocTestTakesList;

#[derive(LintDiagnostic)]
#[diag("unknown `doc(test)` attribute `{$name}`")]
pub(crate) struct DocTestUnknown {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("`#![doc(test(...)]` does not take a literal")]
pub(crate) struct DocTestLiteral;

#[derive(LintDiagnostic)]
#[diag("this attribute can only be applied at the crate level")]
#[note(
    "read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information"
)]
pub(crate) struct AttrCrateLevelOnly;

#[derive(LintDiagnostic)]
#[diag("`#[diagnostic::do_not_recommend]` does not expect any arguments")]
pub(crate) struct DoNotRecommendDoesNotExpectArgs;

#[derive(LintDiagnostic)]
#[diag("invalid `crate_type` value")]
pub(crate) struct UnknownCrateTypes {
    #[subdiagnostic]
    pub sugg: Option<UnknownCrateTypesSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion("did you mean", code = r#""{snippet}""#, applicability = "maybe-incorrect")]
pub(crate) struct UnknownCrateTypesSuggestion {
    #[primary_span]
    pub span: Span,
    pub snippet: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("unreachable configuration predicate")]
pub(crate) struct UnreachableCfgSelectPredicate {
    #[label("this configuration predicate is never reached")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("unreachable configuration predicate")]
pub(crate) struct UnreachableCfgSelectPredicateWildcard {
    #[label("this configuration predicate is never reached")]
    pub span: Span,

    #[label("always matches")]
    pub wildcard_span: Span,
}
