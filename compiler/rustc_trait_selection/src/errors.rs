use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, DiagMessage, DiagStyledString, Diagnostic,
    EmissionGuarantee, IntoDiagArg, Level, MultiSpan, Subdiagnostic, msg,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{Visitor, VisitorExt, walk_ty};
use rustc_hir::{self as hir, AmbigArg, FnRetTy, GenericParamKind, Node};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::print::{PrintTraitRefExt as _, TraitRefPrintOnlyTraitPath};
use rustc_middle::ty::{self, Binder, ClosureKind, FnSig, GenericArg, Region, Ty, TyCtxt};
use rustc_span::{BytePos, Ident, Span, Symbol, kw};

use crate::error_reporting::infer::ObligationCauseAsDiagArg;
use crate::error_reporting::infer::need_type_info::UnderspecifiedArgKind;
use crate::error_reporting::infer::nice_region_error::placeholder_error::Highlighted;

pub mod note_and_explain;

#[derive(Diagnostic)]
#[diag("unable to construct a constant value for the unevaluated constant {$unevaluated}")]
pub struct UnableToConstructConstantValue<'a> {
    #[primary_span]
    pub span: Span,
    pub unevaluated: ty::UnevaluatedConst<'a>,
}

#[derive(Diagnostic)]
#[diag("this attribute must have a value", code = E0232)]
#[note("e.g. `#[rustc_on_unimplemented(message=\"foo\")]`")]
pub struct NoValueInOnUnimplemented {
    #[primary_span]
    #[label("expected value here")]
    pub span: Span,
}

pub struct NegativePositiveConflict<'tcx> {
    pub impl_span: Span,
    pub trait_desc: ty::TraitRef<'tcx>,
    pub self_ty: Option<Ty<'tcx>>,
    pub negative_impl_span: Result<Span, Symbol>,
    pub positive_impl_span: Result<Span, Symbol>,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for NegativePositiveConflict<'_> {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            msg!(
            "found both positive and negative implementation of trait `{$trait_desc}`{$self_desc ->
                [none] {\"\"}
                *[default] {\" \"}for type `{$self_desc}`
            }:"
        ),
        );
        diag.arg("trait_desc", self.trait_desc.print_only_trait_path().to_string());
        diag.arg("self_desc", self.self_ty.map_or_else(|| "none".to_string(), |ty| ty.to_string()));
        diag.span(self.impl_span);
        diag.code(E0751);
        match self.negative_impl_span {
            Ok(span) => {
                diag.span_label(span, msg!("negative implementation here"));
            }
            Err(cname) => {
                diag.note(msg!("negative implementation in crate `{$negative_impl_cname}`"));
                diag.arg("negative_impl_cname", cname.to_string());
            }
        }
        match self.positive_impl_span {
            Ok(span) => {
                diag.span_label(span, msg!("positive implementation here"));
            }
            Err(cname) => {
                diag.note(msg!("positive implementation in crate `{$positive_impl_cname}`"));
                diag.arg("positive_impl_cname", cname.to_string());
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag("overflow evaluating associated type `{$ty}`")]
pub struct InherentProjectionNormalizationOverflow {
    #[primary_span]
    pub span: Span,
    pub ty: String,
}

pub enum AdjustSignatureBorrow {
    Borrow { to_borrow: Vec<(Span, String)> },
    RemoveBorrow { remove_borrow: Vec<(Span, String)> },
}

impl Subdiagnostic for AdjustSignatureBorrow {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            AdjustSignatureBorrow::Borrow { to_borrow } => {
                diag.arg("borrow_len", to_borrow.len());
                diag.multipart_suggestion(
                    msg!(
                        "consider adjusting the signature so it borrows its {$borrow_len ->
                            [one] argument
                            *[other] arguments
                        }"
                    ),
                    to_borrow,
                    Applicability::MaybeIncorrect,
                );
            }
            AdjustSignatureBorrow::RemoveBorrow { remove_borrow } => {
                diag.arg("remove_borrow_len", remove_borrow.len());
                diag.multipart_suggestion(
                    msg!(
                        "consider adjusting the signature so it does not borrow its {$remove_borrow_len ->
                            [one] argument
                            *[other] arguments
                        }"
                    ),
                    remove_borrow,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag("expected a closure that implements the `{$trait_prefix}{$expected}` trait, but this closure only implements `{$trait_prefix}{$found}`", code = E0525)]
pub struct ClosureKindMismatch {
    #[primary_span]
    #[label("this closure implements `{$trait_prefix}{$found}`, not `{$trait_prefix}{$expected}`")]
    pub closure_span: Span,
    pub expected: ClosureKind,
    pub found: ClosureKind,
    #[label("the requirement to implement `{$trait_prefix}{$expected}` derives from here")]
    pub cause_span: Span,

    pub trait_prefix: &'static str,

    #[subdiagnostic]
    pub fn_once_label: Option<ClosureFnOnceLabel>,

    #[subdiagnostic]
    pub fn_mut_label: Option<ClosureFnMutLabel>,
}

#[derive(Subdiagnostic)]
#[label(
    "closure is `{$trait_prefix}FnOnce` because it moves the variable `{$place}` out of its environment"
)]
pub struct ClosureFnOnceLabel {
    #[primary_span]
    pub span: Span,
    pub place: String,
    pub trait_prefix: &'static str,
}

#[derive(Subdiagnostic)]
#[label("closure is `{$trait_prefix}FnMut` because it mutates the variable `{$place}` here")]
pub struct ClosureFnMutLabel {
    #[primary_span]
    pub span: Span,
    pub place: String,
    pub trait_prefix: &'static str,
}

#[derive(Diagnostic)]
#[diag(
    "{$coro_kind}closure does not implement `{$kind}` because it captures state from its environment"
)]
pub(crate) struct CoroClosureNotFn {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
    pub coro_kind: String,
}

#[derive(Diagnostic)]
#[diag("{$source_kind ->
[closure] type annotations needed for the closure `{$source_name}`
[normal] type annotations needed for `{$source_name}`
*[other] type annotations needed
}", code = E0282)]
pub struct AnnotationRequired<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label("type must be known at this point")]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
}

// Copy of `AnnotationRequired` for E0283
#[derive(Diagnostic)]
#[diag("{$source_kind ->
[closure] type annotations needed for the closure `{$source_name}`
[normal] type annotations needed for `{$source_name}`
*[other] type annotations needed
}", code = E0283)]
pub struct AmbiguousImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label("type must be known at this point")]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
}

// Copy of `AnnotationRequired` for E0284
#[derive(Diagnostic)]
#[diag("{$source_kind ->
[closure] type annotations needed for the closure `{$source_name}`
[normal] type annotations needed for `{$source_name}`
*[other] type annotations needed
}", code = E0284)]
pub struct AmbiguousReturn<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label("type must be known at this point")]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
}

// Used when a better one isn't available
#[derive(Subdiagnostic)]
#[label(
    "{$bad_kind ->
*[other] cannot infer type
[more_info] cannot infer {$prefix_kind ->
*[type] type for {$prefix}
[const_with_param] the value of const parameter
[const] the value of the constant
} `{$name}`{$has_parent ->
[true] {\" \"}declared on the {$parent_prefix} `{$parent_name}`
*[false] {\"\"}
}
}"
)]
pub struct InferenceBadError<'a> {
    #[primary_span]
    pub span: Span,
    pub bad_kind: &'static str,
    pub prefix_kind: UnderspecifiedArgKind,
    pub has_parent: bool,
    pub prefix: &'a str,
    pub parent_prefix: &'a str,
    pub parent_name: String,
    pub name: String,
}

#[derive(Subdiagnostic)]
pub enum SourceKindSubdiag<'a> {
    #[suggestion(
        "{$kind ->
            [with_pattern] consider giving `{$name}` an explicit type
            [closure] consider giving this closure parameter an explicit type
            *[other] consider giving this pattern a type
        }{$x_kind ->
            [has_name] , where the {$prefix_kind ->
                *[type] type for {$prefix}
                [const_with_param] value of const parameter
                [const] value of the constant
            } `{$arg_name}` is specified
            [underscore] , where the placeholders `_` are specified
            *[empty] {\"\"}
        }",
        style = "verbose",
        code = ": {type_name}",
        applicability = "has-placeholders"
    )]
    LetLike {
        #[primary_span]
        span: Span,
        name: String,
        type_name: String,
        kind: &'static str,
        x_kind: &'static str,
        prefix_kind: UnderspecifiedArgKind,
        prefix: &'a str,
        arg_name: String,
    },
    #[label(
        "cannot infer {$is_type ->
            [true] type
            *[false] the value
        } of the {$is_type ->
            [true] type
            *[false] const
        } {$parent_exists ->
            [true] parameter `{$param_name}` declared on the {$parent_prefix} `{$parent_name}`
            *[false] parameter {$param_name}
        }"
    )]
    GenericLabel {
        #[primary_span]
        span: Span,
        is_type: bool,
        param_name: String,
        parent_exists: bool,
        parent_prefix: String,
        parent_name: String,
    },
    #[suggestion(
        "consider specifying the generic {$arg_count ->
            [one] argument
            *[other] arguments
        }",
        style = "verbose",
        code = "::<{args}>",
        applicability = "has-placeholders"
    )]
    GenericSuggestion {
        #[primary_span]
        span: Span,
        arg_count: usize,
        args: String,
    },
}

#[derive(Subdiagnostic)]
pub enum SourceKindMultiSuggestion<'a> {
    #[multipart_suggestion(
        "try using a fully qualified path to specify the expected types",
        style = "verbose",
        applicability = "has-placeholders"
    )]
    FullyQualified {
        #[suggestion_part(code = "{def_path}({adjustment}")]
        span_lo: Span,
        #[suggestion_part(code = "{successor_pos}")]
        span_hi: Span,
        def_path: String,
        adjustment: &'a str,
        successor_pos: &'a str,
    },
    #[multipart_suggestion(
        "try giving this closure an explicit return type",
        style = "verbose",
        applicability = "has-placeholders"
    )]
    ClosureReturn {
        #[suggestion_part(code = "{start_span_code}")]
        start_span: Span,
        start_span_code: String,
        #[suggestion_part(code = " }}")]
        end_span: Option<Span>,
    },
}

impl<'a> SourceKindMultiSuggestion<'a> {
    pub fn new_fully_qualified(
        span: Span,
        def_path: String,
        adjustment: &'a str,
        successor: (&'a str, BytePos),
    ) -> Self {
        Self::FullyQualified {
            span_lo: span.shrink_to_lo(),
            span_hi: span.shrink_to_hi().with_hi(successor.1),
            def_path,
            adjustment,
            successor_pos: successor.0,
        }
    }

    pub fn new_closure_return(
        ty_info: String,
        data: &'a FnRetTy<'a>,
        should_wrap_expr: Option<Span>,
    ) -> Self {
        let arrow = match data {
            FnRetTy::DefaultReturn(_) => " -> ",
            _ => "",
        };
        let (start_span, start_span_code, end_span) = match should_wrap_expr {
            Some(end_span) => (data.span(), format!("{arrow}{ty_info} {{"), Some(end_span)),
            None => (data.span(), format!("{arrow}{ty_info}"), None),
        };
        Self::ClosureReturn { start_span, start_span_code, end_span }
    }
}

pub enum RegionOriginNote<'a> {
    Plain {
        span: Span,
        msg: DiagMessage,
    },
    WithName {
        span: Span,
        msg: DiagMessage,
        name: &'a str,
        continues: bool,
    },
    WithRequirement {
        span: Span,
        requirement: ObligationCauseAsDiagArg<'a>,
        expected_found: Option<(DiagStyledString, DiagStyledString)>,
    },
}

impl Subdiagnostic for RegionOriginNote<'_> {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let label_or_note = |diag: &mut Diag<'_, G>, span, msg: DiagMessage| {
            let sub_count = diag.children.iter().filter(|d| d.span.is_dummy()).count();
            let expanded_sub_count = diag.children.iter().filter(|d| !d.span.is_dummy()).count();
            let span_is_primary = diag.span.primary_spans().iter().all(|&sp| sp == span);
            if span_is_primary && sub_count == 0 && expanded_sub_count == 0 {
                diag.span_label(span, msg);
            } else if span_is_primary && expanded_sub_count == 0 {
                diag.note(msg);
            } else {
                diag.span_note(span, msg);
            }
        };
        match self {
            RegionOriginNote::Plain { span, msg } => {
                label_or_note(diag, span, msg);
            }
            RegionOriginNote::WithName { span, msg, name, continues } => {
                diag.arg("name", name);
                diag.arg("continues", continues);
                label_or_note(diag, span, msg);
            }
            RegionOriginNote::WithRequirement {
                span,
                requirement,
                expected_found: Some((expected, found)),
            } => {
                // `RegionOriginNote` can appear multiple times on one diagnostic with different
                // `requirement` values. Scope args per-note and eagerly translate to avoid
                // cross-note arg collisions.
                // See https://github.com/rust-lang/rust/issues/143872 for details.
                diag.store_args();
                diag.arg("requirement", requirement);
                let msg = diag.eagerly_translate(msg!(
                    "...so that the {$requirement ->
                            [method_compat] method type is compatible with trait
                            [type_compat] associated type is compatible with trait
                            [const_compat] const is compatible with trait
                            [expr_assignable] expression is assignable
                            [if_else_different] `if` and `else` have incompatible types
                            [no_else] `if` missing an `else` returns `()`
                            [fn_main_correct_type] `main` function has the correct type
                            [fn_lang_correct_type] lang item function has the correct type
                            [intrinsic_correct_type] intrinsic has the correct type
                            [method_correct_type] method receiver has the correct type
                            *[other] types are compatible
                        }"
                ));
                diag.restore_args();
                label_or_note(diag, span, msg);

                diag.note_expected_found("", expected, "", found);
            }
            RegionOriginNote::WithRequirement { span, requirement, expected_found: None } => {
                // FIXME: this really should be handled at some earlier stage. Our
                // handling of region checking when type errors are present is
                // *terrible*.
                diag.store_args();
                diag.arg("requirement", requirement);
                let msg = diag.eagerly_translate(msg!(
                    "...so that {$requirement ->
                            [method_compat] method type is compatible with trait
                            [type_compat] associated type is compatible with trait
                            [const_compat] const is compatible with trait
                            [expr_assignable] expression is assignable
                            [if_else_different] `if` and `else` have incompatible types
                            [no_else] `if` missing an `else` returns `()`
                            [fn_main_correct_type] `main` function has the correct type
                            [fn_lang_correct_type] lang item function has the correct type
                            [intrinsic_correct_type] intrinsic has the correct type
                            [method_correct_type] method receiver has the correct type
                            *[other] types are compatible
                        }"
                ));
                diag.restore_args();
                label_or_note(diag, span, msg);
            }
        };
    }
}

pub enum LifetimeMismatchLabels {
    InRet {
        param_span: Span,
        ret_span: Span,
        span: Span,
        label_var1: Option<Ident>,
    },
    Normal {
        hir_equal: bool,
        ty_sup: Span,
        ty_sub: Span,
        span: Span,
        sup: Option<Ident>,
        sub: Option<Ident>,
    },
}

impl Subdiagnostic for LifetimeMismatchLabels {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            LifetimeMismatchLabels::InRet { param_span, ret_span, span, label_var1 } => {
                diag.span_label(param_span, msg!("this parameter and the return type are declared with different lifetimes..."));
                diag.span_label(ret_span, msg!("{\"\"}"));
                diag.span_label(
                    span,
                    msg!(
                        "...but data{$label_var1_exists ->
                            [true] {\" \"}from `{$label_var1}`
                            *[false] {\"\"}
                        } is returned here"
                    ),
                );
                diag.arg("label_var1_exists", label_var1.is_some());
                diag.arg("label_var1", label_var1.map(|x| x.to_string()).unwrap_or_default());
            }
            LifetimeMismatchLabels::Normal {
                hir_equal,
                ty_sup,
                ty_sub,
                span,
                sup: label_var1,
                sub: label_var2,
            } => {
                if hir_equal {
                    diag.span_label(
                        ty_sup,
                        msg!("this type is declared with multiple lifetimes..."),
                    );
                    diag.span_label(ty_sub, msg!("{\"\"}"));
                    diag.span_label(
                        span,
                        msg!("...but data with one lifetime flows into the other here"),
                    );
                } else {
                    diag.span_label(
                        ty_sup,
                        msg!("these two types are declared with different lifetimes..."),
                    );
                    diag.span_label(ty_sub, msg!("{\"\"}"));
                    diag.span_label(
                        span,
                        msg!(
                            "...but data{$label_var1_exists ->
                                [true] {\" \"}from `{$label_var1}`
                                *[false] {\"\"}
                            } flows{$label_var2_exists ->
                                [true] {\" \"}into `{$label_var2}`
                                *[false] {\"\"}
                            } here"
                        ),
                    );
                    diag.arg("label_var1_exists", label_var1.is_some());
                    diag.arg("label_var1", label_var1.map(|x| x.to_string()).unwrap_or_default());
                    diag.arg("label_var2_exists", label_var2.is_some());
                    diag.arg("label_var2", label_var2.map(|x| x.to_string()).unwrap_or_default());
                }
            }
        }
    }
}

pub struct AddLifetimeParamsSuggestion<'a> {
    pub tcx: TyCtxt<'a>,
    pub generic_param_scope: LocalDefId,
    pub sub: Region<'a>,
    pub ty_sup: &'a hir::Ty<'a>,
    pub ty_sub: &'a hir::Ty<'a>,
    pub add_note: bool,
}

impl Subdiagnostic for AddLifetimeParamsSuggestion<'_> {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut mk_suggestion = || {
            let Some(anon_reg) = self.tcx.is_suitable_region(self.generic_param_scope, self.sub)
            else {
                return false;
            };

            let node = self.tcx.hir_node_by_def_id(anon_reg.scope);
            let is_impl = matches!(&node, hir::Node::ImplItem(_));
            let (generics, parent_generics) = match node {
                hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { generics, .. }, .. })
                | hir::Node::TraitItem(hir::TraitItem { generics, .. })
                | hir::Node::ImplItem(hir::ImplItem { generics, .. }) => (
                    generics,
                    match self.tcx.parent_hir_node(self.tcx.local_def_id_to_hir_id(anon_reg.scope))
                    {
                        hir::Node::Item(hir::Item {
                            kind: hir::ItemKind::Trait(_, _, _, _, generics, ..),
                            ..
                        })
                        | hir::Node::Item(hir::Item {
                            kind: hir::ItemKind::Impl(hir::Impl { generics, .. }),
                            ..
                        }) => Some(generics),
                        _ => None,
                    },
                ),
                _ => return false,
            };

            let suggestion_param_name = generics
                .params
                .iter()
                .filter(|p| matches!(p.kind, GenericParamKind::Lifetime { .. }))
                .map(|p| p.name.ident().name)
                .find(|i| *i != kw::UnderscoreLifetime);
            let introduce_new = suggestion_param_name.is_none();

            let mut default = "'a".to_string();
            if let Some(parent_generics) = parent_generics {
                let used: FxHashSet<_> = parent_generics
                    .params
                    .iter()
                    .filter(|p| matches!(p.kind, GenericParamKind::Lifetime { .. }))
                    .map(|p| p.name.ident().name)
                    .filter(|i| *i != kw::UnderscoreLifetime)
                    .map(|l| l.to_string())
                    .collect();
                if let Some(lt) =
                    ('a'..='z').map(|it| format!("'{it}")).find(|it| !used.contains(it))
                {
                    // We want a lifetime that *isn't* present in the `trait` or `impl` that assoc
                    // `fn` belongs to. We could suggest reusing one of their lifetimes, but it is
                    // likely to be an over-constraining lifetime requirement, so we always add a
                    // lifetime to the `fn`.
                    default = lt;
                }
            }
            let suggestion_param_name =
                suggestion_param_name.map(|n| n.to_string()).unwrap_or_else(|| default);

            struct ImplicitLifetimeFinder {
                suggestions: Vec<(Span, String)>,
                suggestion_param_name: String,
            }

            impl<'v> Visitor<'v> for ImplicitLifetimeFinder {
                fn visit_ty(&mut self, ty: &'v hir::Ty<'v, AmbigArg>) {
                    match ty.kind {
                        hir::TyKind::Path(hir::QPath::Resolved(_, path)) => {
                            for segment in path.segments {
                                if let Some(args) = segment.args {
                                    if args.args.iter().all(|arg| {
                                        matches!(
                                            arg,
                                            hir::GenericArg::Lifetime(lifetime)
                                                if lifetime.is_implicit()
                                        )
                                    }) {
                                        self.suggestions.push((
                                            segment.ident.span.shrink_to_hi(),
                                            format!(
                                                "<{}>",
                                                args.args
                                                    .iter()
                                                    .map(|_| self.suggestion_param_name.clone())
                                                    .collect::<Vec<_>>()
                                                    .join(", ")
                                            ),
                                        ));
                                    } else {
                                        for arg in args.args {
                                            if let hir::GenericArg::Lifetime(lifetime) = arg
                                                && lifetime.is_anonymous()
                                            {
                                                self.suggestions.push(
                                                    lifetime
                                                        .suggestion(&self.suggestion_param_name),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        hir::TyKind::Ref(lifetime, ..) if lifetime.is_anonymous() => {
                            self.suggestions.push(lifetime.suggestion(&self.suggestion_param_name));
                        }
                        _ => {}
                    }
                    walk_ty(self, ty);
                }
            }
            let mut visitor = ImplicitLifetimeFinder {
                suggestions: vec![],
                suggestion_param_name: suggestion_param_name.clone(),
            };
            if let Some(fn_decl) = node.fn_decl()
                && let hir::FnRetTy::Return(ty) = fn_decl.output
            {
                visitor.visit_ty_unambig(ty);
            }
            if visitor.suggestions.is_empty() {
                // Do not suggest constraining the `&self` param, but rather the return type.
                // If that is wrong (because it is not sufficient), a follow up error will tell the
                // user to fix it. This way we lower the chances of *over* constraining, but still
                // get the cake of "correctly" constrained in two steps.
                visitor.visit_ty_unambig(self.ty_sup);
            }
            visitor.visit_ty_unambig(self.ty_sub);
            if visitor.suggestions.is_empty() {
                return false;
            }
            if introduce_new {
                let new_param_suggestion = if let Some(first) =
                    generics.params.iter().find(|p| !p.name.ident().span.is_empty())
                {
                    (first.span.shrink_to_lo(), format!("{suggestion_param_name}, "))
                } else {
                    (generics.span, format!("<{suggestion_param_name}>"))
                };

                visitor.suggestions.push(new_param_suggestion);
            }
            diag.multipart_suggestion(
                msg!(
                    "consider {$is_reuse ->
                        [true] reusing
                        *[false] introducing
                    } a named lifetime parameter{$is_impl ->
                        [true] {\" \"}and update trait if needed
                        *[false] {\"\"}
                    }"
                ),
                visitor.suggestions,
                Applicability::MaybeIncorrect,
            );
            diag.arg("is_impl", is_impl);
            diag.arg("is_reuse", !introduce_new);

            true
        };
        if mk_suggestion() && self.add_note {
            diag.note(msg!("each elided lifetime in input position becomes a distinct lifetime"));
        }
    }
}

#[derive(Diagnostic)]
#[diag("lifetime mismatch", code = E0623)]
pub struct LifetimeMismatch<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub labels: LifetimeMismatchLabels,
    #[subdiagnostic]
    pub suggestion: AddLifetimeParamsSuggestion<'a>,
}

pub struct IntroducesStaticBecauseUnmetLifetimeReq {
    pub unmet_requirements: MultiSpan,
    pub binding_span: Span,
}

impl Subdiagnostic for IntroducesStaticBecauseUnmetLifetimeReq {
    fn add_to_diag<G: EmissionGuarantee>(mut self, diag: &mut Diag<'_, G>) {
        self.unmet_requirements.push_span_label(
            self.binding_span,
            msg!("introduces a `'static` lifetime requirement"),
        );
        diag.span_note(
            self.unmet_requirements,
            msg!("because this has an unmet lifetime requirement"),
        );
    }
}

// FIXME(#100717): replace with a `Option<Span>` when subdiagnostic supports that
#[derive(Subdiagnostic)]
pub enum DoesNotOutliveStaticFromImpl {
    #[note(
        "...does not necessarily outlive the static lifetime introduced by the compatible `impl`"
    )]
    Spanned {
        #[primary_span]
        span: Span,
    },
    #[note(
        "...does not necessarily outlive the static lifetime introduced by the compatible `impl`"
    )]
    Unspanned,
}

#[derive(Subdiagnostic)]
pub enum ImplicitStaticLifetimeSubdiag {
    #[note("this has an implicit `'static` lifetime requirement")]
    Note {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        "consider relaxing the implicit `'static` requirement",
        style = "verbose",
        code = " + '_",
        applicability = "maybe-incorrect"
    )]
    Sugg {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("incompatible lifetime on type")]
pub struct MismatchedStaticLifetime<'a> {
    #[primary_span]
    pub cause_span: Span,
    #[subdiagnostic]
    pub unmet_lifetime_reqs: IntroducesStaticBecauseUnmetLifetimeReq,
    #[subdiagnostic]
    pub expl: Option<note_and_explain::RegionExplanation<'a>>,
    #[subdiagnostic]
    pub does_not_outlive_static_from_impl: DoesNotOutliveStaticFromImpl,
    #[subdiagnostic]
    pub implicit_static_lifetimes: Vec<ImplicitStaticLifetimeSubdiag>,
}

#[derive(Diagnostic)]
pub enum ExplicitLifetimeRequired<'a> {
    #[diag("explicit lifetime required in the type of `{$simple_ident}`", code = E0621)]
    WithIdent {
        #[primary_span]
        #[label("lifetime `{$named}` required")]
        span: Span,
        simple_ident: Ident,
        named: String,
        #[suggestion(
            "add explicit lifetime `{$named}` to the type of `{$simple_ident}`",
            code = "{new_ty}",
            applicability = "unspecified",
            style = "verbose"
        )]
        new_ty_span: Span,
        #[skip_arg]
        new_ty: Ty<'a>,
    },
    #[diag("explicit lifetime required in parameter type", code = E0621)]
    WithParamType {
        #[primary_span]
        #[label("lifetime `{$named}` required")]
        span: Span,
        named: String,
        #[suggestion(
            "add explicit lifetime `{$named}` to type",
            code = "{new_ty}",
            applicability = "unspecified",
            style = "verbose"
        )]
        new_ty_span: Span,
        #[skip_arg]
        new_ty: Ty<'a>,
    },
}

pub enum TyOrSig<'tcx> {
    Ty(Highlighted<'tcx, Ty<'tcx>>),
    ClosureSig(Highlighted<'tcx, Binder<'tcx, FnSig<'tcx>>>),
}

impl IntoDiagArg for TyOrSig<'_> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        match self {
            TyOrSig::Ty(ty) => ty.into_diag_arg(path),
            TyOrSig::ClosureSig(sig) => sig.into_diag_arg(path),
        }
    }
}

#[derive(Subdiagnostic)]
pub enum ActualImplExplNotes<'tcx> {
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...")]
    ExpectedSignatureTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for any lifetime `'{$lifetime_1}`...")]
    ExpectedSignatureAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for some specific lifetime `'{$lifetime_1}`...")]
    ExpectedSignatureSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(
        "{$leading_ellipsis ->
            [true] ...
            *[false] {\"\"}
        }closure with signature `{$ty_or_sig}` must implement `{$trait_path}`"
    )]
    ExpectedSignatureNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...")]
    ExpectedPassiveTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for any lifetime `'{$lifetime_1}`...")]
    ExpectedPassiveAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for some specific lifetime `'{$lifetime_1}`...")]
    ExpectedPassiveSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(
        "{$leading_ellipsis ->
            [true] ...
            *[false] {\"\"}
        }`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`"
    )]
    ExpectedPassiveNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note("{$leading_ellipsis ->
        [true] ...
        *[false] {\"\"}
    }`{$ty_or_sig}` must implement `{$trait_path}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...")]
    ExpectedOtherTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(
        "{$leading_ellipsis ->
            [true] ...
            *[false] {\"\"}
        }`{$ty_or_sig}` must implement `{$trait_path}`, for any lifetime `'{$lifetime_1}`..."
    )]
    ExpectedOtherAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(
        "{$leading_ellipsis ->
            [true] ...
            *[false] {\"\"}
        }`{$ty_or_sig}` must implement `{$trait_path}`, for some specific lifetime `'{$lifetime_1}`..."
    )]
    ExpectedOtherSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(
        "{$leading_ellipsis ->
            [true] ...
            *[false] {\"\"}
        }`{$ty_or_sig}` must implement `{$trait_path}`"
    )]
    ExpectedOtherNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(
        "...but it actually implements `{$trait_path}`{$has_lifetime ->
            [true] , for some specific lifetime `'{$lifetime}`
            *[false] {\"\"}
        }"
    )]
    ButActuallyImplementsTrait {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
    },
    #[note(
        "...but `{$trait_path}` is actually implemented for the type `{$ty}`{$has_lifetime ->
            [true] , for some specific lifetime `'{$lifetime}`
            *[false] {\"\"}
        }"
    )]
    ButActuallyImplementedForTy {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
        ty: String,
    },
    #[note(
        "...but `{$ty}` actually implements `{$trait_path}`{$has_lifetime ->
            [true] , for some specific lifetime `'{$lifetime}`
            *[false] {\"\"}
        }"
    )]
    ButActuallyTyImplements {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
        ty: String,
    },
}

pub enum ActualImplExpectedKind {
    Signature,
    Passive,
    Other,
}

pub enum ActualImplExpectedLifetimeKind {
    Two,
    Any,
    Some,
    Nothing,
}

impl<'tcx> ActualImplExplNotes<'tcx> {
    pub fn new_expected(
        kind: ActualImplExpectedKind,
        lt_kind: ActualImplExpectedLifetimeKind,
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    ) -> Self {
        match (kind, lt_kind) {
            (ActualImplExpectedKind::Signature, ActualImplExpectedLifetimeKind::Two) => {
                Self::ExpectedSignatureTwo {
                    leading_ellipsis,
                    ty_or_sig,
                    trait_path,
                    lifetime_1,
                    lifetime_2,
                }
            }
            (ActualImplExpectedKind::Signature, ActualImplExpectedLifetimeKind::Any) => {
                Self::ExpectedSignatureAny { leading_ellipsis, ty_or_sig, trait_path, lifetime_1 }
            }
            (ActualImplExpectedKind::Signature, ActualImplExpectedLifetimeKind::Some) => {
                Self::ExpectedSignatureSome { leading_ellipsis, ty_or_sig, trait_path, lifetime_1 }
            }
            (ActualImplExpectedKind::Signature, ActualImplExpectedLifetimeKind::Nothing) => {
                Self::ExpectedSignatureNothing { leading_ellipsis, ty_or_sig, trait_path }
            }
            (ActualImplExpectedKind::Passive, ActualImplExpectedLifetimeKind::Two) => {
                Self::ExpectedPassiveTwo {
                    leading_ellipsis,
                    ty_or_sig,
                    trait_path,
                    lifetime_1,
                    lifetime_2,
                }
            }
            (ActualImplExpectedKind::Passive, ActualImplExpectedLifetimeKind::Any) => {
                Self::ExpectedPassiveAny { leading_ellipsis, ty_or_sig, trait_path, lifetime_1 }
            }
            (ActualImplExpectedKind::Passive, ActualImplExpectedLifetimeKind::Some) => {
                Self::ExpectedPassiveSome { leading_ellipsis, ty_or_sig, trait_path, lifetime_1 }
            }
            (ActualImplExpectedKind::Passive, ActualImplExpectedLifetimeKind::Nothing) => {
                Self::ExpectedPassiveNothing { leading_ellipsis, ty_or_sig, trait_path }
            }
            (ActualImplExpectedKind::Other, ActualImplExpectedLifetimeKind::Two) => {
                Self::ExpectedOtherTwo {
                    leading_ellipsis,
                    ty_or_sig,
                    trait_path,
                    lifetime_1,
                    lifetime_2,
                }
            }
            (ActualImplExpectedKind::Other, ActualImplExpectedLifetimeKind::Any) => {
                Self::ExpectedOtherAny { leading_ellipsis, ty_or_sig, trait_path, lifetime_1 }
            }
            (ActualImplExpectedKind::Other, ActualImplExpectedLifetimeKind::Some) => {
                Self::ExpectedOtherSome { leading_ellipsis, ty_or_sig, trait_path, lifetime_1 }
            }
            (ActualImplExpectedKind::Other, ActualImplExpectedLifetimeKind::Nothing) => {
                Self::ExpectedOtherNothing { leading_ellipsis, ty_or_sig, trait_path }
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag("implementation of `{$trait_def_id}` is not general enough")]
pub struct TraitPlaceholderMismatch<'tcx> {
    #[primary_span]
    pub span: Span,
    #[label("doesn't satisfy where-clause")]
    pub satisfy_span: Option<Span>,
    #[label("due to a where-clause on `{$def_id}`...")]
    pub where_span: Option<Span>,
    #[label("implementation of `{$trait_def_id}` is not general enough")]
    pub dup_span: Option<Span>,
    pub def_id: String,
    pub trait_def_id: String,

    #[subdiagnostic]
    pub actual_impl_expl_notes: Vec<ActualImplExplNotes<'tcx>>,
}

pub struct ConsiderBorrowingParamHelp {
    pub spans: Vec<Span>,
}

impl Subdiagnostic for ConsiderBorrowingParamHelp {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut type_param_span: MultiSpan = self.spans.clone().into();
        for &span in &self.spans {
            // Seems like we can't call f() here as Into<DiagMessage> is required
            type_param_span
                .push_span_label(span, msg!("consider borrowing this type parameter in the trait"));
        }
        let msg = diag.eagerly_translate(msg!("the lifetime requirements from the `impl` do not correspond to the requirements in the `trait`"));
        diag.span_help(type_param_span, msg);
    }
}

#[derive(Diagnostic)]
#[diag("`impl` item signature doesn't match `trait` item signature")]
pub struct TraitImplDiff {
    #[primary_span]
    #[label("found `{$found}`")]
    pub sp: Span,
    #[label("expected `{$expected}`")]
    pub trait_sp: Span,
    #[note(
        "expected signature `{$expected}`
        {\"   \"}found signature `{$found}`"
    )]
    pub note: (),
    #[subdiagnostic]
    pub param_help: ConsiderBorrowingParamHelp,
    #[help(
        "verify the lifetime relationships in the `trait` and `impl` between the `self` argument, the other inputs and its output"
    )]
    pub rel_help: bool,
    pub expected: String,
    pub found: String,
}

pub struct DynTraitConstraintSuggestion {
    pub span: Span,
    pub ident: Ident,
}

impl Subdiagnostic for DynTraitConstraintSuggestion {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut multi_span: MultiSpan = vec![self.span].into();
        multi_span.push_span_label(
            self.span,
            msg!("this has an implicit `'static` lifetime requirement"),
        );
        multi_span.push_span_label(
            self.ident.span,
            msg!("calling this method introduces the `impl`'s `'static` requirement"),
        );
        let msg = diag.eagerly_translate(msg!("the used `impl` has a `'static` requirement"));
        diag.span_note(multi_span, msg);
        let msg =
            diag.eagerly_translate(msg!("consider relaxing the implicit `'static` requirement"));
        diag.span_suggestion_verbose(
            self.span.shrink_to_hi(),
            msg,
            " + '_",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Diagnostic)]
#[diag("{$has_param_name ->
    [true] `{$param_name}`
    *[false] `fn` parameter
} has {$lifetime_kind ->
    [true] lifetime `{$lifetime}`
    *[false] an anonymous lifetime `'_`
} but calling `{$assoc_item}` introduces an implicit `'static` lifetime requirement", code = E0772)]
pub struct ButCallingIntroduces {
    #[label(
        "{$has_lifetime ->
        [true] lifetime `{$lifetime}`
        *[false] an anonymous lifetime `'_`
    }"
    )]
    pub param_ty_span: Span,
    #[primary_span]
    #[label("...is used and required to live as long as `'static` here because of an implicit lifetime bound on the {$has_impl_path ->
        [true] `impl` of `{$impl_path}`
        *[false] inherent `impl`
    }")]
    pub cause_span: Span,

    pub has_param_name: bool,
    pub param_name: String,
    pub has_lifetime: bool,
    pub lifetime: String,
    pub assoc_item: Symbol,
    pub has_impl_path: bool,
    pub impl_path: String,
}

pub struct ReqIntroducedLocations {
    pub span: MultiSpan,
    pub spans: Vec<Span>,
    pub fn_decl_span: Span,
    pub cause_span: Span,
    pub add_label: bool,
}

impl Subdiagnostic for ReqIntroducedLocations {
    fn add_to_diag<G: EmissionGuarantee>(mut self, diag: &mut Diag<'_, G>) {
        for sp in self.spans {
            self.span.push_span_label(sp, msg!("`'static` requirement introduced here"));
        }

        if self.add_label {
            self.span.push_span_label(
                self.fn_decl_span,
                msg!("requirement introduced by this return type"),
            );
        }
        self.span.push_span_label(self.cause_span, msg!("because of this returned expression"));
        let msg = diag.eagerly_translate(msg!(
            "\"`'static` lifetime requirement introduced by the return type"
        ));
        diag.span_note(self.span, msg);
    }
}

#[derive(Diagnostic)]
#[diag("{$has_param_name ->
    [true] `{$param_name}`
    *[false] `fn` parameter
} has {$has_lifetime ->
    [true] lifetime `{$lifetime}`
    *[false] an anonymous lifetime `'_`
} but it needs to satisfy a `'static` lifetime requirement", code = E0759)]
pub struct ButNeedsToSatisfy {
    #[primary_span]
    pub sp: Span,
    #[label(
        "this data with {$has_lifetime ->
        [true] lifetime `{$lifetime}`
        *[false] an anonymous lifetime `'_`
    }..."
    )]
    pub influencer_point: Span,
    #[label("...is used here...")]
    pub spans: Vec<Span>,
    #[label(
        "{$spans_empty ->
        *[true] ...is used and required to live as long as `'static` here
        [false] ...and is required to live as long as `'static` here
    }"
    )]
    pub require_span_as_label: Option<Span>,
    #[note(
        "{$spans_empty ->
        *[true] ...is used and required to live as long as `'static` here
        [false] ...and is required to live as long as `'static` here
    }"
    )]
    pub require_span_as_note: Option<Span>,
    #[note("`'static` lifetime requirement introduced by this bound")]
    pub bound: Option<Span>,

    pub has_param_name: bool,
    pub param_name: String,
    pub spans_empty: bool,
    pub has_lifetime: bool,
    pub lifetime: String,
}

#[derive(Diagnostic)]
#[diag("lifetime of reference outlives lifetime of borrowed content...", code = E0312)]
pub struct OutlivesContent<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag("lifetime of the source pointer does not outlive lifetime bound of the object type", code = E0476)]
pub struct OutlivesBound<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag("the type `{$ty}` does not fulfill the required lifetime", code = E0477)]
pub struct FulfillReqLifetime<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
    #[subdiagnostic]
    pub note: Option<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag("lifetime bound not satisfied", code = E0478)]
pub struct LfBoundNotSatisfied<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag("in type `{$ty}`, reference has a longer lifetime than the data it references", code = E0491)]
pub struct RefLongerThanData<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Subdiagnostic)]
pub enum WhereClauseSuggestions {
    #[suggestion(
        "remove the `where` clause",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    Remove {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        "copy the `where` clause predicates from the trait",
        code = "{space}where {trait_predicates}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    CopyPredicates {
        #[primary_span]
        span: Span,
        space: &'static str,
        trait_predicates: String,
    },
}

#[derive(Subdiagnostic)]
pub enum SuggestRemoveSemiOrReturnBinding {
    #[multipart_suggestion(
        "consider removing this semicolon and boxing the expressions",
        applicability = "machine-applicable"
    )]
    RemoveAndBox {
        #[suggestion_part(code = "Box::new(")]
        first_lo: Span,
        #[suggestion_part(code = ")")]
        first_hi: Span,
        #[suggestion_part(code = "Box::new(")]
        second_lo: Span,
        #[suggestion_part(code = ")")]
        second_hi: Span,
        #[suggestion_part(code = "")]
        sp: Span,
    },
    #[suggestion(
        "consider removing this semicolon",
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    Remove {
        #[primary_span]
        sp: Span,
    },
    #[suggestion(
        "consider returning the local binding `{$ident}`",
        style = "verbose",
        code = "{code}",
        applicability = "maybe-incorrect"
    )]
    Add {
        #[primary_span]
        sp: Span,
        code: String,
        ident: Ident,
    },
    #[note("consider returning one of these bindings")]
    AddOne {
        #[primary_span]
        spans: MultiSpan,
    },
}

#[derive(Subdiagnostic)]
pub enum ConsiderAddingAwait {
    #[help("consider `await`ing on both `Future`s")]
    BothFuturesHelp,
    #[multipart_suggestion(
        "consider `await`ing on both `Future`s",
        applicability = "maybe-incorrect"
    )]
    BothFuturesSugg {
        #[suggestion_part(code = ".await")]
        first: Span,
        #[suggestion_part(code = ".await")]
        second: Span,
    },
    #[suggestion(
        "consider `await`ing on the `Future`",
        code = ".await",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    FutureSugg {
        #[primary_span]
        span: Span,
    },
    #[note("calling an async function returns a future")]
    FutureSuggNote {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "consider `await`ing on the `Future`",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    FutureSuggMultiple {
        #[suggestion_part(code = ".await")]
        spans: Vec<Span>,
    },
}

#[derive(Diagnostic)]
pub enum PlaceholderRelationLfNotSatisfied {
    #[diag("lifetime bound not satisfied")]
    HasBoth {
        #[primary_span]
        span: Span,
        #[note("the lifetime `{$sub_symbol}` defined here...")]
        sub_span: Span,
        #[note("...must outlive the lifetime `{$sup_symbol}` defined here")]
        sup_span: Span,
        sub_symbol: Symbol,
        sup_symbol: Symbol,
        #[note(
            "this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)"
        )]
        note: (),
    },
    #[diag("lifetime bound not satisfied")]
    HasSub {
        #[primary_span]
        span: Span,
        #[note("the lifetime `{$sub_symbol}` defined here...")]
        sub_span: Span,
        #[note("...must outlive the lifetime defined here")]
        sup_span: Span,
        sub_symbol: Symbol,
        #[note(
            "this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)"
        )]
        note: (),
    },
    #[diag("lifetime bound not satisfied")]
    HasSup {
        #[primary_span]
        span: Span,
        #[note("the lifetime defined here...")]
        sub_span: Span,
        #[note("...must outlive the lifetime `{$sup_symbol}` defined here")]
        sup_span: Span,
        sup_symbol: Symbol,
        #[note(
            "this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)"
        )]
        note: (),
    },
    #[diag("lifetime bound not satisfied")]
    HasNone {
        #[primary_span]
        span: Span,
        #[note("the lifetime defined here...")]
        sub_span: Span,
        #[note("...must outlive the lifetime defined here")]
        sup_span: Span,
        #[note(
            "this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)"
        )]
        note: (),
    },
    #[diag("lifetime bound not satisfied")]
    OnlyPrimarySpan {
        #[primary_span]
        span: Span,
        #[note(
            "this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)"
        )]
        note: (),
    },
}

#[derive(Diagnostic)]
#[diag("hidden type for `{$opaque_ty}` captures lifetime that does not appear in bounds", code = E0700)]
pub struct OpaqueCapturesLifetime<'tcx> {
    #[primary_span]
    pub span: Span,
    #[label("opaque type defined here")]
    pub opaque_ty_span: Span,
    pub opaque_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub enum FunctionPointerSuggestion<'a> {
    #[suggestion(
        "consider using a reference",
        code = "&",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    UseRef {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        "consider removing the reference",
        code = "{fn_name}",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    RemoveRef {
        #[primary_span]
        span: Span,
        #[skip_arg]
        fn_name: String,
    },
    #[suggestion(
        "consider casting to a fn pointer",
        code = "&({fn_name} as {sig})",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    CastRef {
        #[primary_span]
        span: Span,
        #[skip_arg]
        fn_name: String,
        #[skip_arg]
        sig: Binder<'a, FnSig<'a>>,
    },
    #[suggestion(
        "consider casting to a fn pointer",
        code = " as {sig}",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    Cast {
        #[primary_span]
        span: Span,
        #[skip_arg]
        sig: Binder<'a, FnSig<'a>>,
    },
    #[suggestion(
        "consider casting both fn items to fn pointers using `as {$expected_sig}`",
        code = " as {found_sig}",
        style = "hidden",
        applicability = "maybe-incorrect"
    )]
    CastBoth {
        #[primary_span]
        span: Span,
        #[skip_arg]
        found_sig: Binder<'a, FnSig<'a>>,
        expected_sig: Binder<'a, FnSig<'a>>,
    },
    #[suggestion(
        "consider casting both fn items to fn pointers using `as {$expected_sig}`",
        code = "&({fn_name} as {found_sig})",
        style = "hidden",
        applicability = "maybe-incorrect"
    )]
    CastBothRef {
        #[primary_span]
        span: Span,
        #[skip_arg]
        fn_name: String,
        #[skip_arg]
        found_sig: Binder<'a, FnSig<'a>>,
        expected_sig: Binder<'a, FnSig<'a>>,
    },
}

#[derive(Subdiagnostic)]
#[note("fn items are distinct from fn pointers")]
pub struct FnItemsAreDistinct;

#[derive(Subdiagnostic)]
#[note("different fn items have unique types, even if their signatures are the same")]
pub struct FnUniqTypes;

#[derive(Subdiagnostic)]
#[help("consider casting the fn item to a fn pointer: `{$casting}`")]
pub struct FnConsiderCasting {
    pub casting: String,
}

#[derive(Subdiagnostic)]
#[help("consider casting both fn items to fn pointers using `as {$sig}`")]
pub struct FnConsiderCastingBoth<'a> {
    pub sig: Binder<'a, FnSig<'a>>,
}

#[derive(Subdiagnostic)]
pub enum SuggestAccessingField<'a> {
    #[suggestion(
        "you might have meant to use field `{$name}` whose type is `{$ty}`",
        code = "{snippet}.{name}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Safe {
        #[primary_span]
        span: Span,
        snippet: String,
        name: Symbol,
        ty: Ty<'a>,
    },
    #[suggestion(
        "you might have meant to use field `{$name}` whose type is `{$ty}`",
        code = "unsafe {{ {snippet}.{name} }}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Unsafe {
        #[primary_span]
        span: Span,
        snippet: String,
        name: Symbol,
        ty: Ty<'a>,
    },
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "try wrapping the pattern in `{$variant}`",
    applicability = "maybe-incorrect"
)]
pub struct SuggestTuplePatternOne {
    pub variant: String,
    #[suggestion_part(code = "{variant}(")]
    pub span_low: Span,
    #[suggestion_part(code = ")")]
    pub span_high: Span,
}

pub struct SuggestTuplePatternMany {
    pub path: String,
    pub cause_span: Span,
    pub compatible_variants: Vec<String>,
}

impl Subdiagnostic for SuggestTuplePatternMany {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("path", self.path);
        let message =
            diag.eagerly_translate(msg!("try wrapping the pattern in a variant of `{$path}`"));
        diag.multipart_suggestions(
            message,
            self.compatible_variants.into_iter().map(|variant| {
                vec![
                    (self.cause_span.shrink_to_lo(), format!("{variant}(")),
                    (self.cause_span.shrink_to_hi(), ")".to_string()),
                ]
            }),
            rustc_errors::Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Subdiagnostic)]
pub enum TypeErrorAdditionalDiags {
    #[suggestion(
        "if you meant to write a byte literal, prefix with `b`",
        code = "b'{code}'",
        applicability = "machine-applicable"
    )]
    MeantByteLiteral {
        #[primary_span]
        span: Span,
        code: String,
    },
    #[suggestion(
        "if you meant to write a `char` literal, use single quotes",
        code = "'{code}'",
        applicability = "machine-applicable"
    )]
    MeantCharLiteral {
        #[primary_span]
        span: Span,
        code: String,
    },
    #[multipart_suggestion(
        "if you meant to write a string literal, use double quotes",
        applicability = "machine-applicable"
    )]
    MeantStrLiteral {
        #[suggestion_part(code = "\"")]
        start: Span,
        #[suggestion_part(code = "\"")]
        end: Span,
    },
    #[suggestion(
        "consider specifying the actual array length",
        code = "{length}",
        applicability = "maybe-incorrect"
    )]
    ConsiderSpecifyingLength {
        #[primary_span]
        span: Span,
        length: u64,
    },
    #[note("`?` operator cannot convert from `{$found}` to `{$expected}`")]
    TryCannotConvert { found: String, expected: String },
    #[suggestion(
        "use a trailing comma to create a tuple with one element",
        code = ",",
        applicability = "machine-applicable"
    )]
    TupleOnlyComma {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "use a trailing comma to create a tuple with one element",
        applicability = "machine-applicable"
    )]
    TupleAlsoParentheses {
        #[suggestion_part(code = "(")]
        span_low: Span,
        #[suggestion_part(code = ",)")]
        span_high: Span,
    },
    #[suggestion(
        "consider adding `let`",
        style = "verbose",
        applicability = "machine-applicable",
        code = "let "
    )]
    AddLetForLetChains {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
pub enum ObligationCauseFailureCode {
    #[diag("method not compatible with trait", code = E0308)]
    MethodCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("type not compatible with trait", code = E0308)]
    TypeCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("const not compatible with trait", code = E0308)]
    ConstCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("`?` operator has incompatible types", code = E0308)]
    TryCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("`match` arms have incompatible types", code = E0308)]
    MatchCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("`if` and `else` have incompatible types", code = E0308)]
    IfElseDifferent {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("`if` may be missing an `else` clause", code = E0317)]
    NoElse {
        #[primary_span]
        span: Span,
    },
    #[diag("`else` clause of `let...else` does not diverge", code = E0308)]
    NoDiverge {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("`main` function has wrong type", code = E0580)]
    FnMainCorrectType {
        #[primary_span]
        span: Span,
    },
    #[diag(
        "{$lang_item_name ->
            [panic_impl] `#[panic_handler]`
            *[lang_item_name] lang item `{$lang_item_name}`
        } function has wrong type"
    , code = E0308)]
    FnLangCorrectType {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
        lang_item_name: Symbol,
    },
    #[diag("intrinsic has wrong type", code = E0308)]
    IntrinsicCorrectType {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("mismatched `self` parameter type", code = E0308)]
    MethodCorrectType {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("closure/coroutine type that references itself", code = E0644)]
    ClosureSelfref {
        #[primary_span]
        span: Span,
    },
    #[diag("cannot coerce functions which must be inlined to function pointers", code = E0308)]
    CantCoerceForceInline {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("cannot coerce intrinsics to function pointers", code = E0308)]
    CantCoerceIntrinsic {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag("mismatched types", code = E0308)]
    Generic {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
}

#[derive(Subdiagnostic)]
pub enum AddPreciseCapturing {
    #[suggestion(
        "add a `use<...>` bound to explicitly capture `{$new_lifetime}`",
        style = "verbose",
        code = " + use<{concatenated_bounds}>",
        applicability = "machine-applicable"
    )]
    New {
        #[primary_span]
        span: Span,
        new_lifetime: Symbol,
        concatenated_bounds: String,
    },
    #[suggestion(
        "add `{$new_lifetime}` to the `use<...>` bound to explicitly capture it",
        style = "verbose",
        code = "{pre}{new_lifetime}{post}",
        applicability = "machine-applicable"
    )]
    Existing {
        #[primary_span]
        span: Span,
        new_lifetime: Symbol,
        pre: &'static str,
        post: &'static str,
    },
}

pub struct AddPreciseCapturingAndParams {
    pub suggs: Vec<(Span, String)>,
    pub new_lifetime: Symbol,
    pub apit_spans: Vec<Span>,
}

impl Subdiagnostic for AddPreciseCapturingAndParams {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("new_lifetime", self.new_lifetime);
        diag.multipart_suggestion(
            msg!("add a `use<...>` bound to explicitly capture `{$new_lifetime}` after turning all argument-position `impl Trait` into type parameters, noting that this possibly affects the API of this crate"),
            self.suggs,
            Applicability::MaybeIncorrect,
        );
        diag.span_note(
            self.apit_spans,
            msg!("you could use a `use<...>` bound to explicitly capture `{$new_lifetime}`, but argument-position `impl Trait`s are not nameable"),
        );
    }
}

/// Given a set of captured `DefId` for an RPIT (opaque_def_id) and a given
/// function (fn_def_id), try to suggest adding `+ use<...>` to capture just
/// the specified parameters. If one of those parameters is an APIT, then try
/// to suggest turning it into a regular type parameter.
pub fn impl_trait_overcapture_suggestion<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: LocalDefId,
    fn_def_id: LocalDefId,
    captured_args: FxIndexSet<DefId>,
) -> Option<AddPreciseCapturingForOvercapture> {
    let generics = tcx.generics_of(fn_def_id);

    let mut captured_lifetimes = FxIndexSet::default();
    let mut captured_non_lifetimes = FxIndexSet::default();
    let mut synthetics = vec![];

    for arg in captured_args {
        if tcx.def_kind(arg) == DefKind::LifetimeParam {
            captured_lifetimes.insert(tcx.item_name(arg));
        } else {
            let idx = generics.param_def_id_to_index(tcx, arg).expect("expected arg in scope");
            let param = generics.param_at(idx as usize, tcx);
            if param.kind.is_synthetic() {
                synthetics.push((tcx.def_span(arg), param.name));
            } else {
                captured_non_lifetimes.insert(tcx.item_name(arg));
            }
        }
    }

    let mut next_fresh_param = || {
        ["T", "U", "V", "W", "X", "Y", "A", "B", "C"]
            .into_iter()
            .map(Symbol::intern)
            .chain((0..).map(|i| Symbol::intern(&format!("T{i}"))))
            .find(|s| captured_non_lifetimes.insert(*s))
            .unwrap()
    };

    let mut suggs = vec![];
    let mut apit_spans = vec![];

    if !synthetics.is_empty() {
        let mut new_params = String::new();
        for (i, (span, name)) in synthetics.into_iter().enumerate() {
            apit_spans.push(span);

            let fresh_param = next_fresh_param();

            // Suggest renaming.
            suggs.push((span, fresh_param.to_string()));

            // Super jank. Turn `impl Trait` into `T: Trait`.
            //
            // This currently involves stripping the `impl` from the name of
            // the parameter, since APITs are always named after how they are
            // rendered in the AST. This sucks! But to recreate the bound list
            // from the APIT itself would be miserable, so we're stuck with
            // this for now!
            if i > 0 {
                new_params += ", ";
            }
            let name_as_bounds = name.as_str().trim_start_matches("impl").trim_start();
            new_params += fresh_param.as_str();
            new_params += ": ";
            new_params += name_as_bounds;
        }

        let Some(generics) = tcx.hir_get_generics(fn_def_id) else {
            // This shouldn't happen, but don't ICE.
            return None;
        };

        // Add generics or concatenate to the end of the list.
        suggs.push(if let Some(params_span) = generics.span_for_param_suggestion() {
            (params_span, format!(", {new_params}"))
        } else {
            (generics.span, format!("<{new_params}>"))
        });
    }

    let concatenated_bounds = captured_lifetimes
        .into_iter()
        .chain(captured_non_lifetimes)
        .map(|sym| sym.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let opaque_hir_id = tcx.local_def_id_to_hir_id(opaque_def_id);
    // FIXME: This is a bit too conservative, since it ignores parens already written in AST.
    let (lparen, rparen) = match tcx
        .hir_parent_iter(opaque_hir_id)
        .nth(1)
        .expect("expected ty to have a parent always")
        .1
    {
        Node::PathSegment(segment)
            if segment.args().paren_sugar_output().is_some_and(|ty| ty.hir_id == opaque_hir_id) =>
        {
            ("(", ")")
        }
        Node::Ty(ty) => match ty.kind {
            rustc_hir::TyKind::Ptr(_) | rustc_hir::TyKind::Ref(..) => ("(", ")"),
            // FIXME: RPITs are not allowed to be nested in `impl Fn() -> ...`,
            // but we eventually could support that, and that would necessitate
            // making this more sophisticated.
            _ => ("", ""),
        },
        _ => ("", ""),
    };

    let rpit_span = tcx.def_span(opaque_def_id);
    if !lparen.is_empty() {
        suggs.push((rpit_span.shrink_to_lo(), lparen.to_string()));
    }
    suggs.push((rpit_span.shrink_to_hi(), format!(" + use<{concatenated_bounds}>{rparen}")));

    Some(AddPreciseCapturingForOvercapture { suggs, apit_spans })
}

pub struct AddPreciseCapturingForOvercapture {
    pub suggs: Vec<(Span, String)>,
    pub apit_spans: Vec<Span>,
}

impl Subdiagnostic for AddPreciseCapturingForOvercapture {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let applicability = if self.apit_spans.is_empty() {
            Applicability::MachineApplicable
        } else {
            // If there are APIT that are converted to regular parameters,
            // then this may make the API turbofishable in ways that were
            // not intended.
            Applicability::MaybeIncorrect
        };
        diag.multipart_suggestion(
            msg!("use the precise capturing `use<...>` syntax to make the captures explicit"),
            self.suggs,
            applicability,
        );
        if !self.apit_spans.is_empty() {
            diag.span_note(
                self.apit_spans,
                msg!("you could use a `use<...>` bound to explicitly specify captures, but argument-position `impl Trait`s are not nameable"),
            );
        }
    }
}

#[derive(Diagnostic)]
#[diag("expected generic {$kind} parameter, found `{$arg}`", code = E0792)]
pub(crate) struct NonGenericOpaqueTypeParam<'a, 'tcx> {
    pub arg: GenericArg<'tcx>,
    pub kind: &'a str,
    #[primary_span]
    pub span: Span,
    #[label("{STREQ($arg, \"'static\") ->
        [true] cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type
        *[other] this generic parameter must be used with a generic {$kind} parameter
    }")]
    pub param_span: Span,
}
