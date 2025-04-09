use std::path::PathBuf;

use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, DiagMessage, DiagStyledString, Diagnostic,
    EmissionGuarantee, IntoDiagArg, Level, MultiSpan, SubdiagMessageOp, Subdiagnostic,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{Visitor, VisitorExt, walk_ty};
use rustc_hir::{self as hir, AmbigArg, FnRetTy, GenericParamKind, IsAnonInPath, Node};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::print::{PrintTraitRefExt as _, TraitRefPrintOnlyTraitPath};
use rustc_middle::ty::{self, Binder, ClosureKind, FnSig, GenericArg, Region, Ty, TyCtxt};
use rustc_span::{BytePos, Ident, Span, Symbol, kw};

use crate::error_reporting::infer::ObligationCauseAsDiagArg;
use crate::error_reporting::infer::need_type_info::UnderspecifiedArgKind;
use crate::error_reporting::infer::nice_region_error::placeholder_error::Highlighted;
use crate::fluent_generated as fluent;

pub mod note_and_explain;

#[derive(Diagnostic)]
#[diag(trait_selection_unable_to_construct_constant_value)]
pub struct UnableToConstructConstantValue<'a> {
    #[primary_span]
    pub span: Span,
    pub unevaluated: ty::UnevaluatedConst<'a>,
}

#[derive(Diagnostic)]
#[diag(trait_selection_empty_on_clause_in_rustc_on_unimplemented, code = E0232)]
pub struct EmptyOnClauseInOnUnimplemented {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(trait_selection_invalid_on_clause_in_rustc_on_unimplemented, code = E0232)]
pub struct InvalidOnClauseInOnUnimplemented {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(trait_selection_no_value_in_rustc_on_unimplemented, code = E0232)]
#[note]
pub struct NoValueInOnUnimplemented {
    #[primary_span]
    #[label]
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
        let mut diag = Diag::new(dcx, level, fluent::trait_selection_negative_positive_conflict);
        diag.arg("trait_desc", self.trait_desc.print_only_trait_path().to_string());
        diag.arg("self_desc", self.self_ty.map_or_else(|| "none".to_string(), |ty| ty.to_string()));
        diag.span(self.impl_span);
        diag.code(E0751);
        match self.negative_impl_span {
            Ok(span) => {
                diag.span_label(span, fluent::trait_selection_negative_implementation_here);
            }
            Err(cname) => {
                diag.note(fluent::trait_selection_negative_implementation_in_crate);
                diag.arg("negative_impl_cname", cname.to_string());
            }
        }
        match self.positive_impl_span {
            Ok(span) => {
                diag.span_label(span, fluent::trait_selection_positive_implementation_here);
            }
            Err(cname) => {
                diag.note(fluent::trait_selection_positive_implementation_in_crate);
                diag.arg("positive_impl_cname", cname.to_string());
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(trait_selection_inherent_projection_normalization_overflow)]
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        match self {
            AdjustSignatureBorrow::Borrow { to_borrow } => {
                diag.arg("len", to_borrow.len());
                diag.multipart_suggestion_verbose(
                    fluent::trait_selection_adjust_signature_borrow,
                    to_borrow,
                    Applicability::MaybeIncorrect,
                );
            }
            AdjustSignatureBorrow::RemoveBorrow { remove_borrow } => {
                diag.arg("len", remove_borrow.len());
                diag.multipart_suggestion_verbose(
                    fluent::trait_selection_adjust_signature_remove_borrow,
                    remove_borrow,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}

#[derive(Diagnostic)]
#[diag(trait_selection_closure_kind_mismatch, code = E0525)]
pub struct ClosureKindMismatch {
    #[primary_span]
    #[label]
    pub closure_span: Span,
    pub expected: ClosureKind,
    pub found: ClosureKind,
    #[label(trait_selection_closure_kind_requirement)]
    pub cause_span: Span,

    pub trait_prefix: &'static str,

    #[subdiagnostic]
    pub fn_once_label: Option<ClosureFnOnceLabel>,

    #[subdiagnostic]
    pub fn_mut_label: Option<ClosureFnMutLabel>,
}

#[derive(Subdiagnostic)]
#[label(trait_selection_closure_fn_once_label)]
pub struct ClosureFnOnceLabel {
    #[primary_span]
    pub span: Span,
    pub place: String,
}

#[derive(Subdiagnostic)]
#[label(trait_selection_closure_fn_mut_label)]
pub struct ClosureFnMutLabel {
    #[primary_span]
    pub span: Span,
    pub place: String,
}

#[derive(Diagnostic)]
#[diag(trait_selection_async_closure_not_fn)]
pub(crate) struct AsyncClosureNotFn {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag(trait_selection_type_annotations_needed, code = E0282)]
pub struct AnnotationRequired<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
    #[note(trait_selection_full_type_written)]
    pub was_written: bool,
    pub path: PathBuf,
    #[note(trait_selection_type_annotations_needed_error_time)]
    pub time_version: bool,
}

// Copy of `AnnotationRequired` for E0283
#[derive(Diagnostic)]
#[diag(trait_selection_type_annotations_needed, code = E0283)]
pub struct AmbiguousImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
    #[note(trait_selection_full_type_written)]
    pub was_written: bool,
    pub path: PathBuf,
}

// Copy of `AnnotationRequired` for E0284
#[derive(Diagnostic)]
#[diag(trait_selection_type_annotations_needed, code = E0284)]
pub struct AmbiguousReturn<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
    #[note(trait_selection_full_type_written)]
    pub was_written: bool,
    pub path: PathBuf,
}

// Used when a better one isn't available
#[derive(Subdiagnostic)]
#[label(trait_selection_label_bad)]
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
        trait_selection_source_kind_subdiag_let,
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
    #[label(trait_selection_source_kind_subdiag_generic_label)]
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
        trait_selection_source_kind_subdiag_generic_suggestion,
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
        trait_selection_source_kind_fully_qualified,
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
        trait_selection_source_kind_closure_return,
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        let mut label_or_note = |span, msg: DiagMessage| {
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
                label_or_note(span, msg);
            }
            RegionOriginNote::WithName { span, msg, name, continues } => {
                label_or_note(span, msg);
                diag.arg("name", name);
                diag.arg("continues", continues);
            }
            RegionOriginNote::WithRequirement {
                span,
                requirement,
                expected_found: Some((expected, found)),
            } => {
                label_or_note(span, fluent::trait_selection_subtype);
                diag.arg("requirement", requirement);

                diag.note_expected_found("", expected, "", found);
            }
            RegionOriginNote::WithRequirement { span, requirement, expected_found: None } => {
                // FIXME: this really should be handled at some earlier stage. Our
                // handling of region checking when type errors are present is
                // *terrible*.
                label_or_note(span, fluent::trait_selection_subtype_2);
                diag.arg("requirement", requirement);
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        match self {
            LifetimeMismatchLabels::InRet { param_span, ret_span, span, label_var1 } => {
                diag.span_label(param_span, fluent::trait_selection_declared_different);
                diag.span_label(ret_span, fluent::trait_selection_nothing);
                diag.span_label(span, fluent::trait_selection_data_returned);
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
                    diag.span_label(ty_sup, fluent::trait_selection_declared_multiple);
                    diag.span_label(ty_sub, fluent::trait_selection_nothing);
                    diag.span_label(span, fluent::trait_selection_data_lifetime_flow);
                } else {
                    diag.span_label(ty_sup, fluent::trait_selection_types_declared_different);
                    diag.span_label(ty_sub, fluent::trait_selection_nothing);
                    diag.span_label(span, fluent::trait_selection_data_flows);
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
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
                            kind: hir::ItemKind::Trait(_, _, _, generics, ..),
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
                    let make_suggestion = |lifetime: &hir::Lifetime| {
                        if lifetime.is_anon_in_path == IsAnonInPath::Yes
                            && lifetime.ident.span.is_empty()
                        {
                            format!("{}, ", self.suggestion_param_name)
                        } else if lifetime.ident.name == kw::UnderscoreLifetime
                            && lifetime.ident.span.is_empty()
                        {
                            format!("{} ", self.suggestion_param_name)
                        } else {
                            self.suggestion_param_name.clone()
                        }
                    };
                    match ty.kind {
                        hir::TyKind::Path(hir::QPath::Resolved(_, path)) => {
                            for segment in path.segments {
                                if let Some(args) = segment.args {
                                    if args.args.iter().all(|arg| {
                                        matches!(
                                            arg,
                                            hir::GenericArg::Lifetime(lifetime)
                                                if lifetime.is_anon_in_path == IsAnonInPath::Yes
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
                                                self.suggestions.push((
                                                    lifetime.ident.span,
                                                    make_suggestion(lifetime),
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        hir::TyKind::Ref(lifetime, ..) if lifetime.is_anonymous() => {
                            self.suggestions.push((lifetime.ident.span, make_suggestion(lifetime)));
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
                // get the cake of "correctly" contrained in two steps.
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
            diag.multipart_suggestion_verbose(
                fluent::trait_selection_lifetime_param_suggestion,
                visitor.suggestions,
                Applicability::MaybeIncorrect,
            );
            diag.arg("is_impl", is_impl);
            diag.arg("is_reuse", !introduce_new);

            true
        };
        if mk_suggestion() && self.add_note {
            diag.note(fluent::trait_selection_lifetime_param_suggestion_elided);
        }
    }
}

#[derive(Diagnostic)]
#[diag(trait_selection_lifetime_mismatch, code = E0623)]
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        mut self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        self.unmet_requirements
            .push_span_label(self.binding_span, fluent::trait_selection_msl_introduces_static);
        diag.span_note(self.unmet_requirements, fluent::trait_selection_msl_unmet_req);
    }
}

// FIXME(#100717): replace with a `Option<Span>` when subdiagnostic supports that
#[derive(Subdiagnostic)]
pub enum DoesNotOutliveStaticFromImpl {
    #[note(trait_selection_does_not_outlive_static_from_impl)]
    Spanned {
        #[primary_span]
        span: Span,
    },
    #[note(trait_selection_does_not_outlive_static_from_impl)]
    Unspanned,
}

#[derive(Subdiagnostic)]
pub enum ImplicitStaticLifetimeSubdiag {
    #[note(trait_selection_implicit_static_lifetime_note)]
    Note {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        trait_selection_implicit_static_lifetime_suggestion,
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
#[diag(trait_selection_mismatched_static_lifetime)]
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
    #[diag(trait_selection_explicit_lifetime_required_with_ident, code = E0621)]
    WithIdent {
        #[primary_span]
        #[label]
        span: Span,
        simple_ident: Ident,
        named: String,
        #[suggestion(
            trait_selection_explicit_lifetime_required_sugg_with_ident,
            code = "{new_ty}",
            applicability = "unspecified"
        )]
        new_ty_span: Span,
        #[skip_arg]
        new_ty: Ty<'a>,
    },
    #[diag(trait_selection_explicit_lifetime_required_with_param_type, code = E0621)]
    WithParamType {
        #[primary_span]
        #[label]
        span: Span,
        named: String,
        #[suggestion(
            trait_selection_explicit_lifetime_required_sugg_with_param_type,
            code = "{new_ty}",
            applicability = "unspecified"
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
    #[note(trait_selection_actual_impl_expl_expected_signature_two)]
    ExpectedSignatureTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_signature_any)]
    ExpectedSignatureAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_signature_some)]
    ExpectedSignatureSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_signature_nothing)]
    ExpectedSignatureNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(trait_selection_actual_impl_expl_expected_passive_two)]
    ExpectedPassiveTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_passive_any)]
    ExpectedPassiveAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_passive_some)]
    ExpectedPassiveSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_passive_nothing)]
    ExpectedPassiveNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(trait_selection_actual_impl_expl_expected_other_two)]
    ExpectedOtherTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_other_any)]
    ExpectedOtherAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_other_some)]
    ExpectedOtherSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(trait_selection_actual_impl_expl_expected_other_nothing)]
    ExpectedOtherNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(trait_selection_actual_impl_expl_but_actually_implements_trait)]
    ButActuallyImplementsTrait {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
    },
    #[note(trait_selection_actual_impl_expl_but_actually_implemented_for_ty)]
    ButActuallyImplementedForTy {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
        ty: String,
    },
    #[note(trait_selection_actual_impl_expl_but_actually_ty_implements)]
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
#[diag(trait_selection_trait_placeholder_mismatch)]
pub struct TraitPlaceholderMismatch<'tcx> {
    #[primary_span]
    pub span: Span,
    #[label(trait_selection_label_satisfy)]
    pub satisfy_span: Option<Span>,
    #[label(trait_selection_label_where)]
    pub where_span: Option<Span>,
    #[label(trait_selection_label_dup)]
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        f: &F,
    ) {
        let mut type_param_span: MultiSpan = self.spans.clone().into();
        for &span in &self.spans {
            // Seems like we can't call f() here as Into<DiagMessage> is required
            type_param_span.push_span_label(span, fluent::trait_selection_tid_consider_borrowing);
        }
        let msg = f(diag, fluent::trait_selection_tid_param_help.into());
        diag.span_help(type_param_span, msg);
    }
}

#[derive(Subdiagnostic)]
#[help(trait_selection_tid_rel_help)]
pub struct RelationshipHelp;

#[derive(Diagnostic)]
#[diag(trait_selection_trait_impl_diff)]
pub struct TraitImplDiff {
    #[primary_span]
    #[label(trait_selection_found)]
    pub sp: Span,
    #[label(trait_selection_expected)]
    pub trait_sp: Span,
    #[note(trait_selection_expected_found)]
    pub note: (),
    #[subdiagnostic]
    pub param_help: ConsiderBorrowingParamHelp,
    #[subdiagnostic]
    // Seems like subdiagnostics are always pushed to the end, so this one
    // also has to be a subdiagnostic to maintain order.
    pub rel_help: Option<RelationshipHelp>,
    pub expected: String,
    pub found: String,
}

pub struct DynTraitConstraintSuggestion {
    pub span: Span,
    pub ident: Ident,
}

impl Subdiagnostic for DynTraitConstraintSuggestion {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        f: &F,
    ) {
        let mut multi_span: MultiSpan = vec![self.span].into();
        multi_span.push_span_label(self.span, fluent::trait_selection_dtcs_has_lifetime_req_label);
        multi_span
            .push_span_label(self.ident.span, fluent::trait_selection_dtcs_introduces_requirement);
        let msg = f(diag, fluent::trait_selection_dtcs_has_req_note.into());
        diag.span_note(multi_span, msg);
        let msg = f(diag, fluent::trait_selection_dtcs_suggestion.into());
        diag.span_suggestion_verbose(
            self.span.shrink_to_hi(),
            msg,
            " + '_",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Diagnostic)]
#[diag(trait_selection_but_calling_introduces, code = E0772)]
pub struct ButCallingIntroduces {
    #[label(trait_selection_label1)]
    pub param_ty_span: Span,
    #[primary_span]
    #[label(trait_selection_label2)]
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        mut self,
        diag: &mut Diag<'_, G>,
        f: &F,
    ) {
        for sp in self.spans {
            self.span.push_span_label(sp, fluent::trait_selection_ril_introduced_here);
        }

        if self.add_label {
            self.span.push_span_label(self.fn_decl_span, fluent::trait_selection_ril_introduced_by);
        }
        self.span.push_span_label(self.cause_span, fluent::trait_selection_ril_because_of);
        let msg = f(diag, fluent::trait_selection_ril_static_introduced_by.into());
        diag.span_note(self.span, msg);
    }
}

#[derive(Diagnostic)]
#[diag(trait_selection_but_needs_to_satisfy, code = E0759)]
pub struct ButNeedsToSatisfy {
    #[primary_span]
    pub sp: Span,
    #[label(trait_selection_influencer)]
    pub influencer_point: Span,
    #[label(trait_selection_used_here)]
    pub spans: Vec<Span>,
    #[label(trait_selection_require)]
    pub require_span_as_label: Option<Span>,
    #[note(trait_selection_require)]
    pub require_span_as_note: Option<Span>,
    #[note(trait_selection_introduced_by_bound)]
    pub bound: Option<Span>,

    pub has_param_name: bool,
    pub param_name: String,
    pub spans_empty: bool,
    pub has_lifetime: bool,
    pub lifetime: String,
}

#[derive(Diagnostic)]
#[diag(trait_selection_outlives_content, code = E0312)]
pub struct OutlivesContent<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(trait_selection_outlives_bound, code = E0476)]
pub struct OutlivesBound<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(trait_selection_fulfill_req_lifetime, code = E0477)]
pub struct FulfillReqLifetime<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
    #[subdiagnostic]
    pub note: Option<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(trait_selection_lf_bound_not_satisfied, code = E0478)]
pub struct LfBoundNotSatisfied<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(trait_selection_ref_longer_than_data, code = E0491)]
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
        trait_selection_where_remove,
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    Remove {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        trait_selection_where_copy_predicates,
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
        trait_selection_srs_remove_and_box,
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
        trait_selection_srs_remove,
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    Remove {
        #[primary_span]
        sp: Span,
    },
    #[suggestion(
        trait_selection_srs_add,
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
    #[note(trait_selection_srs_add_one)]
    AddOne {
        #[primary_span]
        spans: MultiSpan,
    },
}

#[derive(Subdiagnostic)]
pub enum ConsiderAddingAwait {
    #[help(trait_selection_await_both_futures)]
    BothFuturesHelp,
    #[multipart_suggestion(trait_selection_await_both_futures, applicability = "maybe-incorrect")]
    BothFuturesSugg {
        #[suggestion_part(code = ".await")]
        first: Span,
        #[suggestion_part(code = ".await")]
        second: Span,
    },
    #[suggestion(
        trait_selection_await_future,
        code = ".await",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    FutureSugg {
        #[primary_span]
        span: Span,
    },
    #[note(trait_selection_await_note)]
    FutureSuggNote {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        trait_selection_await_future,
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
    #[diag(trait_selection_lf_bound_not_satisfied)]
    HasBoth {
        #[primary_span]
        span: Span,
        #[note(trait_selection_prlf_defined_with_sub)]
        sub_span: Span,
        #[note(trait_selection_prlf_must_outlive_with_sup)]
        sup_span: Span,
        sub_symbol: Symbol,
        sup_symbol: Symbol,
        #[note(trait_selection_prlf_known_limitation)]
        note: (),
    },
    #[diag(trait_selection_lf_bound_not_satisfied)]
    HasSub {
        #[primary_span]
        span: Span,
        #[note(trait_selection_prlf_defined_with_sub)]
        sub_span: Span,
        #[note(trait_selection_prlf_must_outlive_without_sup)]
        sup_span: Span,
        sub_symbol: Symbol,
        #[note(trait_selection_prlf_known_limitation)]
        note: (),
    },
    #[diag(trait_selection_lf_bound_not_satisfied)]
    HasSup {
        #[primary_span]
        span: Span,
        #[note(trait_selection_prlf_defined_without_sub)]
        sub_span: Span,
        #[note(trait_selection_prlf_must_outlive_with_sup)]
        sup_span: Span,
        sup_symbol: Symbol,
        #[note(trait_selection_prlf_known_limitation)]
        note: (),
    },
    #[diag(trait_selection_lf_bound_not_satisfied)]
    HasNone {
        #[primary_span]
        span: Span,
        #[note(trait_selection_prlf_defined_without_sub)]
        sub_span: Span,
        #[note(trait_selection_prlf_must_outlive_without_sup)]
        sup_span: Span,
        #[note(trait_selection_prlf_known_limitation)]
        note: (),
    },
    #[diag(trait_selection_lf_bound_not_satisfied)]
    OnlyPrimarySpan {
        #[primary_span]
        span: Span,
        #[note(trait_selection_prlf_known_limitation)]
        note: (),
    },
}

#[derive(Diagnostic)]
#[diag(trait_selection_opaque_captures_lifetime, code = E0700)]
pub struct OpaqueCapturesLifetime<'tcx> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub opaque_ty_span: Span,
    pub opaque_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub enum FunctionPointerSuggestion<'a> {
    #[suggestion(
        trait_selection_fps_use_ref,
        code = "&",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    UseRef {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        trait_selection_fps_remove_ref,
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
        trait_selection_fps_cast,
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
        trait_selection_fps_cast,
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
        trait_selection_fps_cast_both,
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
        trait_selection_fps_cast_both,
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
#[note(trait_selection_fps_items_are_distinct)]
pub struct FnItemsAreDistinct;

#[derive(Subdiagnostic)]
#[note(trait_selection_fn_uniq_types)]
pub struct FnUniqTypes;

#[derive(Subdiagnostic)]
#[help(trait_selection_fn_consider_casting)]
pub struct FnConsiderCasting {
    pub casting: String,
}

#[derive(Subdiagnostic)]
#[help(trait_selection_fn_consider_casting_both)]
pub struct FnConsiderCastingBoth<'a> {
    pub sig: Binder<'a, FnSig<'a>>,
}

#[derive(Subdiagnostic)]
pub enum SuggestAccessingField<'a> {
    #[suggestion(
        trait_selection_suggest_accessing_field,
        code = "{snippet}.{name}",
        applicability = "maybe-incorrect"
    )]
    Safe {
        #[primary_span]
        span: Span,
        snippet: String,
        name: Symbol,
        ty: Ty<'a>,
    },
    #[suggestion(
        trait_selection_suggest_accessing_field,
        code = "unsafe {{ {snippet}.{name} }}",
        applicability = "maybe-incorrect"
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
#[multipart_suggestion(trait_selection_stp_wrap_one, applicability = "maybe-incorrect")]
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        f: &F,
    ) {
        diag.arg("path", self.path);
        let message = f(diag, crate::fluent_generated::trait_selection_stp_wrap_many.into());
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
        trait_selection_meant_byte_literal,
        code = "b'{code}'",
        applicability = "machine-applicable"
    )]
    MeantByteLiteral {
        #[primary_span]
        span: Span,
        code: String,
    },
    #[suggestion(
        trait_selection_meant_char_literal,
        code = "'{code}'",
        applicability = "machine-applicable"
    )]
    MeantCharLiteral {
        #[primary_span]
        span: Span,
        code: String,
    },
    #[multipart_suggestion(trait_selection_meant_str_literal, applicability = "machine-applicable")]
    MeantStrLiteral {
        #[suggestion_part(code = "\"")]
        start: Span,
        #[suggestion_part(code = "\"")]
        end: Span,
    },
    #[suggestion(
        trait_selection_consider_specifying_length,
        code = "{length}",
        applicability = "maybe-incorrect"
    )]
    ConsiderSpecifyingLength {
        #[primary_span]
        span: Span,
        length: u64,
    },
    #[note(trait_selection_try_cannot_convert)]
    TryCannotConvert { found: String, expected: String },
    #[suggestion(
        trait_selection_tuple_trailing_comma,
        code = ",",
        applicability = "machine-applicable"
    )]
    TupleOnlyComma {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        trait_selection_tuple_trailing_comma,
        applicability = "machine-applicable"
    )]
    TupleAlsoParentheses {
        #[suggestion_part(code = "(")]
        span_low: Span,
        #[suggestion_part(code = ",)")]
        span_high: Span,
    },
    #[suggestion(
        trait_selection_suggest_add_let_for_letchains,
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
    #[diag(trait_selection_oc_method_compat, code = E0308)]
    MethodCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_type_compat, code = E0308)]
    TypeCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_const_compat, code = E0308)]
    ConstCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_try_compat, code = E0308)]
    TryCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_match_compat, code = E0308)]
    MatchCompat {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_if_else_different, code = E0308)]
    IfElseDifferent {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_no_else, code = E0317)]
    NoElse {
        #[primary_span]
        span: Span,
    },
    #[diag(trait_selection_oc_no_diverge, code = E0308)]
    NoDiverge {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_fn_main_correct_type, code = E0580)]
    FnMainCorrectType {
        #[primary_span]
        span: Span,
    },
    #[diag(trait_selection_oc_fn_lang_correct_type, code = E0308)]
    FnLangCorrectType {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
        lang_item_name: Symbol,
    },
    #[diag(trait_selection_oc_intrinsic_correct_type, code = E0308)]
    IntrinsicCorrectType {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_method_correct_type, code = E0308)]
    MethodCorrectType {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_closure_selfref, code = E0644)]
    ClosureSelfref {
        #[primary_span]
        span: Span,
    },
    #[diag(trait_selection_oc_cant_coerce_force_inline, code = E0308)]
    CantCoerceForceInline {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_cant_coerce_intrinsic, code = E0308)]
    CantCoerceIntrinsic {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiags: Vec<TypeErrorAdditionalDiags>,
    },
    #[diag(trait_selection_oc_generic, code = E0308)]
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
        trait_selection_precise_capturing_new,
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
        trait_selection_precise_capturing_existing,
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        diag.arg("new_lifetime", self.new_lifetime);
        diag.multipart_suggestion_verbose(
            fluent::trait_selection_precise_capturing_new_but_apit,
            self.suggs,
            Applicability::MaybeIncorrect,
        );
        diag.span_note(
            self.apit_spans,
            fluent::trait_selection_warn_removing_apit_params_for_undercapture,
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
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _f: &F,
    ) {
        let applicability = if self.apit_spans.is_empty() {
            Applicability::MachineApplicable
        } else {
            // If there are APIT that are converted to regular parameters,
            // then this may make the API turbofishable in ways that were
            // not intended.
            Applicability::MaybeIncorrect
        };
        diag.multipart_suggestion_verbose(
            fluent::trait_selection_precise_capturing_overcaptures,
            self.suggs,
            applicability,
        );
        if !self.apit_spans.is_empty() {
            diag.span_note(
                self.apit_spans,
                fluent::trait_selection_warn_removing_apit_params_for_overcapture,
            );
        }
    }
}

#[derive(Diagnostic)]
#[diag(trait_selection_opaque_type_non_generic_param, code = E0792)]
pub(crate) struct NonGenericOpaqueTypeParam<'a, 'tcx> {
    pub ty: GenericArg<'tcx>,
    pub kind: &'a str,
    #[primary_span]
    pub span: Span,
    #[label]
    pub param_span: Span,
}
