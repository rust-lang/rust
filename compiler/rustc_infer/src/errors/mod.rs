use hir::GenericParamKind;
use rustc_errors::{
    fluent, AddToDiagnostic, Applicability, Diagnostic, DiagnosticMessage, DiagnosticStyledString,
    IntoDiagnosticArg, MultiSpan, SubdiagnosticMessage,
};
use rustc_hir as hir;
use rustc_hir::FnRetTy;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::print::TraitRefPrintOnlyTraitPath;
use rustc_middle::ty::{Binder, FnSig, Region, Ty, TyCtxt};
use rustc_span::symbol::kw;
use rustc_span::Symbol;
use rustc_span::{symbol::Ident, BytePos, Span};

use crate::infer::error_reporting::nice_region_error::placeholder_error::Highlighted;
use crate::infer::error_reporting::{
    need_type_info::{GeneratorKindAsDiagArg, UnderspecifiedArgKind},
    ObligationCauseAsDiagArg,
};

pub mod note_and_explain;

#[derive(Diagnostic)]
#[diag(infer_opaque_hidden_type)]
pub struct OpaqueHiddenTypeDiag {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(opaque_type)]
    pub opaque_type: Span,
    #[note(hidden_type)]
    pub hidden_type: Span,
}

#[derive(Diagnostic)]
#[diag(infer_type_annotations_needed, code = "E0282")]
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
}

// Copy of `AnnotationRequired` for E0283
#[derive(Diagnostic)]
#[diag(infer_type_annotations_needed, code = "E0283")]
pub struct AmbigousImpl<'a> {
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
}

// Copy of `AnnotationRequired` for E0284
#[derive(Diagnostic)]
#[diag(infer_type_annotations_needed, code = "E0284")]
pub struct AmbigousReturn<'a> {
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
}

#[derive(Diagnostic)]
#[diag(infer_need_type_info_in_generator, code = "E0698")]
pub struct NeedTypeInfoInGenerator<'a> {
    #[primary_span]
    pub span: Span,
    pub generator_kind: GeneratorKindAsDiagArg,
    #[subdiagnostic]
    pub bad_label: InferenceBadError<'a>,
}

// Used when a better one isn't available
#[derive(Subdiagnostic)]
#[label(infer_label_bad)]
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
        infer_source_kind_subdiag_let,
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
    #[label(infer_source_kind_subdiag_generic_label)]
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
        infer_source_kind_subdiag_generic_suggestion,
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
        infer_source_kind_fully_qualified,
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
        infer_source_kind_closure_return,
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

#[derive(Subdiagnostic)]
#[suggestion(
    infer_suggest_add_let_for_letchains,
    style = "verbose",
    applicability = "machine-applicable",
    code = "let "
)]
pub(crate) struct SuggAddLetForLetChains {
    #[primary_span]
    pub span: Span,
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
        let (arrow, post) = match data {
            FnRetTy::DefaultReturn(_) => ("-> ", " "),
            _ => ("", ""),
        };
        let (start_span, start_span_code, end_span) = match should_wrap_expr {
            Some(end_span) => {
                (data.span(), format!("{}{}{}{{ ", arrow, ty_info, post), Some(end_span))
            }
            None => (data.span(), format!("{}{}{}", arrow, ty_info, post), None),
        };
        Self::ClosureReturn { start_span, start_span_code, end_span }
    }
}

pub enum RegionOriginNote<'a> {
    Plain {
        span: Span,
        msg: DiagnosticMessage,
    },
    WithName {
        span: Span,
        msg: DiagnosticMessage,
        name: &'a str,
        continues: bool,
    },
    WithRequirement {
        span: Span,
        requirement: ObligationCauseAsDiagArg<'a>,
        expected_found: Option<(DiagnosticStyledString, DiagnosticStyledString)>,
    },
}

impl AddToDiagnostic for RegionOriginNote<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        let mut label_or_note = |span, msg: DiagnosticMessage| {
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
                diag.set_arg("name", name);
                diag.set_arg("continues", continues);
            }
            RegionOriginNote::WithRequirement {
                span,
                requirement,
                expected_found: Some((expected, found)),
            } => {
                label_or_note(span, fluent::infer_subtype);
                diag.set_arg("requirement", requirement);

                diag.note_expected_found(&"", expected, &"", found);
            }
            RegionOriginNote::WithRequirement { span, requirement, expected_found: None } => {
                // FIXME: this really should be handled at some earlier stage. Our
                // handling of region checking when type errors are present is
                // *terrible*.
                label_or_note(span, fluent::infer_subtype_2);
                diag.set_arg("requirement", requirement);
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

impl AddToDiagnostic for LifetimeMismatchLabels {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        match self {
            LifetimeMismatchLabels::InRet { param_span, ret_span, span, label_var1 } => {
                diag.span_label(param_span, fluent::infer_declared_different);
                diag.span_label(ret_span, fluent::infer_nothing);
                diag.span_label(span, fluent::infer_data_returned);
                diag.set_arg("label_var1_exists", label_var1.is_some());
                diag.set_arg("label_var1", label_var1.map(|x| x.to_string()).unwrap_or_default());
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
                    diag.span_label(ty_sup, fluent::infer_declared_multiple);
                    diag.span_label(ty_sub, fluent::infer_nothing);
                    diag.span_label(span, fluent::infer_data_lifetime_flow);
                } else {
                    diag.span_label(ty_sup, fluent::infer_types_declared_different);
                    diag.span_label(ty_sub, fluent::infer_nothing);
                    diag.span_label(span, fluent::infer_data_flows);
                    diag.set_arg("label_var1_exists", label_var1.is_some());
                    diag.set_arg(
                        "label_var1",
                        label_var1.map(|x| x.to_string()).unwrap_or_default(),
                    );
                    diag.set_arg("label_var2_exists", label_var2.is_some());
                    diag.set_arg(
                        "label_var2",
                        label_var2.map(|x| x.to_string()).unwrap_or_default(),
                    );
                }
            }
        }
    }
}

pub struct AddLifetimeParamsSuggestion<'a> {
    pub tcx: TyCtxt<'a>,
    pub sub: Region<'a>,
    pub ty_sup: &'a hir::Ty<'a>,
    pub ty_sub: &'a hir::Ty<'a>,
    pub add_note: bool,
}

impl AddToDiagnostic for AddLifetimeParamsSuggestion<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        let mut mk_suggestion = || {
            let (
                hir::Ty { kind: hir::TyKind::Ref(lifetime_sub, _), .. },
                hir::Ty { kind: hir::TyKind::Ref(lifetime_sup, _), .. },
            ) = (self.ty_sub, self.ty_sup) else {
                return false;
            };

            if !lifetime_sub.is_anonymous() || !lifetime_sup.is_anonymous() {
                return false;
            };

            let Some(anon_reg) = self.tcx.is_suitable_region(self.sub) else {
                return false;
            };

            let hir_id = self.tcx.hir().local_def_id_to_hir_id(anon_reg.def_id);

            let node = self.tcx.hir().get(hir_id);
            let is_impl = matches!(&node, hir::Node::ImplItem(_));
            let generics = match node {
                hir::Node::Item(&hir::Item {
                    kind: hir::ItemKind::Fn(_, ref generics, ..),
                    ..
                })
                | hir::Node::TraitItem(&hir::TraitItem { ref generics, .. })
                | hir::Node::ImplItem(&hir::ImplItem { ref generics, .. }) => generics,
                _ => return false,
            };

            let suggestion_param_name = generics
                .params
                .iter()
                .filter(|p| matches!(p.kind, GenericParamKind::Lifetime { .. }))
                .map(|p| p.name.ident().name)
                .find(|i| *i != kw::UnderscoreLifetime);
            let introduce_new = suggestion_param_name.is_none();
            let suggestion_param_name =
                suggestion_param_name.map(|n| n.to_string()).unwrap_or_else(|| "'a".to_owned());

            debug!(?lifetime_sup.ident.span);
            debug!(?lifetime_sub.ident.span);
            let make_suggestion = |ident: Ident| {
                let sugg = if ident.name == kw::Empty {
                    format!("{}, ", suggestion_param_name)
                } else if ident.name == kw::UnderscoreLifetime && ident.span.is_empty() {
                    format!("{} ", suggestion_param_name)
                } else {
                    suggestion_param_name.clone()
                };
                (ident.span, sugg)
            };
            let mut suggestions =
                vec![make_suggestion(lifetime_sub.ident), make_suggestion(lifetime_sup.ident)];

            if introduce_new {
                let new_param_suggestion = if let Some(first) =
                    generics.params.iter().find(|p| !p.name.ident().span.is_empty())
                {
                    (first.span.shrink_to_lo(), format!("{}, ", suggestion_param_name))
                } else {
                    (generics.span, format!("<{}>", suggestion_param_name))
                };

                suggestions.push(new_param_suggestion);
            }

            diag.multipart_suggestion(
                fluent::infer_lifetime_param_suggestion,
                suggestions,
                Applicability::MaybeIncorrect,
            );
            diag.set_arg("is_impl", is_impl);
            true
        };
        if mk_suggestion() && self.add_note {
            diag.note(fluent::infer_lifetime_param_suggestion_elided);
        }
    }
}

#[derive(Diagnostic)]
#[diag(infer_lifetime_mismatch, code = "E0623")]
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

impl AddToDiagnostic for IntroducesStaticBecauseUnmetLifetimeReq {
    fn add_to_diagnostic_with<F>(mut self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        self.unmet_requirements
            .push_span_label(self.binding_span, fluent::infer_msl_introduces_static);
        diag.span_note(self.unmet_requirements, fluent::infer_msl_unmet_req);
    }
}

// FIXME(#100717): replace with a `Option<Span>` when subdiagnostic supports that
#[derive(Subdiagnostic)]
pub enum DoesNotOutliveStaticFromImpl {
    #[note(infer_does_not_outlive_static_from_impl)]
    Spanned {
        #[primary_span]
        span: Span,
    },
    #[note(infer_does_not_outlive_static_from_impl)]
    Unspanned,
}

#[derive(Subdiagnostic)]
pub enum ImplicitStaticLifetimeSubdiag {
    #[note(infer_implicit_static_lifetime_note)]
    Note {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        infer_implicit_static_lifetime_suggestion,
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
#[diag(infer_mismatched_static_lifetime)]
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
    #[diag(infer_explicit_lifetime_required_with_ident, code = "E0621")]
    WithIdent {
        #[primary_span]
        #[label]
        span: Span,
        simple_ident: Ident,
        named: String,
        #[suggestion(
            infer_explicit_lifetime_required_sugg_with_ident,
            code = "{new_ty}",
            applicability = "unspecified"
        )]
        new_ty_span: Span,
        #[skip_arg]
        new_ty: Ty<'a>,
    },
    #[diag(infer_explicit_lifetime_required_with_param_type, code = "E0621")]
    WithParamType {
        #[primary_span]
        #[label]
        span: Span,
        named: String,
        #[suggestion(
            infer_explicit_lifetime_required_sugg_with_param_type,
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

impl IntoDiagnosticArg for TyOrSig<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        match self {
            TyOrSig::Ty(ty) => ty.into_diagnostic_arg(),
            TyOrSig::ClosureSig(sig) => sig.into_diagnostic_arg(),
        }
    }
}

#[derive(Subdiagnostic)]
pub enum ActualImplExplNotes<'tcx> {
    #[note(infer_actual_impl_expl_expected_signature_two)]
    ExpectedSignatureTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(infer_actual_impl_expl_expected_signature_any)]
    ExpectedSignatureAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(infer_actual_impl_expl_expected_signature_some)]
    ExpectedSignatureSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(infer_actual_impl_expl_expected_signature_nothing)]
    ExpectedSignatureNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(infer_actual_impl_expl_expected_passive_two)]
    ExpectedPassiveTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(infer_actual_impl_expl_expected_passive_any)]
    ExpectedPassiveAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(infer_actual_impl_expl_expected_passive_some)]
    ExpectedPassiveSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(infer_actual_impl_expl_expected_passive_nothing)]
    ExpectedPassiveNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(infer_actual_impl_expl_expected_other_two)]
    ExpectedOtherTwo {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
        lifetime_2: usize,
    },
    #[note(infer_actual_impl_expl_expected_other_any)]
    ExpectedOtherAny {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(infer_actual_impl_expl_expected_other_some)]
    ExpectedOtherSome {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        lifetime_1: usize,
    },
    #[note(infer_actual_impl_expl_expected_other_nothing)]
    ExpectedOtherNothing {
        leading_ellipsis: bool,
        ty_or_sig: TyOrSig<'tcx>,
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
    },
    #[note(infer_actual_impl_expl_but_actually_implements_trait)]
    ButActuallyImplementsTrait {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
    },
    #[note(infer_actual_impl_expl_but_actually_implemented_for_ty)]
    ButActuallyImplementedForTy {
        trait_path: Highlighted<'tcx, TraitRefPrintOnlyTraitPath<'tcx>>,
        has_lifetime: bool,
        lifetime: usize,
        ty: String,
    },
    #[note(infer_actual_impl_expl_but_actually_ty_implements)]
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
#[diag(infer_trait_placeholder_mismatch)]
pub struct TraitPlaceholderMismatch<'tcx> {
    #[primary_span]
    pub span: Span,
    #[label(label_satisfy)]
    pub satisfy_span: Option<Span>,
    #[label(label_where)]
    pub where_span: Option<Span>,
    #[label(label_dup)]
    pub dup_span: Option<Span>,
    pub def_id: String,
    pub trait_def_id: String,

    #[subdiagnostic]
    pub actual_impl_expl_notes: Vec<ActualImplExplNotes<'tcx>>,
}

pub struct ConsiderBorrowingParamHelp {
    pub spans: Vec<Span>,
}

impl AddToDiagnostic for ConsiderBorrowingParamHelp {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, f: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        let mut type_param_span: MultiSpan = self.spans.clone().into();
        for &span in &self.spans {
            // Seems like we can't call f() here as Into<DiagnosticMessage> is required
            type_param_span.push_span_label(span, fluent::infer_tid_consider_borrowing);
        }
        let msg = f(diag, fluent::infer_tid_param_help.into());
        diag.span_help(type_param_span, msg);
    }
}

#[derive(Subdiagnostic)]
#[help(infer_tid_rel_help)]
pub struct RelationshipHelp;

#[derive(Diagnostic)]
#[diag(infer_trait_impl_diff)]
pub struct TraitImplDiff {
    #[primary_span]
    #[label(found)]
    pub sp: Span,
    #[label(expected)]
    pub trait_sp: Span,
    #[note(expected_found)]
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

impl AddToDiagnostic for DynTraitConstraintSuggestion {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, f: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        let mut multi_span: MultiSpan = vec![self.span].into();
        multi_span.push_span_label(self.span, fluent::infer_dtcs_has_lifetime_req_label);
        multi_span.push_span_label(self.ident.span, fluent::infer_dtcs_introduces_requirement);
        let msg = f(diag, fluent::infer_dtcs_has_req_note.into());
        diag.span_note(multi_span, msg);
        let msg = f(diag, fluent::infer_dtcs_suggestion.into());
        diag.span_suggestion_verbose(
            self.span.shrink_to_hi(),
            msg,
            " + '_",
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(Diagnostic)]
#[diag(infer_but_calling_introduces, code = "E0772")]
pub struct ButCallingIntroduces {
    #[label(label1)]
    pub param_ty_span: Span,
    #[primary_span]
    #[label(label2)]
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

impl AddToDiagnostic for ReqIntroducedLocations {
    fn add_to_diagnostic_with<F>(mut self, diag: &mut Diagnostic, f: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        for sp in self.spans {
            self.span.push_span_label(sp, fluent::infer_ril_introduced_here);
        }

        if self.add_label {
            self.span.push_span_label(self.fn_decl_span, fluent::infer_ril_introduced_by);
        }
        self.span.push_span_label(self.cause_span, fluent::infer_ril_because_of);
        let msg = f(diag, fluent::infer_ril_static_introduced_by.into());
        diag.span_note(self.span, msg);
    }
}

pub struct MoreTargeted {
    pub ident: Symbol,
}

impl AddToDiagnostic for MoreTargeted {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _f: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        diag.code(rustc_errors::error_code!(E0772));
        diag.set_primary_message(fluent::infer_more_targeted);
        diag.set_arg("ident", self.ident);
    }
}

#[derive(Diagnostic)]
#[diag(infer_but_needs_to_satisfy, code = "E0759")]
pub struct ButNeedsToSatisfy {
    #[primary_span]
    pub sp: Span,
    #[label(influencer)]
    pub influencer_point: Span,
    #[label(used_here)]
    pub spans: Vec<Span>,
    #[label(require)]
    pub require_span_as_label: Option<Span>,
    #[note(require)]
    pub require_span_as_note: Option<Span>,
    #[note(introduced_by_bound)]
    pub bound: Option<Span>,

    #[subdiagnostic]
    pub req_introduces_loc: Option<ReqIntroducedLocations>,

    pub has_param_name: bool,
    pub param_name: String,
    pub spans_empty: bool,
    pub has_lifetime: bool,
    pub lifetime: String,
}

#[derive(Diagnostic)]
#[diag(infer_outlives_content, code = "E0312")]
pub struct OutlivesContent<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(infer_outlives_bound, code = "E0476")]
pub struct OutlivesBound<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(infer_fullfill_req_lifetime, code = "E0477")]
pub struct FullfillReqLifetime<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
    #[subdiagnostic]
    pub note: Option<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(infer_lf_bound_not_satisfied, code = "E0478")]
pub struct LfBoundNotSatisfied<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub notes: Vec<note_and_explain::RegionExplanation<'a>>,
}

#[derive(Diagnostic)]
#[diag(infer_ref_longer_than_data, code = "E0491")]
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
        infer_where_remove,
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    Remove {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        infer_where_copy_predicates,
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
    #[multipart_suggestion(infer_srs_remove_and_box, applicability = "machine-applicable")]
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
        infer_srs_remove,
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    Remove {
        #[primary_span]
        sp: Span,
    },
    #[suggestion(
        infer_srs_add,
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
    #[note(infer_srs_add_one)]
    AddOne {
        #[primary_span]
        spans: MultiSpan,
    },
}

#[derive(Subdiagnostic)]
pub enum ConsiderAddingAwait {
    #[help(infer_await_both_futures)]
    BothFuturesHelp,
    #[multipart_suggestion(infer_await_both_futures, applicability = "maybe-incorrect")]
    BothFuturesSugg {
        #[suggestion_part(code = ".await")]
        first: Span,
        #[suggestion_part(code = ".await")]
        second: Span,
    },
    #[suggestion(
        infer_await_future,
        code = ".await",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    FutureSugg {
        #[primary_span]
        span: Span,
    },
    #[note(infer_await_note)]
    FutureSuggNote {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        infer_await_future,
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
    #[diag(infer_lf_bound_not_satisfied)]
    HasBoth {
        #[primary_span]
        span: Span,
        #[note(infer_prlf_defined_with_sub)]
        sub_span: Span,
        #[note(infer_prlf_must_oultive_with_sup)]
        sup_span: Span,
        sub_symbol: Symbol,
        sup_symbol: Symbol,
        #[note(infer_prlf_known_limitation)]
        note: (),
    },
    #[diag(infer_lf_bound_not_satisfied)]
    HasSub {
        #[primary_span]
        span: Span,
        #[note(infer_prlf_defined_with_sub)]
        sub_span: Span,
        #[note(infer_prlf_must_oultive_without_sup)]
        sup_span: Span,
        sub_symbol: Symbol,
        #[note(infer_prlf_known_limitation)]
        note: (),
    },
    #[diag(infer_lf_bound_not_satisfied)]
    HasSup {
        #[primary_span]
        span: Span,
        #[note(infer_prlf_defined_without_sub)]
        sub_span: Span,
        #[note(infer_prlf_must_oultive_with_sup)]
        sup_span: Span,
        sup_symbol: Symbol,
        #[note(infer_prlf_known_limitation)]
        note: (),
    },
    #[diag(infer_lf_bound_not_satisfied)]
    HasNone {
        #[primary_span]
        span: Span,
        #[note(infer_prlf_defined_without_sub)]
        sub_span: Span,
        #[note(infer_prlf_must_oultive_without_sup)]
        sup_span: Span,
        #[note(infer_prlf_known_limitation)]
        note: (),
    },
    #[diag(infer_lf_bound_not_satisfied)]
    OnlyPrimarySpan {
        #[primary_span]
        span: Span,
        #[note(infer_prlf_known_limitation)]
        note: (),
    },
}
