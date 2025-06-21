use rustc_errors::{Diag, EmissionGuarantee, IntoDiagArg, Subdiagnostic};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::bug;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{Span, kw};

use crate::error_reporting::infer::nice_region_error::find_anon_type;
use crate::fluent_generated as fluent;

struct DescriptionCtx<'a> {
    span: Option<Span>,
    kind: &'a str,
    arg: String,
}

impl<'a> DescriptionCtx<'a> {
    fn new<'tcx>(
        tcx: TyCtxt<'tcx>,
        generic_param_scope: LocalDefId,
        region: ty::Region<'tcx>,
        alt_span: Option<Span>,
    ) -> Option<Self> {
        let (span, kind, arg) = match region.kind() {
            ty::ReEarlyParam(br) => {
                let scope = tcx
                    .parent(tcx.generics_of(generic_param_scope).region_param(br, tcx).def_id)
                    .expect_local();
                let span = if let Some(param) =
                    tcx.hir_get_generics(scope).and_then(|generics| generics.get_named(br.name))
                {
                    param.span
                } else {
                    tcx.def_span(scope)
                };
                if br.has_name() {
                    (Some(span), "as_defined", br.name.to_string())
                } else {
                    (Some(span), "as_defined_anon", String::new())
                }
            }
            ty::ReLateParam(ref fr) => {
                if !fr.kind.is_named()
                    && let Some((ty, _)) = find_anon_type(tcx, generic_param_scope, region)
                {
                    (Some(ty.span), "defined_here", String::new())
                } else {
                    let scope = fr.scope.expect_local();
                    match fr.kind {
                        ty::LateParamRegionKind::Named(_, name) => {
                            let span = if let Some(param) = tcx
                                .hir_get_generics(scope)
                                .and_then(|generics| generics.get_named(name))
                            {
                                param.span
                            } else {
                                tcx.def_span(scope)
                            };
                            if name == kw::UnderscoreLifetime {
                                (Some(span), "as_defined_anon", String::new())
                            } else {
                                (Some(span), "as_defined", name.to_string())
                            }
                        }
                        ty::LateParamRegionKind::Anon(_) => {
                            let span = Some(tcx.def_span(scope));
                            (span, "defined_here", String::new())
                        }
                        _ => (Some(tcx.def_span(scope)), "defined_here_reg", region.to_string()),
                    }
                }
            }

            ty::ReStatic => (alt_span, "restatic", String::new()),

            ty::RePlaceholder(_) | ty::ReError(_) => return None,

            ty::ReVar(_) | ty::ReBound(..) | ty::ReErased => {
                bug!("unexpected region for DescriptionCtx: {:?}", region);
            }
        };
        Some(DescriptionCtx { span, kind, arg })
    }
}

pub enum PrefixKind {
    Empty,
    RefValidFor,
    ContentValidFor,
    TypeObjValidFor,
    SourcePointerValidFor,
    TypeSatisfy,
    TypeOutlive,
    LfParamInstantiatedWith,
    LfParamMustOutlive,
    LfInstantiatedWith,
    LfMustOutlive,
    PointerValidFor,
    DataValidFor,
}

pub enum SuffixKind {
    Empty,
    Continues,
    ReqByBinding,
}

impl IntoDiagArg for PrefixKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        let kind = match self {
            Self::Empty => "empty",
            Self::RefValidFor => "ref_valid_for",
            Self::ContentValidFor => "content_valid_for",
            Self::TypeObjValidFor => "type_obj_valid_for",
            Self::SourcePointerValidFor => "source_pointer_valid_for",
            Self::TypeSatisfy => "type_satisfy",
            Self::TypeOutlive => "type_outlive",
            Self::LfParamInstantiatedWith => "lf_param_instantiated_with",
            Self::LfParamMustOutlive => "lf_param_must_outlive",
            Self::LfInstantiatedWith => "lf_instantiated_with",
            Self::LfMustOutlive => "lf_must_outlive",
            Self::PointerValidFor => "pointer_valid_for",
            Self::DataValidFor => "data_valid_for",
        }
        .into();
        rustc_errors::DiagArgValue::Str(kind)
    }
}

impl IntoDiagArg for SuffixKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        let kind = match self {
            Self::Empty => "empty",
            Self::Continues => "continues",
            Self::ReqByBinding => "req_by_binding",
        }
        .into();
        rustc_errors::DiagArgValue::Str(kind)
    }
}

pub struct RegionExplanation<'a> {
    desc: DescriptionCtx<'a>,
    prefix: PrefixKind,
    suffix: SuffixKind,
}

impl RegionExplanation<'_> {
    pub fn new<'tcx>(
        tcx: TyCtxt<'tcx>,
        generic_param_scope: LocalDefId,
        region: ty::Region<'tcx>,
        alt_span: Option<Span>,
        prefix: PrefixKind,
        suffix: SuffixKind,
    ) -> Option<Self> {
        Some(Self {
            desc: DescriptionCtx::new(tcx, generic_param_scope, region, alt_span)?,
            prefix,
            suffix,
        })
    }
}

impl Subdiagnostic for RegionExplanation<'_> {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.store_args();
        diag.arg("pref_kind", self.prefix);
        diag.arg("suff_kind", self.suffix);
        diag.arg("desc_kind", self.desc.kind);
        diag.arg("desc_arg", self.desc.arg);

        let msg = diag.eagerly_translate(fluent::trait_selection_region_explanation);
        diag.restore_args();
        if let Some(span) = self.desc.span {
            diag.span_note(span, msg);
        } else {
            diag.note(msg);
        }
    }
}
