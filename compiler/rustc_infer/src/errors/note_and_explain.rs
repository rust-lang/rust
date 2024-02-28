use crate::fluent_generated as fluent;
use crate::infer::error_reporting::nice_region_error::find_anon_type;
use rustc_errors::{
    AddToDiagnostic, Diag, EmissionGuarantee, IntoDiagnosticArg, SubdiagnosticMessageOp,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{symbol::kw, Span};

struct DescriptionCtx<'a> {
    span: Option<Span>,
    kind: &'a str,
    arg: String,
}

impl<'a> DescriptionCtx<'a> {
    fn new<'tcx>(
        tcx: TyCtxt<'tcx>,
        region: ty::Region<'tcx>,
        alt_span: Option<Span>,
    ) -> Option<Self> {
        let (span, kind, arg) = match *region {
            ty::ReEarlyParam(ref br) => {
                let scope = region.free_region_binding_scope(tcx).expect_local();
                let span = if let Some(param) =
                    tcx.hir().get_generics(scope).and_then(|generics| generics.get_named(br.name))
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
                if !fr.bound_region.is_named()
                    && let Some((ty, _)) = find_anon_type(tcx, region, &fr.bound_region)
                {
                    (Some(ty.span), "defined_here", String::new())
                } else {
                    let scope = region.free_region_binding_scope(tcx).expect_local();
                    match fr.bound_region {
                        ty::BoundRegionKind::BrNamed(_, name) => {
                            let span = if let Some(param) = tcx
                                .hir()
                                .get_generics(scope)
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
                        ty::BrAnon => {
                            let span = Some(tcx.def_span(scope));
                            (span, "defined_here", String::new())
                        }
                        _ => (Some(tcx.def_span(scope)), "defined_here_reg", region.to_string()),
                    }
                }
            }

            ty::ReStatic => (alt_span, "restatic", String::new()),

            ty::RePlaceholder(_) | ty::ReError(_) => return None,

            // FIXME(#13998) RePlaceholder should probably print like
            // ReLateParam rather than dumping Debug output on the user.
            //
            // We shouldn't really be having unification failures with ReVar
            // and ReBound though.
            //
            // FIXME(@lcnr): figure out why we have to handle `ReBound`
            // here, this feels somewhat off.
            ty::ReVar(_) | ty::ReBound(..) | ty::ReErased => {
                (alt_span, "revar", format!("{region:?}"))
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

impl IntoDiagnosticArg for PrefixKind {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagArgValue {
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

impl IntoDiagnosticArg for SuffixKind {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagArgValue {
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
        region: ty::Region<'tcx>,
        alt_span: Option<Span>,
        prefix: PrefixKind,
        suffix: SuffixKind,
    ) -> Option<Self> {
        Some(Self { desc: DescriptionCtx::new(tcx, region, alt_span)?, prefix, suffix })
    }
}

impl AddToDiagnostic for RegionExplanation<'_> {
    fn add_to_diagnostic_with<G: EmissionGuarantee, F: SubdiagnosticMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        f: F,
    ) {
        diag.arg("pref_kind", self.prefix);
        diag.arg("suff_kind", self.suffix);
        diag.arg("desc_kind", self.desc.kind);
        diag.arg("desc_arg", self.desc.arg);

        let msg = f(diag, fluent::infer_region_explanation.into());
        if let Some(span) = self.desc.span {
            diag.span_note(span, msg);
        } else {
            diag.note(msg);
        }
    }
}
