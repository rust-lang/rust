use crate::infer::error_reporting::nice_region_error::find_anon_type;
use rustc_errors::{
    self, fluent, AddToDiagnostic, Diagnostic, IntoDiagnosticArg, SubdiagnosticMessage,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{symbol::kw, Span};

#[derive(Default)]
struct DescriptionCtx<'a> {
    span: Option<Span>,
    kind: &'a str,
    arg: String,
    num_arg: u32,
}

impl<'a> DescriptionCtx<'a> {
    fn new<'tcx>(
        tcx: TyCtxt<'tcx>,
        region: ty::Region<'tcx>,
        alt_span: Option<Span>,
    ) -> Option<Self> {
        let mut me = DescriptionCtx::default();
        me.span = alt_span;
        match *region {
            ty::ReEarlyBound(_) | ty::ReFree(_) => {
                return Self::from_early_bound_and_free_regions(tcx, region);
            }
            ty::ReStatic => {
                me.kind = "restatic";
            }

            ty::RePlaceholder(_) => return None,

            ty::ReError(_) => return None,

            // FIXME(#13998) RePlaceholder should probably print like
            // ReFree rather than dumping Debug output on the user.
            //
            // We shouldn't really be having unification failures with ReVar
            // and ReLateBound though.
            ty::ReVar(_) | ty::ReLateBound(..) | ty::ReErased => {
                me.kind = "revar";
                me.arg = format!("{:?}", region);
            }
        };
        Some(me)
    }

    fn from_early_bound_and_free_regions<'tcx>(
        tcx: TyCtxt<'tcx>,
        region: ty::Region<'tcx>,
    ) -> Option<Self> {
        let mut me = DescriptionCtx::default();
        let scope = region.free_region_binding_scope(tcx).expect_local();
        match *region {
            ty::ReEarlyBound(ref br) => {
                let mut sp = tcx.def_span(scope);
                if let Some(param) =
                    tcx.hir().get_generics(scope).and_then(|generics| generics.get_named(br.name))
                {
                    sp = param.span;
                }
                if br.has_name() {
                    me.kind = "as_defined";
                    me.arg = br.name.to_string();
                } else {
                    me.kind = "as_defined_anon";
                };
                me.span = Some(sp)
            }
            ty::ReFree(ref fr) => {
                if !fr.bound_region.is_named()
                    && let Some((ty, _)) = find_anon_type(tcx, region, &fr.bound_region)
                {
                    me.kind = "defined_here";
                    me.span = Some(ty.span);
                } else {
                    match fr.bound_region {
                        ty::BoundRegionKind::BrNamed(_, name) => {
                            let mut sp = tcx.def_span(scope);
                            if let Some(param) =
                                tcx.hir().get_generics(scope).and_then(|generics| generics.get_named(name))
                            {
                                sp = param.span;
                            }
                            if name == kw::UnderscoreLifetime {
                                me.kind = "as_defined_anon";
                            } else {
                                me.kind = "as_defined";
                                me.arg = name.to_string();
                            };
                            me.span = Some(sp);
                        }
                        ty::BrAnon(idx, span) => {
                            me.kind = "anon_num_here";
                            me.num_arg = idx+1;
                            me.span = match span {
                                Some(_) => span,
                                None => Some(tcx.def_span(scope)),
                            }
                        },
                        _ => {
                            me.kind = "defined_here_reg";
                            me.arg = region.to_string();
                            me.span = Some(tcx.def_span(scope));
                        },
                    }
                }
            }
            _ => bug!(),
        }
        Some(me)
    }

    fn add_to(self, diag: &mut rustc_errors::Diagnostic) {
        diag.set_arg("desc_kind", self.kind);
        diag.set_arg("desc_arg", self.arg);
        diag.set_arg("desc_num_arg", self.num_arg);
    }
}

pub enum PrefixKind {
    Empty,
}

pub enum SuffixKind {
    Continues,
}

impl IntoDiagnosticArg for PrefixKind {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        let kind = match self {
            Self::Empty => "empty",
        }
        .into();
        rustc_errors::DiagnosticArgValue::Str(kind)
    }
}

impl IntoDiagnosticArg for SuffixKind {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        let kind = match self {
            Self::Continues => "continues",
        }
        .into();
        rustc_errors::DiagnosticArgValue::Str(kind)
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
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        if let Some(span) = self.desc.span {
            diag.span_note(span, fluent::infer_region_explanation);
        } else {
            diag.note(fluent::infer_region_explanation);
        }
        self.desc.add_to(diag);
        diag.set_arg("pref_kind", self.prefix);
        diag.set_arg("suff_kind", self.suffix);
    }
}
