use crate::infer::type_variable::TypeVariableOriginKind;
use crate::infer::{InferCtxt, Symbol};
use rustc_errors::{
    pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Body, Expr, ExprKind, FnRetTy, HirId, Local, MatchSource, Pat};
use rustc_middle::hir::nested_filter;
use rustc_middle::infer::unify_key::ConstVariableOriginKind;
use rustc_middle::ty::print::Print;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind};
use rustc_middle::ty::{self, Const, DefIdTree, InferConst, Ty, TyCtxt, TypeFoldable, TypeFolder};
use rustc_span::symbol::kw;
use rustc_span::{sym, Span};
use std::borrow::Cow;

pub enum TypeAnnotationNeeded {
    /// ```compile_fail,E0282
    /// let x = "hello".chars().rev().collect();
    /// ```
    E0282,
    /// An implementation cannot be chosen unambiguously because of lack of information.
    /// ```compile_fail,E0283
    /// let _ = Default::default();
    /// ```
    E0283,
    /// ```compile_fail,E0284
    /// let mut d: u64 = 2;
    /// d = d % 1u32.into();
    /// ```
    E0284,
}

impl Into<rustc_errors::DiagnosticId> for TypeAnnotationNeeded {
    fn into(self) -> rustc_errors::DiagnosticId {
        match self {
            Self::E0282 => rustc_errors::error_code!(E0282),
            Self::E0283 => rustc_errors::error_code!(E0283),
            Self::E0284 => rustc_errors::error_code!(E0284),
        }
    }
}

/// Information about a constant or a type containing inference variables.
pub struct InferenceDiagnosticsData {
    pub name: String,
    pub span: Option<Span>,
    pub kind: UnderspecifiedArgKind,
    pub parent: Option<InferenceDiagnosticsParentData>,
}

/// Data on the parent definition where a generic argument was declared.
pub struct InferenceDiagnosticsParentData {
    pub prefix: &'static str,
    pub name: String,
    pub def_id: DefId,
}

pub enum UnderspecifiedArgKind {
    Type { prefix: Cow<'static, str> },
    Const { is_parameter: bool },
}

impl InferenceDiagnosticsData {
    /// Generate a label for a generic argument which can't be inferred. When not
    /// much is known about the argument, `use_diag` may be used to describe the
    /// labeled value.
    fn cannot_infer_msg(&self) -> String {
        if self.name == "_" && matches!(self.kind, UnderspecifiedArgKind::Type { .. }) {
            return "cannot infer type".to_string();
        }

        let suffix = match &self.parent {
            Some(parent) => parent.suffix_string(),
            None => String::new(),
        };

        // For example: "cannot infer type for type parameter `T`"
        format!("cannot infer {} `{}`{}", self.kind.prefix_string(), self.name, suffix)
    }
}

impl InferenceDiagnosticsParentData {
    fn for_parent_def_id(
        tcx: TyCtxt<'_>,
        parent_def_id: DefId,
    ) -> Option<InferenceDiagnosticsParentData> {
        let parent_name =
            tcx.def_key(parent_def_id).disambiguated_data.data.get_opt_name()?.to_string();

        Some(InferenceDiagnosticsParentData {
            prefix: tcx.def_kind(parent_def_id).descr(parent_def_id),
            name: parent_name,
            def_id: parent_def_id,
        })
    }

    fn for_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Option<InferenceDiagnosticsParentData> {
        Self::for_parent_def_id(tcx, tcx.parent(def_id))
    }

    fn suffix_string(&self) -> String {
        format!(" declared on the {} `{}`", self.prefix, self.name)
    }
}

impl UnderspecifiedArgKind {
    fn prefix_string(&self) -> Cow<'static, str> {
        match self {
            Self::Type { prefix } => format!("type for {}", prefix).into(),
            Self::Const { is_parameter: true } => "the value of const parameter".into(),
            Self::Const { is_parameter: false } => "the value of the constant".into(),
        }
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    /// Extracts data used by diagnostic for either types or constants
    /// which were stuck during inference.
    pub fn extract_inference_diagnostics_data(
        &self,
        arg: GenericArg<'tcx>,
        highlight: Option<ty::print::RegionHighlightMode<'tcx>>,
    ) -> InferenceDiagnosticsData {
        match arg.unpack() {
            GenericArgKind::Type(ty) => {
                if let ty::Infer(ty::TyVar(ty_vid)) = *ty.kind() {
                    let mut inner = self.inner.borrow_mut();
                    let ty_vars = &inner.type_variables();
                    let var_origin = ty_vars.var_origin(ty_vid);
                    if let TypeVariableOriginKind::TypeParameterDefinition(name, def_id) =
                        var_origin.kind
                    {
                        if name != kw::SelfUpper {
                            return InferenceDiagnosticsData {
                                name: name.to_string(),
                                span: Some(var_origin.span),
                                kind: UnderspecifiedArgKind::Type {
                                    prefix: "type parameter".into(),
                                },
                                parent: def_id.and_then(|def_id| {
                                    InferenceDiagnosticsParentData::for_def_id(self.tcx, def_id)
                                }),
                            };
                        }
                    }
                }

                let mut printer = ty::print::FmtPrinter::new(self.tcx, Namespace::TypeNS);
                if let Some(highlight) = highlight {
                    printer.region_highlight_mode = highlight;
                }
                let name = ty.print(printer).unwrap().into_buffer();
                InferenceDiagnosticsData {
                    name,
                    span: None,
                    kind: UnderspecifiedArgKind::Type { prefix: ty.prefix_string(self.tcx) },
                    parent: None,
                }
            }
            GenericArgKind::Const(ct) => {
                if let ty::ConstKind::Infer(InferConst::Var(vid)) = ct.val() {
                    let origin =
                        self.inner.borrow_mut().const_unification_table().probe_value(vid).origin;
                    if let ConstVariableOriginKind::ConstParameterDefinition(name, def_id) =
                        origin.kind
                    {
                        return InferenceDiagnosticsData {
                            name: name.to_string(),
                            span: Some(origin.span),
                            kind: UnderspecifiedArgKind::Const { is_parameter: true },
                            parent: InferenceDiagnosticsParentData::for_def_id(self.tcx, def_id),
                        };
                    }

                    debug_assert!(!origin.span.is_dummy());
                    let mut printer =
                        ty::print::FmtPrinter::new(self.tcx, Namespace::ValueNS);
                    if let Some(highlight) = highlight {
                        printer.region_highlight_mode = highlight;
                    }
                    InferenceDiagnosticsData {
                        name: ct.print(printer).unwrap().into_buffer(),
                        span: Some(origin.span),
                        kind: UnderspecifiedArgKind::Const { is_parameter: false },
                        parent: None,
                    }
                } else {
                    bug!("unexpect const: {:?}", ct);
                }
            }
            GenericArgKind::Lifetime(_) => bug!("unexpected lifetime"),
        }
    }

    pub fn emit_inference_failure_err(
        &self,
        body_id: Option<hir::BodyId>,
        span: Span,
        arg: GenericArg<'tcx>,
        impl_candidates: Vec<ty::TraitRef<'tcx>>,
        error_code: TypeAnnotationNeeded,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let arg = self.resolve_vars_if_possible(arg);

        let error_code = error_code.into();
        let err = self.tcx.sess.struct_span_err_with_code(
            span,
            &format!("type annotations needed"),
            error_code,
        );
        err
    }

    pub fn need_type_info_err_in_generator(
        &self,
        kind: hir::GeneratorKind,
        span: Span,
        ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let ty = self.resolve_vars_if_possible(ty);
        let data = self.extract_inference_diagnostics_data(ty.into(), None);

        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0698,
            "type inside {} must be known in this context",
            kind,
        );
        err.span_label(span, data.cannot_infer_msg());
        err
    }
}
