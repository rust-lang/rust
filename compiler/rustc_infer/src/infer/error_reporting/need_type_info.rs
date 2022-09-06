use crate::errors::{
    AmbigousImpl, AmbigousReturn, AnnotationRequired, InferenceBadError, NeedTypeInfoInGenerator,
    SourceKindMultiSuggestion, SourceKindSubdiag,
};
use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::InferCtxt;
use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed, IntoDiagnosticArg};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def::{CtorOf, DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Body, Closure, Expr, ExprKind, FnRetTy, HirId, Local, LocalSource};
use rustc_middle::hir::nested_filter;
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::print::{FmtPrinter, PrettyPrinter, Print, Printer};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, Subst, SubstsRef};
use rustc_middle::ty::{self, DefIdTree, InferConst};
use rustc_middle::ty::{IsSuggestable, Ty, TyCtxt, TypeckResults};
use rustc_session::SessionDiagnostic;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{BytePos, Span};
use std::borrow::Cow;
use std::iter;

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
    prefix: &'static str,
    name: String,
}

#[derive(Clone)]
pub enum UnderspecifiedArgKind {
    Type { prefix: Cow<'static, str> },
    Const { is_parameter: bool },
}

impl InferenceDiagnosticsData {
    fn can_add_more_info(&self) -> bool {
        !(self.name == "_" && matches!(self.kind, UnderspecifiedArgKind::Type { .. }))
    }

    fn where_x_is_kind(&self, in_type: Ty<'_>) -> &'static str {
        if in_type.is_ty_infer() {
            "empty"
        } else if self.name == "_" {
            // FIXME: Consider specializing this message if there is a single `_`
            // in the type.
            "underscore"
        } else {
            "has_name"
        }
    }

    /// Generate a label for a generic argument which can't be inferred. When not
    /// much is known about the argument, `use_diag` may be used to describe the
    /// labeled value.
    fn make_bad_error(&self, span: Span) -> InferenceBadError<'_> {
        let has_parent = self.parent.is_some();
        let bad_kind = if self.can_add_more_info() { "more_info" } else { "other" };
        let (parent_prefix, parent_name) = self
            .parent
            .as_ref()
            .map(|parent| (parent.prefix, parent.name.clone()))
            .unwrap_or_default();
        InferenceBadError {
            span,
            bad_kind,
            prefix_kind: self.kind.clone(),
            prefix: self.kind.try_get_prefix().unwrap_or_default(),
            name: self.name.clone(),
            has_parent,
            parent_prefix,
            parent_name,
        }
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
        })
    }

    fn for_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Option<InferenceDiagnosticsParentData> {
        Self::for_parent_def_id(tcx, tcx.parent(def_id))
    }
}

impl IntoDiagnosticArg for UnderspecifiedArgKind {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        let kind = match self {
            Self::Type { .. } => "type",
            Self::Const { is_parameter: true } => "const_with_param",
            Self::Const { is_parameter: false } => "const",
        };
        rustc_errors::DiagnosticArgValue::Str(kind.into())
    }
}

impl UnderspecifiedArgKind {
    fn try_get_prefix(&self) -> Option<&str> {
        match self {
            Self::Type { prefix } => Some(prefix.as_ref()),
            Self::Const { .. } => None,
        }
    }
}

fn fmt_printer<'a, 'tcx>(infcx: &'a InferCtxt<'_, 'tcx>, ns: Namespace) -> FmtPrinter<'a, 'tcx> {
    let mut printer = FmtPrinter::new(infcx.tcx, ns);
    let ty_getter = move |ty_vid| {
        if infcx.probe_ty_var(ty_vid).is_ok() {
            warn!("resolved ty var in error message");
        }
        if let TypeVariableOriginKind::TypeParameterDefinition(name, _) =
            infcx.inner.borrow_mut().type_variables().var_origin(ty_vid).kind
        {
            Some(name)
        } else {
            None
        }
    };
    printer.ty_infer_name_resolver = Some(Box::new(ty_getter));
    let const_getter = move |ct_vid| {
        if infcx.probe_const_var(ct_vid).is_ok() {
            warn!("resolved const var in error message");
        }
        if let ConstVariableOriginKind::ConstParameterDefinition(name, _) =
            infcx.inner.borrow_mut().const_unification_table().probe_value(ct_vid).origin.kind
        {
            return Some(name);
        } else {
            None
        }
    };
    printer.const_infer_name_resolver = Some(Box::new(const_getter));
    printer
}

fn ty_to_string<'tcx>(infcx: &InferCtxt<'_, 'tcx>, ty: Ty<'tcx>) -> String {
    let printer = fmt_printer(infcx, Namespace::TypeNS);
    let ty = infcx.resolve_vars_if_possible(ty);
    match ty.kind() {
        // We don't want the regular output for `fn`s because it includes its path in
        // invalid pseudo-syntax, we want the `fn`-pointer output instead.
        ty::FnDef(..) => ty.fn_sig(infcx.tcx).print(printer).unwrap().into_buffer(),
        // FIXME: The same thing for closures, but this only works when the closure
        // does not capture anything.
        //
        // We do have to hide the `extern "rust-call"` ABI in that case though,
        // which is too much of a bother for now.
        _ => ty.print(printer).unwrap().into_buffer(),
    }
}

/// We don't want to directly use `ty_to_string` for closures as their type isn't really
/// something users are familiar with. Directly printing the `fn_sig` of closures also
/// doesn't work as they actually use the "rust-call" API.
fn closure_as_fn_str<'tcx>(infcx: &InferCtxt<'_, 'tcx>, ty: Ty<'tcx>) -> String {
    let ty::Closure(_, substs) = ty.kind() else { unreachable!() };
    let fn_sig = substs.as_closure().sig();
    let args = fn_sig
        .inputs()
        .skip_binder()
        .iter()
        .next()
        .map(|args| {
            args.tuple_fields()
                .iter()
                .map(|arg| ty_to_string(infcx, arg))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let ret = if fn_sig.output().skip_binder().is_unit() {
        String::new()
    } else {
        format!(" -> {}", ty_to_string(infcx, fn_sig.output().skip_binder()))
    };
    format!("fn({}){}", args, ret)
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
                InferenceDiagnosticsData {
                    name: ty.print(printer).unwrap().into_buffer(),
                    span: None,
                    kind: UnderspecifiedArgKind::Type { prefix: ty.prefix_string(self.tcx) },
                    parent: None,
                }
            }
            GenericArgKind::Const(ct) => {
                if let ty::ConstKind::Infer(InferConst::Var(vid)) = ct.kind() {
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
                    let mut printer = ty::print::FmtPrinter::new(self.tcx, Namespace::ValueNS);
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
                    // If we end up here the `FindInferSourceVisitor`
                    // won't work, as its expected argument isn't an inference variable.
                    //
                    // FIXME: Ideally we should look into the generic constant
                    // to figure out which inference var is actually unresolved so that
                    // this path is unreachable.
                    let mut printer = ty::print::FmtPrinter::new(self.tcx, Namespace::ValueNS);
                    if let Some(highlight) = highlight {
                        printer.region_highlight_mode = highlight;
                    }
                    InferenceDiagnosticsData {
                        name: ct.print(printer).unwrap().into_buffer(),
                        span: None,
                        kind: UnderspecifiedArgKind::Const { is_parameter: false },
                        parent: None,
                    }
                }
            }
            GenericArgKind::Lifetime(_) => bug!("unexpected lifetime"),
        }
    }

    /// Used as a fallback in [InferCtxt::emit_inference_failure_err]
    /// in case we weren't able to get a better error.
    fn bad_inference_failure_err(
        &self,
        span: Span,
        arg_data: InferenceDiagnosticsData,
        error_code: TypeAnnotationNeeded,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let source_kind = "other";
        let source_name = "";
        let failure_span = None;
        let infer_subdiags = Vec::new();
        let multi_suggestions = Vec::new();
        let bad_label = Some(arg_data.make_bad_error(span));
        match error_code {
            TypeAnnotationNeeded::E0282 => AnnotationRequired {
                span,
                source_kind,
                source_name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label,
            }
            .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic),
            TypeAnnotationNeeded::E0283 => AmbigousImpl {
                span,
                source_kind,
                source_name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label,
            }
            .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic),
            TypeAnnotationNeeded::E0284 => AmbigousReturn {
                span,
                source_kind,
                source_name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label,
            }
            .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic),
        }
    }

    pub fn emit_inference_failure_err(
        &self,
        body_id: Option<hir::BodyId>,
        failure_span: Span,
        arg: GenericArg<'tcx>,
        error_code: TypeAnnotationNeeded,
        should_label_span: bool,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let arg = self.resolve_vars_if_possible(arg);
        let arg_data = self.extract_inference_diagnostics_data(arg, None);

        let Some(typeck_results) = self.in_progress_typeck_results else {
            // If we don't have any typeck results we're outside
            // of a body, so we won't be able to get better info
            // here.
            return self.bad_inference_failure_err(failure_span, arg_data, error_code);
        };
        let typeck_results = typeck_results.borrow();
        let typeck_results = &typeck_results;

        let mut local_visitor = FindInferSourceVisitor::new(&self, typeck_results, arg);
        if let Some(body_id) = body_id {
            let expr = self.tcx.hir().expect_expr(body_id.hir_id);
            local_visitor.visit_expr(expr);
        }

        let Some(InferSource { span, kind }) = local_visitor.infer_source else {
            return self.bad_inference_failure_err(failure_span, arg_data, error_code)
        };

        let (source_kind, name) = kind.ty_localized_msg(self);
        let failure_span = if should_label_span && !failure_span.overlaps(span) {
            Some(failure_span)
        } else {
            None
        };

        let mut infer_subdiags = Vec::new();
        let mut multi_suggestions = Vec::new();
        match kind {
            InferSourceKind::LetBinding { insert_span, pattern_name, ty } => {
                infer_subdiags.push(SourceKindSubdiag::LetLike {
                    span: insert_span,
                    name: pattern_name.map(|name| name.to_string()).unwrap_or_else(String::new),
                    x_kind: arg_data.where_x_is_kind(ty),
                    prefix_kind: arg_data.kind.clone(),
                    prefix: arg_data.kind.try_get_prefix().unwrap_or_default(),
                    arg_name: arg_data.name,
                    kind: if pattern_name.is_some() { "with_pattern" } else { "other" },
                    type_name: ty_to_string(self, ty),
                });
            }
            InferSourceKind::ClosureArg { insert_span, ty } => {
                infer_subdiags.push(SourceKindSubdiag::LetLike {
                    span: insert_span,
                    name: String::new(),
                    x_kind: arg_data.where_x_is_kind(ty),
                    prefix_kind: arg_data.kind.clone(),
                    prefix: arg_data.kind.try_get_prefix().unwrap_or_default(),
                    arg_name: arg_data.name,
                    kind: "closure",
                    type_name: ty_to_string(self, ty),
                });
            }
            InferSourceKind::GenericArg {
                insert_span,
                argument_index,
                generics_def_id,
                def_id: _,
                generic_args,
            } => {
                let generics = self.tcx.generics_of(generics_def_id);
                let is_type = matches!(arg.unpack(), GenericArgKind::Type(_));

                let (parent_exists, parent_prefix, parent_name) =
                    InferenceDiagnosticsParentData::for_parent_def_id(self.tcx, generics_def_id)
                        .map_or((false, String::new(), String::new()), |parent| {
                            (true, parent.prefix.to_string(), parent.name)
                        });

                infer_subdiags.push(SourceKindSubdiag::GenericLabel {
                    span,
                    is_type,
                    param_name: generics.params[argument_index].name.to_string(),
                    parent_exists,
                    parent_prefix,
                    parent_name,
                });

                let args = fmt_printer(self, Namespace::TypeNS)
                    .comma_sep(generic_args.iter().copied().map(|arg| {
                        if arg.is_suggestable(self.tcx, true) {
                            return arg;
                        }

                        match arg.unpack() {
                            GenericArgKind::Lifetime(_) => bug!("unexpected lifetime"),
                            GenericArgKind::Type(_) => self
                                .next_ty_var(TypeVariableOrigin {
                                    span: rustc_span::DUMMY_SP,
                                    kind: TypeVariableOriginKind::MiscVariable,
                                })
                                .into(),
                            GenericArgKind::Const(arg) => self
                                .next_const_var(
                                    arg.ty(),
                                    ConstVariableOrigin {
                                        span: rustc_span::DUMMY_SP,
                                        kind: ConstVariableOriginKind::MiscVariable,
                                    },
                                )
                                .into(),
                        }
                    }))
                    .unwrap()
                    .into_buffer();

                infer_subdiags.push(SourceKindSubdiag::GenericSuggestion {
                    span: insert_span,
                    arg_count: generic_args.len(),
                    args,
                });
            }
            InferSourceKind::FullyQualifiedMethodCall { receiver, successor, substs, def_id } => {
                let printer = fmt_printer(self, Namespace::ValueNS);
                let def_path = printer.print_def_path(def_id, substs).unwrap().into_buffer();

                // We only care about whether we have to add `&` or `&mut ` for now.
                // This is the case if the last adjustment is a borrow and the
                // first adjustment was not a builtin deref.
                let adjustment = match typeck_results.expr_adjustments(receiver) {
                    [
                        Adjustment { kind: Adjust::Deref(None), target: _ },
                        ..,
                        Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), target: _ },
                    ] => "",
                    [
                        ..,
                        Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(_, mut_)), target: _ },
                    ] => match mut_ {
                        AutoBorrowMutability::Mut { .. } => "&mut ",
                        AutoBorrowMutability::Not => "&",
                    },
                    _ => "",
                };

                multi_suggestions.push(SourceKindMultiSuggestion::FullyQualified {
                    span: receiver.span,
                    def_path,
                    adjustment,
                    successor,
                });
            }
            InferSourceKind::ClosureReturn { ty, data, should_wrap_expr } => {
                let ty_info = ty_to_string(self, ty);
                multi_suggestions.push(SourceKindMultiSuggestion::ClosureReturn {
                    ty_info,
                    data,
                    should_wrap_expr,
                });
            }
        }
        match error_code {
            TypeAnnotationNeeded::E0282 => AnnotationRequired {
                span,
                source_kind,
                source_name: &name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label: None,
            }
            .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic),
            TypeAnnotationNeeded::E0283 => AmbigousImpl {
                span,
                source_kind,
                source_name: &name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label: None,
            }
            .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic),
            TypeAnnotationNeeded::E0284 => AmbigousReturn {
                span,
                source_kind,
                source_name: &name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label: None,
            }
            .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic),
        }
    }

    pub fn need_type_info_err_in_generator(
        &self,
        kind: hir::GeneratorKind,
        span: Span,
        ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let ty = self.resolve_vars_if_possible(ty);
        let data = self.extract_inference_diagnostics_data(ty.into(), None);

        NeedTypeInfoInGenerator {
            bad_label: data.make_bad_error(span),
            span,
            generator_kind: GeneratorKindAsDiagArg(kind),
        }
        .into_diagnostic(&self.tcx.sess.parse_sess.span_diagnostic)
    }
}

pub struct GeneratorKindAsDiagArg(pub hir::GeneratorKind);

impl IntoDiagnosticArg for GeneratorKindAsDiagArg {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        let kind = match self.0 {
            hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Block) => "async_block",
            hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Closure) => "async_closure",
            hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Fn) => "async_fn",
            hir::GeneratorKind::Gen => "generator",
        };
        rustc_errors::DiagnosticArgValue::Str(kind.into())
    }
}

#[derive(Debug)]
struct InferSource<'tcx> {
    span: Span,
    kind: InferSourceKind<'tcx>,
}

#[derive(Debug)]
enum InferSourceKind<'tcx> {
    LetBinding {
        insert_span: Span,
        pattern_name: Option<Ident>,
        ty: Ty<'tcx>,
    },
    ClosureArg {
        insert_span: Span,
        ty: Ty<'tcx>,
    },
    GenericArg {
        insert_span: Span,
        argument_index: usize,
        generics_def_id: DefId,
        def_id: DefId,
        generic_args: &'tcx [GenericArg<'tcx>],
    },
    FullyQualifiedMethodCall {
        receiver: &'tcx Expr<'tcx>,
        /// If the method has other arguments, this is ", " and the start of the first argument,
        /// while for methods without arguments this is ")" and the end of the method call.
        successor: (&'static str, BytePos),
        substs: SubstsRef<'tcx>,
        def_id: DefId,
    },
    ClosureReturn {
        ty: Ty<'tcx>,
        data: &'tcx FnRetTy<'tcx>,
        should_wrap_expr: Option<Span>,
    },
}

impl<'tcx> InferSource<'tcx> {
    fn from_expansion(&self) -> bool {
        let source_from_expansion = match self.kind {
            InferSourceKind::LetBinding { insert_span, .. }
            | InferSourceKind::ClosureArg { insert_span, .. }
            | InferSourceKind::GenericArg { insert_span, .. } => insert_span.from_expansion(),
            InferSourceKind::FullyQualifiedMethodCall { receiver, .. } => {
                receiver.span.from_expansion()
            }
            InferSourceKind::ClosureReturn { data, should_wrap_expr, .. } => {
                data.span().from_expansion() || should_wrap_expr.map_or(false, Span::from_expansion)
            }
        };
        source_from_expansion || self.span.from_expansion()
    }
}

impl<'tcx> InferSourceKind<'tcx> {
    fn ty_localized_msg(&self, infcx: &InferCtxt<'_, 'tcx>) -> (&'static str, String) {
        match *self {
            InferSourceKind::LetBinding { ty, .. }
            | InferSourceKind::ClosureArg { ty, .. }
            | InferSourceKind::ClosureReturn { ty, .. } => {
                if ty.is_closure() {
                    ("closure", closure_as_fn_str(infcx, ty))
                } else if !ty.is_ty_infer() {
                    ("normal", ty_to_string(infcx, ty))
                } else {
                    ("other", String::new())
                }
            }
            // FIXME: We should be able to add some additional info here.
            InferSourceKind::GenericArg { .. }
            | InferSourceKind::FullyQualifiedMethodCall { .. } => ("other", String::new()),
        }
    }
}

#[derive(Debug)]
struct InsertableGenericArgs<'tcx> {
    insert_span: Span,
    substs: SubstsRef<'tcx>,
    generics_def_id: DefId,
    def_id: DefId,
}

/// A visitor which searches for the "best" spot to use in the inference error.
///
/// For this it walks over the hir body and tries to check all places where
/// inference variables could be bound.
///
/// While doing so, the currently best spot is stored in `infer_source`.
/// For details on how we rank spots, see [Self::source_cost]
struct FindInferSourceVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    typeck_results: &'a TypeckResults<'tcx>,

    target: GenericArg<'tcx>,

    attempt: usize,
    infer_source_cost: usize,
    infer_source: Option<InferSource<'tcx>>,
}

impl<'a, 'tcx> FindInferSourceVisitor<'a, 'tcx> {
    fn new(
        infcx: &'a InferCtxt<'a, 'tcx>,
        typeck_results: &'a TypeckResults<'tcx>,
        target: GenericArg<'tcx>,
    ) -> Self {
        FindInferSourceVisitor {
            infcx,
            typeck_results,

            target,

            attempt: 0,
            infer_source_cost: usize::MAX,
            infer_source: None,
        }
    }

    /// Computes cost for the given source.
    ///
    /// Sources with a small cost are prefer and should result
    /// in a clearer and idiomatic suggestion.
    fn source_cost(&self, source: &InferSource<'tcx>) -> usize {
        #[derive(Clone, Copy)]
        struct CostCtxt<'tcx> {
            tcx: TyCtxt<'tcx>,
        }
        impl<'tcx> CostCtxt<'tcx> {
            fn arg_cost(self, arg: GenericArg<'tcx>) -> usize {
                match arg.unpack() {
                    GenericArgKind::Lifetime(_) => 0, // erased
                    GenericArgKind::Type(ty) => self.ty_cost(ty),
                    GenericArgKind::Const(_) => 3, // some non-zero value
                }
            }
            fn ty_cost(self, ty: Ty<'tcx>) -> usize {
                match *ty.kind() {
                    ty::Closure(..) => 1000,
                    ty::FnDef(..) => 150,
                    ty::FnPtr(..) => 30,
                    ty::Adt(def, substs) => {
                        5 + self
                            .tcx
                            .generics_of(def.did())
                            .own_substs_no_defaults(self.tcx, substs)
                            .iter()
                            .map(|&arg| self.arg_cost(arg))
                            .sum::<usize>()
                    }
                    ty::Tuple(args) => 5 + args.iter().map(|arg| self.ty_cost(arg)).sum::<usize>(),
                    ty::Ref(_, ty, _) => 2 + self.ty_cost(ty),
                    ty::Infer(..) => 0,
                    _ => 1,
                }
            }
        }

        // The sources are listed in order of preference here.
        let tcx = self.infcx.tcx;
        let ctx = CostCtxt { tcx };
        let base_cost = match source.kind {
            InferSourceKind::LetBinding { ty, .. } => ctx.ty_cost(ty),
            InferSourceKind::ClosureArg { ty, .. } => ctx.ty_cost(ty),
            InferSourceKind::GenericArg { def_id, generic_args, .. } => {
                let variant_cost = match tcx.def_kind(def_id) {
                    // `None::<u32>` and friends are ugly.
                    DefKind::Variant | DefKind::Ctor(CtorOf::Variant, _) => 15,
                    _ => 10,
                };
                variant_cost + generic_args.iter().map(|&arg| ctx.arg_cost(arg)).sum::<usize>()
            }
            InferSourceKind::FullyQualifiedMethodCall { substs, .. } => {
                20 + substs.iter().map(|arg| ctx.arg_cost(arg)).sum::<usize>()
            }
            InferSourceKind::ClosureReturn { ty, should_wrap_expr, .. } => {
                30 + ctx.ty_cost(ty) + if should_wrap_expr.is_some() { 10 } else { 0 }
            }
        };

        let suggestion_may_apply = if source.from_expansion() { 10000 } else { 0 };

        base_cost + suggestion_may_apply
    }

    /// Uses `fn source_cost` to determine whether this inference source is preferable to
    /// previous sources. We generally prefer earlier sources.
    #[instrument(level = "debug", skip(self))]
    fn update_infer_source(&mut self, new_source: InferSource<'tcx>) {
        let cost = self.source_cost(&new_source) + self.attempt;
        debug!(?cost);
        self.attempt += 1;
        if cost < self.infer_source_cost {
            self.infer_source_cost = cost;
            self.infer_source = Some(new_source);
        }
    }

    fn node_substs_opt(&self, hir_id: HirId) -> Option<SubstsRef<'tcx>> {
        let substs = self.typeck_results.node_substs_opt(hir_id);
        self.infcx.resolve_vars_if_possible(substs)
    }

    fn opt_node_type(&self, hir_id: HirId) -> Option<Ty<'tcx>> {
        let ty = self.typeck_results.node_type_opt(hir_id);
        self.infcx.resolve_vars_if_possible(ty)
    }

    // Check whether this generic argument is the inference variable we
    // are looking for.
    fn generic_arg_is_target(&self, arg: GenericArg<'tcx>) -> bool {
        if arg == self.target {
            return true;
        }

        match (arg.unpack(), self.target.unpack()) {
            (GenericArgKind::Type(inner_ty), GenericArgKind::Type(target_ty)) => {
                use ty::{Infer, TyVar};
                match (inner_ty.kind(), target_ty.kind()) {
                    (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => {
                        self.infcx.inner.borrow_mut().type_variables().sub_unified(a_vid, b_vid)
                    }
                    _ => false,
                }
            }
            (GenericArgKind::Const(inner_ct), GenericArgKind::Const(target_ct)) => {
                use ty::InferConst::*;
                match (inner_ct.kind(), target_ct.kind()) {
                    (ty::ConstKind::Infer(Var(a_vid)), ty::ConstKind::Infer(Var(b_vid))) => self
                        .infcx
                        .inner
                        .borrow_mut()
                        .const_unification_table()
                        .unioned(a_vid, b_vid),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Does this generic argument contain our target inference variable
    /// in a way which can be written by the user.
    fn generic_arg_contains_target(&self, arg: GenericArg<'tcx>) -> bool {
        let mut walker = arg.walk();
        while let Some(inner) = walker.next() {
            if self.generic_arg_is_target(inner) {
                return true;
            }
            match inner.unpack() {
                GenericArgKind::Lifetime(_) => {}
                GenericArgKind::Type(ty) => {
                    if matches!(ty.kind(), ty::Opaque(..) | ty::Closure(..) | ty::Generator(..)) {
                        // Opaque types can't be named by the user right now.
                        //
                        // Both the generic arguments of closures and generators can
                        // also not be named. We may want to only look into the closure
                        // signature in case it has no captures, as that can be represented
                        // using `fn(T) -> R`.

                        // FIXME(type_alias_impl_trait): These opaque types
                        // can actually be named, so it would make sense to
                        // adjust this case and add a test for it.
                        walker.skip_current_subtree();
                    }
                }
                GenericArgKind::Const(ct) => {
                    if matches!(ct.kind(), ty::ConstKind::Unevaluated(..)) {
                        // You can't write the generic arguments for
                        // unevaluated constants.
                        walker.skip_current_subtree();
                    }
                }
            }
        }
        false
    }

    fn expr_inferred_subst_iter(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Box<dyn Iterator<Item = InsertableGenericArgs<'tcx>> + 'a> {
        let tcx = self.infcx.tcx;
        match expr.kind {
            hir::ExprKind::Path(ref path) => {
                if let Some(substs) = self.node_substs_opt(expr.hir_id) {
                    return self.path_inferred_subst_iter(expr.hir_id, substs, path);
                }
            }
            // FIXME(#98711): Ideally we would also deal with type relative
            // paths here, even if that is quite rare.
            //
            // See the `need_type_info/expr-struct-type-relative-gat.rs` test
            // for an example where that would be needed.
            //
            // However, the `type_dependent_def_id` for `Self::Output` in an
            // impl is currently the `DefId` of `Output` in the trait definition
            // which makes this somewhat difficult and prevents us from just
            // using `self.path_inferred_subst_iter` here.
            hir::ExprKind::Struct(&hir::QPath::Resolved(_self_ty, path), _, _) => {
                if let Some(ty) = self.opt_node_type(expr.hir_id) {
                    if let ty::Adt(_, substs) = ty.kind() {
                        return Box::new(self.resolved_path_inferred_subst_iter(path, substs));
                    }
                }
            }
            hir::ExprKind::MethodCall(segment, ..) => {
                if let Some(def_id) = self.typeck_results.type_dependent_def_id(expr.hir_id) {
                    let generics = tcx.generics_of(def_id);
                    let insertable: Option<_> = try {
                        if generics.has_impl_trait() {
                            None?
                        }
                        let substs = self.node_substs_opt(expr.hir_id)?;
                        let span = tcx.hir().span(segment.hir_id);
                        let insert_span = segment.ident.span.shrink_to_hi().with_hi(span.hi());
                        InsertableGenericArgs {
                            insert_span,
                            substs,
                            generics_def_id: def_id,
                            def_id,
                        }
                    };
                    return Box::new(insertable.into_iter());
                }
            }
            _ => {}
        }

        Box::new(iter::empty())
    }

    fn resolved_path_inferred_subst_iter(
        &self,
        path: &'tcx hir::Path<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> impl Iterator<Item = InsertableGenericArgs<'tcx>> + 'a {
        let tcx = self.infcx.tcx;
        // The last segment of a path often has `Res::Err` and the
        // correct `Res` is the one of the whole path.
        //
        // FIXME: We deal with that one separately for now,
        // would be good to remove this special case.
        let last_segment_using_path_data: Option<_> = try {
            let generics_def_id = tcx.res_generics_def_id(path.res)?;
            let generics = tcx.generics_of(generics_def_id);
            if generics.has_impl_trait() {
                None?
            }
            let insert_span =
                path.segments.last().unwrap().ident.span.shrink_to_hi().with_hi(path.span.hi());
            InsertableGenericArgs {
                insert_span,
                substs,
                generics_def_id,
                def_id: path.res.def_id(),
            }
        };

        path.segments
            .iter()
            .filter_map(move |segment| {
                let res = segment.res;
                let generics_def_id = tcx.res_generics_def_id(res)?;
                let generics = tcx.generics_of(generics_def_id);
                if generics.has_impl_trait() {
                    return None;
                }
                let span = tcx.hir().span(segment.hir_id);
                let insert_span = segment.ident.span.shrink_to_hi().with_hi(span.hi());
                Some(InsertableGenericArgs {
                    insert_span,
                    substs,
                    generics_def_id,
                    def_id: res.def_id(),
                })
            })
            .chain(last_segment_using_path_data)
    }

    fn path_inferred_subst_iter(
        &self,
        hir_id: HirId,
        substs: SubstsRef<'tcx>,
        qpath: &'tcx hir::QPath<'tcx>,
    ) -> Box<dyn Iterator<Item = InsertableGenericArgs<'tcx>> + 'a> {
        let tcx = self.infcx.tcx;
        match qpath {
            hir::QPath::Resolved(_self_ty, path) => {
                Box::new(self.resolved_path_inferred_subst_iter(path, substs))
            }
            hir::QPath::TypeRelative(ty, segment) => {
                let Some(def_id) = self.typeck_results.type_dependent_def_id(hir_id) else {
                    return Box::new(iter::empty());
                };

                let generics = tcx.generics_of(def_id);
                let segment: Option<_> = try {
                    if !segment.infer_args || generics.has_impl_trait() {
                        None?;
                    }
                    let span = tcx.hir().span(segment.hir_id);
                    let insert_span = segment.ident.span.shrink_to_hi().with_hi(span.hi());
                    InsertableGenericArgs { insert_span, substs, generics_def_id: def_id, def_id }
                };

                let parent_def_id = generics.parent.unwrap();
                if tcx.def_kind(parent_def_id) == DefKind::Impl {
                    let parent_ty = tcx.bound_type_of(parent_def_id).subst(tcx, substs);
                    match (parent_ty.kind(), &ty.kind) {
                        (
                            ty::Adt(def, substs),
                            hir::TyKind::Path(hir::QPath::Resolved(_self_ty, path)),
                        ) => {
                            if tcx.res_generics_def_id(path.res) != Some(def.did()) {
                                match path.res {
                                    Res::Def(DefKind::TyAlias, _) => {
                                        // FIXME: Ideally we should support this. For that
                                        // we have to map back from the self type to the
                                        // type alias though. That's difficult.
                                        //
                                        // See the `need_type_info/type-alias.rs` test for
                                        // some examples.
                                    }
                                    // There cannot be inference variables in the self type,
                                    // so there's nothing for us to do here.
                                    Res::SelfTy { .. } => {}
                                    _ => warn!(
                                        "unexpected path: def={:?} substs={:?} path={:?}",
                                        def, substs, path,
                                    ),
                                }
                            } else {
                                return Box::new(
                                    self.resolved_path_inferred_subst_iter(path, substs)
                                        .chain(segment),
                                );
                            }
                        }
                        _ => (),
                    }
                }

                Box::new(segment.into_iter())
            }
            hir::QPath::LangItem(_, _, _) => Box::new(iter::empty()),
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for FindInferSourceVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.infcx.tcx.hir()
    }

    fn visit_local(&mut self, local: &'tcx Local<'tcx>) {
        intravisit::walk_local(self, local);

        if let Some(ty) = self.opt_node_type(local.hir_id) {
            if self.generic_arg_contains_target(ty.into()) {
                match local.source {
                    LocalSource::Normal if local.ty.is_none() => {
                        self.update_infer_source(InferSource {
                            span: local.pat.span,
                            kind: InferSourceKind::LetBinding {
                                insert_span: local.pat.span.shrink_to_hi(),
                                pattern_name: local.pat.simple_ident(),
                                ty,
                            },
                        })
                    }
                    _ => {}
                }
            }
        }
    }

    /// For closures, we first visit the parameters and then the content,
    /// as we prefer those.
    fn visit_body(&mut self, body: &'tcx Body<'tcx>) {
        for param in body.params {
            debug!(
                "param: span {:?}, ty_span {:?}, pat.span {:?}",
                param.span, param.ty_span, param.pat.span
            );
            if param.ty_span != param.pat.span {
                debug!("skipping param: has explicit type");
                continue;
            }

            let Some(param_ty) = self.opt_node_type(param.hir_id) else {
                continue
            };

            if self.generic_arg_contains_target(param_ty.into()) {
                self.update_infer_source(InferSource {
                    span: param.pat.span,
                    kind: InferSourceKind::ClosureArg {
                        insert_span: param.pat.span.shrink_to_hi(),
                        ty: param_ty,
                    },
                })
            }
        }
        intravisit::walk_body(self, body);
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        let tcx = self.infcx.tcx;
        match expr.kind {
            // When encountering `func(arg)` first look into `arg` and then `func`,
            // as `arg` is "more specific".
            ExprKind::Call(func, args) => {
                for arg in args {
                    self.visit_expr(arg);
                }
                self.visit_expr(func);
            }
            _ => intravisit::walk_expr(self, expr),
        }

        for args in self.expr_inferred_subst_iter(expr) {
            debug!(?args);
            let InsertableGenericArgs { insert_span, substs, generics_def_id, def_id } = args;
            let generics = tcx.generics_of(generics_def_id);
            if let Some(argument_index) = generics
                .own_substs(substs)
                .iter()
                .position(|&arg| self.generic_arg_contains_target(arg))
            {
                let substs = self.infcx.resolve_vars_if_possible(substs);
                let generic_args = &generics.own_substs_no_defaults(tcx, substs)
                    [generics.own_counts().lifetimes..];
                let span = match expr.kind {
                    ExprKind::MethodCall(path, ..) => path.ident.span,
                    _ => expr.span,
                };

                self.update_infer_source(InferSource {
                    span,
                    kind: InferSourceKind::GenericArg {
                        insert_span,
                        argument_index,
                        generics_def_id,
                        def_id,
                        generic_args,
                    },
                });
            }
        }

        if let Some(node_ty) = self.opt_node_type(expr.hir_id) {
            if let (
                &ExprKind::Closure(&Closure { fn_decl, body, fn_decl_span, .. }),
                ty::Closure(_, substs),
            ) = (&expr.kind, node_ty.kind())
            {
                let output = substs.as_closure().sig().output().skip_binder();
                if self.generic_arg_contains_target(output.into()) {
                    let body = self.infcx.tcx.hir().body(body);
                    let should_wrap_expr = if matches!(body.value.kind, ExprKind::Block(..)) {
                        None
                    } else {
                        Some(body.value.span.shrink_to_hi())
                    };
                    self.update_infer_source(InferSource {
                        span: fn_decl_span,
                        kind: InferSourceKind::ClosureReturn {
                            ty: output,
                            data: &fn_decl.output,
                            should_wrap_expr,
                        },
                    })
                }
            }
        }

        let has_impl_trait = |def_id| {
            iter::successors(Some(tcx.generics_of(def_id)), |generics| {
                generics.parent.map(|def_id| tcx.generics_of(def_id))
            })
            .any(|generics| generics.has_impl_trait())
        };
        if let ExprKind::MethodCall(path, receiver, args, span) = expr.kind
            && let Some(substs) = self.node_substs_opt(expr.hir_id)
            && substs.iter().any(|arg| self.generic_arg_contains_target(arg))
            && let Some(def_id) = self.typeck_results.type_dependent_def_id(expr.hir_id)
            && self.infcx.tcx.trait_of_item(def_id).is_some()
            && !has_impl_trait(def_id)
        {
            let successor =
                args.get(0).map_or_else(|| (")", span.hi()), |arg| (", ", arg.span.lo()));
            let substs = self.infcx.resolve_vars_if_possible(substs);
            self.update_infer_source(InferSource {
                span: path.ident.span,
                kind: InferSourceKind::FullyQualifiedMethodCall {
                    receiver,
                    successor,
                    substs,
                    def_id,
                }
            })
        }
    }
}
