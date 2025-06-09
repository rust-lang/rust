use std::borrow::Cow;
use std::iter;
use std::path::PathBuf;

use rustc_errors::codes::*;
use rustc_errors::{Diag, IntoDiagArg};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Namespace, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Body, Closure, Expr, ExprKind, FnRetTy, HirId, LetStmt, LocalSource};
use rustc_middle::bug;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow};
use rustc_middle::ty::print::{FmtPrinter, PrettyPrinter, Print, Printer};
use rustc_middle::ty::{
    self, GenericArg, GenericArgKind, GenericArgsRef, InferConst, IsSuggestable, Term, TermKind,
    Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt, TypeckResults,
};
use rustc_span::{BytePos, DUMMY_SP, Ident, Span, sym};
use tracing::{debug, instrument, warn};

use super::nice_region_error::placeholder_error::Highlighted;
use crate::error_reporting::TypeErrCtxt;
use crate::errors::{
    AmbiguousImpl, AmbiguousReturn, AnnotationRequired, InferenceBadError,
    SourceKindMultiSuggestion, SourceKindSubdiag,
};
use crate::infer::InferCtxt;

pub enum TypeAnnotationNeeded {
    /// ```compile_fail,E0282
    /// let x;
    /// ```
    E0282,
    /// An implementation cannot be chosen unambiguously because of lack of information.
    /// ```compile_fail,E0790
    /// let _ = Default::default();
    /// ```
    E0283,
    /// ```compile_fail,E0284
    /// let mut d: u64 = 2;
    /// d = d % 1u32.into();
    /// ```
    E0284,
}

impl From<TypeAnnotationNeeded> for ErrCode {
    fn from(val: TypeAnnotationNeeded) -> Self {
        match val {
            TypeAnnotationNeeded::E0282 => E0282,
            TypeAnnotationNeeded::E0283 => E0283,
            TypeAnnotationNeeded::E0284 => E0284,
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
        if in_type.is_ty_or_numeric_infer() {
            ""
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
            prefix: tcx.def_descr(parent_def_id),
            name: parent_name,
        })
    }

    fn for_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Option<InferenceDiagnosticsParentData> {
        Self::for_parent_def_id(tcx, tcx.parent(def_id))
    }
}

impl IntoDiagArg for UnderspecifiedArgKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        let kind = match self {
            Self::Type { .. } => "type",
            Self::Const { is_parameter: true } => "const_with_param",
            Self::Const { is_parameter: false } => "const",
        };
        rustc_errors::DiagArgValue::Str(kind.into())
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

struct ClosureEraser<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> ClosureEraser<'a, 'tcx> {
    fn new_infer(&mut self) -> Ty<'tcx> {
        self.infcx.next_ty_var(DUMMY_SP)
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for ClosureEraser<'a, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match ty.kind() {
            ty::Closure(_, args) => {
                // For a closure type, we turn it into a function pointer so that it gets rendered
                // as `fn(args) -> Ret`.
                let closure_sig = args.as_closure().sig();
                Ty::new_fn_ptr(
                    self.cx(),
                    self.cx().signature_unclosure(closure_sig, hir::Safety::Safe),
                )
            }
            ty::Adt(_, args) if !args.iter().any(|a| a.has_infer()) => {
                // We have a type that doesn't have any inference variables, so we replace
                // the whole thing with `_`. The type system already knows about this type in
                // its entirety and it is redundant to specify it for the user. The user only
                // needs to specify the type parameters that we *couldn't* figure out.
                self.new_infer()
            }
            ty::Adt(def, args) => {
                let generics = self.cx().generics_of(def.did());
                let generics: Vec<bool> = generics
                    .own_params
                    .iter()
                    .map(|param| param.default_value(self.cx()).is_some())
                    .collect();
                let ty = Ty::new_adt(
                    self.cx(),
                    *def,
                    self.cx().mk_args_from_iter(generics.into_iter().zip(args.iter()).map(
                        |(has_default, arg)| {
                            if arg.has_infer() {
                                // This param has an unsubstituted type variable, meaning that this
                                // type has a (potentially deeply nested) type parameter from the
                                // corresponding type's definition. We have explicitly asked this
                                // type to not be hidden. In either case, we keep the type and don't
                                // substitute with `_` just yet.
                                arg.fold_with(self)
                            } else if has_default {
                                // We have a type param that has a default type, like the allocator
                                // in Vec. We decided to show `Vec` itself, because it hasn't yet
                                // been replaced by an `_` `Infer`, but we want to ensure that the
                                // type parameter with default types does *not* get replaced with
                                // `_` because then we'd end up with `Vec<_, _>`, instead of
                                // `Vec<_>`.
                                arg
                            } else if let GenericArgKind::Type(_) = arg.kind() {
                                // We don't replace lifetime or const params, only type params.
                                self.new_infer().into()
                            } else {
                                arg.fold_with(self)
                            }
                        },
                    )),
                );
                ty
            }
            _ if ty.has_infer() => {
                // This type has a (potentially nested) type parameter that we couldn't figure out.
                // We will print this depth of type, so at least the type name and at least one of
                // its type parameters.
                ty.super_fold_with(self)
            }
            // We don't have an unknown type parameter anywhere, replace with `_`.
            _ => self.new_infer(),
        }
    }

    fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        // Avoid accidentally erasing the type of the const.
        c
    }
}

fn fmt_printer<'a, 'tcx>(infcx: &'a InferCtxt<'tcx>, ns: Namespace) -> FmtPrinter<'a, 'tcx> {
    let mut printer = FmtPrinter::new(infcx.tcx, ns);
    let ty_getter = move |ty_vid| {
        if infcx.probe_ty_var(ty_vid).is_ok() {
            warn!("resolved ty var in error message");
        }

        let var_origin = infcx.type_var_origin(ty_vid);
        if let Some(def_id) = var_origin.param_def_id
            // The `Self` param of a trait has the def-id of the trait,
            // since it's a synthetic parameter.
            && infcx.tcx.def_kind(def_id) == DefKind::TyParam
            && let name = infcx.tcx.item_name(def_id)
            && !var_origin.span.from_expansion()
        {
            let generics = infcx.tcx.generics_of(infcx.tcx.parent(def_id));
            let idx = generics.param_def_id_to_index(infcx.tcx, def_id).unwrap();
            let generic_param_def = generics.param_at(idx as usize, infcx.tcx);
            if let ty::GenericParamDefKind::Type { synthetic: true, .. } = generic_param_def.kind {
                None
            } else {
                Some(name)
            }
        } else {
            None
        }
    };
    printer.ty_infer_name_resolver = Some(Box::new(ty_getter));
    let const_getter =
        move |ct_vid| Some(infcx.tcx.item_name(infcx.const_var_origin(ct_vid)?.param_def_id?));
    printer.const_infer_name_resolver = Some(Box::new(const_getter));
    printer
}

fn ty_to_string<'tcx>(
    infcx: &InferCtxt<'tcx>,
    ty: Ty<'tcx>,
    called_method_def_id: Option<DefId>,
) -> String {
    let mut printer = fmt_printer(infcx, Namespace::TypeNS);
    let ty = infcx.resolve_vars_if_possible(ty);
    // We use `fn` ptr syntax for closures, but this only works when the closure does not capture
    // anything. We also remove all type parameters that are fully known to the type system.
    let ty = ty.fold_with(&mut ClosureEraser { infcx });

    match (ty.kind(), called_method_def_id) {
        // We don't want the regular output for `fn`s because it includes its path in
        // invalid pseudo-syntax, we want the `fn`-pointer output instead.
        (ty::FnDef(..), _) => {
            ty.fn_sig(infcx.tcx).print(&mut printer).unwrap();
            printer.into_buffer()
        }
        (_, Some(def_id))
            if ty.is_ty_or_numeric_infer()
                && infcx.tcx.get_diagnostic_item(sym::iterator_collect_fn) == Some(def_id) =>
        {
            "Vec<_>".to_string()
        }
        _ if ty.is_ty_or_numeric_infer() => "/* Type */".to_string(),
        _ => {
            ty.print(&mut printer).unwrap();
            printer.into_buffer()
        }
    }
}

/// We don't want to directly use `ty_to_string` for closures as their type isn't really
/// something users are familiar with. Directly printing the `fn_sig` of closures also
/// doesn't work as they actually use the "rust-call" API.
fn closure_as_fn_str<'tcx>(infcx: &InferCtxt<'tcx>, ty: Ty<'tcx>) -> String {
    let ty::Closure(_, args) = ty.kind() else {
        bug!("cannot convert non-closure to fn str in `closure_as_fn_str`")
    };
    let fn_sig = args.as_closure().sig();
    let args = fn_sig
        .inputs()
        .skip_binder()
        .iter()
        .next()
        .map(|args| {
            args.tuple_fields()
                .iter()
                .map(|arg| ty_to_string(infcx, arg, None))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let ret = if fn_sig.output().skip_binder().is_unit() {
        String::new()
    } else {
        format!(" -> {}", ty_to_string(infcx, fn_sig.output().skip_binder(), None))
    };
    format!("fn({args}){ret}")
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    /// Extracts data used by diagnostic for either types or constants
    /// which were stuck during inference.
    pub fn extract_inference_diagnostics_data(
        &self,
        term: Term<'tcx>,
        highlight: ty::print::RegionHighlightMode<'tcx>,
    ) -> InferenceDiagnosticsData {
        let tcx = self.tcx;
        match term.kind() {
            TermKind::Ty(ty) => {
                if let ty::Infer(ty::TyVar(ty_vid)) = *ty.kind() {
                    let var_origin = self.infcx.type_var_origin(ty_vid);
                    if let Some(def_id) = var_origin.param_def_id
                        // The `Self` param of a trait has the def-id of the trait,
                        // since it's a synthetic parameter.
                        && self.tcx.def_kind(def_id) == DefKind::TyParam
                        && !var_origin.span.from_expansion()
                    {
                        return InferenceDiagnosticsData {
                            name: self.tcx.item_name(def_id).to_string(),
                            span: Some(var_origin.span),
                            kind: UnderspecifiedArgKind::Type { prefix: "type parameter".into() },
                            parent: InferenceDiagnosticsParentData::for_def_id(self.tcx, def_id),
                        };
                    }
                }

                InferenceDiagnosticsData {
                    name: Highlighted { highlight, ns: Namespace::TypeNS, tcx, value: ty }
                        .to_string(),
                    span: None,
                    kind: UnderspecifiedArgKind::Type { prefix: ty.prefix_string(self.tcx) },
                    parent: None,
                }
            }
            TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(InferConst::Var(vid)) = ct.kind() {
                    let origin = self.const_var_origin(vid).expect("expected unresolved const var");
                    if let Some(def_id) = origin.param_def_id {
                        return InferenceDiagnosticsData {
                            name: self.tcx.item_name(def_id).to_string(),
                            span: Some(origin.span),
                            kind: UnderspecifiedArgKind::Const { is_parameter: true },
                            parent: InferenceDiagnosticsParentData::for_def_id(self.tcx, def_id),
                        };
                    }

                    debug_assert!(!origin.span.is_dummy());
                    InferenceDiagnosticsData {
                        name: Highlighted { highlight, ns: Namespace::ValueNS, tcx, value: ct }
                            .to_string(),
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
                    InferenceDiagnosticsData {
                        name: Highlighted { highlight, ns: Namespace::ValueNS, tcx, value: ct }
                            .to_string(),
                        span: None,
                        kind: UnderspecifiedArgKind::Const { is_parameter: false },
                        parent: None,
                    }
                }
            }
        }
    }

    /// Used as a fallback in [TypeErrCtxt::emit_inference_failure_err]
    /// in case we weren't able to get a better error.
    fn bad_inference_failure_err(
        &self,
        span: Span,
        arg_data: InferenceDiagnosticsData,
        error_code: TypeAnnotationNeeded,
    ) -> Diag<'a> {
        let source_kind = "other";
        let source_name = "";
        let failure_span = None;
        let infer_subdiags = Vec::new();
        let multi_suggestions = Vec::new();
        let bad_label = Some(arg_data.make_bad_error(span));
        match error_code {
            TypeAnnotationNeeded::E0282 => self.dcx().create_err(AnnotationRequired {
                span,
                source_kind,
                source_name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label,
                was_written: false,
                path: Default::default(),
            }),
            TypeAnnotationNeeded::E0283 => self.dcx().create_err(AmbiguousImpl {
                span,
                source_kind,
                source_name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label,
                was_written: false,
                path: Default::default(),
            }),
            TypeAnnotationNeeded::E0284 => self.dcx().create_err(AmbiguousReturn {
                span,
                source_kind,
                source_name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label,
                was_written: false,
                path: Default::default(),
            }),
        }
    }

    #[instrument(level = "debug", skip(self, error_code))]
    pub fn emit_inference_failure_err(
        &self,
        body_def_id: LocalDefId,
        failure_span: Span,
        term: Term<'tcx>,
        error_code: TypeAnnotationNeeded,
        should_label_span: bool,
    ) -> Diag<'a> {
        let term = self.resolve_vars_if_possible(term);
        let arg_data = self
            .extract_inference_diagnostics_data(term, ty::print::RegionHighlightMode::default());

        let Some(typeck_results) = &self.typeck_results else {
            // If we don't have any typeck results we're outside
            // of a body, so we won't be able to get better info
            // here.
            return self.bad_inference_failure_err(failure_span, arg_data, error_code);
        };

        let mut local_visitor = FindInferSourceVisitor::new(self, typeck_results, term);
        if let Some(body) = self.tcx.hir_maybe_body_owned_by(
            self.tcx.typeck_root_def_id(body_def_id.to_def_id()).expect_local(),
        ) {
            let expr = body.value;
            local_visitor.visit_expr(expr);
        }

        let Some(InferSource { span, kind }) = local_visitor.infer_source else {
            return self.bad_inference_failure_err(failure_span, arg_data, error_code);
        };

        let (source_kind, name, path) = kind.ty_localized_msg(self);
        let failure_span = if should_label_span && !failure_span.overlaps(span) {
            Some(failure_span)
        } else {
            None
        };

        let mut infer_subdiags = Vec::new();
        let mut multi_suggestions = Vec::new();
        match kind {
            InferSourceKind::LetBinding { insert_span, pattern_name, ty, def_id } => {
                infer_subdiags.push(SourceKindSubdiag::LetLike {
                    span: insert_span,
                    name: pattern_name.map(|name| name.to_string()).unwrap_or_else(String::new),
                    x_kind: arg_data.where_x_is_kind(ty),
                    prefix_kind: arg_data.kind.clone(),
                    prefix: arg_data.kind.try_get_prefix().unwrap_or_default(),
                    arg_name: arg_data.name,
                    kind: if pattern_name.is_some() { "with_pattern" } else { "other" },
                    type_name: ty_to_string(self, ty, def_id),
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
                    type_name: ty_to_string(self, ty, None),
                });
            }
            InferSourceKind::GenericArg {
                insert_span,
                argument_index,
                generics_def_id,
                def_id: _,
                generic_args,
                have_turbofish,
            } => {
                let generics = self.tcx.generics_of(generics_def_id);
                let is_type = term.as_type().is_some();

                let (parent_exists, parent_prefix, parent_name) =
                    InferenceDiagnosticsParentData::for_parent_def_id(self.tcx, generics_def_id)
                        .map_or((false, String::new(), String::new()), |parent| {
                            (true, parent.prefix.to_string(), parent.name)
                        });

                infer_subdiags.push(SourceKindSubdiag::GenericLabel {
                    span,
                    is_type,
                    param_name: generics.own_params[argument_index].name.to_string(),
                    parent_exists,
                    parent_prefix,
                    parent_name,
                });

                let args = if self.tcx.get_diagnostic_item(sym::iterator_collect_fn)
                    == Some(generics_def_id)
                {
                    "Vec<_>".to_string()
                } else {
                    let mut printer = fmt_printer(self, Namespace::TypeNS);
                    printer
                        .comma_sep(generic_args.iter().copied().map(|arg| {
                            if arg.is_suggestable(self.tcx, true) {
                                return arg;
                            }

                            match arg.kind() {
                                GenericArgKind::Lifetime(_) => bug!("unexpected lifetime"),
                                GenericArgKind::Type(_) => self.next_ty_var(DUMMY_SP).into(),
                                GenericArgKind::Const(_) => self.next_const_var(DUMMY_SP).into(),
                            }
                        }))
                        .unwrap();
                    printer.into_buffer()
                };

                if !have_turbofish {
                    infer_subdiags.push(SourceKindSubdiag::GenericSuggestion {
                        span: insert_span,
                        arg_count: generic_args.len(),
                        args,
                    });
                }
            }
            InferSourceKind::FullyQualifiedMethodCall { receiver, successor, args, def_id } => {
                let placeholder = Some(self.next_ty_var(DUMMY_SP));
                if let Some(args) = args.make_suggestable(self.infcx.tcx, true, placeholder) {
                    let mut printer = fmt_printer(self, Namespace::ValueNS);
                    printer.print_def_path(def_id, args).unwrap();
                    let def_path = printer.into_buffer();

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
                            Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(mut_)), target: _ },
                        ] => hir::Mutability::from(*mut_).ref_prefix_str(),
                        _ => "",
                    };

                    multi_suggestions.push(SourceKindMultiSuggestion::new_fully_qualified(
                        receiver.span,
                        def_path,
                        adjustment,
                        successor,
                    ));
                }
            }
            InferSourceKind::ClosureReturn { ty, data, should_wrap_expr } => {
                let placeholder = Some(self.next_ty_var(DUMMY_SP));
                if let Some(ty) = ty.make_suggestable(self.infcx.tcx, true, placeholder) {
                    let ty_info = ty_to_string(self, ty, None);
                    multi_suggestions.push(SourceKindMultiSuggestion::new_closure_return(
                        ty_info,
                        data,
                        should_wrap_expr,
                    ));
                }
            }
        }
        match error_code {
            TypeAnnotationNeeded::E0282 => self.dcx().create_err(AnnotationRequired {
                span,
                source_kind,
                source_name: &name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label: None,
                was_written: path.is_some(),
                path: path.unwrap_or_default(),
            }),
            TypeAnnotationNeeded::E0283 => self.dcx().create_err(AmbiguousImpl {
                span,
                source_kind,
                source_name: &name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label: None,
                was_written: path.is_some(),
                path: path.unwrap_or_default(),
            }),
            TypeAnnotationNeeded::E0284 => self.dcx().create_err(AmbiguousReturn {
                span,
                source_kind,
                source_name: &name,
                failure_span,
                infer_subdiags,
                multi_suggestions,
                bad_label: None,
                was_written: path.is_some(),
                path: path.unwrap_or_default(),
            }),
        }
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
        def_id: Option<DefId>,
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
        have_turbofish: bool,
    },
    FullyQualifiedMethodCall {
        receiver: &'tcx Expr<'tcx>,
        /// If the method has other arguments, this is ", " and the start of the first argument,
        /// while for methods without arguments this is ")" and the end of the method call.
        successor: (&'static str, BytePos),
        args: GenericArgsRef<'tcx>,
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
                data.span().from_expansion() || should_wrap_expr.is_some_and(Span::from_expansion)
            }
        };
        source_from_expansion || self.span.from_expansion()
    }
}

impl<'tcx> InferSourceKind<'tcx> {
    fn ty_localized_msg(&self, infcx: &InferCtxt<'tcx>) -> (&'static str, String, Option<PathBuf>) {
        let mut path = None;
        match *self {
            InferSourceKind::LetBinding { ty, .. }
            | InferSourceKind::ClosureArg { ty, .. }
            | InferSourceKind::ClosureReturn { ty, .. } => {
                if ty.is_closure() {
                    ("closure", closure_as_fn_str(infcx, ty), path)
                } else if !ty.is_ty_or_numeric_infer() {
                    ("normal", infcx.tcx.short_string(ty, &mut path), path)
                } else {
                    ("other", String::new(), path)
                }
            }
            // FIXME: We should be able to add some additional info here.
            InferSourceKind::GenericArg { .. }
            | InferSourceKind::FullyQualifiedMethodCall { .. } => ("other", String::new(), path),
        }
    }
}

#[derive(Debug)]
struct InsertableGenericArgs<'tcx> {
    insert_span: Span,
    args: GenericArgsRef<'tcx>,
    generics_def_id: DefId,
    def_id: DefId,
    have_turbofish: bool,
}

/// A visitor which searches for the "best" spot to use in the inference error.
///
/// For this it walks over the hir body and tries to check all places where
/// inference variables could be bound.
///
/// While doing so, the currently best spot is stored in `infer_source`.
/// For details on how we rank spots, see [Self::source_cost]
struct FindInferSourceVisitor<'a, 'tcx> {
    tecx: &'a TypeErrCtxt<'a, 'tcx>,
    typeck_results: &'a TypeckResults<'tcx>,

    target: Term<'tcx>,

    attempt: usize,
    infer_source_cost: usize,
    infer_source: Option<InferSource<'tcx>>,
}

impl<'a, 'tcx> FindInferSourceVisitor<'a, 'tcx> {
    fn new(
        tecx: &'a TypeErrCtxt<'a, 'tcx>,
        typeck_results: &'a TypeckResults<'tcx>,
        target: Term<'tcx>,
    ) -> Self {
        FindInferSourceVisitor {
            tecx,
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
                match arg.kind() {
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
                    ty::Adt(def, args) => {
                        5 + self
                            .tcx
                            .generics_of(def.did())
                            .own_args_no_defaults(self.tcx, args)
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
        let tcx = self.tecx.tcx;
        let ctx = CostCtxt { tcx };
        match source.kind {
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
            InferSourceKind::FullyQualifiedMethodCall { args, .. } => {
                20 + args.iter().map(|arg| ctx.arg_cost(arg)).sum::<usize>()
            }
            InferSourceKind::ClosureReturn { ty, should_wrap_expr, .. } => {
                30 + ctx.ty_cost(ty) + if should_wrap_expr.is_some() { 10 } else { 0 }
            }
        }
    }

    /// Uses `fn source_cost` to determine whether this inference source is preferable to
    /// previous sources. We generally prefer earlier sources.
    #[instrument(level = "debug", skip(self))]
    fn update_infer_source(&mut self, mut new_source: InferSource<'tcx>) {
        if new_source.from_expansion() {
            return;
        }

        let cost = self.source_cost(&new_source) + self.attempt;
        debug!(?cost);
        self.attempt += 1;
        if let Some(InferSource { kind: InferSourceKind::GenericArg { def_id: did, .. }, .. }) =
            self.infer_source
            && let InferSourceKind::LetBinding { ref ty, ref mut def_id, .. } = new_source.kind
            && ty.is_ty_or_numeric_infer()
        {
            // Customize the output so we talk about `let x: Vec<_> = iter.collect();` instead of
            // `let x: _ = iter.collect();`, as this is a very common case.
            *def_id = Some(did);
        }

        if cost < self.infer_source_cost {
            self.infer_source_cost = cost;
            self.infer_source = Some(new_source);
        }
    }

    fn node_args_opt(&self, hir_id: HirId) -> Option<GenericArgsRef<'tcx>> {
        let args = self.typeck_results.node_args_opt(hir_id);
        self.tecx.resolve_vars_if_possible(args)
    }

    fn opt_node_type(&self, hir_id: HirId) -> Option<Ty<'tcx>> {
        let ty = self.typeck_results.node_type_opt(hir_id);
        self.tecx.resolve_vars_if_possible(ty)
    }

    // Check whether this generic argument is the inference variable we
    // are looking for.
    fn generic_arg_is_target(&self, arg: GenericArg<'tcx>) -> bool {
        if arg == self.target.into() {
            return true;
        }

        match (arg.kind(), self.target.kind()) {
            (GenericArgKind::Type(inner_ty), TermKind::Ty(target_ty)) => {
                use ty::{Infer, TyVar};
                match (inner_ty.kind(), target_ty.kind()) {
                    (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => {
                        self.tecx.sub_relations.borrow_mut().unified(self.tecx, a_vid, b_vid)
                    }
                    _ => false,
                }
            }
            (GenericArgKind::Const(inner_ct), TermKind::Const(target_ct)) => {
                use ty::InferConst::*;
                match (inner_ct.kind(), target_ct.kind()) {
                    (ty::ConstKind::Infer(Var(a_vid)), ty::ConstKind::Infer(Var(b_vid))) => {
                        self.tecx.root_const_var(a_vid) == self.tecx.root_const_var(b_vid)
                    }
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
            match inner.kind() {
                GenericArgKind::Lifetime(_) => {}
                GenericArgKind::Type(ty) => {
                    if matches!(
                        ty.kind(),
                        ty::Alias(ty::Opaque, ..)
                            | ty::Closure(..)
                            | ty::CoroutineClosure(..)
                            | ty::Coroutine(..)
                    ) {
                        // Opaque types can't be named by the user right now.
                        //
                        // Both the generic arguments of closures and coroutines can
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

    fn expr_inferred_arg_iter(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Box<dyn Iterator<Item = InsertableGenericArgs<'tcx>> + 'a> {
        let tcx = self.tecx.tcx;
        match expr.kind {
            hir::ExprKind::Path(ref path) => {
                if let Some(args) = self.node_args_opt(expr.hir_id) {
                    return self.path_inferred_arg_iter(expr.hir_id, args, path);
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
            // using `self.path_inferred_arg_iter` here.
            hir::ExprKind::Struct(&hir::QPath::Resolved(_self_ty, path), _, _)
            // FIXME(TaKO8Ki): Ideally we should support other kinds,
            // such as `TyAlias` or `AssocTy`. For that we have to map
            // back from the self type to the type alias though. That's difficult.
            //
            // See the `need_type_info/issue-103053.rs` test for
            // a example.
            if matches!(path.res, Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, _)) => {
                if let Some(ty) = self.opt_node_type(expr.hir_id)
                    && let ty::Adt(_, args) = ty.kind()
                {
                    return Box::new(self.resolved_path_inferred_arg_iter(path, args));
                }
            }
            hir::ExprKind::MethodCall(segment, ..) => {
                if let Some(def_id) = self.typeck_results.type_dependent_def_id(expr.hir_id) {
                    let generics = tcx.generics_of(def_id);
                    let insertable: Option<_> = try {
                        if generics.has_impl_trait() {
                            None?
                        }
                        let args = self.node_args_opt(expr.hir_id)?;
                        let span = tcx.hir_span(segment.hir_id);
                        let insert_span = segment.ident.span.shrink_to_hi().with_hi(span.hi());
                        InsertableGenericArgs {
                            insert_span,
                            args,
                            generics_def_id: def_id,
                            def_id,
                            have_turbofish: false,
                        }
                    };
                    return Box::new(insertable.into_iter());
                }
            }
            _ => {}
        }

        Box::new(iter::empty())
    }

    fn resolved_path_inferred_arg_iter(
        &self,
        path: &'tcx hir::Path<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> impl Iterator<Item = InsertableGenericArgs<'tcx>> + 'tcx {
        let tcx = self.tecx.tcx;
        let have_turbofish = path.segments.iter().any(|segment| {
            segment.args.is_some_and(|args| args.args.iter().any(|arg| arg.is_ty_or_const()))
        });
        // The last segment of a path often has `Res::Err` and the
        // correct `Res` is the one of the whole path.
        //
        // FIXME: We deal with that one separately for now,
        // would be good to remove this special case.
        let last_segment_using_path_data: Option<_> = try {
            let generics_def_id = tcx.res_generics_def_id(path.res)?;
            let generics = tcx.generics_of(generics_def_id);
            if generics.has_impl_trait() {
                do yeet ();
            }
            let insert_span =
                path.segments.last().unwrap().ident.span.shrink_to_hi().with_hi(path.span.hi());
            InsertableGenericArgs {
                insert_span,
                args,
                generics_def_id,
                def_id: path.res.def_id(),
                have_turbofish,
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
                let span = tcx.hir_span(segment.hir_id);
                let insert_span = segment.ident.span.shrink_to_hi().with_hi(span.hi());
                Some(InsertableGenericArgs {
                    insert_span,
                    args,
                    generics_def_id,
                    def_id: res.def_id(),
                    have_turbofish,
                })
            })
            .chain(last_segment_using_path_data)
    }

    fn path_inferred_arg_iter(
        &self,
        hir_id: HirId,
        args: GenericArgsRef<'tcx>,
        qpath: &'tcx hir::QPath<'tcx>,
    ) -> Box<dyn Iterator<Item = InsertableGenericArgs<'tcx>> + 'a> {
        let tcx = self.tecx.tcx;
        match qpath {
            hir::QPath::Resolved(_self_ty, path) => {
                Box::new(self.resolved_path_inferred_arg_iter(path, args))
            }
            hir::QPath::TypeRelative(ty, segment) => {
                let Some(def_id) = self.typeck_results.type_dependent_def_id(hir_id) else {
                    return Box::new(iter::empty());
                };

                let generics = tcx.generics_of(def_id);
                let segment: Option<_> = try {
                    if !segment.infer_args || generics.has_impl_trait() {
                        do yeet ();
                    }
                    let span = tcx.hir_span(segment.hir_id);
                    let insert_span = segment.ident.span.shrink_to_hi().with_hi(span.hi());
                    InsertableGenericArgs {
                        insert_span,
                        args,
                        generics_def_id: def_id,
                        def_id,
                        have_turbofish: false,
                    }
                };

                let parent_def_id = generics.parent.unwrap();
                if let DefKind::Impl { .. } = tcx.def_kind(parent_def_id) {
                    let parent_ty = tcx.type_of(parent_def_id).instantiate(tcx, args);
                    match (parent_ty.kind(), &ty.kind) {
                        (
                            ty::Adt(def, args),
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
                                    Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } => {}
                                    _ => warn!(
                                        "unexpected path: def={:?} args={:?} path={:?}",
                                        def, args, path,
                                    ),
                                }
                            } else {
                                return Box::new(
                                    self.resolved_path_inferred_arg_iter(path, args).chain(segment),
                                );
                            }
                        }
                        _ => (),
                    }
                }

                Box::new(segment.into_iter())
            }
            hir::QPath::LangItem(_, _) => Box::new(iter::empty()),
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for FindInferSourceVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tecx.tcx
    }

    fn visit_local(&mut self, local: &'tcx LetStmt<'tcx>) {
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
                                def_id: None,
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
    fn visit_body(&mut self, body: &Body<'tcx>) {
        for param in body.params {
            debug!(
                "param: span {:?}, ty_span {:?}, pat.span {:?}",
                param.span, param.ty_span, param.pat.span
            );
            if param.ty_span != param.pat.span {
                debug!("skipping param: has explicit type");
                continue;
            }

            let Some(param_ty) = self.opt_node_type(param.hir_id) else { continue };

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
        let tcx = self.tecx.tcx;
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

        for args in self.expr_inferred_arg_iter(expr) {
            debug!(?args);
            let InsertableGenericArgs {
                insert_span,
                args,
                generics_def_id,
                def_id,
                have_turbofish,
            } = args;
            let generics = tcx.generics_of(generics_def_id);
            if let Some(mut argument_index) = generics
                .own_args(args)
                .iter()
                .position(|&arg| self.generic_arg_contains_target(arg))
            {
                if generics.parent.is_none() && generics.has_self {
                    argument_index += 1;
                }
                let args = self.tecx.resolve_vars_if_possible(args);
                let generic_args =
                    &generics.own_args_no_defaults(tcx, args)[generics.own_counts().lifetimes..];
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
                        have_turbofish,
                    },
                });
            }
        }

        if let Some(node_ty) = self.opt_node_type(expr.hir_id) {
            if let (
                &ExprKind::Closure(&Closure { fn_decl, body, fn_decl_span, .. }),
                ty::Closure(_, args),
            ) = (&expr.kind, node_ty.kind())
            {
                let output = args.as_closure().sig().output().skip_binder();
                if self.generic_arg_contains_target(output.into()) {
                    let body = self.tecx.tcx.hir_body(body);
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
        if let ExprKind::MethodCall(path, receiver, method_args, span) = expr.kind
            && let Some(args) = self.node_args_opt(expr.hir_id)
            && args.iter().any(|arg| self.generic_arg_contains_target(arg))
            && let Some(def_id) = self.typeck_results.type_dependent_def_id(expr.hir_id)
            && self.tecx.tcx.trait_of_item(def_id).is_some()
            && !has_impl_trait(def_id)
            // FIXME(fn_delegation): In delegation item argument spans are equal to last path
            // segment. This leads to ICE's when emitting `multipart_suggestion`.
            && tcx.hir_opt_delegation_sig_id(expr.hir_id.owner.def_id).is_none()
        {
            let successor =
                method_args.get(0).map_or_else(|| (")", span.hi()), |arg| (", ", arg.span.lo()));
            let args = self.tecx.resolve_vars_if_possible(args);
            self.update_infer_source(InferSource {
                span: path.ident.span,
                kind: InferSourceKind::FullyQualifiedMethodCall {
                    receiver,
                    successor,
                    args,
                    def_id,
                },
            })
        }
    }
}
