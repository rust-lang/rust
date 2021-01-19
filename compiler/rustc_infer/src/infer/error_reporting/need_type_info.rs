use crate::infer::type_variable::TypeVariableOriginKind;
use crate::infer::InferCtxt;
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Body, Expr, ExprKind, FnRetTy, HirId, Local, Pat};
use rustc_middle::hir::map::Map;
use rustc_middle::infer::unify_key::ConstVariableOriginKind;
use rustc_middle::ty::print::Print;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind};
use rustc_middle::ty::{self, DefIdTree, InferConst, Ty, TyCtxt};
use rustc_span::source_map::DesugaringKind;
use rustc_span::symbol::kw;
use rustc_span::Span;
use std::borrow::Cow;

struct FindHirNodeVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    target: GenericArg<'tcx>,
    target_span: Span,
    found_node_ty: Option<Ty<'tcx>>,
    found_local_pattern: Option<&'tcx Pat<'tcx>>,
    found_arg_pattern: Option<&'tcx Pat<'tcx>>,
    found_closure: Option<&'tcx Expr<'tcx>>,
    found_method_call: Option<&'tcx Expr<'tcx>>,
    found_exact_method_call: Option<&'tcx Expr<'tcx>>,
    found_use_diagnostic: Option<UseDiagnostic<'tcx>>,
}

impl<'a, 'tcx> FindHirNodeVisitor<'a, 'tcx> {
    fn new(infcx: &'a InferCtxt<'a, 'tcx>, target: GenericArg<'tcx>, target_span: Span) -> Self {
        Self {
            infcx,
            target,
            target_span,
            found_node_ty: None,
            found_local_pattern: None,
            found_arg_pattern: None,
            found_closure: None,
            found_method_call: None,
            found_exact_method_call: None,
            found_use_diagnostic: None,
        }
    }

    fn node_type_opt(&self, hir_id: HirId) -> Option<Ty<'tcx>> {
        self.infcx.in_progress_typeck_results?.borrow().node_type_opt(hir_id)
    }

    fn node_ty_contains_target(&self, hir_id: HirId) -> Option<Ty<'tcx>> {
        self.node_type_opt(hir_id).map(|ty| self.infcx.resolve_vars_if_possible(ty)).filter(|ty| {
            ty.walk().any(|inner| {
                inner == self.target
                    || match (inner.unpack(), self.target.unpack()) {
                        (GenericArgKind::Type(inner_ty), GenericArgKind::Type(target_ty)) => {
                            use ty::{Infer, TyVar};
                            match (inner_ty.kind(), target_ty.kind()) {
                                (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => self
                                    .infcx
                                    .inner
                                    .borrow_mut()
                                    .type_variables()
                                    .sub_unified(a_vid, b_vid),
                                _ => false,
                            }
                        }
                        _ => false,
                    }
            })
        })
    }

    /// Determine whether the expression, assumed to be the callee within a `Call`,
    /// corresponds to the `From::from` emitted in desugaring of the `?` operator.
    fn is_try_conversion(&self, callee: &Expr<'tcx>) -> bool {
        self.infcx
            .trait_def_from_hir_fn(callee.hir_id)
            .map_or(false, |def_id| self.infcx.is_try_conversion(callee.span, def_id))
    }
}

impl<'a, 'tcx> Visitor<'tcx> for FindHirNodeVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.infcx.tcx.hir())
    }

    fn visit_local(&mut self, local: &'tcx Local<'tcx>) {
        if let (None, Some(ty)) =
            (self.found_local_pattern, self.node_ty_contains_target(local.hir_id))
        {
            self.found_local_pattern = Some(&*local.pat);
            self.found_node_ty = Some(ty);
        }
        intravisit::walk_local(self, local);
    }

    fn visit_body(&mut self, body: &'tcx Body<'tcx>) {
        for param in body.params {
            if let (None, Some(ty)) =
                (self.found_arg_pattern, self.node_ty_contains_target(param.hir_id))
            {
                self.found_arg_pattern = Some(&*param.pat);
                self.found_node_ty = Some(ty);
            }
        }
        intravisit::walk_body(self, body);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::MethodCall(_, call_span, exprs, _) = expr.kind {
            if call_span == self.target_span
                && Some(self.target)
                    == self.infcx.in_progress_typeck_results.and_then(|typeck_results| {
                        typeck_results
                            .borrow()
                            .node_type_opt(exprs.first().unwrap().hir_id)
                            .map(Into::into)
                    })
            {
                self.found_exact_method_call = Some(&expr);
                return;
            }
        }

        // FIXME(const_generics): Currently, any uninferred `const` generics arguments
        // are handled specially, but instead they should be handled in `annotate_method_call`,
        // which currently doesn't work because this evaluates to `false` for const arguments.
        // See https://github.com/rust-lang/rust/pull/77758 for more details.
        if let Some(ty) = self.node_ty_contains_target(expr.hir_id) {
            match expr.kind {
                ExprKind::Closure(..) => self.found_closure = Some(&expr),
                ExprKind::MethodCall(..) => self.found_method_call = Some(&expr),

                // If the given expression falls within the target span and is a
                // `From::from(e)` call emitted during desugaring of the `?` operator,
                // extract the types inferred before and after the call
                ExprKind::Call(callee, [arg])
                    if self.target_span.contains(expr.span)
                        && self.found_use_diagnostic.is_none()
                        && self.is_try_conversion(callee) =>
                {
                    self.found_use_diagnostic = self.node_type_opt(arg.hir_id).map(|pre_ty| {
                        UseDiagnostic::TryConversion { pre_ty, post_ty: ty, span: callee.span }
                    });
                }
                _ => {}
            }
        }
        intravisit::walk_expr(self, expr);
    }
}

/// An observation about the use site of a type to be emitted as an additional
/// note in an inference failure error.
enum UseDiagnostic<'tcx> {
    /// Records the types inferred before and after `From::from` is called on the
    /// error value within the desugaring of the `?` operator.
    TryConversion { pre_ty: Ty<'tcx>, post_ty: Ty<'tcx>, span: Span },
}

impl UseDiagnostic<'_> {
    /// Return a descriptor of the value at the use site
    fn descr(&self) -> &'static str {
        match self {
            Self::TryConversion { .. } => "error for `?` operator",
        }
    }

    /// Return a descriptor of the type at the use site
    fn type_descr(&self) -> &'static str {
        match self {
            Self::TryConversion { .. } => "error type for `?` operator",
        }
    }

    fn applies_to(&self, span: Span) -> bool {
        match *self {
            // In some cases the span for an inference failure due to try
            // conversion contains the antecedent expression as well as the `?`
            Self::TryConversion { span: s, .. } => span.contains(s) && span.hi() == s.hi(),
        }
    }

    fn attach_note(&self, err: &mut DiagnosticBuilder<'_>) {
        match *self {
            Self::TryConversion { pre_ty, post_ty, .. } => {
                let intro = "`?` implicitly converts the error value";

                let msg = match (pre_ty.is_ty_infer(), post_ty.is_ty_infer()) {
                    (true, true) => format!("{} using the `From` trait", intro),
                    (false, true) => {
                        format!("{} into a type implementing `From<{}>`", intro, pre_ty)
                    }
                    (true, false) => {
                        format!("{} into `{}` using the `From` trait", intro, post_ty)
                    }
                    (false, false) => {
                        format!(
                            "{} into `{}` using its implementation of `From<{}>`",
                            intro, post_ty, pre_ty
                        )
                    }
                };

                err.note(&msg);
            }
        }
    }
}

/// Suggest giving an appropriate return type to a closure expression.
fn closure_return_type_suggestion(
    err: &mut DiagnosticBuilder<'_>,
    output: &FnRetTy<'_>,
    body: &Body<'_>,
    ret: &str,
) {
    let (arrow, post) = match output {
        FnRetTy::DefaultReturn(_) => ("-> ", " "),
        _ => ("", ""),
    };
    let suggestion = match body.value.kind {
        ExprKind::Block(..) => vec![(output.span(), format!("{}{}{}", arrow, ret, post))],
        _ => vec![
            (output.span(), format!("{}{}{}{{ ", arrow, ret, post)),
            (body.value.span.shrink_to_hi(), " }".to_string()),
        ],
    };
    err.multipart_suggestion(
        "give this closure an explicit return type without `_` placeholders",
        suggestion,
        Applicability::HasPlaceholders,
    );
}

/// Given a closure signature, return a `String` containing a list of all its argument types.
fn closure_args(fn_sig: &ty::PolyFnSig<'_>) -> String {
    fn_sig
        .inputs()
        .skip_binder()
        .iter()
        .next()
        .map(|args| args.tuple_fields().map(|arg| arg.to_string()).collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}

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
}

pub enum UnderspecifiedArgKind {
    Type { prefix: Cow<'static, str> },
    Const { is_parameter: bool },
}

impl InferenceDiagnosticsData {
    /// Generate a label for a generic argument which can't be inferred. When not
    /// much is known about the argument, `use_diag` may be used to describe the
    /// labeled value.
    fn cannot_infer_msg(&self, use_diag: Option<&UseDiagnostic<'_>>) -> String {
        if self.name == "_" && matches!(self.kind, UnderspecifiedArgKind::Type { .. }) {
            if let Some(use_diag) = use_diag {
                return format!("cannot infer type of {}", use_diag.descr());
            }

            return "cannot infer type".to_string();
        }

        let suffix = match (&self.parent, use_diag) {
            (Some(parent), _) => format!(" declared on the {} `{}`", parent.prefix, parent.name),
            (None, Some(use_diag)) => format!(" in {}", use_diag.type_descr()),
            (None, None) => String::new(),
        };

        // For example: "cannot infer type for type parameter `T`"
        format!("cannot infer {} `{}`{}", self.kind.prefix_string(), self.name, suffix)
    }
}

impl InferenceDiagnosticsParentData {
    fn for_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Option<InferenceDiagnosticsParentData> {
        let parent_def_id = tcx.parent(def_id)?;

        let parent_name =
            tcx.def_key(parent_def_id).disambiguated_data.data.get_opt_name()?.to_string();

        Some(InferenceDiagnosticsParentData {
            prefix: tcx.def_kind(parent_def_id).descr(parent_def_id),
            name: parent_name,
        })
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
        highlight: Option<ty::print::RegionHighlightMode>,
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

                let mut s = String::new();
                let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
                if let Some(highlight) = highlight {
                    printer.region_highlight_mode = highlight;
                }
                let _ = ty.print(printer);
                InferenceDiagnosticsData {
                    name: s,
                    span: None,
                    kind: UnderspecifiedArgKind::Type { prefix: ty.prefix_string() },
                    parent: None,
                }
            }
            GenericArgKind::Const(ct) => {
                if let ty::ConstKind::Infer(InferConst::Var(vid)) = ct.val {
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
                    let mut s = String::new();
                    let mut printer =
                        ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::ValueNS);
                    if let Some(highlight) = highlight {
                        printer.region_highlight_mode = highlight;
                    }
                    let _ = ct.print(printer);
                    InferenceDiagnosticsData {
                        name: s,
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
    ) -> DiagnosticBuilder<'tcx> {
        let arg = self.resolve_vars_if_possible(arg);
        let arg_data = self.extract_inference_diagnostics_data(arg, None);

        let mut local_visitor = FindHirNodeVisitor::new(&self, arg, span);
        let ty_to_string = |ty: Ty<'tcx>| -> String {
            let mut s = String::new();
            let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
            let mut inner = self.inner.borrow_mut();
            let ty_vars = inner.type_variables();
            let getter = move |ty_vid| {
                let var_origin = ty_vars.var_origin(ty_vid);
                if let TypeVariableOriginKind::TypeParameterDefinition(name, _) = var_origin.kind {
                    return Some(name.to_string());
                }
                None
            };
            printer.name_resolver = Some(Box::new(&getter));
            let _ = if let ty::FnDef(..) = ty.kind() {
                // We don't want the regular output for `fn`s because it includes its path in
                // invalid pseudo-syntax, we want the `fn`-pointer output instead.
                ty.fn_sig(self.tcx).print(printer)
            } else {
                ty.print(printer)
            };
            s
        };

        if let Some(body_id) = body_id {
            let expr = self.tcx.hir().expect_expr(body_id.hir_id);
            local_visitor.visit_expr(expr);
        }
        let err_span = if let Some(pattern) = local_visitor.found_arg_pattern {
            pattern.span
        } else if let Some(span) = arg_data.span {
            // `span` here lets us point at `sum` instead of the entire right hand side expr:
            // error[E0282]: type annotations needed
            //  --> file2.rs:3:15
            //   |
            // 3 |     let _ = x.sum() as f64;
            //   |               ^^^ cannot infer type for `S`
            span
        } else if let Some(ExprKind::MethodCall(_, call_span, _, _)) =
            local_visitor.found_method_call.map(|e| &e.kind)
        {
            // Point at the call instead of the whole expression:
            // error[E0284]: type annotations needed
            //  --> file.rs:2:5
            //   |
            // 2 |     vec![Ok(2)].into_iter().collect()?;
            //   |                             ^^^^^^^ cannot infer type
            //   |
            //   = note: cannot resolve `<_ as std::ops::Try>::Ok == _`
            if span.contains(*call_span) { *call_span } else { span }
        } else {
            span
        };

        let is_named_and_not_impl_trait = |ty: Ty<'_>| {
            &ty.to_string() != "_" &&
                // FIXME: Remove this check after `impl_trait_in_bindings` is stabilized. #63527
                (!ty.is_impl_trait() || self.tcx.features().impl_trait_in_bindings)
        };

        let ty_msg = match (local_visitor.found_node_ty, local_visitor.found_exact_method_call) {
            (_, Some(_)) => String::new(),
            (Some(ty), _) if ty.is_closure() => {
                let substs =
                    if let ty::Closure(_, substs) = *ty.kind() { substs } else { unreachable!() };
                let fn_sig = substs.as_closure().sig();
                let args = closure_args(&fn_sig);
                let ret = fn_sig.output().skip_binder().to_string();
                format!(" for the closure `fn({}) -> {}`", args, ret)
            }
            (Some(ty), _) if is_named_and_not_impl_trait(ty) => {
                let ty = ty_to_string(ty);
                format!(" for `{}`", ty)
            }
            _ => String::new(),
        };

        // When `arg_data.name` corresponds to a type argument, show the path of the full type we're
        // trying to infer. In the following example, `ty_msg` contains
        // " for `std::result::Result<i32, E>`":
        // ```
        // error[E0282]: type annotations needed for `std::result::Result<i32, E>`
        //  --> file.rs:L:CC
        //   |
        // L |     let b = Ok(4);
        //   |         -   ^^ cannot infer type for `E` in `std::result::Result<i32, E>`
        //   |         |
        //   |         consider giving `b` the explicit type `std::result::Result<i32, E>`, where
        //   |         the type parameter `E` is specified
        // ```
        let error_code = error_code.into();
        let mut err = self.tcx.sess.struct_span_err_with_code(
            err_span,
            &format!("type annotations needed{}", ty_msg),
            error_code,
        );

        let use_diag = local_visitor.found_use_diagnostic.as_ref();
        if let Some(use_diag) = use_diag {
            if use_diag.applies_to(err_span) {
                use_diag.attach_note(&mut err);
            }
        }

        let suffix = match local_visitor.found_node_ty {
            Some(ty) if ty.is_closure() => {
                let substs =
                    if let ty::Closure(_, substs) = *ty.kind() { substs } else { unreachable!() };
                let fn_sig = substs.as_closure().sig();
                let ret = fn_sig.output().skip_binder().to_string();

                let closure_decl_and_body_id =
                    local_visitor.found_closure.and_then(|closure| match &closure.kind {
                        ExprKind::Closure(_, decl, body_id, ..) => Some((decl, *body_id)),
                        _ => None,
                    });

                if let Some((decl, body_id)) = closure_decl_and_body_id {
                    closure_return_type_suggestion(
                        &mut err,
                        &decl.output,
                        self.tcx.hir().body(body_id),
                        &ret,
                    );
                    // We don't want to give the other suggestions when the problem is the
                    // closure return type.
                    err.span_label(
                        span,
                        arg_data.cannot_infer_msg(use_diag.filter(|d| d.applies_to(span))),
                    );
                    return err;
                }

                // This shouldn't be reachable, but just in case we leave a reasonable fallback.
                let args = closure_args(&fn_sig);
                // This suggestion is incomplete, as the user will get further type inference
                // errors due to the `_` placeholders and the introduction of `Box`, but it does
                // nudge them in the right direction.
                format!("a boxed closure type like `Box<dyn Fn({}) -> {}>`", args, ret)
            }
            Some(ty) if is_named_and_not_impl_trait(ty) && arg_data.name == "_" => {
                let ty = ty_to_string(ty);
                format!("the explicit type `{}`, with the type parameters specified", ty)
            }
            Some(ty) if is_named_and_not_impl_trait(ty) && ty.to_string() != arg_data.name => {
                let ty = ty_to_string(ty);
                format!(
                    "the explicit type `{}`, where the type parameter `{}` is specified",
                    ty, arg_data.name,
                )
            }
            _ => "a type".to_string(),
        };

        if let Some(e) = local_visitor.found_exact_method_call {
            if let ExprKind::MethodCall(segment, ..) = &e.kind {
                // Suggest specifying type params or point out the return type of the call:
                //
                // error[E0282]: type annotations needed
                //   --> $DIR/type-annotations-needed-expr.rs:2:39
                //    |
                // LL |     let _ = x.into_iter().sum() as f64;
                //    |                           ^^^
                //    |                           |
                //    |                           cannot infer type for `S`
                //    |                           help: consider specifying the type argument in
                //    |                           the method call: `sum::<S>`
                //    |
                //    = note: type must be known at this point
                //
                // or
                //
                // error[E0282]: type annotations needed
                //   --> $DIR/issue-65611.rs:59:20
                //    |
                // LL |     let x = buffer.last().unwrap().0.clone();
                //    |             -------^^^^--
                //    |             |      |
                //    |             |      cannot infer type for `T`
                //    |             this method call resolves to `std::option::Option<&T>`
                //    |
                //    = note: type must be known at this point
                self.annotate_method_call(segment, e, &mut err);
            }
        } else if let Some(pattern) = local_visitor.found_arg_pattern {
            // We don't want to show the default label for closures.
            //
            // So, before clearing, the output would look something like this:
            // ```
            // let x = |_| {  };
            //          -  ^^^^ cannot infer type for `[_; 0]`
            //          |
            //          consider giving this closure parameter a type
            // ```
            //
            // After clearing, it looks something like this:
            // ```
            // let x = |_| {  };
            //          ^ consider giving this closure parameter the type `[_; 0]`
            //            with the type parameter `_` specified
            // ```
            err.span_label(
                pattern.span,
                format!("consider giving this closure parameter {}", suffix),
            );
        } else if let Some(pattern) = local_visitor.found_local_pattern {
            let msg = if let Some(simple_ident) = pattern.simple_ident() {
                match pattern.span.desugaring_kind() {
                    None => format!("consider giving `{}` {}", simple_ident, suffix),
                    Some(DesugaringKind::ForLoop(_)) => {
                        "the element type for this iterator is not specified".to_string()
                    }
                    _ => format!("this needs {}", suffix),
                }
            } else {
                format!("consider giving this pattern {}", suffix)
            };
            err.span_label(pattern.span, msg);
        } else if let Some(e) = local_visitor.found_method_call {
            if let ExprKind::MethodCall(segment, _, exprs, _) = &e.kind {
                // Suggest impl candidates:
                //
                // error[E0283]: type annotations needed
                //   --> $DIR/E0283.rs:35:24
                //    |
                // LL |     let bar = foo_impl.into() * 1u32;
                //    |               ---------^^^^--
                //    |               |        |
                //    |               |        cannot infer type for type parameter `T` declared on the trait `Into`
                //    |               this method call resolves to `T`
                //    |               help: specify type like: `<Impl as Into<u32>>::into(foo_impl)`
                //    |
                //    = note: cannot satisfy `Impl: Into<_>`
                if !impl_candidates.is_empty() && e.span.contains(span) {
                    if let Some(expr) = exprs.first() {
                        if let ExprKind::Path(hir::QPath::Resolved(_, path)) = expr.kind {
                            if let [path_segment] = &path.segments[..] {
                                let candidate_len = impl_candidates.len();
                                let suggestions = impl_candidates.iter().map(|candidate| {
                                    format!(
                                        "{}::{}({})",
                                        candidate, segment.ident, path_segment.ident
                                    )
                                });
                                err.span_suggestions(
                                    e.span,
                                    &format!(
                                        "use the fully qualified path for the potential candidate{}",
                                        pluralize!(candidate_len),
                                    ),
                                    suggestions,
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                    };
                }
                // Suggest specifying type params or point out the return type of the call:
                //
                // error[E0282]: type annotations needed
                //   --> $DIR/type-annotations-needed-expr.rs:2:39
                //    |
                // LL |     let _ = x.into_iter().sum() as f64;
                //    |                           ^^^
                //    |                           |
                //    |                           cannot infer type for `S`
                //    |                           help: consider specifying the type argument in
                //    |                           the method call: `sum::<S>`
                //    |
                //    = note: type must be known at this point
                //
                // or
                //
                // error[E0282]: type annotations needed
                //   --> $DIR/issue-65611.rs:59:20
                //    |
                // LL |     let x = buffer.last().unwrap().0.clone();
                //    |             -------^^^^--
                //    |             |      |
                //    |             |      cannot infer type for `T`
                //    |             this method call resolves to `std::option::Option<&T>`
                //    |
                //    = note: type must be known at this point
                self.annotate_method_call(segment, e, &mut err);
            }
        }
        // Instead of the following:
        // error[E0282]: type annotations needed
        //  --> file2.rs:3:15
        //   |
        // 3 |     let _ = x.sum() as f64;
        //   |             --^^^--------- cannot infer type for `S`
        //   |
        //   = note: type must be known at this point
        // We want:
        // error[E0282]: type annotations needed
        //  --> file2.rs:3:15
        //   |
        // 3 |     let _ = x.sum() as f64;
        //   |               ^^^ cannot infer type for `S`
        //   |
        //   = note: type must be known at this point
        let span = arg_data.span.unwrap_or(err_span);

        // Avoid multiple labels pointing at `span`.
        if !err
            .span
            .span_labels()
            .iter()
            .any(|span_label| span_label.label.is_some() && span_label.span == span)
            && local_visitor.found_arg_pattern.is_none()
        {
            // FIXME(const_generics): we would like to handle const arguments
            // as part of the normal diagnostics flow below, but there appear to
            // be subtleties in doing so, so for now we special-case const args
            // here.
            if let (UnderspecifiedArgKind::Const { .. }, Some(parent_data)) =
                (&arg_data.kind, &arg_data.parent)
            {
                err.span_suggestion_verbose(
                    span,
                    "consider specifying the const argument",
                    format!("{}::<{}>", parent_data.name, arg_data.name),
                    Applicability::MaybeIncorrect,
                );
            }

            err.span_label(
                span,
                arg_data.cannot_infer_msg(use_diag.filter(|d| d.applies_to(span))),
            );
        }

        err
    }

    fn trait_def_from_hir_fn(&self, hir_id: hir::HirId) -> Option<DefId> {
        // The DefId will be the method's trait item ID unless this is an inherent impl
        if let Some((DefKind::AssocFn, def_id)) =
            self.in_progress_typeck_results?.borrow().type_dependent_def(hir_id)
        {
            return self
                .tcx
                .parent(def_id)
                .filter(|&parent_def_id| self.tcx.is_trait(parent_def_id));
        }

        None
    }

    /// If the `FnSig` for the method call can be found and type arguments are identified as
    /// needed, suggest annotating the call, otherwise point out the resulting type of the call.
    fn annotate_method_call(
        &self,
        segment: &hir::PathSegment<'_>,
        e: &Expr<'_>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        if let (Some(typeck_results), None) = (self.in_progress_typeck_results, &segment.args) {
            let borrow = typeck_results.borrow();
            if let Some((DefKind::AssocFn, did)) = borrow.type_dependent_def(e.hir_id) {
                let generics = self.tcx.generics_of(did);
                if !generics.params.is_empty() {
                    err.span_suggestion_verbose(
                        segment.ident.span.shrink_to_hi(),
                        &format!(
                            "consider specifying the type argument{} in the method call",
                            pluralize!(generics.params.len()),
                        ),
                        format!(
                            "::<{}>",
                            generics
                                .params
                                .iter()
                                .map(|p| p.name.to_string())
                                .collect::<Vec<String>>()
                                .join(", ")
                        ),
                        Applicability::HasPlaceholders,
                    );
                } else {
                    let sig = self.tcx.fn_sig(did);
                    let bound_output = sig.output();
                    let output = bound_output.skip_binder();
                    err.span_label(e.span, &format!("this method call resolves to `{}`", output));
                    let kind = output.kind();
                    if let ty::Projection(proj) = kind {
                        if let Some(span) = self.tcx.hir().span_if_local(proj.item_def_id) {
                            err.span_label(span, &format!("`{}` defined here", output));
                        }
                    }
                }
            }
        }
    }

    pub fn need_type_info_err_in_generator(
        &self,
        kind: hir::GeneratorKind,
        span: Span,
        ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        let ty = self.resolve_vars_if_possible(ty);
        let data = self.extract_inference_diagnostics_data(ty.into(), None);

        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0698,
            "type inside {} must be known in this context",
            kind,
        );
        err.span_label(span, data.cannot_infer_msg(None));
        err
    }
}
