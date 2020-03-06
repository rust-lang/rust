use crate::infer::type_variable::TypeVariableOriginKind;
use crate::infer::InferCtxt;
use rustc::hir::map::Map;
use rustc::ty::print::Print;
use rustc::ty::{self, DefIdTree, Infer, Ty, TyVar};
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Body, Expr, ExprKind, FnRetTy, HirId, Local, Pat};
use rustc_span::source_map::DesugaringKind;
use rustc_span::symbol::kw;
use rustc_span::Span;
use std::borrow::Cow;

struct FindLocalByTypeVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    target_ty: Ty<'tcx>,
    hir_map: &'a Map<'tcx>,
    found_local_pattern: Option<&'tcx Pat<'tcx>>,
    found_arg_pattern: Option<&'tcx Pat<'tcx>>,
    found_ty: Option<Ty<'tcx>>,
    found_closure: Option<&'tcx ExprKind<'tcx>>,
    found_method_call: Option<&'tcx Expr<'tcx>>,
}

impl<'a, 'tcx> FindLocalByTypeVisitor<'a, 'tcx> {
    fn new(infcx: &'a InferCtxt<'a, 'tcx>, target_ty: Ty<'tcx>, hir_map: &'a Map<'tcx>) -> Self {
        Self {
            infcx,
            target_ty,
            hir_map,
            found_local_pattern: None,
            found_arg_pattern: None,
            found_ty: None,
            found_closure: None,
            found_method_call: None,
        }
    }

    fn node_matches_type(&mut self, hir_id: HirId) -> Option<Ty<'tcx>> {
        let ty_opt =
            self.infcx.in_progress_tables.and_then(|tables| tables.borrow().node_type_opt(hir_id));
        match ty_opt {
            Some(ty) => {
                let ty = self.infcx.resolve_vars_if_possible(&ty);
                if ty.walk().any(|inner_ty| {
                    inner_ty == self.target_ty
                        || match (&inner_ty.kind, &self.target_ty.kind) {
                            (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => self
                                .infcx
                                .inner
                                .borrow_mut()
                                .type_variables
                                .sub_unified(a_vid, b_vid),
                            _ => false,
                        }
                }) {
                    Some(ty)
                } else {
                    None
                }
            }
            None => None,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for FindLocalByTypeVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, Self::Map> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_local(&mut self, local: &'tcx Local<'tcx>) {
        if let (None, Some(ty)) = (self.found_local_pattern, self.node_matches_type(local.hir_id)) {
            self.found_local_pattern = Some(&*local.pat);
            self.found_ty = Some(ty);
        }
        intravisit::walk_local(self, local);
    }

    fn visit_body(&mut self, body: &'tcx Body<'tcx>) {
        for param in body.params {
            if let (None, Some(ty)) = (self.found_arg_pattern, self.node_matches_type(param.hir_id))
            {
                self.found_arg_pattern = Some(&*param.pat);
                self.found_ty = Some(ty);
            }
        }
        intravisit::walk_body(self, body);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if self.node_matches_type(expr.hir_id).is_some() {
            match expr.kind {
                ExprKind::Closure(..) => self.found_closure = Some(&expr.kind),
                ExprKind::MethodCall(..) => self.found_method_call = Some(&expr),
                _ => {}
            }
        }
        intravisit::walk_expr(self, expr);
    }
}

/// Suggest giving an appropriate return type to a closure expression.
fn closure_return_type_suggestion(
    span: Span,
    err: &mut DiagnosticBuilder<'_>,
    output: &FnRetTy<'_>,
    body: &Body<'_>,
    descr: &str,
    name: &str,
    ret: &str,
    parent_name: Option<String>,
    parent_descr: Option<&str>,
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
    err.span_label(span, InferCtxt::missing_type_msg(&name, &descr, parent_name, parent_descr));
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
    E0282,
    E0283,
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

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn extract_type_name(
        &self,
        ty: Ty<'tcx>,
        highlight: Option<ty::print::RegionHighlightMode>,
    ) -> (String, Option<Span>, Cow<'static, str>, Option<String>, Option<&'static str>) {
        if let ty::Infer(ty::TyVar(ty_vid)) = ty.kind {
            let ty_vars = &self.inner.borrow().type_variables;
            let var_origin = ty_vars.var_origin(ty_vid);
            if let TypeVariableOriginKind::TypeParameterDefinition(name, def_id) = var_origin.kind {
                let parent_def_id = def_id.and_then(|def_id| self.tcx.parent(def_id));
                let (parent_name, parent_desc) = if let Some(parent_def_id) = parent_def_id {
                    let parent_name = self
                        .tcx
                        .def_key(parent_def_id)
                        .disambiguated_data
                        .data
                        .get_opt_name()
                        .map(|parent_symbol| parent_symbol.to_string());

                    let type_parent_desc = self
                        .tcx
                        .def_kind(parent_def_id)
                        .map(|parent_def_kind| parent_def_kind.descr(parent_def_id));

                    (parent_name, type_parent_desc)
                } else {
                    (None, None)
                };

                if name != kw::SelfUpper {
                    return (
                        name.to_string(),
                        Some(var_origin.span),
                        "type parameter".into(),
                        parent_name,
                        parent_desc,
                    );
                }
            }
        }

        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
        if let Some(highlight) = highlight {
            printer.region_highlight_mode = highlight;
        }
        let _ = ty.print(printer);
        (s, None, ty.prefix_string(), None, None)
    }

    pub fn need_type_info_err(
        &self,
        body_id: Option<hir::BodyId>,
        span: Span,
        ty: Ty<'tcx>,
        error_code: TypeAnnotationNeeded,
    ) -> DiagnosticBuilder<'tcx> {
        let ty = self.resolve_vars_if_possible(&ty);
        let (name, name_sp, descr, parent_name, parent_descr) = self.extract_type_name(&ty, None);

        let mut local_visitor = FindLocalByTypeVisitor::new(&self, ty, &self.tcx.hir());
        let ty_to_string = |ty: Ty<'tcx>| -> String {
            let mut s = String::new();
            let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
            let ty_vars = &self.inner.borrow().type_variables;
            let getter = move |ty_vid| {
                let var_origin = ty_vars.var_origin(ty_vid);
                if let TypeVariableOriginKind::TypeParameterDefinition(name, _) = var_origin.kind {
                    return Some(name.to_string());
                }
                None
            };
            printer.name_resolver = Some(Box::new(&getter));
            let _ = ty.print(printer);
            s
        };

        if let Some(body_id) = body_id {
            let expr = self.tcx.hir().expect_expr(body_id.hir_id);
            local_visitor.visit_expr(expr);
        }
        let err_span = if let Some(pattern) = local_visitor.found_arg_pattern {
            pattern.span
        } else if let Some(span) = name_sp {
            // `span` here lets us point at `sum` instead of the entire right hand side expr:
            // error[E0282]: type annotations needed
            //  --> file2.rs:3:15
            //   |
            // 3 |     let _ = x.sum() as f64;
            //   |               ^^^ cannot infer type for `S`
            span
        } else if let Some(ExprKind::MethodCall(_, call_span, _)) =
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

        let ty_msg = match local_visitor.found_ty {
            Some(ty::TyS { kind: ty::Closure(def_id, substs), .. }) => {
                let fn_sig = substs.as_closure().sig(*def_id, self.tcx);
                let args = closure_args(&fn_sig);
                let ret = fn_sig.output().skip_binder().to_string();
                format!(" for the closure `fn({}) -> {}`", args, ret)
            }
            Some(ty) if is_named_and_not_impl_trait(ty) => {
                let ty = ty_to_string(ty);
                format!(" for `{}`", ty)
            }
            _ => String::new(),
        };

        // When `name` corresponds to a type argument, show the path of the full type we're
        // trying to infer. In the following example, `ty_msg` contains
        // " in `std::result::Result<i32, E>`":
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

        let suffix = match local_visitor.found_ty {
            Some(ty::TyS { kind: ty::Closure(def_id, substs), .. }) => {
                let fn_sig = substs.as_closure().sig(*def_id, self.tcx);
                let ret = fn_sig.output().skip_binder().to_string();

                if let Some(ExprKind::Closure(_, decl, body_id, ..)) = local_visitor.found_closure {
                    if let Some(body) = self.tcx.hir().krate().bodies.get(body_id) {
                        closure_return_type_suggestion(
                            span,
                            &mut err,
                            &decl.output,
                            &body,
                            &descr,
                            &name,
                            &ret,
                            parent_name,
                            parent_descr,
                        );
                        // We don't want to give the other suggestions when the problem is the
                        // closure return type.
                        return err;
                    }
                }

                // This shouldn't be reachable, but just in case we leave a reasonable fallback.
                let args = closure_args(&fn_sig);
                // This suggestion is incomplete, as the user will get further type inference
                // errors due to the `_` placeholders and the introduction of `Box`, but it does
                // nudge them in the right direction.
                format!("a boxed closure type like `Box<dyn Fn({}) -> {}>`", args, ret)
            }
            Some(ty) if is_named_and_not_impl_trait(ty) && name == "_" => {
                let ty = ty_to_string(ty);
                format!("the explicit type `{}`, with the type parameters specified", ty)
            }
            Some(ty) if is_named_and_not_impl_trait(ty) && ty.to_string() != name => {
                let ty = ty_to_string(ty);
                format!(
                    "the explicit type `{}`, where the type parameter `{}` is specified",
                    ty, name,
                )
            }
            _ => "a type".to_string(),
        };

        if let Some(pattern) = local_visitor.found_arg_pattern {
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
                    Some(DesugaringKind::ForLoop) => {
                        "the element type for this iterator is not specified".to_string()
                    }
                    _ => format!("this needs {}", suffix),
                }
            } else {
                format!("consider giving this pattern {}", suffix)
            };
            err.span_label(pattern.span, msg);
        } else if let Some(e) = local_visitor.found_method_call {
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
        let span = name_sp.unwrap_or(err_span);
        if !err
            .span
            .span_labels()
            .iter()
            .any(|span_label| span_label.label.is_some() && span_label.span == span)
            && local_visitor.found_arg_pattern.is_none()
        {
            // Avoid multiple labels pointing at `span`.
            err.span_label(
                span,
                InferCtxt::missing_type_msg(&name, &descr, parent_name, parent_descr),
            );
        }

        err
    }

    /// If the `FnSig` for the method call can be found and type arguments are identified as
    /// needed, suggest annotating the call, otherwise point out the resulting type of the call.
    fn annotate_method_call(
        &self,
        segment: &hir::PathSegment<'_>,
        e: &Expr<'_>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        if let (Ok(snippet), Some(tables), None) = (
            self.tcx.sess.source_map().span_to_snippet(segment.ident.span),
            self.in_progress_tables,
            &segment.args,
        ) {
            let borrow = tables.borrow();
            if let Some((DefKind::Method, did)) = borrow.type_dependent_def(e.hir_id) {
                let generics = self.tcx.generics_of(did);
                if !generics.params.is_empty() {
                    err.span_suggestion(
                        segment.ident.span,
                        &format!(
                            "consider specifying the type argument{} in the method call",
                            if generics.params.len() > 1 { "s" } else { "" },
                        ),
                        format!(
                            "{}::<{}>",
                            snippet,
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
                    err.span_label(e.span, &format!("this method call resolves to `{:?}`", output));
                    let kind = &output.kind;
                    if let ty::Projection(proj) | ty::UnnormalizedProjection(proj) = kind {
                        if let Some(span) = self.tcx.hir().span_if_local(proj.item_def_id) {
                            err.span_label(span, &format!("`{:?}` defined here", output));
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
        let ty = self.resolve_vars_if_possible(&ty);
        let (name, _, descr, parent_name, parent_descr) = self.extract_type_name(&ty, None);

        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0698,
            "type inside {} must be known in this context",
            kind,
        );
        err.span_label(span, InferCtxt::missing_type_msg(&name, &descr, parent_name, parent_descr));
        err
    }

    fn missing_type_msg(
        type_name: &str,
        descr: &str,
        parent_name: Option<String>,
        parent_descr: Option<&str>,
    ) -> Cow<'static, str> {
        if type_name == "_" {
            "cannot infer type".into()
        } else {
            let parent_desc = if let Some(parent_name) = parent_name {
                let parent_type_descr = if let Some(parent_descr) = parent_descr {
                    format!(" the {}", parent_descr)
                } else {
                    "".into()
                };

                format!(" declared on{} `{}`", parent_type_descr, parent_name)
            } else {
                "".to_string()
            };

            format!("cannot infer type for {} `{}`{}", descr, type_name, parent_desc).into()
        }
    }
}
