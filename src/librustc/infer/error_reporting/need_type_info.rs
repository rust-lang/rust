use crate::hir::def::Namespace;
use crate::hir::{self, Body, FunctionRetTy, Expr, ExprKind, HirId, Local, Pat};
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::infer::InferCtxt;
use crate::infer::type_variable::TypeVariableOriginKind;
use crate::ty::{self, Ty, Infer, TyVar};
use crate::ty::print::Print;
use syntax::source_map::DesugaringKind;
use syntax_pos::Span;
use errors::{Applicability, DiagnosticBuilder};

struct FindLocalByTypeVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    target_ty: Ty<'tcx>,
    hir_map: &'a hir::map::Map<'tcx>,
    found_local_pattern: Option<&'tcx Pat>,
    found_arg_pattern: Option<&'tcx Pat>,
    found_ty: Option<Ty<'tcx>>,
    found_closure: Option<&'tcx ExprKind>,
}

impl<'a, 'tcx> FindLocalByTypeVisitor<'a, 'tcx> {
    fn new(
        infcx: &'a InferCtxt<'a, 'tcx>,
        target_ty: Ty<'tcx>,
        hir_map: &'a hir::map::Map<'tcx>,
    ) -> Self {
        Self {
            infcx,
            target_ty,
            hir_map,
            found_local_pattern: None,
            found_arg_pattern: None,
            found_ty: None,
            found_closure: None,
        }
    }

    fn node_matches_type(&mut self, hir_id: HirId) -> Option<Ty<'tcx>> {
        let ty_opt = self.infcx.in_progress_tables.and_then(|tables| {
            tables.borrow().node_type_opt(hir_id)
        });
        match ty_opt {
            Some(ty) => {
                let ty = self.infcx.resolve_vars_if_possible(&ty);
                if ty.walk().any(|inner_ty| {
                    inner_ty == self.target_ty || match (&inner_ty.kind, &self.target_ty.kind) {
                        (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => {
                            self.infcx
                                .type_variables
                                .borrow_mut()
                                .sub_unified(a_vid, b_vid)
                        }
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
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_local(&mut self, local: &'tcx Local) {
        if let (None, Some(ty)) = (self.found_local_pattern, self.node_matches_type(local.hir_id)) {
            self.found_local_pattern = Some(&*local.pat);
            self.found_ty = Some(ty);
        }
        intravisit::walk_local(self, local);
    }

    fn visit_body(&mut self, body: &'tcx Body) {
        for param in &body.params {
            if let (None, Some(ty)) = (
                self.found_arg_pattern,
                self.node_matches_type(param.hir_id),
            ) {
                self.found_arg_pattern = Some(&*param.pat);
                self.found_ty = Some(ty);
            }
        }
        intravisit::walk_body(self, body);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if let (ExprKind::Closure(_, _fn_decl, _id, _sp, _), Some(_)) = (
            &expr.node,
            self.node_matches_type(expr.hir_id),
        ) {
            self.found_closure = Some(&expr.node);
        }
        intravisit::walk_expr(self, expr);
    }
}

/// Suggest giving an appropriate return type to a closure expression.
fn closure_return_type_suggestion(
    span: Span,
    err: &mut DiagnosticBuilder<'_>,
    output: &FunctionRetTy,
    body: &Body,
    name: &str,
    ret: &str,
) {
    let (arrow, post) = match output {
        FunctionRetTy::DefaultReturn(_) => ("-> ", " "),
        _ => ("", ""),
    };
    let suggestion = match body.value.node {
        ExprKind::Block(..) => {
            vec![(output.span(), format!("{}{}{}", arrow, ret, post))]
        }
        _ => {
            vec![
                (output.span(), format!("{}{}{}{{ ", arrow, ret, post)),
                (body.value.span.shrink_to_hi(), " }".to_string()),
            ]
        }
    };
    err.multipart_suggestion(
        "give this closure an explicit return type without `_` placeholders",
        suggestion,
        Applicability::HasPlaceholders,
    );
    err.span_label(span, InferCtxt::missing_type_msg(&name));
}

/// Given a closure signature, return a `String` containing a list of all its argument types.
fn closure_args(fn_sig: &ty::PolyFnSig<'_>) -> String {
    fn_sig.inputs()
        .skip_binder()
        .iter()
        .next()
        .map(|args| args.tuple_fields()
            .map(|arg| arg.to_string())
            .collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn extract_type_name(
        &self,
        ty: Ty<'tcx>,
        highlight: Option<ty::print::RegionHighlightMode>,
    ) -> (String, Option<Span>) {
        if let ty::Infer(ty::TyVar(ty_vid)) = ty.kind {
            let ty_vars = self.type_variables.borrow();
            let var_origin = ty_vars.var_origin(ty_vid);
            if let TypeVariableOriginKind::TypeParameterDefinition(name) = var_origin.kind {
                return (name.to_string(), Some(var_origin.span));
            }
        }

        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
        if let Some(highlight) = highlight {
            printer.region_highlight_mode = highlight;
        }
        let _ = ty.print(printer);
        (s, None)
    }

    pub fn need_type_info_err(
        &self,
        body_id: Option<hir::BodyId>,
        span: Span,
        ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        let ty = self.resolve_vars_if_possible(&ty);
        let (name, name_sp) = self.extract_type_name(&ty, None);

        let mut local_visitor = FindLocalByTypeVisitor::new(&self, ty, &self.tcx.hir());
        let ty_to_string = |ty: Ty<'tcx>| -> String {
            let mut s = String::new();
            let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
            let ty_vars = self.type_variables.borrow();
            let getter = move |ty_vid| {
                if let TypeVariableOriginKind::TypeParameterDefinition(name) =
                    ty_vars.var_origin(ty_vid).kind {
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
                let fn_sig = substs.closure_sig(*def_id, self.tcx);
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
        let mut err = struct_span_err!(
            self.tcx.sess,
            err_span,
            E0282,
            "type annotations needed{}",
            ty_msg,
        );

        let suffix = match local_visitor.found_ty {
            Some(ty::TyS { kind: ty::Closure(def_id, substs), .. }) => {
                let fn_sig = substs.closure_sig(*def_id, self.tcx);
                let ret = fn_sig.output().skip_binder().to_string();

                if let Some(ExprKind::Closure(_, decl, body_id, ..)) = local_visitor.found_closure {
                    if let Some(body) = self.tcx.hir().krate().bodies.get(body_id) {
                        closure_return_type_suggestion(
                            span,
                            &mut err,
                            &decl.output,
                            &body,
                            &name,
                            &ret,
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
                    ty,
                    name,
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
                    None => {
                        format!("consider giving `{}` {}", simple_ident, suffix)
                    }
                    Some(DesugaringKind::ForLoop) => {
                        "the element type for this iterator is not specified".to_string()
                    }
                    _ => format!("this needs {}", suffix),
                }
            } else {
                format!("consider giving this pattern {}", suffix)
            };
            err.span_label(pattern.span, msg);
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
        let span = name_sp.unwrap_or(span);
        if !err.span.span_labels().iter().any(|span_label| {
                span_label.label.is_some() && span_label.span == span
            }) && local_visitor.found_arg_pattern.is_none()
        { // Avoid multiple labels pointing at `span`.
            err.span_label(span, InferCtxt::missing_type_msg(&name));
        }

        err
    }

    pub fn need_type_info_err_in_generator(
        &self,
        kind: hir::GeneratorKind,
        span: Span,
        ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        let ty = self.resolve_vars_if_possible(&ty);
        let name = self.extract_type_name(&ty, None).0;
        let mut err = struct_span_err!(
            self.tcx.sess, span, E0698, "type inside {} must be known in this context", kind,
        );
        err.span_label(span, InferCtxt::missing_type_msg(&name));
        err
    }

    fn missing_type_msg(type_name: &str) -> String {
        if type_name == "_" {
            "cannot infer type".to_owned()
        } else {
            format!("cannot infer type for `{}`", type_name)
        }
    }
}
