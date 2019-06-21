use crate::hir::def::Namespace;
use crate::hir::{self, Local, Pat, Body, HirId};
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::infer::InferCtxt;
use crate::infer::type_variable::TypeVariableOriginKind;
use crate::ty::{self, Ty, Infer, TyVar};
use crate::ty::print::Print;
use syntax::source_map::CompilerDesugaringKind;
use syntax_pos::Span;
use errors::DiagnosticBuilder;

struct FindLocalByTypeVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    target_ty: Ty<'tcx>,
    hir_map: &'a hir::map::Map<'tcx>,
    found_local_pattern: Option<&'tcx Pat>,
    found_arg_pattern: Option<&'tcx Pat>,
    found_ty: Option<Ty<'tcx>>,
}

impl<'a, 'tcx> FindLocalByTypeVisitor<'a, 'tcx> {
    fn node_matches_type(&mut self, hir_id: HirId) -> Option<Ty<'tcx>> {
        let ty_opt = self.infcx.in_progress_tables.and_then(|tables| {
            tables.borrow().node_type_opt(hir_id)
        });
        match ty_opt {
            Some(ty) => {
                let ty = self.infcx.resolve_vars_if_possible(&ty);
                if ty.walk().any(|inner_ty| {
                    inner_ty == self.target_ty || match (&inner_ty.sty, &self.target_ty.sty) {
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
        for argument in &body.arguments {
            if let (None, Some(ty)) = (
                self.found_arg_pattern,
                self.node_matches_type(argument.hir_id),
            ) {
                self.found_arg_pattern = Some(&*argument.pat);
                self.found_ty = Some(ty);
            }
        }
        intravisit::walk_body(self, body);
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn extract_type_name(
        &self,
        ty: Ty<'tcx>,
        highlight: Option<ty::print::RegionHighlightMode>,
    ) -> String {
        if let ty::Infer(ty::TyVar(ty_vid)) = ty.sty {
            let ty_vars = self.type_variables.borrow();
            if let TypeVariableOriginKind::TypeParameterDefinition(name) =
                ty_vars.var_origin(ty_vid).kind {
                return name.to_string();
            }
        }

        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS);
        if let Some(highlight) = highlight {
            printer.region_highlight_mode = highlight;
        }
        let _ = ty.print(printer);
        s
    }

    pub fn need_type_info_err(
        &self,
        body_id: Option<hir::BodyId>,
        span: Span,
        ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        let ty = self.resolve_vars_if_possible(&ty);
        let name = self.extract_type_name(&ty, None);

        let mut err_span = span;

        let mut local_visitor = FindLocalByTypeVisitor {
            infcx: &self,
            target_ty: ty,
            hir_map: &self.tcx.hir(),
            found_local_pattern: None,
            found_arg_pattern: None,
            found_ty: None,
        };
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
        let (ty_msg, suffix) = match &local_visitor.found_ty {
            Some(ty) if &ty.to_string() != "_" && name == "_" => {
                let ty = ty_to_string(ty);
                (format!(" for `{}`", ty),
                 format!("the explicit type `{}`, with the type parameters specified", ty))
            }
            Some(ty) if &ty.to_string() != "_" && ty.to_string() != name => {
                let ty = ty_to_string(ty);
                (format!(" for `{}`", ty),
                 format!(
                     "the explicit type `{}`, where the type parameter `{}` is specified",
                    ty,
                    name,
                 ))
            }
            _ => (String::new(), "a type".to_owned()),
        };
        let mut labels = vec![(span, InferCtxt::missing_type_msg(&name))];

        if let Some(pattern) = local_visitor.found_arg_pattern {
            err_span = pattern.span;
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
            labels.clear();
            labels.push((
                pattern.span,
                format!("consider giving this closure parameter {}", suffix),
            ));
        } else if let Some(pattern) = local_visitor.found_local_pattern {
            if let Some(simple_ident) = pattern.simple_ident() {
                match pattern.span.compiler_desugaring_kind() {
                    None => labels.push((
                        pattern.span,
                        format!("consider giving `{}` {}", simple_ident, suffix),
                    )),
                    Some(CompilerDesugaringKind::ForLoop) => labels.push((
                        pattern.span,
                        "the element type for this iterator is not specified".to_owned(),
                    )),
                    _ => {}
                }
            } else {
                labels.push((pattern.span, format!("consider giving this pattern {}", suffix)));
            }
        };

        let mut err = struct_span_err!(
            self.tcx.sess,
            err_span,
            E0282,
            "type annotations needed{}",
            ty_msg,
        );

        for (target_span, label_message) in labels {
            err.span_label(target_span, label_message);
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
        let name = self.extract_type_name(&ty, None);
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
