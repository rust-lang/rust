use crate::hir::def::Namespace;
use crate::hir::{self, Local, Pat, Body, HirId};
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::infer::InferCtxt;
use crate::infer::type_variable::TypeVariableOrigin;
use crate::ty::{self, Ty, Infer, TyVar};
use crate::ty::print::Print;
use syntax::source_map::CompilerDesugaringKind;
use syntax_pos::Span;
use errors::DiagnosticBuilder;

struct FindLocalByTypeVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    target_ty: &'a Ty<'tcx>,
    hir_map: &'a hir::map::Map<'gcx>,
    found_local_pattern: Option<&'gcx Pat>,
    found_arg_pattern: Option<&'gcx Pat>,
}

impl<'a, 'gcx, 'tcx> FindLocalByTypeVisitor<'a, 'gcx, 'tcx> {
    fn node_matches_type(&mut self, hir_id: HirId) -> bool {
        let ty_opt = self.infcx.in_progress_tables.and_then(|tables| {
            tables.borrow().node_type_opt(hir_id)
        });
        match ty_opt {
            Some(ty) => {
                let ty = self.infcx.resolve_type_vars_if_possible(&ty);
                ty.walk().any(|inner_ty| {
                    inner_ty == *self.target_ty || match (&inner_ty.sty, &self.target_ty.sty) {
                        (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => {
                            self.infcx
                                .type_variables
                                .borrow_mut()
                                .sub_unified(a_vid, b_vid)
                        }
                        _ => false,
                    }
                })
            }
            None => false,
        }
    }
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for FindLocalByTypeVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_local(&mut self, local: &'gcx Local) {
        if self.found_local_pattern.is_none() && self.node_matches_type(local.hir_id) {
            self.found_local_pattern = Some(&*local.pat);
        }
        intravisit::walk_local(self, local);
    }

    fn visit_body(&mut self, body: &'gcx Body) {
        for argument in &body.arguments {
            if self.found_arg_pattern.is_none() && self.node_matches_type(argument.hir_id) {
                self.found_arg_pattern = Some(&*argument.pat);
            }
        }
        intravisit::walk_body(self, body);
    }
}


impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    pub fn extract_type_name(
        &self,
        ty: &'a Ty<'tcx>,
        highlight: Option<ty::print::RegionHighlightMode>,
    ) -> String {
        if let ty::Infer(ty::TyVar(ty_vid)) = (*ty).sty {
            let ty_vars = self.type_variables.borrow();
            if let TypeVariableOrigin::TypeParameterDefinition(_, name) =
                *ty_vars.var_origin(ty_vid) {
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

    pub fn need_type_info_err(&self,
                            body_id: Option<hir::BodyId>,
                            span: Span,
                            ty: Ty<'tcx>)
                            -> DiagnosticBuilder<'gcx> {
        let ty = self.resolve_type_vars_if_possible(&ty);
        let name = self.extract_type_name(&ty, None);

        let mut err_span = span;
        let mut labels = vec![(
            span,
            if &name == "_" {
                "cannot infer type".to_owned()
            } else {
                format!("cannot infer type for `{}`", name)
            },
        )];

        let mut local_visitor = FindLocalByTypeVisitor {
            infcx: &self,
            target_ty: &ty,
            hir_map: &self.tcx.hir(),
            found_local_pattern: None,
            found_arg_pattern: None,
        };

        if let Some(body_id) = body_id {
            let expr = self.tcx.hir().expect_expr_by_hir_id(body_id.hir_id);
            local_visitor.visit_expr(expr);
        }

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
            //          ^ consider giving this closure parameter a type
            // ```
            labels.clear();
            labels.push(
                (pattern.span, "consider giving this closure parameter a type".to_owned()));
        } else if let Some(pattern) = local_visitor.found_local_pattern {
            if let Some(simple_ident) = pattern.simple_ident() {
                match pattern.span.compiler_desugaring_kind() {
                    None => labels.push((pattern.span,
                                         format!("consider giving `{}` a type", simple_ident))),
                    Some(CompilerDesugaringKind::ForLoop) => labels.push((
                        pattern.span,
                        "the element type for this iterator is not specified".to_owned(),
                    )),
                    _ => {}
                }
            } else {
                labels.push((pattern.span, "consider giving the pattern a type".to_owned()));
            }
        }

        let mut err = struct_span_err!(self.tcx.sess,
                                       err_span,
                                       E0282,
                                       "type annotations needed");

        for (target_span, label_message) in labels {
            err.span_label(target_span, label_message);
        }

        err
    }
}
