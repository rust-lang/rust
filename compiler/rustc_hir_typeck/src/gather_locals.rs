use crate::FnCtxt;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::PatKind;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::ty::Ty;
use rustc_middle::ty::UserType;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;
use rustc_trait_selection::traits;

/// A declaration is an abstraction of [hir::Local] and [hir::Let].
///
/// It must have a hir_id, as this is how we connect gather_locals to the check functions.
pub(super) struct Declaration<'a> {
    pub hir_id: hir::HirId,
    pub pat: &'a hir::Pat<'a>,
    pub ty: Option<&'a hir::Ty<'a>>,
    pub span: Span,
    pub init: Option<&'a hir::Expr<'a>>,
    pub els: Option<&'a hir::Block<'a>>,
}

impl<'a> From<&'a hir::Local<'a>> for Declaration<'a> {
    fn from(local: &'a hir::Local<'a>) -> Self {
        let hir::Local { hir_id, pat, ty, span, init, els, source: _ } = *local;
        Declaration { hir_id, pat, ty, span, init, els }
    }
}

impl<'a> From<&'a hir::Let<'a>> for Declaration<'a> {
    fn from(let_expr: &'a hir::Let<'a>) -> Self {
        let hir::Let { hir_id, pat, ty, span, init } = *let_expr;
        Declaration { hir_id, pat, ty, span, init: Some(init), els: None }
    }
}

pub(super) struct GatherLocalsVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    // parameters are special cases of patterns, but we want to handle them as
    // *distinct* cases. so track when we are hitting a pattern *within* an fn
    // parameter.
    outermost_fn_param_pat: Option<Span>,
}

impl<'a, 'tcx> GatherLocalsVisitor<'a, 'tcx> {
    pub(super) fn new(fcx: &'a FnCtxt<'a, 'tcx>) -> Self {
        Self { fcx, outermost_fn_param_pat: None }
    }

    fn assign(&mut self, span: Span, nid: hir::HirId, ty_opt: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // Infer the variable's type.
                let var_ty = self.fcx.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeInference,
                    span,
                });
                self.fcx.locals.borrow_mut().insert(nid, var_ty);
                var_ty
            }
            Some(typ) => {
                // Take type that the user specified.
                self.fcx.locals.borrow_mut().insert(nid, typ);
                typ
            }
        }
    }

    /// Allocates a type for a declaration, which may have a type annotation. If it does have
    /// a type annotation, then the [`Ty`] stored will be the resolved type. This may be found
    /// again during type checking by querying [`FnCtxt::local_ty`] for the same hir_id.
    fn declare(&mut self, decl: Declaration<'tcx>) {
        let local_ty = match decl.ty {
            Some(ref ty) => {
                let o_ty = self.fcx.to_ty(&ty);

                let c_ty =
                    self.fcx.inh.infcx.canonicalize_user_type_annotation(UserType::Ty(o_ty.raw));
                debug!("visit_local: ty.hir_id={:?} o_ty={:?} c_ty={:?}", ty.hir_id, o_ty, c_ty);
                self.fcx
                    .typeck_results
                    .borrow_mut()
                    .user_provided_types_mut()
                    .insert(ty.hir_id, c_ty);

                Some(o_ty.normalized)
            }
            None => None,
        };
        self.assign(decl.span, decl.hir_id, local_ty);

        debug!(
            "local variable {:?} is assigned type {}",
            decl.pat,
            self.fcx.ty_to_string(*self.fcx.locals.borrow().get(&decl.hir_id).unwrap())
        );
    }
}

impl<'a, 'tcx> Visitor<'tcx> for GatherLocalsVisitor<'a, 'tcx> {
    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'tcx hir::Local<'tcx>) {
        self.declare(local.into());
        intravisit::walk_local(self, local)
    }

    fn visit_let_expr(&mut self, let_expr: &'tcx hir::Let<'tcx>) {
        self.declare(let_expr.into());
        intravisit::walk_let_expr(self, let_expr);
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        let old_outermost_fn_param_pat = self.outermost_fn_param_pat.replace(param.ty_span);
        intravisit::walk_param(self, param);
        self.outermost_fn_param_pat = old_outermost_fn_param_pat;
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        if let PatKind::Binding(_, _, ident, _) = p.kind {
            let var_ty = self.assign(p.span, p.hir_id, None);

            if let Some(ty_span) = self.outermost_fn_param_pat {
                if !self.fcx.tcx.features().unsized_fn_params {
                    self.fcx.require_type_is_sized(
                        var_ty,
                        p.span,
                        // ty_span == ident.span iff this is a closure parameter with no type
                        // ascription, or if it's an implicit `self` parameter
                        traits::SizedArgumentType(
                            if ty_span == ident.span
                                && self.fcx.tcx.is_closure(self.fcx.body_id.into())
                            {
                                None
                            } else {
                                Some(ty_span)
                            },
                        ),
                    );
                }
            } else {
                if !self.fcx.tcx.features().unsized_locals {
                    self.fcx.require_type_is_sized(var_ty, p.span, traits::VariableType(p.hir_id));
                }
            }

            debug!(
                "pattern binding {} is assigned to {} with type {:?}",
                ident,
                self.fcx.ty_to_string(*self.fcx.locals.borrow().get(&p.hir_id).unwrap()),
                var_ty
            );
        }
        let old_outermost_fn_param_pat = self.outermost_fn_param_pat.take();
        intravisit::walk_pat(self, p);
        self.outermost_fn_param_pat = old_outermost_fn_param_pat;
    }

    // Don't descend into the bodies of nested closures.
    fn visit_fn(
        &mut self,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        _: hir::BodyId,
        _: Span,
        _: LocalDefId,
    ) {
    }
}
