use crate::check::{FnCtxt, LocalTy, UserType};
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::PatKind;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::ty::Ty;
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
}

impl<'a> From<&'a hir::Local<'a>> for Declaration<'a> {
    fn from(local: &'a hir::Local<'a>) -> Self {
        let hir::Local { hir_id, pat, ty, span, init, .. } = *local;
        Declaration { hir_id, pat, ty, span, init }
    }
}

impl<'a> From<&'a hir::Let<'a>> for Declaration<'a> {
    fn from(let_expr: &'a hir::Let<'a>) -> Self {
        let hir::Let { hir_id, pat, ty, span, init } = *let_expr;
        Declaration { hir_id, pat, ty, span, init: Some(init) }
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

    fn assign(&mut self, span: Span, nid: hir::HirId, ty_opt: Option<LocalTy<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // Infer the variable's type.
                let var_ty = self.fcx.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeInference,
                    span,
                });
                self.fcx
                    .locals
                    .borrow_mut()
                    .insert(nid, LocalTy { decl_ty: var_ty, revealed_ty: var_ty });
                var_ty
            }
            Some(typ) => {
                // Take type that the user specified.
                self.fcx.locals.borrow_mut().insert(nid, typ);
                typ.revealed_ty
            }
        }
    }

    /// Allocates a [LocalTy] for a declaration, which may have a type annotation. If it does have
    /// a type annotation, then the LocalTy stored will be the resolved type. This may be found
    /// again during type checking by querying [FnCtxt::local_ty] for the same hir_id.
    fn declare(&mut self, decl: Declaration<'tcx>) {
        let local_ty = match decl.ty {
            Some(ref ty) => {
                let o_ty = self.fcx.to_ty(&ty);

                let c_ty = self.fcx.inh.infcx.canonicalize_user_type_annotation(UserType::Ty(o_ty));
                debug!("visit_local: ty.hir_id={:?} o_ty={:?} c_ty={:?}", ty.hir_id, o_ty, c_ty);
                self.fcx
                    .typeck_results
                    .borrow_mut()
                    .user_provided_types_mut()
                    .insert(ty.hir_id, c_ty);

                Some(LocalTy { decl_ty: o_ty, revealed_ty: o_ty })
            }
            None => None,
        };
        self.assign(decl.span, decl.hir_id, local_ty);

        debug!(
            "local variable {:?} is assigned type {}",
            decl.pat,
            self.fcx.ty_to_string(self.fcx.locals.borrow().get(&decl.hir_id).unwrap().decl_ty)
        );
    }
}

impl<'a, 'tcx> Visitor<'tcx> for GatherLocalsVisitor<'a, 'tcx> {
    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'tcx hir::Local<'tcx>) {
        self.declare(local.into());
        intravisit::walk_local(self, local);
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
                        traits::SizedArgumentType(Some(ty_span)),
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
                self.fcx.ty_to_string(self.fcx.locals.borrow().get(&p.hir_id).unwrap().decl_ty),
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
        _: hir::HirId,
    ) {
    }
}
