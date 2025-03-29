use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{HirId, PatKind};
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use tracing::debug;

use crate::FnCtxt;

/// Provides context for checking patterns in declarations. More specifically this
/// allows us to infer array types if the pattern is irrefutable and allows us to infer
/// the size of the array. See issue #76342.
#[derive(Debug, Copy, Clone)]
pub(super) enum DeclOrigin<'a> {
    // from an `if let` expression
    LetExpr,
    // from `let x = ..`
    LocalDecl { els: Option<&'a hir::Block<'a>> },
}

impl<'a> DeclOrigin<'a> {
    pub(super) fn try_get_else(&self) -> Option<&'a hir::Block<'a>> {
        match self {
            Self::LocalDecl { els } => *els,
            Self::LetExpr => None,
        }
    }
}

/// A declaration is an abstraction of [hir::LetStmt] and [hir::LetExpr].
///
/// It must have a hir_id, as this is how we connect gather_locals to the check functions.
pub(super) struct Declaration<'a> {
    pub hir_id: HirId,
    pub pat: &'a hir::Pat<'a>,
    pub ty: Option<&'a hir::Ty<'a>>,
    pub span: Span,
    pub init: Option<&'a hir::Expr<'a>>,
    pub origin: DeclOrigin<'a>,
}

impl<'a> From<&'a hir::LetStmt<'a>> for Declaration<'a> {
    fn from(local: &'a hir::LetStmt<'a>) -> Self {
        let hir::LetStmt { hir_id, super_: _, pat, ty, span, init, els, source: _ } = *local;
        Declaration { hir_id, pat, ty, span, init, origin: DeclOrigin::LocalDecl { els } }
    }
}

impl<'a> From<(&'a hir::LetExpr<'a>, HirId)> for Declaration<'a> {
    fn from((let_expr, hir_id): (&'a hir::LetExpr<'a>, HirId)) -> Self {
        let hir::LetExpr { pat, ty, span, init, recovered: _ } = *let_expr;
        Declaration { hir_id, pat, ty, span, init: Some(init), origin: DeclOrigin::LetExpr }
    }
}

pub(super) struct GatherLocalsVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    // parameters are special cases of patterns, but we want to handle them as
    // *distinct* cases. so track when we are hitting a pattern *within* an fn
    // parameter.
    outermost_fn_param_pat: Option<(Span, HirId)>,
}

impl<'a, 'tcx> GatherLocalsVisitor<'a, 'tcx> {
    pub(super) fn new(fcx: &'a FnCtxt<'a, 'tcx>) -> Self {
        Self { fcx, outermost_fn_param_pat: None }
    }

    fn assign(&mut self, span: Span, nid: HirId, ty_opt: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // Infer the variable's type.
                let var_ty = self.fcx.next_ty_var(span);
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
                let o_ty = self.fcx.lower_ty(ty);

                let c_ty = self.fcx.infcx.canonicalize_user_type_annotation(
                    ty::UserType::new_with_bounds(
                        ty::UserTypeKind::Ty(o_ty.raw),
                        self.fcx.collect_impl_trait_clauses_from_hir_ty(ty),
                    ),
                );
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
    fn visit_local(&mut self, local: &'tcx hir::LetStmt<'tcx>) {
        self.declare(local.into());
        intravisit::walk_local(self, local)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Let(let_expr) = expr.kind {
            self.declare((let_expr, expr.hir_id).into());
        }
        intravisit::walk_expr(self, expr)
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        let old_outermost_fn_param_pat =
            self.outermost_fn_param_pat.replace((param.ty_span, param.hir_id));
        intravisit::walk_param(self, param);
        self.outermost_fn_param_pat = old_outermost_fn_param_pat;
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        if let PatKind::Binding(_, _, ident, _) = p.kind {
            let var_ty = self.assign(p.span, p.hir_id, None);

            if let Some((ty_span, hir_id)) = self.outermost_fn_param_pat {
                if !self.fcx.tcx.features().unsized_fn_params() {
                    self.fcx.require_type_is_sized(
                        var_ty,
                        ty_span,
                        // ty_span == ident.span iff this is a closure parameter with no type
                        // ascription, or if it's an implicit `self` parameter
                        ObligationCauseCode::SizedArgumentType(
                            if ty_span == ident.span
                                && self.fcx.tcx.is_closure_like(self.fcx.body_id.into())
                            {
                                None
                            } else {
                                Some(hir_id)
                            },
                        ),
                    );
                }
            } else if !self.fcx.tcx.features().unsized_locals() {
                self.fcx.require_type_is_sized(
                    var_ty,
                    p.span,
                    ObligationCauseCode::VariableType(p.hir_id),
                );
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
