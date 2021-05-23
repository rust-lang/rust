//! This module contains the functionality to convert from the wacky tcx data
//! structures into the THIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::thir::util::UserAnnotatedTyHelpers;
use crate::thir::*;

use rustc_ast as ast;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::Node;
use rustc_middle::middle::region;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::ty::{self, Ty, TyCtxt};

pub fn build_thir<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def: ty::WithOptConstParam<LocalDefId>,
    expr: &'tcx hir::Expr<'tcx>,
) -> (Thir<'tcx>, ExprId) {
    let mut cx = Cx::new(tcx, owner_def);
    let expr = cx.mirror_expr(expr);
    (cx.thir, expr)
}

struct Cx<'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: Thir<'tcx>,

    crate param_env: ty::ParamEnv<'tcx>,

    crate region_scope_tree: &'tcx region::ScopeTree,
    crate typeck_results: &'tcx ty::TypeckResults<'tcx>,

    /// The `DefId` of the owner of this body.
    body_owner: DefId,
}

impl<'tcx> Cx<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def: ty::WithOptConstParam<LocalDefId>) -> Cx<'tcx> {
        let typeck_results = tcx.typeck_opt_const_arg(def);
        Cx {
            tcx,
            thir: Thir::new(),
            param_env: tcx.param_env(def.did),
            region_scope_tree: tcx.region_scope_tree(def.did),
            typeck_results,
            body_owner: def.did.to_def_id(),
        }
    }

    crate fn const_eval_literal(
        &mut self,
        lit: &'tcx ast::LitKind,
        ty: Ty<'tcx>,
        sp: Span,
        neg: bool,
    ) -> &'tcx ty::Const<'tcx> {
        trace!("const_eval_literal: {:#?}, {:?}, {:?}, {:?}", lit, ty, sp, neg);

        match self.tcx.at(sp).lit_to_const(LitToConstInput { lit, ty, neg }) {
            Ok(c) => c,
            Err(LitToConstError::UnparseableFloat) => {
                // FIXME(#31407) this is only necessary because float parsing is buggy
                self.tcx.sess.span_err(sp, "could not evaluate float literal (see issue #31407)");
                // create a dummy value and continue compiling
                self.tcx.const_error(ty)
            }
            Err(LitToConstError::Reported) => {
                // create a dummy value and continue compiling
                self.tcx.const_error(ty)
            }
            Err(LitToConstError::TypeError) => bug!("const_eval_literal: had type error"),
        }
    }

    crate fn pattern_from_hir(&mut self, p: &hir::Pat<'_>) -> Pat<'tcx> {
        let p = match self.tcx.hir().get(p.hir_id) {
            Node::Pat(p) | Node::Binding(p) => p,
            node => bug!("pattern became {:?}", node),
        };
        Pat::from_hir(self.tcx, self.param_env, self.typeck_results(), p)
    }
}

impl<'tcx> UserAnnotatedTyHelpers<'tcx> for Cx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn typeck_results(&self) -> &ty::TypeckResults<'tcx> {
        self.typeck_results
    }
}

mod block;
mod expr;
