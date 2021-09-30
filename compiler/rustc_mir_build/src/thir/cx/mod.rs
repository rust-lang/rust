//! This module contains the functionality to convert from the wacky tcx data
//! structures into the THIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::thir::pattern::pat_from_hir;
use crate::thir::util::UserAnnotatedTyHelpers;

use rustc_ast as ast;
use rustc_data_structures::steal::Steal;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::HirId;
use rustc_hir::Node;
use rustc_middle::middle::region;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::thir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

crate fn thir_body<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def: ty::WithOptConstParam<LocalDefId>,
) -> (&'tcx Steal<Thir<'tcx>>, ExprId) {
    let hir = tcx.hir();
    let body = hir.body(hir.body_owned_by(hir.local_def_id_to_hir_id(owner_def.did)));
    let mut cx = Cx::new(tcx, owner_def);
    if cx.typeck_results.tainted_by_errors.is_some() {
        return (tcx.alloc_steal_thir(Thir::new()), ExprId::from_u32(0));
    }
    let expr = cx.mirror_expr(&body.value);
    (tcx.alloc_steal_thir(cx.thir), expr)
}

crate fn thir_tree<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def: ty::WithOptConstParam<LocalDefId>,
) -> String {
    format!("{:#?}", thir_body(tcx, owner_def).0.steal())
}

struct Cx<'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: Thir<'tcx>,

    crate param_env: ty::ParamEnv<'tcx>,

    crate region_scope_tree: &'tcx region::ScopeTree,
    crate typeck_results: &'tcx ty::TypeckResults<'tcx>,

    /// When applying adjustments to the expression
    /// with the given `HirId`, use the given `Span`,
    /// instead of the usual span. This is used to
    /// assign the span of an overall method call
    /// (e.g. `my_val.foo()`) to the adjustment expressions
    /// for the receiver.
    adjustment_span: Option<(HirId, Span)>,

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
            adjustment_span: None,
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
        pat_from_hir(self.tcx, self.param_env, self.typeck_results(), p)
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
