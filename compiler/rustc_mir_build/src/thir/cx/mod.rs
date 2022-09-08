//! This module contains the functionality to convert from the wacky tcx data
//! structures into the THIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::thir::pattern::pat_from_hir;
use crate::thir::util::UserAnnotatedTyHelpers;

use rustc_data_structures::steal::Steal;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::HirId;
use rustc_hir::Node;
use rustc_middle::middle::region;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, RvalueScopes, TyCtxt};
use rustc_span::Span;

pub(crate) fn thir_body<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def: ty::WithOptConstParam<LocalDefId>,
) -> Result<(&'tcx Steal<Thir<'tcx>>, ExprId), ErrorGuaranteed> {
    let hir = tcx.hir();
    let body = hir.body(hir.body_owned_by(owner_def.did));
    let mut cx = Cx::new(tcx, owner_def);
    if let Some(reported) = cx.typeck_results.tainted_by_errors {
        return Err(reported);
    }
    let expr = cx.mirror_expr(&body.value);
    Ok((tcx.alloc_steal_thir(cx.thir), expr))
}

pub(crate) fn thir_tree<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_def: ty::WithOptConstParam<LocalDefId>,
) -> String {
    match thir_body(tcx, owner_def) {
        Ok((thir, _)) => format!("{:#?}", thir.steal()),
        Err(_) => "error".into(),
    }
}

struct Cx<'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: Thir<'tcx>,

    pub(crate) param_env: ty::ParamEnv<'tcx>,

    pub(crate) region_scope_tree: &'tcx region::ScopeTree,
    pub(crate) typeck_results: &'tcx ty::TypeckResults<'tcx>,
    pub(crate) rvalue_scopes: &'tcx RvalueScopes,

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
            rvalue_scopes: &typeck_results.rvalue_scopes,
            body_owner: def.did.to_def_id(),
            adjustment_span: None,
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn pattern_from_hir(&mut self, p: &hir::Pat<'_>) -> Box<Pat<'tcx>> {
        let p = match self.tcx.hir().get(p.hir_id) {
            Node::Pat(p) => p,
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
