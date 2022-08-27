//! This module contains the functionality to convert from the wacky tcx data
//! structures into the THIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::thir::pattern::pat_from_hir;
use crate::thir::util::UserAnnotatedTyHelpers;

use rustc_data_structures::steal::Steal;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::HirId;
use rustc_hir::Node;
use rustc_middle::middle::region;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, RvalueScopes, Subst, TyCtxt};
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

    let owner_id = hir.local_def_id_to_hir_id(owner_def.did);
    if let Some(ref fn_decl) = hir.fn_decl_by_hir_id(owner_id) {
        let explicit_params = cx.explicit_params(owner_id, fn_decl, body);
        cx.thir.params = explicit_params.collect();
    }

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
    pub(crate) fn pattern_from_hir(&mut self, p: &hir::Pat<'_>) -> Pat<'tcx> {
        let p = match self.tcx.hir().get(p.hir_id) {
            Node::Pat(p) => p,
            node => bug!("pattern became {:?}", node),
        };
        pat_from_hir(self.tcx, self.param_env, self.typeck_results(), p)
    }

    fn explicit_params<'a>(
        &'a mut self,
        owner_id: HirId,
        fn_decl: &'tcx hir::FnDecl<'tcx>,
        body: &'tcx hir::Body<'tcx>,
    ) -> impl Iterator<Item = Param<'tcx>> + 'a {
        let fn_sig = self.typeck_results.liberated_fn_sigs()[owner_id];

        body.params.iter().enumerate().map(move |(index, param)| {
            let ty_span = fn_decl
                .inputs
                .get(index)
                // Make sure that inferred closure args have no type span
                .and_then(|ty| if param.pat.span != ty.span { Some(ty.span) } else { None });

            let self_kind = if index == 0 && fn_decl.implicit_self.has_implicit_self() {
                Some(fn_decl.implicit_self)
            } else {
                None
            };

            // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
            // (as it's created inside the body itself, not passed in from outside).
            let ty = if fn_decl.c_variadic && index == fn_decl.inputs.len() {
                let va_list_did = self.tcx.require_lang_item(LangItem::VaList, Some(param.span));

                self.tcx
                    .bound_type_of(va_list_did)
                    .subst(self.tcx, &[self.tcx.lifetimes.re_erased.into()])
            } else {
                fn_sig.inputs()[index]
            };

            let pat = self.pattern_from_hir(param.pat);
            Param { pat, ty, ty_span, self_kind, hir_id: param.hir_id }
        })
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
