//! This module contains the functionality to convert from the wacky tcx data
//! structures into the THIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::thir::pattern::pat_from_hir;
use crate::thir::util::UserAnnotatedTyHelpers;

use rustc_data_structures::steal::Steal;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::HirId;
use rustc_hir::Node;
use rustc_middle::middle::region;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, RvalueScopes, Ty, TyCtxt};

pub(crate) fn thir_body(
    tcx: TyCtxt<'_>,
    owner_def: LocalDefId,
) -> Result<(&Steal<Thir<'_>>, ExprId), ErrorGuaranteed> {
    let hir = tcx.hir();
    let body = hir.body(hir.body_owned_by(owner_def));
    let mut cx = Cx::new(tcx, owner_def);
    if let Some(reported) = cx.typeck_results.tainted_by_errors {
        return Err(reported);
    }
    let expr = cx.mirror_expr(&body.value);

    let owner_id = hir.local_def_id_to_hir_id(owner_def);
    if let Some(ref fn_decl) = hir.fn_decl_by_hir_id(owner_id) {
        let closure_env_param = cx.closure_env_param(owner_def, owner_id);
        let explicit_params = cx.explicit_params(owner_id, fn_decl, body);
        cx.thir.params = closure_env_param.into_iter().chain(explicit_params).collect();

        // The resume argument may be missing, in that case we need to provide it here.
        // It will always be `()` in this case.
        if tcx.def_kind(owner_def) == DefKind::Generator && body.params.is_empty() {
            cx.thir.params.push(Param {
                ty: Ty::new_unit(tcx),
                pat: None,
                ty_span: None,
                self_kind: None,
                hir_id: None,
            });
        }
    }

    Ok((tcx.alloc_steal_thir(cx.thir), expr))
}

struct Cx<'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: Thir<'tcx>,

    param_env: ty::ParamEnv<'tcx>,

    region_scope_tree: &'tcx region::ScopeTree,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
    rvalue_scopes: &'tcx RvalueScopes,

    /// False to indicate that adjustments should not be applied. Only used for `custom_mir`
    apply_adjustments: bool,

    /// The `DefId` of the owner of this body.
    body_owner: DefId,
}

impl<'tcx> Cx<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def: LocalDefId) -> Cx<'tcx> {
        let typeck_results = tcx.typeck(def);
        let hir = tcx.hir();
        let hir_id = hir.local_def_id_to_hir_id(def);

        let body_type = if hir.body_owner_kind(def).is_fn_or_closure() {
            // fetch the fully liberated fn signature (that is, all bound
            // types/lifetimes replaced)
            BodyTy::Fn(typeck_results.liberated_fn_sigs()[hir_id])
        } else {
            // Get the revealed type of this const. This is *not* the adjusted
            // type of its body, which may be a subtype of this type. For
            // example:
            //
            // fn foo(_: &()) {}
            // static X: fn(&'static ()) = foo;
            //
            // The adjusted type of the body of X is `for<'a> fn(&'a ())` which
            // is not the same as the type of X. We need the type of the return
            // place to be the type of the constant because NLL typeck will
            // equate them.
            BodyTy::Const(typeck_results.node_type(hir_id))
        };

        Cx {
            tcx,
            thir: Thir::new(body_type),
            param_env: tcx.param_env(def),
            region_scope_tree: tcx.region_scope_tree(def),
            typeck_results,
            rvalue_scopes: &typeck_results.rvalue_scopes,
            body_owner: def.to_def_id(),
            apply_adjustments: hir
                .attrs(hir_id)
                .iter()
                .all(|attr| attr.name_or_empty() != rustc_span::sym::custom_mir),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn pattern_from_hir(&mut self, p: &hir::Pat<'_>) -> Box<Pat<'tcx>> {
        let p = match self.tcx.hir().get(p.hir_id) {
            Node::Pat(p) => p,
            node => bug!("pattern became {:?}", node),
        };
        pat_from_hir(self.tcx, self.param_env, self.typeck_results(), p)
    }

    fn closure_env_param(&self, owner_def: LocalDefId, owner_id: HirId) -> Option<Param<'tcx>> {
        match self.tcx.def_kind(owner_def) {
            DefKind::Closure => {
                let closure_ty = self.typeck_results.node_type(owner_id);

                let ty::Closure(closure_def_id, closure_args) = *closure_ty.kind() else {
                    bug!("closure expr does not have closure type: {:?}", closure_ty);
                };

                let bound_vars =
                    self.tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Region(ty::BrEnv)]);
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                    kind: ty::BrEnv,
                };
                let env_region = ty::Region::new_late_bound(self.tcx, ty::INNERMOST, br);
                let closure_env_ty =
                    self.tcx.closure_env_ty(closure_def_id, closure_args, env_region).unwrap();
                let liberated_closure_env_ty = self.tcx.erase_late_bound_regions(
                    ty::Binder::bind_with_vars(closure_env_ty, bound_vars),
                );
                let env_param = Param {
                    ty: liberated_closure_env_ty,
                    pat: None,
                    ty_span: None,
                    self_kind: None,
                    hir_id: None,
                };

                Some(env_param)
            }
            DefKind::Generator => {
                let gen_ty = self.typeck_results.node_type(owner_id);
                let gen_param =
                    Param { ty: gen_ty, pat: None, ty_span: None, self_kind: None, hir_id: None };
                Some(gen_param)
            }
            _ => None,
        }
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
                    .type_of(va_list_did)
                    .instantiate(self.tcx, &[self.tcx.lifetimes.re_erased.into()])
            } else {
                fn_sig.inputs()[index]
            };

            let pat = self.pattern_from_hir(param.pat);
            Param { pat: Some(pat), ty, ty_span, self_kind, hir_id: Some(param.hir_id) }
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
