use crate::check::{FnCtxt, InferredPath};
use crate::rustc::ty::TypeFoldable;
use rustc::infer::type_variable::TypeVariableOriginKind;
use rustc::infer::InferCtxt;
use rustc::ty;
use rustc::ty::fold::TypeFolder;
use rustc::ty::{Ty, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::HirId;

/// Code to detect cases where using `!` (never-type) fallback instead of `()` fallback
/// may result in the introduction of undefined behavior
///

pub struct NeverCompatHandler<'tcx> {
    unresolved_paths: FxHashMap<HirId, InferredPath<'tcx>>,
    unconstrained_diverging: Vec<Ty<'tcx>>,
}

struct TyVarFinder<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    vars: Vec<Ty<'tcx>>,
}
impl<'a, 'tcx> TypeFolder<'tcx> for TyVarFinder<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Infer(ty::InferTy::TyVar(_)) = t.kind {
            self.vars.push(t);
        }
        t.super_fold_with(self)
    }
}

fn find_questionable_call(
    path: InferredPath<'tcx>,
    fcx: &FnCtxt<'a, 'tcx>,
) -> Option<Vec<Ty<'tcx>>> {
    let tcx = fcx.tcx;
    let ty = fcx.infcx.resolve_vars_if_possible(&path.ty);
    debug!("post_fallback: Fully resolved ty: {:?}", ty);

    let ty = ty.unwrap_or_else(|| bug!("Missing ty in path: {:?}", path));

    if let ty::FnDef(_, substs) = ty.kind {
        debug!("Got substs: {:?}", substs);
        let mut args_inhabited = true;

        for arg in &*path.args.unwrap() {
            let resolved_arg = fcx.infcx.resolve_vars_if_possible(arg);

            if resolved_arg.conservative_is_privately_uninhabited(tcx) {
                debug!("post_fallback: Arg is uninhabited: {:?}", resolved_arg);
                args_inhabited = false;
                break;
            } else {
                debug!("post_fallback: Arg is inhabited: {:?}", resolved_arg);
            }
        }

        if !args_inhabited {
            debug!("post_fallback: Not all arguments are inhabited");
            return None;
        }

        for (subst_ty, vars) in substs.types().zip(path.unresolved_vars.into_iter()) {
            let resolved_subst = fcx.infcx.resolve_vars_if_possible(&subst_ty);
            if resolved_subst.conservative_is_privately_uninhabited(tcx) {
                debug!("post_fallback: Subst is uninhabited: {:?}", resolved_subst);
                if !vars.is_empty() {
                    debug!("Found fallback vars: {:?}", vars);
                    debug!(
                        "post_fallback: All arguments are inhabited, at least one subst is not inhabited!"
                    );
                    return Some(vars);
                } else {
                    debug!("No fallback vars")
                }
            } else {
                debug!("post_fallback: Subst is inhabited: {:?}", resolved_subst);
            }
        }
    }
    return None;
}

impl<'tcx> NeverCompatHandler<'tcx> {
    pub fn pre_fallback(fcx: &FnCtxt<'a, 'tcx>) -> NeverCompatHandler<'tcx> {
        let unresolved_paths: FxHashMap<HirId, InferredPath<'tcx>> = fcx
            .inferred_paths
            .borrow()
            .iter()
            .map(|(id, path)| (*id, path.clone()))
            .filter_map(|(hir_id, mut path)| {
                debug!("pre_fallback: inspecting path ({:?}, {:?})", hir_id, path);

                let ty_resolved = fcx.infcx.resolve_vars_if_possible(&path.ty);

                let fn_substs = match ty_resolved {
                    Some(ty::TyS { kind: ty::FnDef(_, substs), .. }) => substs,
                    _ => {
                        debug!("pre_fallback: non-fn ty {:?}, skipping", ty_resolved);
                        return None;
                    }
                };

                if fcx.infcx.unresolved_type_vars(fn_substs).is_some() {
                    for subst in fn_substs.types() {
                        let mut finder = TyVarFinder { infcx: &fcx.infcx, vars: vec![] };
                        subst.fold_with(&mut finder);
                        path.unresolved_vars.push(finder.vars);
                    }

                    debug!(
                        "pre_fallback: unresolved vars in ty {:?} : {:?}",
                        ty_resolved, path.unresolved_vars
                    );

                    Some((hir_id, path))
                } else {
                    debug!("pre_fallback: all vars resolved in ty: {:?}", ty_resolved);
                    None
                }
            })
            .collect();

        let unconstrained_diverging: Vec<_> = fcx
            .unsolved_variables()
            .iter()
            .cloned()
            .filter(|ty| fcx.infcx.type_var_diverges(ty))
            .collect();

        NeverCompatHandler { unresolved_paths, unconstrained_diverging }
    }

    pub fn post_fallback(self, fcx: &FnCtxt<'a, 'tcx>) {
        let tcx = fcx.tcx;
        for (call_id, path) in self.unresolved_paths {
            debug!(
                "post_fallback: resolved ty: {:?} at span {:?} : expr={:?} parent={:?} path={:?}",
                path.span,
                path.ty,
                tcx.hir().get(call_id),
                tcx.hir().get(tcx.hir().get_parent_node(call_id)),
                path
            );

            let span = path.span;
            if let Some(vars) = find_questionable_call(path, fcx) {
                let mut best_diverging_var = None;
                let mut best_var = None;

                for var in vars {
                    for diverging_var in &self.unconstrained_diverging {
                        match (&var.kind, &diverging_var.kind) {
                            (
                                ty::Infer(ty::InferTy::TyVar(vid1)),
                                ty::Infer(ty::InferTy::TyVar(vid2)),
                            ) => {
                                if fcx.infcx.type_variables.borrow_mut().sub_unified(*vid1, *vid2) {
                                    debug!(
                                        "Type variable {:?} is equal to diverging var {:?}",
                                        var, diverging_var
                                    );

                                    debug!(
                                        "Var origin: {:?}",
                                        fcx.infcx.type_variables.borrow().var_origin(*vid1)
                                    );
                                    best_var = Some(vid1);
                                    best_diverging_var = Some(vid2);
                                }
                            }
                            _ => bug!(
                                "Unexpected types: var={:?} diverging_var={:?}",
                                var,
                                diverging_var
                            ),
                        }
                    }
                }

                let var_origin = *fcx.infcx.type_variables.borrow().var_origin(*best_var.unwrap());
                let diverging_var_span =
                    fcx.infcx.type_variables.borrow().var_origin(*best_diverging_var.unwrap()).span;

                let mut err = tcx
                    .sess
                    .struct_span_warn(span, "Fallback to `!` may introduce undefined behavior");

                match var_origin.kind {
                    TypeVariableOriginKind::TypeParameterDefinition(name, did) => {
                        err.span_note(
                            var_origin.span,
                            &format!("the type parameter {} here was inferred to `!`", name),
                        );
                        if let Some(did) = did {
                            err.span_note(fcx.tcx.def_span(did), "(type parameter defined here)");
                        }
                    }
                    _ => {
                        err.span_note(var_origin.span, "the type here was inferred to `!`");
                    }
                }

                err.span_note(diverging_var_span, "... due to this expression evaluating to `!`")
                    .note("If you want the `!` type to be used here, add explicit type annotations")
                    .emit();
            }
        }
    }
}
