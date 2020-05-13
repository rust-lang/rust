#![allow(dead_code)]
#![allow(unused_imports)]
use crate::transform::{MirPass, MirSource};
use crate::util::patch::MirPatch;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::*;
use rustc_middle::traits;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::predicate_for_trait_def;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;

static VEC_CONSTRUCTOR: &str = "std::vec::Vec::<T>::new";

pub struct GcPreallocator;

impl<'tcx> MirPass<'tcx> for GcPreallocator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        let param_env = tcx.param_env(source.def_id()).with_reveal_all();
        let mut patch = MirPatch::new(body);

        for (bb, bb_data) in body.basic_blocks().iter_enumerated() {
            let terminator = bb_data.terminator();
            if let TerminatorKind::Call {
                func: ref op,
                ref args,
                ref destination,
                cleanup,
                from_hir_call,
            } = terminator.kind
            {
                if let ty::FnDef(callee_def_id, substs) = op.ty(&**body, tcx).kind {
                    let name = tcx.def_path_str(callee_def_id);
                    if name == VEC_CONSTRUCTOR {
                        let elem_ty = substs.type_at(0);
                        let impls_manageable_contents =
                            ty_impls_manageable_contents(elem_ty, tcx, param_env);
                        if impls_manageable_contents {
                            // If our T in Vec<T> implements ManageableContents, then we patch
                            // Vec::new to use BoehmAllocator instead of Global alloc.
                            let new_in_col_fn = tcx.lang_items().new_in_collector_fn().unwrap();
                            let new_in_col_fn_op =
                                Operand::function_handle(tcx, new_in_col_fn, substs, DUMMY_SP);
                            let new_term = TerminatorKind::Call {
                                func: new_in_col_fn_op,
                                args: args.clone(),
                                destination: destination.clone(),
                                cleanup,
                                from_hir_call,
                            };
                            patch.patch_terminator(bb, new_term);
                        }
                    }
                }
            }
        }
        patch.apply(body);
    }
}

fn ty_impls_manageable_contents<'tcx>(
    ty: Ty<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> bool {
    let prealloc_trait_id = tcx.lang_items().manageable_contents_trait().unwrap();
    let obligation = predicate_for_trait_def(
        tcx,
        param_env,
        traits::ObligationCause::dummy(),
        prealloc_trait_id,
        0,
        ty,
        &[],
    );
    tcx.infer_ctxt().enter(|infcx| infcx.predicate_must_hold_modulo_regions(&obligation))
}
