use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::{self, ClosureSizeProfileData, Instance, TyCtxt};

/// For a given closure, writes out the data for the profiling the impact of RFC 2229 on
/// closure size into a CSV.
///
/// During the same compile all closures dump the information in the same file
/// "closure_profile_XXXXX.csv", which is created in the directory where the compiler is invoked.
crate fn dump_closure_profile(tcx: TyCtxt<'tcx>, closure_instance: Instance<'tcx>) {
    let closure_def_id = closure_instance.def_id();
    let typeck_results = tcx.typeck(closure_def_id.expect_local());

    let crate_name = tcx.crate_name(LOCAL_CRATE);

    if typeck_results.closure_size_eval.contains_key(&closure_def_id) {
        let param_env = ty::ParamEnv::reveal_all();

        let ClosureSizeProfileData { before_feature_tys, after_feature_tys } =
            typeck_results.closure_size_eval[&closure_def_id];

        let before_feature_tys = tcx.subst_and_normalize_erasing_regions(
            closure_instance.substs,
            param_env,
            before_feature_tys,
        );
        let after_feature_tys = tcx.subst_and_normalize_erasing_regions(
            closure_instance.substs,
            param_env,
            after_feature_tys,
        );

        let new_size = tcx
            .layout_of(param_env.and(after_feature_tys))
            .map(|l| format!("{:?}", l.size.bytes()))
            .unwrap_or_else(|e| format!("Failed {:?}", e));

        let old_size = tcx
            .layout_of(param_env.and(before_feature_tys))
            .map(|l| format!("{:?}", l.size.bytes()))
            .unwrap_or_else(|e| format!("Failed {:?}", e));

        let mut hasher = StableHasher::new();
        closure_instance.hash_stable(&mut tcx.create_stable_hashing_context(), &mut hasher);
        let hash = hasher.finalize();

        eprintln!("SG_CR_Eslkdjf: {}, {:x?}, {}, {}", crate_name, hash, old_size, new_size);
    }
}
