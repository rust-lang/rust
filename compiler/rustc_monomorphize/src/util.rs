use rustc_middle::ty::{self, ClosureSizeProfileData, Instance, TyCtxt};
use std::fs::OpenOptions;
use std::io::prelude::*;

/// For a given closure, writes out the data for the profiling the impact of RFC 2229 on
/// closure size into a CSV.
///
/// During the same compile all closures dump the information in the same file
/// "closure_profile_XXXXX.csv", which is created in the directory where the compiler is invoked.
pub(crate) fn dump_closure_profile<'tcx>(tcx: TyCtxt<'tcx>, closure_instance: Instance<'tcx>) {
    let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&format!("closure_profile_{}.csv", std::process::id()))
    else {
        eprintln!("Couldn't open file for writing closure profile");
        return;
    };

    let closure_def_id = closure_instance.def_id().expect_local();
    let typeck_results = tcx.typeck(closure_def_id);

    if typeck_results.closure_size_eval.contains_key(&closure_def_id) {
        let param_env = ty::ParamEnv::reveal_all();

        let ClosureSizeProfileData { before_feature_tys, after_feature_tys } =
            typeck_results.closure_size_eval[&closure_def_id];

        let before_feature_tys = tcx.subst_and_normalize_erasing_regions(
            closure_instance.args,
            param_env,
            ty::EarlyBinder::bind(before_feature_tys),
        );
        let after_feature_tys = tcx.subst_and_normalize_erasing_regions(
            closure_instance.args,
            param_env,
            ty::EarlyBinder::bind(after_feature_tys),
        );

        let new_size = tcx
            .layout_of(param_env.and(after_feature_tys))
            .map(|l| format!("{:?}", l.size.bytes()))
            .unwrap_or_else(|e| format!("Failed {e:?}"));

        let old_size = tcx
            .layout_of(param_env.and(before_feature_tys))
            .map(|l| format!("{:?}", l.size.bytes()))
            .unwrap_or_else(|e| format!("Failed {e:?}"));

        let closure_span = tcx.def_span(closure_def_id);
        let src_file = tcx.sess.source_map().span_to_filename(closure_span);
        let line_nos = tcx
            .sess
            .source_map()
            .span_to_lines(closure_span)
            .map(|l| format!("{:?} {:?}", l.lines.first(), l.lines.last()))
            .unwrap_or_else(|e| format!("{e:?}"));

        if let Err(e) = writeln!(
            file,
            "{}, {}, {}, {:?}",
            old_size,
            new_size,
            src_file.prefer_local(),
            line_nos
        ) {
            eprintln!("Error writing to file {e}")
        }
    }
}
