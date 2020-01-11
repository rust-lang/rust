use crate::prelude::*;

mod code_layout;
mod stack2reg;

pub fn optimize_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    ctx: &mut Context,
    cold_ebbs: &EntitySet<Ebb>,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
) {
    // The code_layout optimization is very cheap.
    self::code_layout::optimize_function(ctx, cold_ebbs);

    if tcx.sess.opts.optimize == rustc_session::config::OptLevel::No {
        return; // FIXME classify optimizations over opt levels
    }
    self::stack2reg::optimize_function(ctx, clif_comments, instance);
    #[cfg(debug_assertions)]
    crate::pretty_clif::write_clif_file(tcx, "stack2reg", instance, &ctx.func, &*clif_comments, None);
    crate::base::verify_func(tcx, &*clif_comments, &ctx.func);
}
