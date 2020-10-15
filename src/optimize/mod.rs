//! Various optimizations specific to cg_clif

use crate::prelude::*;

mod code_layout;
pub(crate) mod peephole;
mod stack2reg;

pub(crate) fn optimize_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    ctx: &mut Context,
    cold_blocks: &EntitySet<Block>,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
) {
    // The code_layout optimization is very cheap.
    self::code_layout::optimize_function(ctx, cold_blocks);

    if tcx.sess.opts.optimize == rustc_session::config::OptLevel::No {
        return; // FIXME classify optimizations over opt levels
    }
    self::stack2reg::optimize_function(ctx, clif_comments);
    crate::pretty_clif::write_clif_file(tcx, "stack2reg", None, instance, &ctx, &*clif_comments);
    crate::base::verify_func(tcx, &*clif_comments, &ctx.func);
}
