//! Various optimizations specific to cg_clif

use crate::prelude::*;

pub(crate) mod peephole;

pub(crate) fn optimize_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    ctx: &mut Context,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
) {
    // FIXME classify optimizations over opt levels once we have more

    crate::pretty_clif::write_clif_file(tcx, "preopt", None, instance, &ctx, &*clif_comments);
    crate::base::verify_func(tcx, &*clif_comments, &ctx.func);
}
