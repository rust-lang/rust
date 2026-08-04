//! This pass removes `PlaceMention` statement, which has no effect at codegen.

use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::trace;

use crate::PassPolicy;

pub(super) struct RemovePlaceMention;

impl<'tcx> crate::MirPass<'tcx> for RemovePlaceMention {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(!ctx.opts.unstable_opts.mir_preserve_ub)
    }

    fn run_pass(&self, _: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running RemovePlaceMention on {:?}", body.source);
        for data in body.basic_blocks.as_mut_preserves_cfg() {
            data.retain_statements(|statement| match statement.kind {
                StatementKind::PlaceMention(..) | StatementKind::Nop => false,
                _ => true,
            })
        }
    }
}
