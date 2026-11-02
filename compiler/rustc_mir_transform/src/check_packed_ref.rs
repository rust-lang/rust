use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::span_bug;

use crate::{diagnostics, util};

pub(super) struct CheckPackedRef;

impl<'tcx> crate::MirLint<'tcx> for CheckPackedRef {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let mut checker = PackedRefChecker { body, tcx, typing_env };
        for (bb, data) in body.basic_blocks.iter_enumerated() {
            checker.visit_basic_block_data(bb, data);
        }
    }
}

struct PackedRefChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> Visitor<'tcx> for PackedRefChecker<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if context.is_borrow()
            && let Some((adt, pack)) =
                util::place_unalignment(self.tcx, self.body, self.typing_env, *place)
        {
            let source_info = self.body.source_info(location);
            let def_id = self.body.source.instance.def_id();
            if let Some(impl_def_id) = self.tcx.trait_impl_of_assoc(def_id)
                && self.tcx.is_builtin_derived(impl_def_id)
            {
                // If we ever reach here it means that the generated derive
                // code is somehow doing an unaligned reference, which it
                // shouldn't do.
                span_bug!(source_info.span, "builtin derive created an unaligned reference");
            } else {
                self.tcx.dcx().emit_err(diagnostics::UnalignedPackedRef {
                    span: source_info.span,
                    ty_descr: adt.descr(),
                    align: pack.bytes(),
                });
            }
        }
    }
}
