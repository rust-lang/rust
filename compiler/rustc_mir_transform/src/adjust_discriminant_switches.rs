//! This pass lowers calls to core::slice::len to just PtrMetadata op.
//! It should run before inlining!

use rustc_abi as abi;
use rustc_middle::bug;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::trace;

/// Look for `switchInt(Discriminant(place))` where `place` is a niched enum with more than two variants,
/// and update to `switchInt(Discriminant(place) + DELTA)` when that will simplify codegen later.
///
/// Notably, niched enums when calculating `Discriminant(_)` need to adjust the stored tag to produce the
/// discriminant values, so we can adjust the values in the `switchInt` to better match that tag.
pub(super) struct AdjustDiscriminantSwitches;

impl<'tcx> crate::MirPass<'tcx> for AdjustDiscriminantSwitches {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        if tcx.is_coroutine(def_id) {
            trace!("Skipped for coroutine {:?}", def_id);
            return;
        }

        let mut was_modified = false;
        let typing_env = body.typing_env(tcx);
        for block in body.basic_blocks.as_mut_preserves_cfg() {
            let terminator = block.terminator.as_mut().unwrap();
            if let Some(last) = block.statements.last()
                && let TerminatorKind::SwitchInt { discr: switched_op, targets } =
                    &mut terminator.kind
                && let Some(switched_place) = switched_op.place()
                && let StatementKind::Assign(place_and_value) = &last.kind
                && let Rvalue::Discriminant(inspected_place) = place_and_value.1
                && let discr_place = place_and_value.0
                && discr_place == switched_place
                && let ty = inspected_place.ty(&body.local_decls, tcx).ty
                && let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty))
                && let abi::Variants::Multiple { tag_encoding, .. } = &layout.variants
                && let abi::TagEncoding::Niche { niche_start, niche_variants, .. } = tag_encoding
                && niche_variants.clone().count() > 1
            {
                let discr_ty = discr_place.ty(&body.local_decls, tcx).ty;
                let discr_layout = tcx.layout_of(typing_env.as_query_input(discr_ty)).unwrap();
                let abi::BackendRepr::Scalar(discr_scalar) = discr_layout.backend_repr else {
                    bug!()
                };
                let discr_size = discr_scalar.size(&tcx);
                let mask = discr_size.unsigned_int_max();

                let delta = niche_start.wrapping_sub(niche_variants.start().as_u32().into()) & mask;
                if delta == 0 {
                    continue;
                }
                let delta_scalar = Scalar::from_uint(delta, discr_size);

                // No matter what value we use for `delta` this is always sound,
                // since we adjust the scrutinee and targets in the same way.
                //
                // It's just only useful if it allows simplification later.

                was_modified = true;

                let source_info = terminator.source_info;
                let decl = LocalDecl::with_source_info(discr_ty, source_info);
                let local = body.local_decls.push(decl);
                block.statements.push(Statement::new(
                    source_info,
                    StatementKind::Assign(Box::new((
                        Place::from(local),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            Box::new((
                                switched_op.clone(),
                                Operand::const_from_scalar(
                                    tcx,
                                    discr_ty,
                                    delta_scalar,
                                    source_info.span,
                                ),
                            )),
                        ),
                    ))),
                ));
                *switched_op = Operand::Move(Place::from(local));

                for value in targets.all_values_mut() {
                    *value = (value.0.wrapping_add(delta) & mask).into();
                }
            }
        }

        // It's unclear to me whether this is needed since we didn't change the *targets*,
        // but since we did change which *value* goes where, this is safer.
        if was_modified {
            body.basic_blocks.invalidate_cfg_cache();
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}
