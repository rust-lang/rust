use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, TyCtxt};
use rustc_mir_dataflow::impls::SingleEnumVariant;
use rustc_mir_dataflow::Analysis;
use rustc_span::DUMMY_SP;

use crate::MirPass;

pub struct SingleEnum;

impl<'tcx> MirPass<'tcx> for SingleEnum {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut single_enum_variants = SingleEnumVariant::new(tcx, body)
            .into_engine(tcx, body)
            .iterate_to_fixpoint()
            .into_results_cursor(body);

        let mut discrs = vec![];

        for (bb, block) in body.basic_blocks().iter_enumerated() {
            for (i, stmt) in block.statements.iter().enumerate() {
                let stmt_loc = Location { block: bb.clone(), statement_index: i };
                let StatementKind::Assign(box(_, Rvalue::Discriminant(src))) = stmt.kind
                else { continue };
                if !src.ty(body, tcx).ty.is_enum() {
                    continue;
                }
                let Some(src_local) = src.local_or_deref_local() else { continue };

                single_enum_variants.seek_before_primary_effect(stmt_loc);
                match single_enum_variants.get().get(src_local) {
                    None => {}
                    Some((_, &v)) => discrs.push((stmt_loc, v)),
                };
            }
        }

        for (Location { block, statement_index }, val) in discrs {
            let local_decls = &body.local_decls;
            let bbs = body.basic_blocks.as_mut();

            let stmt = &mut bbs[block].statements[statement_index];
            let Some((lhs, rval)) = stmt.kind.as_assign_mut() else { unreachable!() };
            let Rvalue::Discriminant(rhs) = rval else { unreachable!() };

            let Some(disc) = rhs.ty(local_decls, tcx).ty.discriminant_for_variant(tcx, val)
            else { continue };

            let scalar_ty = lhs.ty(local_decls, tcx).ty;
            let layout = tcx.layout_of(ParamEnv::empty().and(scalar_ty)).unwrap().layout;
            let ct = Operand::const_from_scalar(
                tcx,
                scalar_ty,
                Scalar::from_uint(disc.val, layout.size()),
                DUMMY_SP,
            );
            *rval = Rvalue::Use(ct);
        }
    }
}
