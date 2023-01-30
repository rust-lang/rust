//! This pass eliminates casting of arrays into slices when their length
//! is taken using `.len()` method. Handy to preserve information in MIR for const prop

use crate::ssa::SsaLocals;
use crate::MirPass;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_mir_dataflow::impls::borrowed_locals;

pub struct NormalizeArrayLen;

impl<'tcx> MirPass<'tcx> for NormalizeArrayLen {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 3
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());
        normalize_array_len_calls(tcx, body)
    }
}

fn normalize_array_len_calls<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
    let borrowed_locals = borrowed_locals(body);
    let ssa = SsaLocals::new(tcx, param_env, body, &borrowed_locals);

    let slice_lengths = compute_slice_length(tcx, &ssa, body);
    debug!(?slice_lengths);

    Replacer { tcx, slice_lengths }.visit_body_preserves_cfg(body);
}

fn compute_slice_length<'tcx>(
    tcx: TyCtxt<'tcx>,
    ssa: &SsaLocals,
    body: &Body<'tcx>,
) -> IndexVec<Local, Option<ty::Const<'tcx>>> {
    let mut slice_lengths = IndexVec::from_elem(None, &body.local_decls);

    for (local, rvalue) in ssa.assignments(body) {
        match rvalue {
            Rvalue::Cast(
                CastKind::Pointer(ty::adjustment::PointerCast::Unsize),
                operand,
                cast_ty,
            ) => {
                let operand_ty = operand.ty(body, tcx);
                debug!(?operand_ty);
                if let Some(operand_ty) = operand_ty.builtin_deref(true)
                    && let ty::Array(_, len) = operand_ty.ty.kind()
                    && let Some(cast_ty) = cast_ty.builtin_deref(true)
                    && let ty::Slice(..) = cast_ty.ty.kind()
                {
                    slice_lengths[local] = Some(*len);
                }
            }
            // The length information is stored in the fat pointer, so we treat `operand` as a value.
            Rvalue::Use(operand) => {
                if let Some(rhs) = operand.place() && let Some(rhs) = rhs.as_local() {
                    slice_lengths[local] = slice_lengths[rhs];
                }
            }
            // The length information is stored in the fat pointer.
            // Reborrowing copies length information from one pointer to the other.
            Rvalue::Ref(_, _, rhs) | Rvalue::AddressOf(_, rhs) => {
                if let [PlaceElem::Deref] = rhs.projection[..] {
                    slice_lengths[local] = slice_lengths[rhs.local];
                }
            }
            _ => {}
        }
    }

    slice_lengths
}

struct Replacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    slice_lengths: IndexVec<Local, Option<ty::Const<'tcx>>>,
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, loc: Location) {
        if let Rvalue::Len(place) = rvalue
            && let [PlaceElem::Deref] = &place.projection[..]
            && let Some(len) = self.slice_lengths[place.local]
        {
            *rvalue = Rvalue::Use(Operand::Constant(Box::new(Constant {
                span: rustc_span::DUMMY_SP,
                user_ty: None,
                literal: ConstantKind::from_const(len, self.tcx),
            })));
        }
        self.super_rvalue(rvalue, loc);
    }
}
