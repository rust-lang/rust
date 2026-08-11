use rustc_middle::mir::BinOp;
use rustc_middle::{mir, span_bug, ty};
use rustc_span::{Symbol, sym};
use tracing::trace;

use super::{
    AtomicRmwOp, Immediate, InterpCx, InterpResult, Machine, OpTy, PlaceTy, Scalar, interp_ok,
};

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Returns `true` if emulation happened.
    /// Here we implement the intrinsics that are common to all CTFE instances; individual machines can add their own
    /// intrinsic handling.
    pub fn eval_atomic_intrinsic(
        &mut self,
        intrinsic_name: Symbol,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        let get_ord_at = |i: usize| {
            let ordering = generic_args.const_at(i).to_value();
            ordering.to_branch()[0].to_value().to_leaf().to_atomic_ordering()
        };

        match intrinsic_name {
            sym::atomic_load => {
                let ord = get_ord_at(1);
                let [ptr] = args else { span_bug!(self.cur_span(), "invalid `atomic_load` call") };

                let place = self.deref_pointer(ptr)?;
                let val = M::atomic_load(self, &place, ord)?;
                self.write_scalar(val, dest)?;
            }
            sym::atomic_store => {
                let ord = get_ord_at(1);
                let [ptr, val] = args else {
                    span_bug!(self.cur_span(), "invalid `atomic_store` call")
                };

                let place = self.deref_pointer(ptr)?;
                let val = self.read_immediate(val)?;
                M::atomic_store(self, &place, &val, ord)?;
            }
            sym::atomic_or
            | sym::atomic_xor
            | sym::atomic_and
            | sym::atomic_nand
            | sym::atomic_xadd
            | sym::atomic_xsub
            | sym::atomic_min
            | sym::atomic_umin
            | sym::atomic_max
            | sym::atomic_umax
            | sym::atomic_xchg => {
                let num_ty_generics = match intrinsic_name {
                    sym::atomic_min
                    | sym::atomic_umin
                    | sym::atomic_max
                    | sym::atomic_umax
                    | sym::atomic_xchg => 1,
                    _ => 2,
                };
                let ord = get_ord_at(num_ty_generics);
                let [ptr, operand] = args else {
                    span_bug!(self.cur_span(), "invalid `{intrinsic_name}` call")
                };

                let place = self.deref_pointer(ptr)?;
                let operand = self.read_immediate(operand)?;

                let op = match intrinsic_name {
                    sym::atomic_or => AtomicRmwOp::MirOp { op: BinOp::BitOr, neg: false },
                    sym::atomic_xor => AtomicRmwOp::MirOp { op: BinOp::BitXor, neg: false },
                    sym::atomic_and => AtomicRmwOp::MirOp { op: BinOp::BitAnd, neg: false },
                    sym::atomic_nand => AtomicRmwOp::MirOp { op: BinOp::BitAnd, neg: true },
                    sym::atomic_xadd => AtomicRmwOp::MirOp { op: BinOp::Add, neg: false },
                    sym::atomic_xsub => AtomicRmwOp::MirOp { op: BinOp::Sub, neg: false },
                    sym::atomic_min => AtomicRmwOp::Min,
                    sym::atomic_umin => AtomicRmwOp::Min,
                    sym::atomic_max => AtomicRmwOp::Max,
                    sym::atomic_umax => AtomicRmwOp::Max,
                    sym::atomic_xchg => AtomicRmwOp::Swap,
                    _ => unreachable!(),
                };

                let res = M::atomic_rmw(self, &place, op, &operand, ord)?;
                self.write_scalar(res, dest)?;
            }
            sym::atomic_cxchg | sym::atomic_cxchgweak => {
                let success_ord = get_ord_at(1);
                let failure_ord = get_ord_at(2);
                let [ptr, expected_old, new] = args else {
                    span_bug!(self.cur_span(), "invalid `{intrinsic_name}` call")
                };

                let place = self.deref_pointer(ptr)?;
                let expected_old = self.read_immediate(expected_old)?;
                let new = self.read_immediate(new)?;

                let (actual_old, success) = M::atomic_compare_exchange(
                    self,
                    &place,
                    &expected_old,
                    &new,
                    /* can_fail_spuriously */ intrinsic_name == sym::atomic_cxchgweak,
                    success_ord,
                    failure_ord,
                )?;
                let res = Immediate::ScalarPair(actual_old, Scalar::from_bool(success));
                self.write_immediate(res, dest)?;
            }
            sym::atomic_fence | sym::atomic_singlethreadfence => {
                let ord = get_ord_at(0);
                let [] = args else {
                    span_bug!(self.cur_span(), "invalid `{intrinsic_name}` call")
                };

                M::atomic_fence(self, ord, intrinsic_name == sym::atomic_singlethreadfence)?;
            }

            // Unsupported intrinsic: skip the return_to_block below.
            _ => return interp_ok(false),
        }

        trace!("{:?}", self.dump_place(&dest.clone().into()));
        self.return_to_block(ret)?;
        interp_ok(true)
    }
}
