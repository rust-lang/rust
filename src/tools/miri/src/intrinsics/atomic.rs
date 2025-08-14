use rustc_middle::mir::BinOp;
use rustc_middle::ty::AtomicOrdering;
use rustc_middle::{mir, ty};

use super::check_intrinsic_arg_count;
use crate::*;

pub enum AtomicOp {
    /// The `bool` indicates whether the result of the operation should be negated (`UnOp::Not`,
    /// must be a boolean-typed operation).
    MirOp(mir::BinOp, bool),
    Max,
    Min,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Calls the atomic intrinsic `intrinsic`; the `atomic_` prefix has already been removed.
    /// Returns `Ok(true)` if the intrinsic was handled.
    fn emulate_atomic_intrinsic(
        &mut self,
        intrinsic_name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        let get_ord_at = |i: usize| {
            let ordering = generic_args.const_at(i).to_value();
            ordering.valtree.unwrap_branch()[0].unwrap_leaf().to_atomic_ordering()
        };

        fn read_ord(ord: AtomicOrdering) -> AtomicReadOrd {
            match ord {
                AtomicOrdering::SeqCst => AtomicReadOrd::SeqCst,
                AtomicOrdering::Acquire => AtomicReadOrd::Acquire,
                AtomicOrdering::Relaxed => AtomicReadOrd::Relaxed,
                _ => panic!("invalid read ordering `{ord:?}`"),
            }
        }

        fn write_ord(ord: AtomicOrdering) -> AtomicWriteOrd {
            match ord {
                AtomicOrdering::SeqCst => AtomicWriteOrd::SeqCst,
                AtomicOrdering::Release => AtomicWriteOrd::Release,
                AtomicOrdering::Relaxed => AtomicWriteOrd::Relaxed,
                _ => panic!("invalid write ordering `{ord:?}`"),
            }
        }

        fn rw_ord(ord: AtomicOrdering) -> AtomicRwOrd {
            match ord {
                AtomicOrdering::SeqCst => AtomicRwOrd::SeqCst,
                AtomicOrdering::AcqRel => AtomicRwOrd::AcqRel,
                AtomicOrdering::Acquire => AtomicRwOrd::Acquire,
                AtomicOrdering::Release => AtomicRwOrd::Release,
                AtomicOrdering::Relaxed => AtomicRwOrd::Relaxed,
            }
        }

        fn fence_ord(ord: AtomicOrdering) -> AtomicFenceOrd {
            match ord {
                AtomicOrdering::SeqCst => AtomicFenceOrd::SeqCst,
                AtomicOrdering::AcqRel => AtomicFenceOrd::AcqRel,
                AtomicOrdering::Acquire => AtomicFenceOrd::Acquire,
                AtomicOrdering::Release => AtomicFenceOrd::Release,
                _ => panic!("invalid fence ordering `{ord:?}`"),
            }
        }

        match intrinsic_name {
            "load" => {
                let ord = get_ord_at(1);
                this.atomic_load(args, dest, read_ord(ord))?;
            }

            "store" => {
                let ord = get_ord_at(1);
                this.atomic_store(args, write_ord(ord))?
            }

            "fence" => {
                let ord = get_ord_at(0);
                this.atomic_fence_intrinsic(args, fence_ord(ord))?
            }
            "singlethreadfence" => {
                let ord = get_ord_at(0);
                this.compiler_fence_intrinsic(args, fence_ord(ord))?;
            }

            "xchg" => {
                let ord = get_ord_at(1);
                this.atomic_exchange(args, dest, rw_ord(ord))?;
            }
            "cxchg" => {
                let ord1 = get_ord_at(1);
                let ord2 = get_ord_at(2);
                this.atomic_compare_exchange(args, dest, rw_ord(ord1), read_ord(ord2))?;
            }
            "cxchgweak" => {
                let ord1 = get_ord_at(1);
                let ord2 = get_ord_at(2);
                this.atomic_compare_exchange_weak(args, dest, rw_ord(ord1), read_ord(ord2))?;
            }

            "or" => {
                let ord = get_ord_at(2);
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), rw_ord(ord))?;
            }
            "xor" => {
                let ord = get_ord_at(2);
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), rw_ord(ord))?;
            }
            "and" => {
                let ord = get_ord_at(2);
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), rw_ord(ord))?;
            }
            "nand" => {
                let ord = get_ord_at(2);
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), rw_ord(ord))?;
            }
            "xadd" => {
                let ord = get_ord_at(2);
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), rw_ord(ord))?;
            }
            "xsub" => {
                let ord = get_ord_at(2);
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), rw_ord(ord))?;
            }
            "min" => {
                let ord = get_ord_at(1);
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Int(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Min, rw_ord(ord))?;
            }
            "umin" => {
                let ord = get_ord_at(1);
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Uint(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Min, rw_ord(ord))?;
            }
            "max" => {
                let ord = get_ord_at(1);
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Int(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Max, rw_ord(ord))?;
            }
            "umax" => {
                let ord = get_ord_at(1);
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Uint(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Max, rw_ord(ord))?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: MiriInterpCxExt<'tcx> {
    fn atomic_load(
        &mut self,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        atomic: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place] = check_intrinsic_arg_count(args)?;
        let place = this.deref_pointer(place)?;

        // Perform atomic load.
        let val = this.read_scalar_atomic(&place, atomic)?;
        // Perform regular store.
        this.write_scalar(val, dest)?;
        interp_ok(())
    }

    fn atomic_store(&mut self, args: &[OpTy<'tcx>], atomic: AtomicWriteOrd) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, val] = check_intrinsic_arg_count(args)?;
        let place = this.deref_pointer(place)?;

        // Perform regular load.
        let val = this.read_scalar(val)?;
        // Perform atomic store.
        this.write_scalar_atomic(val, &place, atomic)?;
        interp_ok(())
    }

    fn compiler_fence_intrinsic(
        &mut self,
        args: &[OpTy<'tcx>],
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let [] = check_intrinsic_arg_count(args)?;
        let _ = atomic;
        // FIXME, FIXME(GenMC): compiler fences are currently ignored (also ignored in GenMC mode)
        interp_ok(())
    }

    fn atomic_fence_intrinsic(
        &mut self,
        args: &[OpTy<'tcx>],
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [] = check_intrinsic_arg_count(args)?;
        this.atomic_fence(atomic)?;
        interp_ok(())
    }

    fn atomic_rmw_op(
        &mut self,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        atomic_op: AtomicOp,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, rhs] = check_intrinsic_arg_count(args)?;
        let place = this.deref_pointer(place)?;
        let rhs = this.read_immediate(rhs)?;

        if !(place.layout.ty.is_integral() || place.layout.ty.is_raw_ptr())
            || !(rhs.layout.ty.is_integral() || rhs.layout.ty.is_raw_ptr())
        {
            span_bug!(
                this.cur_span(),
                "atomic arithmetic operations only work on integer and raw pointer types",
            );
        }

        let old = match atomic_op {
            AtomicOp::Min =>
                this.atomic_min_max_scalar(&place, rhs, /* min */ true, atomic)?,
            AtomicOp::Max =>
                this.atomic_min_max_scalar(&place, rhs, /* min */ false, atomic)?,
            AtomicOp::MirOp(op, not) =>
                this.atomic_rmw_op_immediate(&place, &rhs, op, not, atomic)?,
        };
        this.write_immediate(*old, dest)?; // old value is returned
        interp_ok(())
    }

    fn atomic_exchange(
        &mut self,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, new] = check_intrinsic_arg_count(args)?;
        let place = this.deref_pointer(place)?;
        let new = this.read_scalar(new)?;

        let old = this.atomic_exchange_scalar(&place, new, atomic)?;
        this.write_scalar(old, dest)?; // old value is returned
        interp_ok(())
    }

    fn atomic_compare_exchange_impl(
        &mut self,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
        can_fail_spuriously: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, expect_old, new] = check_intrinsic_arg_count(args)?;
        let place = this.deref_pointer(place)?;
        let expect_old = this.read_immediate(expect_old)?; // read as immediate for the sake of `binary_op()`
        let new = this.read_scalar(new)?;

        let old = this.atomic_compare_exchange_scalar(
            &place,
            &expect_old,
            new,
            success,
            fail,
            can_fail_spuriously,
        )?;

        // Return old value.
        this.write_immediate(old, dest)?;
        interp_ok(())
    }

    fn atomic_compare_exchange(
        &mut self,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, false)
    }

    fn atomic_compare_exchange_weak(
        &mut self,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, true)
    }
}
