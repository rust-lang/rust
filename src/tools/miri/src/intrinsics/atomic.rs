use rustc_middle::mir::BinOp;
use rustc_middle::ty::AtomicOrdering;
use rustc_middle::{mir, ty};

use self::helpers::check_intrinsic_arg_count;
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

        let intrinsic_structure: Vec<_> = intrinsic_name.split('_').collect();

        fn read_ord(ord: &str) -> AtomicReadOrd {
            match ord {
                "seqcst" => AtomicReadOrd::SeqCst,
                "acquire" => AtomicReadOrd::Acquire,
                "relaxed" => AtomicReadOrd::Relaxed,
                _ => panic!("invalid read ordering `{ord}`"),
            }
        }

        fn read_ord_const_generic(o: AtomicOrdering) -> AtomicReadOrd {
            match o {
                AtomicOrdering::SeqCst => AtomicReadOrd::SeqCst,
                AtomicOrdering::Acquire => AtomicReadOrd::Acquire,
                AtomicOrdering::Relaxed => AtomicReadOrd::Relaxed,
                _ => panic!("invalid read ordering `{o:?}`"),
            }
        }

        fn write_ord(ord: &str) -> AtomicWriteOrd {
            match ord {
                "seqcst" => AtomicWriteOrd::SeqCst,
                "release" => AtomicWriteOrd::Release,
                "relaxed" => AtomicWriteOrd::Relaxed,
                _ => panic!("invalid write ordering `{ord}`"),
            }
        }

        fn rw_ord(ord: &str) -> AtomicRwOrd {
            match ord {
                "seqcst" => AtomicRwOrd::SeqCst,
                "acqrel" => AtomicRwOrd::AcqRel,
                "acquire" => AtomicRwOrd::Acquire,
                "release" => AtomicRwOrd::Release,
                "relaxed" => AtomicRwOrd::Relaxed,
                _ => panic!("invalid read-write ordering `{ord}`"),
            }
        }

        fn fence_ord(ord: &str) -> AtomicFenceOrd {
            match ord {
                "seqcst" => AtomicFenceOrd::SeqCst,
                "acqrel" => AtomicFenceOrd::AcqRel,
                "acquire" => AtomicFenceOrd::Acquire,
                "release" => AtomicFenceOrd::Release,
                _ => panic!("invalid fence ordering `{ord}`"),
            }
        }

        match &*intrinsic_structure {
            // New-style intrinsics that use const generics
            ["load"] => {
                let ordering = generic_args.const_at(1).to_value();
                let ordering =
                    ordering.valtree.unwrap_branch()[0].unwrap_leaf().to_atomic_ordering();
                this.atomic_load(args, dest, read_ord_const_generic(ordering))?;
            }

            // Old-style intrinsics that have the ordering in the intrinsic name
            ["store", ord] => this.atomic_store(args, write_ord(ord))?,

            ["fence", ord] => this.atomic_fence_intrinsic(args, fence_ord(ord))?,
            ["singlethreadfence", ord] => this.compiler_fence_intrinsic(args, fence_ord(ord))?,

            ["xchg", ord] => this.atomic_exchange(args, dest, rw_ord(ord))?,
            ["cxchg", ord1, ord2] =>
                this.atomic_compare_exchange(args, dest, rw_ord(ord1), read_ord(ord2))?,
            ["cxchgweak", ord1, ord2] =>
                this.atomic_compare_exchange_weak(args, dest, rw_ord(ord1), read_ord(ord2))?,

            ["or", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), rw_ord(ord))?,
            ["xor", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), rw_ord(ord))?,
            ["and", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), rw_ord(ord))?,
            ["nand", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), rw_ord(ord))?,
            ["xadd", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), rw_ord(ord))?,
            ["xsub", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), rw_ord(ord))?,
            ["min", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Int(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Min, rw_ord(ord))?;
            }
            ["umin", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Uint(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Min, rw_ord(ord))?;
            }
            ["max", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Int(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Max, rw_ord(ord))?;
            }
            ["umax", ord] => {
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

        if !place.layout.ty.is_integral() && !place.layout.ty.is_raw_ptr() {
            span_bug!(
                this.cur_span(),
                "atomic arithmetic operations only work on integer and raw pointer types",
            );
        }
        if rhs.layout.ty != place.layout.ty {
            span_bug!(this.cur_span(), "atomic arithmetic operation type mismatch");
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
