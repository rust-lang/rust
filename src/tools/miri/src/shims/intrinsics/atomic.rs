use rustc_middle::{mir, mir::BinOp, ty};

use crate::*;
use helpers::check_arg_count;

pub enum AtomicOp {
    /// The `bool` indicates whether the result of the operation should be negated
    /// (must be a boolean-typed operation).
    MirOp(mir::BinOp, bool),
    Max,
    Min,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Calls the atomic intrinsic `intrinsic`; the `atomic_` prefix has already been removed.
    fn emulate_atomic_intrinsic(
        &mut self,
        intrinsic_name: &str,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let intrinsic_structure: Vec<_> = intrinsic_name.split('_').collect();

        fn read_ord<'tcx>(ord: &str) -> InterpResult<'tcx, AtomicReadOrd> {
            Ok(match ord {
                "seqcst" => AtomicReadOrd::SeqCst,
                "acquire" => AtomicReadOrd::Acquire,
                "relaxed" => AtomicReadOrd::Relaxed,
                _ => throw_unsup_format!("unsupported read ordering `{ord}`"),
            })
        }

        fn write_ord<'tcx>(ord: &str) -> InterpResult<'tcx, AtomicWriteOrd> {
            Ok(match ord {
                "seqcst" => AtomicWriteOrd::SeqCst,
                "release" => AtomicWriteOrd::Release,
                "relaxed" => AtomicWriteOrd::Relaxed,
                _ => throw_unsup_format!("unsupported write ordering `{ord}`"),
            })
        }

        fn rw_ord<'tcx>(ord: &str) -> InterpResult<'tcx, AtomicRwOrd> {
            Ok(match ord {
                "seqcst" => AtomicRwOrd::SeqCst,
                "acqrel" => AtomicRwOrd::AcqRel,
                "acquire" => AtomicRwOrd::Acquire,
                "release" => AtomicRwOrd::Release,
                "relaxed" => AtomicRwOrd::Relaxed,
                _ => throw_unsup_format!("unsupported read-write ordering `{ord}`"),
            })
        }

        fn fence_ord<'tcx>(ord: &str) -> InterpResult<'tcx, AtomicFenceOrd> {
            Ok(match ord {
                "seqcst" => AtomicFenceOrd::SeqCst,
                "acqrel" => AtomicFenceOrd::AcqRel,
                "acquire" => AtomicFenceOrd::Acquire,
                "release" => AtomicFenceOrd::Release,
                _ => throw_unsup_format!("unsupported fence ordering `{ord}`"),
            })
        }

        match &*intrinsic_structure {
            ["load", ord] => this.atomic_load(args, dest, read_ord(ord)?)?,
            ["store", ord] => this.atomic_store(args, write_ord(ord)?)?,

            ["fence", ord] => this.atomic_fence_intrinsic(args, fence_ord(ord)?)?,
            ["singlethreadfence", ord] => this.compiler_fence_intrinsic(args, fence_ord(ord)?)?,

            ["xchg", ord] => this.atomic_exchange(args, dest, rw_ord(ord)?)?,
            ["cxchg", ord1, ord2] =>
                this.atomic_compare_exchange(args, dest, rw_ord(ord1)?, read_ord(ord2)?)?,
            ["cxchgweak", ord1, ord2] =>
                this.atomic_compare_exchange_weak(args, dest, rw_ord(ord1)?, read_ord(ord2)?)?,

            ["or", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), rw_ord(ord)?)?,
            ["xor", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), rw_ord(ord)?)?,
            ["and", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), rw_ord(ord)?)?,
            ["nand", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), rw_ord(ord)?)?,
            ["xadd", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), rw_ord(ord)?)?,
            ["xsub", ord] =>
                this.atomic_rmw_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), rw_ord(ord)?)?,
            ["min", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Int(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Min, rw_ord(ord)?)?;
            }
            ["umin", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Uint(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Min, rw_ord(ord)?)?;
            }
            ["max", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Int(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Max, rw_ord(ord)?)?;
            }
            ["umax", ord] => {
                // Later we will use the type to indicate signed vs unsigned,
                // so make sure it matches the intrinsic name.
                assert!(matches!(args[1].layout.ty.kind(), ty::Uint(_)));
                this.atomic_rmw_op(args, dest, AtomicOp::Max, rw_ord(ord)?)?;
            }

            _ => throw_unsup_format!("unimplemented intrinsic: `atomic_{intrinsic_name}`"),
        }
        Ok(())
    }
}

impl<'mir, 'tcx: 'mir> EvalContextPrivExt<'mir, 'tcx> for MiriInterpCx<'mir, 'tcx> {}
trait EvalContextPrivExt<'mir, 'tcx: 'mir>: MiriInterpCxExt<'mir, 'tcx> {
    fn atomic_load(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        atomic: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place] = check_arg_count(args)?;
        let place = this.deref_pointer(place)?;

        // Perform atomic load.
        let val = this.read_scalar_atomic(&place, atomic)?;
        // Perform regular store.
        this.write_scalar(val, dest)?;
        Ok(())
    }

    fn atomic_store(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        atomic: AtomicWriteOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, val] = check_arg_count(args)?;
        let place = this.deref_pointer(place)?;

        // Perform regular load.
        let val = this.read_scalar(val)?;
        // Perform atomic store
        this.write_scalar_atomic(val, &place, atomic)?;
        Ok(())
    }

    fn compiler_fence_intrinsic(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let [] = check_arg_count(args)?;
        let _ = atomic;
        //FIXME: compiler fences are currently ignored
        Ok(())
    }

    fn atomic_fence_intrinsic(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [] = check_arg_count(args)?;
        this.atomic_fence(atomic)?;
        Ok(())
    }

    fn atomic_rmw_op(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        atomic_op: AtomicOp,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, rhs] = check_arg_count(args)?;
        let place = this.deref_pointer(place)?;
        let rhs = this.read_immediate(rhs)?;

        if !place.layout.ty.is_integral() && !place.layout.ty.is_unsafe_ptr() {
            span_bug!(
                this.cur_span(),
                "atomic arithmetic operations only work on integer and raw pointer types",
            );
        }
        if rhs.layout.ty != place.layout.ty {
            span_bug!(this.cur_span(), "atomic arithmetic operation type mismatch");
        }

        match atomic_op {
            AtomicOp::Min => {
                let old = this.atomic_min_max_scalar(&place, rhs, true, atomic)?;
                this.write_immediate(*old, dest)?; // old value is returned
                Ok(())
            }
            AtomicOp::Max => {
                let old = this.atomic_min_max_scalar(&place, rhs, false, atomic)?;
                this.write_immediate(*old, dest)?; // old value is returned
                Ok(())
            }
            AtomicOp::MirOp(op, neg) => {
                let old = this.atomic_rmw_op_immediate(&place, &rhs, op, neg, atomic)?;
                this.write_immediate(*old, dest)?; // old value is returned
                Ok(())
            }
        }
    }

    fn atomic_exchange(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, new] = check_arg_count(args)?;
        let place = this.deref_pointer(place)?;
        let new = this.read_scalar(new)?;

        let old = this.atomic_exchange_scalar(&place, new, atomic)?;
        this.write_scalar(old, dest)?; // old value is returned
        Ok(())
    }

    fn atomic_compare_exchange_impl(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
        can_fail_spuriously: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, expect_old, new] = check_arg_count(args)?;
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
        Ok(())
    }

    fn atomic_compare_exchange(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, false)
    }

    fn atomic_compare_exchange_weak(
        &mut self,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, true)
    }
}
