use rustc_middle::{mir, mir::BinOp};
use rustc_target::abi::Align;

use crate::*;
use helpers::check_arg_count;

pub enum AtomicOp {
    MirOp(mir::BinOp, bool),
    Max,
    Min,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Calls the atomic intrinsic `intrinsic`; the `atomic_` prefix has already been removed.
    fn emulate_atomic_intrinsic(
        &mut self,
        intrinsic_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        match intrinsic_name {
            // Atomic operations
            "load_seqcst" => this.atomic_load(args, dest, AtomicReadOrd::SeqCst)?,
            "load_relaxed" => this.atomic_load(args, dest, AtomicReadOrd::Relaxed)?,
            "load_acquire" => this.atomic_load(args, dest, AtomicReadOrd::Acquire)?,

            "store_seqcst" => this.atomic_store(args, AtomicWriteOrd::SeqCst)?,
            "store_relaxed" => this.atomic_store(args, AtomicWriteOrd::Relaxed)?,
            "store_release" => this.atomic_store(args, AtomicWriteOrd::Release)?,

            "fence_acquire" => this.atomic_fence(args, AtomicFenceOrd::Acquire)?,
            "fence_release" => this.atomic_fence(args, AtomicFenceOrd::Release)?,
            "fence_acqrel" => this.atomic_fence(args, AtomicFenceOrd::AcqRel)?,
            "fence_seqcst" => this.atomic_fence(args, AtomicFenceOrd::SeqCst)?,

            "singlethreadfence_acquire" => this.compiler_fence(args, AtomicFenceOrd::Acquire)?,
            "singlethreadfence_release" => this.compiler_fence(args, AtomicFenceOrd::Release)?,
            "singlethreadfence_acqrel" => this.compiler_fence(args, AtomicFenceOrd::AcqRel)?,
            "singlethreadfence_seqcst" => this.compiler_fence(args, AtomicFenceOrd::SeqCst)?,

            "xchg_seqcst" => this.atomic_exchange(args, dest, AtomicRwOrd::SeqCst)?,
            "xchg_acquire" => this.atomic_exchange(args, dest, AtomicRwOrd::Acquire)?,
            "xchg_release" => this.atomic_exchange(args, dest, AtomicRwOrd::Release)?,
            "xchg_acqrel" => this.atomic_exchange(args, dest, AtomicRwOrd::AcqRel)?,
            "xchg_relaxed" => this.atomic_exchange(args, dest, AtomicRwOrd::Relaxed)?,

            #[rustfmt::skip]
            "cxchg_seqcst_seqcst" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::SeqCst, AtomicReadOrd::SeqCst)?,
            #[rustfmt::skip]
            "cxchg_acquire_acquire" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::Acquire, AtomicReadOrd::Acquire)?,
            #[rustfmt::skip]
            "cxchg_release_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::Release, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchg_acqrel_acquire" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::AcqRel, AtomicReadOrd::Acquire)?,
            #[rustfmt::skip]
            "cxchg_relaxed_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::Relaxed, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchg_acquire_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::Acquire, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchg_acqrel_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::AcqRel, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchg_seqcst_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::SeqCst, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchg_seqcst_acquire" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOrd::SeqCst, AtomicReadOrd::Acquire)?,

            #[rustfmt::skip]
            "cxchgweak_seqcst_seqcst" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::SeqCst, AtomicReadOrd::SeqCst)?,
            #[rustfmt::skip]
            "cxchgweak_acquire_acquire" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::Acquire, AtomicReadOrd::Acquire)?,
            #[rustfmt::skip]
            "cxchgweak_release_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::Release, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchgweak_acqrel_acquire" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::AcqRel, AtomicReadOrd::Acquire)?,
            #[rustfmt::skip]
            "cxchgweak_relaxed_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::Relaxed, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchgweak_acquire_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::Acquire, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchgweak_acqrel_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::AcqRel, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchgweak_seqcst_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::SeqCst, AtomicReadOrd::Relaxed)?,
            #[rustfmt::skip]
            "cxchgweak_seqcst_acquire" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOrd::SeqCst, AtomicReadOrd::Acquire)?,

            #[rustfmt::skip]
            "or_seqcst" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOrd::SeqCst)?,
            #[rustfmt::skip]
            "or_acquire" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOrd::Acquire)?,
            #[rustfmt::skip]
            "or_release" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOrd::Release)?,
            #[rustfmt::skip]
            "or_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOrd::AcqRel)?,
            #[rustfmt::skip]
            "or_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOrd::Relaxed)?,
            #[rustfmt::skip]
            "xor_seqcst" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOrd::SeqCst)?,
            #[rustfmt::skip]
            "xor_acquire" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOrd::Acquire)?,
            #[rustfmt::skip]
            "xor_release" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOrd::Release)?,
            #[rustfmt::skip]
            "xor_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOrd::AcqRel)?,
            #[rustfmt::skip]
            "xor_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOrd::Relaxed)?,
            #[rustfmt::skip]
            "and_seqcst" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOrd::SeqCst)?,
            #[rustfmt::skip]
            "and_acquire" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOrd::Acquire)?,
            #[rustfmt::skip]
            "and_release" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOrd::Release)?,
            #[rustfmt::skip]
            "and_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOrd::AcqRel)?,
            #[rustfmt::skip]
            "and_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOrd::Relaxed)?,
            #[rustfmt::skip]
            "nand_seqcst" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOrd::SeqCst)?,
            #[rustfmt::skip]
            "nand_acquire" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOrd::Acquire)?,
            #[rustfmt::skip]
            "nand_release" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOrd::Release)?,
            #[rustfmt::skip]
            "nand_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOrd::AcqRel)?,
            #[rustfmt::skip]
            "nand_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOrd::Relaxed)?,
            #[rustfmt::skip]
            "xadd_seqcst" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOrd::SeqCst)?,
            #[rustfmt::skip]
            "xadd_acquire" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOrd::Acquire)?,
            #[rustfmt::skip]
            "xadd_release" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOrd::Release)?,
            #[rustfmt::skip]
            "xadd_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOrd::AcqRel)?,
            #[rustfmt::skip]
            "xadd_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOrd::Relaxed)?,
            #[rustfmt::skip]
            "xsub_seqcst" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOrd::SeqCst)?,
            #[rustfmt::skip]
            "xsub_acquire" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOrd::Acquire)?,
            #[rustfmt::skip]
            "xsub_release" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOrd::Release)?,
            #[rustfmt::skip]
            "xsub_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOrd::AcqRel)?,
            #[rustfmt::skip]
            "xsub_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOrd::Relaxed)?,

            "min_seqcst" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::SeqCst)?,
            "min_acquire" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::Acquire)?,
            "min_release" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::Release)?,
            "min_acqrel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::AcqRel)?,
            "min_relaxed" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::Relaxed)?,
            "max_seqcst" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::SeqCst)?,
            "max_acquire" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::Acquire)?,
            "max_release" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::Release)?,
            "max_acqrel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::AcqRel)?,
            "max_relaxed" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::Relaxed)?,
            "umin_seqcst" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::SeqCst)?,
            "umin_acquire" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::Acquire)?,
            "umin_release" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::Release)?,
            "umin_acqrel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::AcqRel)?,
            "umin_relaxed" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOrd::Relaxed)?,
            "umax_seqcst" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::SeqCst)?,
            "umax_acquire" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::Acquire)?,
            "umax_release" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::Release)?,
            "umax_acqrel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::AcqRel)?,
            "umax_relaxed" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOrd::Relaxed)?,

            name => throw_unsup_format!("unimplemented intrinsic: `atomic_{name}`"),
        }
        Ok(())
    }

    fn atomic_load(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;

        // make sure it fits into a scalar; otherwise it cannot be atomic
        let val = this.read_scalar_atomic(&place, atomic)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;
        // Perform regular access.
        this.write_scalar(val, dest)?;
        Ok(())
    }

    fn atomic_store(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicWriteOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, val] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let val = this.read_scalar(val)?; // make sure it fits into a scalar; otherwise it cannot be atomic

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        // Perform atomic store
        this.write_scalar_atomic(val, &place, atomic)?;
        Ok(())
    }

    fn compiler_fence(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let [] = check_arg_count(args)?;
        let _ = atomic;
        //FIXME: compiler fences are currently ignored
        Ok(())
    }

    fn atomic_fence(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [] = check_arg_count(args)?;
        this.validate_atomic_fence(atomic)?;
        Ok(())
    }

    fn atomic_op(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic_op: AtomicOp,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, rhs] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;

        if !place.layout.ty.is_integral() && !place.layout.ty.is_unsafe_ptr() {
            span_bug!(
                this.cur_span(),
                "atomic arithmetic operations only work on integer and raw pointer types",
            );
        }
        let rhs = this.read_immediate(rhs)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

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
                let old = this.atomic_op_immediate(&place, &rhs, op, neg, atomic)?;
                this.write_immediate(*old, dest)?; // old value is returned
                Ok(())
            }
        }
    }

    fn atomic_exchange(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic: AtomicRwOrd,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, new] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let new = this.read_scalar(new)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        let old = this.atomic_exchange_scalar(&place, new, atomic)?;
        this.write_scalar(old, dest)?; // old value is returned
        Ok(())
    }

    fn atomic_compare_exchange_impl(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
        can_fail_spuriously: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [place, expect_old, new] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let expect_old = this.read_immediate(expect_old)?; // read as immediate for the sake of `binary_op()`
        let new = this.read_scalar(new)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

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
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, false)
    }

    fn atomic_compare_exchange_weak(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, true)
    }
}
