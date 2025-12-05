//! Unwinding runtime for Miri.
//!
//! The core pieces of the runtime are:
//! - An implementation of `catch_unwind` that pushes the invoked stack frame with
//!   some extra metadata derived from the panic-catching arguments of `catch_unwind`.
//! - A hack in `libpanic_unwind` that calls the `miri_start_unwind` intrinsic instead of the
//!   target-native panic runtime. (This lives in the rustc repo.)
//! - An implementation of `miri_start_unwind` that stores its argument (the panic payload), and
//!   then immediately returns, but on the *unwind* edge (not the normal return edge), thus
//!   initiating unwinding.
//! - A hook executed each time a frame is popped, such that if the frame pushed by `catch_unwind`
//!   gets popped *during unwinding*, we take the panic payload and store it according to the extra
//!   metadata we remembered when pushing said frame.

use rustc_abi::ExternAbi;
use rustc_middle::mir;
use rustc_target::spec::PanicStrategy;

use crate::*;

/// Holds all of the relevant data for when unwinding hits a `try` frame.
#[derive(Debug)]
pub struct CatchUnwindData<'tcx> {
    /// The `catch_fn` callback to call in case of a panic.
    catch_fn: Pointer,
    /// The `data` argument for that callback.
    data: ImmTy<'tcx>,
    /// The return place from the original call to `try`.
    dest: MPlaceTy<'tcx>,
    /// The return block from the original call to `try`.
    ret: Option<mir::BasicBlock>,
}

impl VisitProvenance for CatchUnwindData<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let CatchUnwindData { catch_fn, data, dest, ret: _ } = self;
        catch_fn.visit_provenance(visit);
        data.visit_provenance(visit);
        dest.visit_provenance(visit);
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Handles the special `miri_start_unwind` intrinsic, which is called
    /// by libpanic_unwind to delegate the actual unwinding process to Miri.
    fn handle_miri_start_unwind(&mut self, payload: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        trace!("miri_start_unwind: {:?}", this.frame().instance());

        let payload = this.read_immediate(payload)?;
        let thread = this.active_thread_mut();
        thread.unwind_payloads.push(payload);

        interp_ok(())
    }

    /// Handles the `catch_unwind` intrinsic.
    fn handle_catch_unwind(
        &mut self,
        try_fn: &OpTy<'tcx>,
        data: &OpTy<'tcx>,
        catch_fn: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Signature:
        //   fn catch_unwind(try_fn: fn(*mut u8), data: *mut u8, catch_fn: fn(*mut u8, *mut u8)) -> i32
        // Calls `try_fn` with `data` as argument. If that executes normally, returns 0.
        // If that unwinds, calls `catch_fn` with the first argument being `data` and
        // then second argument being a target-dependent `payload` (i.e. it is up to us to define
        // what that is), and returns 1.
        // The `payload` is passed (by libstd) to `__rust_panic_cleanup`, which is then expected to
        // return a `Box<dyn Any + Send + 'static>`.
        // In Miri, `miri_start_unwind` is passed exactly that type, so we make the `payload` simply
        // a pointer to `Box<dyn Any + Send + 'static>`.

        // Get all the arguments.
        let try_fn = this.read_pointer(try_fn)?;
        let data = this.read_immediate(data)?;
        let catch_fn = this.read_pointer(catch_fn)?;

        // Now we make a function call, and pass `data` as first and only argument.
        let f_instance = this.get_ptr_fn(try_fn)?.as_instance()?;
        trace!("try_fn: {:?}", f_instance);
        #[allow(clippy::cloned_ref_to_slice_refs)] // the code is clearer as-is
        this.call_function(
            f_instance,
            ExternAbi::Rust,
            &[data.clone()],
            None,
            // Directly return to caller.
            ReturnContinuation::Goto { ret, unwind: mir::UnwindAction::Continue },
        )?;

        // We ourselves will return `0`, eventually (will be overwritten if we catch a panic).
        this.write_null(dest)?;

        // In unwind mode, we tag this frame with the extra data needed to catch unwinding.
        // This lets `handle_stack_pop` (below) know that we should stop unwinding
        // when we pop this frame.
        if this.tcx.sess.panic_strategy() == PanicStrategy::Unwind {
            this.frame_mut().extra.catch_unwind =
                Some(CatchUnwindData { catch_fn, data, dest: dest.clone(), ret });
        }

        interp_ok(())
    }

    fn handle_stack_pop_unwind(
        &mut self,
        mut extra: FrameExtra<'tcx>,
        unwinding: bool,
    ) -> InterpResult<'tcx, ReturnAction> {
        let this = self.eval_context_mut();
        trace!("handle_stack_pop_unwind(extra = {:?}, unwinding = {})", extra, unwinding);

        // We only care about `catch_panic` if we're unwinding - if we're doing a normal
        // return, then we don't need to do anything special.
        if let (true, Some(catch_unwind)) = (unwinding, extra.catch_unwind.take()) {
            // We've just popped a frame that was pushed by `catch_unwind`,
            // and we are unwinding, so we should catch that.
            trace!(
                "unwinding: found catch_panic frame during unwinding: {:?}",
                this.frame().instance()
            );

            // We set the return value of `catch_unwind` to 1, since there was a panic.
            this.write_scalar(Scalar::from_i32(1), &catch_unwind.dest)?;

            // The Thread's `panic_payload` holds what was passed to `miri_start_unwind`.
            // This is exactly the second argument we need to pass to `catch_fn`.
            let payload = this.active_thread_mut().unwind_payloads.pop().unwrap();

            // Push the `catch_fn` stackframe.
            let f_instance = this.get_ptr_fn(catch_unwind.catch_fn)?.as_instance()?;
            trace!("catch_fn: {:?}", f_instance);
            this.call_function(
                f_instance,
                ExternAbi::Rust,
                &[catch_unwind.data, payload],
                None,
                // Directly return to caller of `catch_unwind`.
                ReturnContinuation::Goto {
                    ret: catch_unwind.ret,
                    // `catch_fn` must not unwind.
                    unwind: mir::UnwindAction::Unreachable,
                },
            )?;

            // We pushed a new stack frame, the engine should not do any jumping now!
            interp_ok(ReturnAction::NoJump)
        } else {
            interp_ok(ReturnAction::Normal)
        }
    }
}
