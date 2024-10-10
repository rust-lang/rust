use rustc_ast::InlineAsmOptions;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, TyCtxt, layout};
use rustc_target::spec::PanicStrategy;
use rustc_target::spec::abi::Abi;

/// A pass that runs which is targeted at ensuring that codegen guarantees about
/// unwinding are upheld for compilations of panic=abort programs.
///
/// When compiling with panic=abort codegen backends generally want to assume
/// that all Rust-defined functions do not unwind, and it's UB if they actually
/// do unwind. Foreign functions, however, can be declared as "may unwind" via
/// their ABI (e.g. `extern "C-unwind"`). To uphold the guarantees that
/// Rust-defined functions never unwind a well-behaved Rust program needs to
/// catch unwinding from foreign functions and force them to abort.
///
/// This pass walks over all functions calls which may possibly unwind,
/// and if any are found sets their cleanup to a block that aborts the process.
/// This forces all unwinds, in panic=abort mode happening in foreign code, to
/// trigger a process abort.
#[derive(PartialEq)]
pub(super) struct AbortUnwindingCalls;

impl<'tcx> crate::MirPass<'tcx> for AbortUnwindingCalls {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let kind = tcx.def_kind(def_id);

        // We don't simplify the MIR of constants at this time because that
        // namely results in a cyclic query when we call `tcx.type_of` below.
        if !kind.is_fn_like() {
            return;
        }

        // Here we test for this function itself whether its ABI allows
        // unwinding or not.
        let body_ty = tcx.type_of(def_id).skip_binder();
        let body_abi = match body_ty.kind() {
            ty::FnDef(..) => body_ty.fn_sig(tcx).abi(),
            ty::Closure(..) => Abi::RustCall,
            ty::CoroutineClosure(..) => Abi::RustCall,
            ty::Coroutine(..) => Abi::Rust,
            ty::Error(_) => return,
            _ => span_bug!(body.span, "unexpected body ty: {:?}", body_ty),
        };
        let body_can_unwind = layout::fn_can_unwind(tcx, Some(def_id), body_abi);

        // Look in this function body for any basic blocks which are terminated
        // with a function call, and whose function we're calling may unwind.
        // This will filter to functions with `extern "C-unwind"` ABIs, for
        // example.
        for block in body.basic_blocks.as_mut() {
            let Some(terminator) = &mut block.terminator else { continue };
            let span = terminator.source_info.span;

            // If we see an `UnwindResume` terminator inside a function that cannot unwind, we need
            // to replace it with `UnwindTerminate`.
            if let TerminatorKind::UnwindResume = &terminator.kind
                && !body_can_unwind
            {
                terminator.kind = TerminatorKind::UnwindTerminate(UnwindTerminateReason::Abi);
            }

            if block.is_cleanup {
                continue;
            }

            let call_can_unwind = match &terminator.kind {
                TerminatorKind::Call { func, .. } => {
                    let ty = func.ty(&body.local_decls, tcx);
                    let sig = ty.fn_sig(tcx);
                    let fn_def_id = match ty.kind() {
                        ty::FnPtr(..) => None,
                        &ty::FnDef(def_id, _) => Some(def_id),
                        _ => span_bug!(span, "invalid callee of type {:?}", ty),
                    };
                    layout::fn_can_unwind(tcx, fn_def_id, sig.abi())
                }
                TerminatorKind::Drop { .. } => {
                    tcx.sess.opts.unstable_opts.panic_in_drop == PanicStrategy::Unwind
                        && layout::fn_can_unwind(tcx, None, Abi::Rust)
                }
                TerminatorKind::Assert { .. } | TerminatorKind::FalseUnwind { .. } => {
                    layout::fn_can_unwind(tcx, None, Abi::Rust)
                }
                TerminatorKind::InlineAsm { options, .. } => {
                    options.contains(InlineAsmOptions::MAY_UNWIND)
                }
                _ if terminator.unwind().is_some() => {
                    span_bug!(span, "unexpected terminator that may unwind {:?}", terminator)
                }
                _ => continue,
            };

            if !call_can_unwind {
                // If this function call can't unwind, then there's no need for it
                // to have a landing pad. This means that we can remove any cleanup
                // registered for it (and turn it into `UnwindAction::Unreachable`).
                let cleanup = block.terminator_mut().unwind_mut().unwrap();
                *cleanup = UnwindAction::Unreachable;
            } else if !body_can_unwind
                && matches!(terminator.unwind(), Some(UnwindAction::Continue))
            {
                // Otherwise if this function can unwind, then if the outer function
                // can also unwind there's nothing to do. If the outer function
                // can't unwind, however, we need to ensure that any `UnwindAction::Continue`
                // is replaced with terminate. For those with `UnwindAction::Cleanup`,
                // cleanup will still happen, and terminate will happen afterwards handled by
                // the `UnwindResume` -> `UnwindTerminate` terminator replacement.
                let cleanup = block.terminator_mut().unwind_mut().unwrap();
                *cleanup = UnwindAction::Terminate(UnwindTerminateReason::Abi);
            }
        }

        // We may have invalidated some `cleanup` blocks so clean those up now.
        super::simplify::remove_dead_blocks(body);
    }
}
