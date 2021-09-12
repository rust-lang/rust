use crate::MirPass;
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::*;
use rustc_middle::ty::layout;
use rustc_middle::ty::{self, TyCtxt};
use rustc_target::spec::abi::Abi;
use rustc_target::spec::PanicStrategy;

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
pub struct AbortUnwindingCalls;

impl<'tcx> MirPass<'tcx> for AbortUnwindingCalls {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let kind = tcx.def_kind(def_id);

        // We don't simplify the MIR of constants at this time because that
        // namely results in a cyclic query when we call `tcx.type_of` below.
        let is_function = match kind {
            DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(..) => true,
            _ => tcx.is_closure(def_id),
        };
        if !is_function {
            return;
        }

        // This pass only runs on functions which themselves cannot unwind,
        // forcibly changing the body of the function to structurally provide
        // this guarantee by aborting on an unwind. If this function can unwind,
        // then there's nothing to do because it already should work correctly.
        //
        // Here we test for this function itself whether its ABI allows
        // unwinding or not.
        let body_flags = tcx.codegen_fn_attrs(def_id).flags;
        let body_ty = tcx.type_of(def_id);
        let body_abi = match body_ty.kind() {
            ty::FnDef(..) => body_ty.fn_sig(tcx).abi(),
            ty::Closure(..) => Abi::RustCall,
            ty::Generator(..) => Abi::Rust,
            _ => span_bug!(body.span, "unexpected body ty: {:?}", body_ty),
        };
        let body_can_unwind = layout::fn_can_unwind(tcx, body_flags, body_abi);

        // Look in this function body for any basic blocks which are terminated
        // with a function call, and whose function we're calling may unwind.
        // This will filter to functions with `extern "C-unwind"` ABIs, for
        // example.
        let mut calls_to_terminate = Vec::new();
        let mut cleanups_to_remove = Vec::new();
        for (id, block) in body.basic_blocks().iter_enumerated() {
            if block.is_cleanup {
                continue;
            }
            let terminator = match &block.terminator {
                Some(terminator) => terminator,
                None => continue,
            };
            let span = terminator.source_info.span;

            let call_can_unwind = match &terminator.kind {
                TerminatorKind::Call { func, .. } => {
                    let ty = func.ty(body, tcx);
                    let sig = ty.fn_sig(tcx);
                    let flags = match ty.kind() {
                        ty::FnPtr(_) => CodegenFnAttrFlags::empty(),
                        ty::FnDef(def_id, _) => tcx.codegen_fn_attrs(*def_id).flags,
                        _ => span_bug!(span, "invalid callee of type {:?}", ty),
                    };
                    layout::fn_can_unwind(tcx, flags, sig.abi())
                }
                TerminatorKind::Drop { .. } | TerminatorKind::DropAndReplace { .. } => {
                    tcx.sess.opts.debugging_opts.panic_in_drop == PanicStrategy::Unwind
                        && layout::fn_can_unwind(tcx, CodegenFnAttrFlags::empty(), Abi::Rust)
                }
                TerminatorKind::Assert { .. } | TerminatorKind::FalseUnwind { .. } => {
                    layout::fn_can_unwind(tcx, CodegenFnAttrFlags::empty(), Abi::Rust)
                }
                _ => continue,
            };

            // If this function call can't unwind, then there's no need for it
            // to have a landing pad. This means that we can remove any cleanup
            // registered for it.
            if !call_can_unwind {
                cleanups_to_remove.push(id);
                continue;
            }

            // Otherwise if this function can unwind, then if the outer function
            // can also unwind there's nothing to do. If the outer function
            // can't unwind, however, we need to change the landing pad for this
            // function call to one that aborts.
            if !body_can_unwind {
                calls_to_terminate.push(id);
            }
        }

        // For call instructions which need to be terminated, we insert a
        // singular basic block which simply terminates, and then configure the
        // `cleanup` attribute for all calls we found to this basic block we
        // insert which means that any unwinding that happens in the functions
        // will force an abort of the process.
        if !calls_to_terminate.is_empty() {
            let bb = BasicBlockData {
                statements: Vec::new(),
                is_cleanup: true,
                terminator: Some(Terminator {
                    source_info: SourceInfo::outermost(body.span),
                    kind: TerminatorKind::Abort,
                }),
            };
            let abort_bb = body.basic_blocks_mut().push(bb);

            for bb in calls_to_terminate {
                let cleanup = body.basic_blocks_mut()[bb].terminator_mut().unwind_mut().unwrap();
                *cleanup = Some(abort_bb);
            }
        }

        for id in cleanups_to_remove {
            let cleanup = body.basic_blocks_mut()[id].terminator_mut().unwind_mut().unwrap();
            *cleanup = None;
        }

        // We may have invalidated some `cleanup` blocks so clean those up now.
        super::simplify::remove_dead_blocks(tcx, body);
    }
}
