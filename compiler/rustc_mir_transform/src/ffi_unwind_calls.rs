use rustc_abi::ExternAbi;
use rustc_hir::def_id::{LOCAL_CRATE, LocalDefId};
use rustc_middle::mir::*;
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::{self, TyCtxt, layout};
use rustc_middle::{bug, span_bug};
use rustc_session::lint::builtin::FFI_UNWIND_CALLS;
use rustc_target::spec::PanicStrategy;
use tracing::debug;

use crate::errors;

// Check if the body of this def_id can possibly leak a foreign unwind into Rust code.
fn has_ffi_unwind_calls(tcx: TyCtxt<'_>, local_def_id: LocalDefId) -> bool {
    debug!("has_ffi_unwind_calls({local_def_id:?})");

    // Only perform check on functions because constants cannot call FFI functions.
    let def_id = local_def_id.to_def_id();
    let kind = tcx.def_kind(def_id);
    if !kind.is_fn_like() {
        return false;
    }

    let body = &*tcx.mir_built(local_def_id).borrow();

    let body_ty = tcx.type_of(def_id).skip_binder();
    let body_abi = match body_ty.kind() {
        ty::FnDef(..) => body_ty.fn_sig(tcx).abi(),
        ty::Closure(..) => ExternAbi::RustCall,
        ty::CoroutineClosure(..) => ExternAbi::RustCall,
        ty::Coroutine(..) => ExternAbi::Rust,
        ty::Error(_) => return false,
        _ => span_bug!(body.span, "unexpected body ty: {:?}", body_ty),
    };
    let body_can_unwind = layout::fn_can_unwind(tcx, Some(def_id), body_abi);

    // Foreign unwinds cannot leak past functions that themselves cannot unwind.
    if !body_can_unwind {
        return false;
    }

    let mut tainted = false;

    for block in body.basic_blocks.iter() {
        if block.is_cleanup {
            continue;
        }
        let Some(terminator) = &block.terminator else { continue };
        let TerminatorKind::Call { func, .. } = &terminator.kind else { continue };

        let ty = func.ty(body, tcx);
        let sig = ty.fn_sig(tcx);

        // Rust calls cannot themselves create foreign unwinds.
        // We assume this is true for intrinsics as well.
        if sig.abi().is_rustic_abi() {
            continue;
        };

        let fn_def_id = match ty.kind() {
            ty::FnPtr(..) => None,
            &ty::FnDef(def_id, _) => {
                // Rust calls cannot themselves create foreign unwinds (even if they use a non-Rust
                // ABI). So the leak of the foreign unwind into Rust can only be elsewhere, not
                // here.
                if !tcx.is_foreign_item(def_id) {
                    continue;
                }
                Some(def_id)
            }
            _ => bug!("invalid callee of type {:?}", ty),
        };

        if layout::fn_can_unwind(tcx, fn_def_id, sig.abi()) {
            // We have detected a call that can possibly leak foreign unwind.
            //
            // Because the function body itself can unwind, we are not aborting this function call
            // upon unwind, so this call can possibly leak foreign unwind into Rust code if the
            // panic runtime linked is panic-abort.

            let lint_root = body.source_scopes[terminator.source_info.scope]
                .local_data
                .as_ref()
                .unwrap_crate_local()
                .lint_root;
            let span = terminator.source_info.span;

            let foreign = fn_def_id.is_some();
            tcx.emit_node_span_lint(
                FFI_UNWIND_CALLS,
                lint_root,
                span,
                errors::FfiUnwindCall { span, foreign },
            );

            tainted = true;
        }
    }

    tainted
}

fn required_panic_strategy(tcx: TyCtxt<'_>, _: LocalCrate) -> Option<PanicStrategy> {
    let local_strategy = tcx.sess.panic_strategy();

    if tcx.is_panic_runtime(LOCAL_CRATE) {
        return Some(local_strategy);
    }

    match local_strategy {
        PanicStrategy::Abort | PanicStrategy::ImmediateAbort => return Some(local_strategy),
        _ => {}
    }

    for def_id in tcx.hir_body_owners() {
        if tcx.has_ffi_unwind_calls(def_id) {
            // Given that this crate is compiled in `-C panic=unwind`, the `AbortUnwindingCalls`
            // MIR pass will not be run on FFI-unwind call sites, therefore a foreign exception
            // can enter Rust through these sites.
            //
            // On the other hand, crates compiled with `-C panic=abort` expects that all Rust
            // functions cannot unwind (whether it's caused by Rust panic or foreign exception),
            // and this expectation mismatch can cause unsoundness (#96926).
            //
            // To address this issue, we enforce that if FFI-unwind calls are used in a crate
            // compiled with `panic=unwind`, then the final panic strategy must be `panic=unwind`.
            // This will ensure that no crates will have wrong unwindability assumption.
            //
            // It should be noted that it is okay to link `panic=unwind` into a `panic=abort`
            // program if it contains no FFI-unwind calls. In such case foreign exception can only
            // enter Rust in a `panic=abort` crate, which will lead to an abort. There will also
            // be no exceptions generated from Rust, so the assumption which `panic=abort` crates
            // make, that no Rust function can unwind, indeed holds for crates compiled with
            // `panic=unwind` as well. In such case this function returns `None`, indicating that
            // the crate does not require a particular final panic strategy, and can be freely
            // linked to crates with either strategy (we need such ability for libstd and its
            // dependencies).
            return Some(PanicStrategy::Unwind);
        }
    }

    // This crate can be linked with either runtime.
    None
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { has_ffi_unwind_calls, required_panic_strategy, ..*providers };
}
