use std::cell::Cell;
use std::ptr::{self, NonNull};

use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::svh::Svh;
use rustc_middle::ty::TyCtxt;
use rustc_span::LocalExpnId;
use rustc_span::profiling::SpannedEventArgRecorder;

use crate::base::ExtCtxt;
use crate::errors;

pub(super) fn provide_derive_macro_expansion<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (LocalExpnId, Svh, &'tcx TokenStream),
) -> Result<&'tcx TokenStream, ()> {
    let (invoc_id, _macro_crate_hash, input) = key;

    let res = with_context(|(ecx, client)| {
        let span = invoc_id.expn_data().call_site;
        let _timer = ecx.sess.prof.generic_activity_with_arg_recorder(
            "expand_derive_proc_macro_inner",
            |recorder| {
                recorder.record_arg_with_span(ecx.sess.source_map(), ecx.expansion_descr(), span);
            },
        );
        let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
        let strategy = crate::proc_macro::exec_strategy(ecx);
        let server = crate::proc_macro_server::Rustc::new(ecx);
        let res = match client.run(&strategy, server, input.clone(), proc_macro_backtrace) {
            // FIXME(pr-time): without flattened some (weird) tests fail, but no idea if it's correct/enough
            Ok(stream) => Ok(tcx.arena.alloc(stream.flattened()) as &TokenStream),
            Err(e) => {
                ecx.dcx().emit_err({
                    errors::ProcMacroDerivePanicked {
                        span,
                        message: e.as_str().map(|message| errors::ProcMacroDerivePanickedHelp {
                            message: message.into(),
                        }),
                    }
                });
                Err(())
            }
        };
        res
    });

    res
}

type CLIENT = pm::bridge::client::Client<pm::TokenStream, pm::TokenStream>;

// based on rust/compiler/rustc_middle/src/ty/context/tls.rs
thread_local! {
    /// A thread local variable that stores a pointer to the current `CONTEXT`.
    static TLV: Cell<(*mut (), Option<CLIENT>)> = const { Cell::new((ptr::null_mut(), None)) };
}

/// Sets `context` as the new current `CONTEXT` for the duration of the function `f`.
#[inline]
pub(crate) fn enter_context<'a, F, R>(context: (&mut ExtCtxt<'a>, CLIENT), f: F) -> R
where
    F: FnOnce() -> R,
{
    let (ectx, client) = context;
    let erased = (ectx as *mut _ as *mut (), Some(client));
    TLV.with(|tlv| {
        let old = tlv.replace(erased);
        let _reset = rustc_data_structures::defer(move || tlv.set(old));
        f()
    })
}

/// Allows access to the current `CONTEXT`.
/// Panics if there is no `CONTEXT` available.
#[inline]
#[track_caller]
fn with_context<F, R>(f: F) -> R
where
    F: for<'a, 'b> FnOnce(&'b mut (&mut ExtCtxt<'a>, CLIENT)) -> R,
{
    let (ectx, client_opt) = TLV.get();
    let ectx = NonNull::new(ectx).expect("no CONTEXT stored in tls");

    // We could get an `CONTEXT` pointer from another thread.
    // Ensure that `CONTEXT` is `DynSync`.
    // FIXME(pr-time): we should not be able to?
    // sync::assert_dyn_sync::<CONTEXT<'_>>();

    // prevent double entering, as that would allow creating two `&mut ExtCtxt`s
    // FIXME(pr-time): probably use a RefCell instead (which checks this properly)?
    TLV.with(|tlv| {
        let old = tlv.replace((ptr::null_mut(), None));
        let _reset = rustc_data_structures::defer(move || tlv.set(old));
        let ectx = {
            let mut casted = ectx.cast::<ExtCtxt<'_>>();
            unsafe { casted.as_mut() }
        };

        f(&mut (ectx, client_opt.unwrap()))
    })
}
