// TODO: remove
#![allow(dead_code)]

use std::cell::Cell;
use std::ptr;

use rustc_ast::tokenstream::TokenStream;
use rustc_middle::ty::TyCtxt;
use rustc_span::profiling::SpannedEventArgRecorder;
use rustc_span::LocalExpnId;

use crate::base::ExtCtxt;
use crate::errors;

pub(super) fn expand<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (LocalExpnId, &'tcx TokenStream),
) -> Result<&'tcx TokenStream, ()> {
    let (invoc_id, input) = key;

    let res = with_context(|(ecx, client)| {
        let span = invoc_id.expn_data().call_site;
        let _timer =
            ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                recorder.record_arg_with_span(ecx.sess.source_map(), ecx.expansion_descr(), span);
            });
        let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
        let strategy = crate::proc_macro::exec_strategy(ecx);
        let server = crate::proc_macro_server::Rustc::new(ecx);
        let res = match client.run(&strategy, server, input.clone(), proc_macro_backtrace) {
            // TODO: without flattened some (weird) tests fail, but no idea if it's correct/enough
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
// #[cfg(not(parallel_compiler))]
thread_local! {
    /// A thread local variable that stores a pointer to the current `CONTEXT`.
    static TLV: Cell<(*mut (), Option<CLIENT>)> = const { Cell::new((ptr::null_mut(), None)) };
}

#[inline]
fn erase(context: &mut ExtCtxt<'_>) -> *mut () {
    context as *mut _ as *mut ()
}

#[inline]
unsafe fn downcast<'a>(context: *mut ()) -> &'a mut ExtCtxt<'a> {
    unsafe { &mut *(context as *mut ExtCtxt<'a>) }
}

/// Sets `context` as the new current `CONTEXT` for the duration of the function `f`.
#[inline]
pub fn enter_context<'a, F, R>(context: (&mut ExtCtxt<'a>, CLIENT), f: F) -> R
where
    F: FnOnce() -> R,
{
    let (ectx, client) = context;
    let erased = (erase(ectx), Some(client));
    TLV.with(|tlv| {
        let old = tlv.replace(erased);
        let _reset = rustc_data_structures::defer(move || tlv.set(old));
        f()
    })
}

/// Allows access to the current `CONTEXT` in a closure if one is available.
#[inline]
#[track_caller]
pub fn with_context_opt<F, R>(f: F) -> R
where
    F: for<'a, 'b> FnOnce(Option<&'b mut (&mut ExtCtxt<'a>, CLIENT)>) -> R,
{
    let (ectx, client_opt) = TLV.get();
    if ectx.is_null() {
        f(None)
    } else {
        // We could get an `CONTEXT` pointer from another thread.
        // Ensure that `CONTEXT` is `DynSync`.
        // TODO: we should not be able to?
        // sync::assert_dyn_sync::<CONTEXT<'_>>();

        unsafe { f(Some(&mut (downcast(ectx), client_opt.unwrap()))) }
    }
}

/// Allows access to the current `CONTEXT`.
/// Panics if there is no `CONTEXT` available.
#[inline]
pub fn with_context<F, R>(f: F) -> R
where
    F: for<'a, 'b> FnOnce(&'b mut (&mut ExtCtxt<'a>, CLIENT)) -> R,
{
    with_context_opt(|opt_context| f(opt_context.expect("no CONTEXT stored in tls")))
}
