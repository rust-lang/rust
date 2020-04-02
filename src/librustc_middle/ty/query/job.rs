use crate::ty::tls;

use rustc_query_system::query::deadlock;
use rustc_rayon_core as rayon_core;
use std::thread;

/// Creates a new thread and forwards information in thread locals to it.
/// The new thread runs the deadlock handler.
/// Must only be called when a deadlock is about to happen.
pub unsafe fn handle_deadlock() {
    let registry = rayon_core::Registry::current();

    let gcx_ptr = tls::GCX_PTR.with(|gcx_ptr| gcx_ptr as *const _);
    let gcx_ptr = &*gcx_ptr;

    let rustc_span_globals =
        rustc_span::GLOBALS.with(|rustc_span_globals| rustc_span_globals as *const _);
    let rustc_span_globals = &*rustc_span_globals;
    let syntax_globals = rustc_ast::attr::GLOBALS.with(|syntax_globals| syntax_globals as *const _);
    let syntax_globals = &*syntax_globals;
    thread::spawn(move || {
        tls::GCX_PTR.set(gcx_ptr, || {
            rustc_ast::attr::GLOBALS.set(syntax_globals, || {
                rustc_span::GLOBALS
                    .set(rustc_span_globals, || tls::with_global(|tcx| deadlock(tcx, &registry)))
            });
        })
    });
}
