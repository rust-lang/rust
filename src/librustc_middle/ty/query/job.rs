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

    let span_session_globals = rustc_span::SESSION_GLOBALS.with(|ssg| ssg as *const _);
    let span_session_globals = &*span_session_globals;
    let ast_session_globals = rustc_ast::attr::SESSION_GLOBALS.with(|asg| asg as *const _);
    let ast_session_globals = &*ast_session_globals;
    thread::spawn(move || {
        tls::GCX_PTR.set(gcx_ptr, || {
            rustc_ast::attr::SESSION_GLOBALS.set(ast_session_globals, || {
                rustc_span::SESSION_GLOBALS
                    .set(span_session_globals, || tls::with_global(|tcx| deadlock(tcx, &registry)))
            });
        })
    });
}
