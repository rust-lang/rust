use std::{mem, ptr};

use rustc_data_structures::sync;

use super::{GlobalCtxt, TyCtxt};
use crate::dep_graph::TaskDepsRef;
use crate::query::plumbing::QueryJobId;

/// This is the implicit state of rustc. It contains the current
/// `TyCtxt` and query. It is updated when creating a local interner or
/// executing a new query. Whenever there's a `TyCtxt` value available
/// you should also have access to an `ImplicitCtxt` through the functions
/// in this module.
#[derive(Clone)]
pub struct ImplicitCtxt<'a, 'tcx> {
    /// The current `TyCtxt`.
    pub tcx: TyCtxt<'tcx>,

    /// The current query job, if any. This is updated by `JobOwner::start` in
    /// `ty::query::plumbing` when executing a query.
    pub query: Option<QueryJobId>,

    /// Used to prevent queries from calling too deeply.
    pub query_depth: usize,

    /// The current dep graph task. This is used to add dependencies to queries
    /// when executing them.
    pub task_deps: TaskDepsRef<'a>,
}

impl<'a, 'tcx> ImplicitCtxt<'a, 'tcx> {
    pub fn new(gcx: &'tcx GlobalCtxt<'tcx>) -> Self {
        let tcx = TyCtxt { gcx };
        ImplicitCtxt { tcx, query: None, query_depth: 0, task_deps: TaskDepsRef::Ignore }
    }
}

// Import the thread-local variable from Rayon, which is preserved for Rayon jobs.
use rustc_thread_pool::tlv::TLV;

#[inline]
fn erase(context: &ImplicitCtxt<'_, '_>) -> *const () {
    context as *const _ as *const ()
}

#[inline]
unsafe fn downcast<'a, 'tcx>(context: *const ()) -> &'a ImplicitCtxt<'a, 'tcx> {
    unsafe { &*(context as *const ImplicitCtxt<'a, 'tcx>) }
}

/// Sets `context` as the new current `ImplicitCtxt` for the duration of the function `f`.
#[inline]
pub fn enter_context<'a, 'tcx, F, R>(context: &ImplicitCtxt<'a, 'tcx>, f: F) -> R
where
    F: FnOnce() -> R,
{
    TLV.with(|tlv| {
        let old = tlv.replace(erase(context));
        let _reset = rustc_data_structures::defer(move || tlv.set(old));
        f()
    })
}

/// Allows access to the current `ImplicitCtxt` in a closure if one is available.
#[inline]
#[track_caller]
pub fn with_context_opt<F, R>(f: F) -> R
where
    F: for<'a, 'tcx> FnOnce(Option<&ImplicitCtxt<'a, 'tcx>>) -> R,
{
    let context = TLV.get();
    if context.is_null() {
        f(None)
    } else {
        // We could get an `ImplicitCtxt` pointer from another thread.
        // Ensure that `ImplicitCtxt` is `DynSync`.
        sync::assert_dyn_sync::<ImplicitCtxt<'_, '_>>();

        unsafe { f(Some(downcast(context))) }
    }
}

/// Allows access to the current `ImplicitCtxt`.
/// Panics if there is no `ImplicitCtxt` available.
#[inline]
pub fn with_context<F, R>(f: F) -> R
where
    F: for<'a, 'tcx> FnOnce(&ImplicitCtxt<'a, 'tcx>) -> R,
{
    with_context_opt(|opt_context| f(opt_context.expect("no ImplicitCtxt stored in tls")))
}

/// Allows access to the current `ImplicitCtxt` whose tcx field is the same as the tcx argument
/// passed in. This means the closure is given an `ImplicitCtxt` with the same `'tcx` lifetime
/// as the `TyCtxt` passed in.
/// This will panic if you pass it a `TyCtxt` which is different from the current
/// `ImplicitCtxt`'s `tcx` field.
#[inline]
pub fn with_related_context<'tcx, F, R>(tcx: TyCtxt<'tcx>, f: F) -> R
where
    F: FnOnce(&ImplicitCtxt<'_, 'tcx>) -> R,
{
    with_context(|context| {
        // The two gcx have different invariant lifetimes, so we need to erase them for the comparison.
        assert!(ptr::eq(
            context.tcx.gcx as *const _ as *const (),
            tcx.gcx as *const _ as *const ()
        ));

        let context: &ImplicitCtxt<'_, '_> = unsafe { mem::transmute(context) };

        f(context)
    })
}

/// Allows access to the `TyCtxt` in the current `ImplicitCtxt`.
/// Panics if there is no `ImplicitCtxt` available.
#[inline]
pub fn with<F, R>(f: F) -> R
where
    F: for<'tcx> FnOnce(TyCtxt<'tcx>) -> R,
{
    with_context(|context| f(context.tcx))
}

/// Allows access to the `TyCtxt` in the current `ImplicitCtxt`.
/// The closure is passed None if there is no `ImplicitCtxt` available.
#[inline]
#[track_caller]
pub fn with_opt<F, R>(f: F) -> R
where
    F: for<'tcx> FnOnce(Option<TyCtxt<'tcx>>) -> R,
{
    with_context_opt(
        #[track_caller]
        |opt_context| f(opt_context.map(|context| context.tcx)),
    )
}
