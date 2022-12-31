use super::{ptr_eq, GlobalCtxt, TyCtxt};

use crate::dep_graph::TaskDepsRef;
use crate::ty::query;
use rustc_data_structures::sync::{self, Lock};
use rustc_errors::Diagnostic;
use std::mem;
use thin_vec::ThinVec;

#[cfg(not(parallel_compiler))]
use std::cell::Cell;

#[cfg(parallel_compiler)]
use rustc_rayon_core as rayon_core;

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
    pub query: Option<query::QueryJobId>,

    /// Where to store diagnostics for the current query job, if any.
    /// This is updated by `JobOwner::start` in `ty::query::plumbing` when executing a query.
    pub diagnostics: Option<&'a Lock<ThinVec<Diagnostic>>>,

    /// Used to prevent queries from calling too deeply.
    pub query_depth: usize,

    /// The current dep graph task. This is used to add dependencies to queries
    /// when executing them.
    pub task_deps: TaskDepsRef<'a>,
}

impl<'a, 'tcx> ImplicitCtxt<'a, 'tcx> {
    pub fn new(gcx: &'tcx GlobalCtxt<'tcx>) -> Self {
        let tcx = TyCtxt { gcx };
        ImplicitCtxt {
            tcx,
            query: None,
            diagnostics: None,
            query_depth: 0,
            task_deps: TaskDepsRef::Ignore,
        }
    }
}

/// Sets Rayon's thread-local variable, which is preserved for Rayon jobs
/// to `value` during the call to `f`. It is restored to its previous value after.
/// This is used to set the pointer to the new `ImplicitCtxt`.
#[cfg(parallel_compiler)]
#[inline]
fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
    rayon_core::tlv::with(value, f)
}

/// Gets Rayon's thread-local variable, which is preserved for Rayon jobs.
/// This is used to get the pointer to the current `ImplicitCtxt`.
#[cfg(parallel_compiler)]
#[inline]
pub fn get_tlv() -> usize {
    rayon_core::tlv::get()
}

#[cfg(not(parallel_compiler))]
thread_local! {
    /// A thread local variable that stores a pointer to the current `ImplicitCtxt`.
    static TLV: Cell<usize> = const { Cell::new(0) };
}

/// Sets TLV to `value` during the call to `f`.
/// It is restored to its previous value after.
/// This is used to set the pointer to the new `ImplicitCtxt`.
#[cfg(not(parallel_compiler))]
#[inline]
fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
    let old = get_tlv();
    let _reset = rustc_data_structures::OnDrop(move || TLV.with(|tlv| tlv.set(old)));
    TLV.with(|tlv| tlv.set(value));
    f()
}

/// Gets the pointer to the current `ImplicitCtxt`.
#[cfg(not(parallel_compiler))]
#[inline]
fn get_tlv() -> usize {
    TLV.with(|tlv| tlv.get())
}

/// Sets `context` as the new current `ImplicitCtxt` for the duration of the function `f`.
#[inline]
pub fn enter_context<'a, 'tcx, F, R>(context: &ImplicitCtxt<'a, 'tcx>, f: F) -> R
where
    F: FnOnce(&ImplicitCtxt<'a, 'tcx>) -> R,
{
    set_tlv(context as *const _ as usize, || f(&context))
}

/// Allows access to the current `ImplicitCtxt` in a closure if one is available.
#[inline]
pub fn with_context_opt<F, R>(f: F) -> R
where
    F: for<'a, 'tcx> FnOnce(Option<&ImplicitCtxt<'a, 'tcx>>) -> R,
{
    let context = get_tlv();
    if context == 0 {
        f(None)
    } else {
        // We could get an `ImplicitCtxt` pointer from another thread.
        // Ensure that `ImplicitCtxt` is `Sync`.
        sync::assert_sync::<ImplicitCtxt<'_, '_>>();

        unsafe { f(Some(&*(context as *const ImplicitCtxt<'_, '_>))) }
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
    with_context(|context| unsafe {
        assert!(ptr_eq(context.tcx.gcx, tcx.gcx));
        let context: &ImplicitCtxt<'_, '_> = mem::transmute(context);
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
pub fn with_opt<F, R>(f: F) -> R
where
    F: for<'tcx> FnOnce(Option<TyCtxt<'tcx>>) -> R,
{
    with_context_opt(|opt_context| f(opt_context.map(|context| context.tcx)))
}
