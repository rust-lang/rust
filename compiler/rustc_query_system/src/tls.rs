use crate::dep_graph::TaskDepsRef;
use crate::query::QueryJobId;
use rustc_data_structures::sync::{self, Lock};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;

#[cfg(not(parallel_compiler))]
use std::cell::Cell;

#[cfg(parallel_compiler)]
use rustc_rayon_core as rayon_core;

/// This is the implicit state of rustc. It contains the current
/// query. It is updated when
/// executing a new query.
#[derive(Clone, Default)]
struct ImplicitCtxt<'a> {
    /// The current query job, if any. This is updated by `JobOwner::start` in
    /// `ty::query::plumbing` when executing a query.
    query: Option<QueryJobId>,

    /// Where to store diagnostics for the current query job, if any.
    /// This is updated by `JobOwner::start` in `ty::query::plumbing` when executing a query.
    diagnostics: Option<&'a Lock<ThinVec<Diagnostic>>>,

    /// Used to prevent layout from recursing too deeply.
    layout_depth: usize,

    /// The current dep graph task. This is used to add dependencies to queries
    /// when executing them.
    task_deps: TaskDepsRef<'a>,
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
fn enter_context<'a, F, R>(context: &ImplicitCtxt<'a>, f: F) -> R
where
    F: FnOnce() -> R,
{
    set_tlv(context as *const _ as usize, || f())
}

/// Allows access to the current `ImplicitCtxt` in a closure if one is available.
#[inline]
fn with_context_opt<F, R>(f: F) -> R
where
    F: for<'a> FnOnce(Option<&ImplicitCtxt<'a>>) -> R,
{
    // We could get a `ImplicitCtxt` pointer from another thread.
    // Ensure that `ImplicitCtxt` is `Sync`.
    sync::assert_sync::<ImplicitCtxt<'_>>();

    let context = get_tlv();
    let context =
        if context == 0 { None } else { unsafe { Some(&*(context as *const ImplicitCtxt<'_>)) } };
    f(context)
}

/// This is a callback from `rustc_ast` as it cannot access the implicit state
/// in `rustc_middle` otherwise. It is used to when diagnostic messages are
/// emitted and stores them in the current query, if there is one.
pub fn track_diagnostic(diagnostic: &Diagnostic) {
    with_context_opt(|icx| {
        if let Some(icx) = icx {
            if let Some(ref diagnostics) = icx.diagnostics {
                let mut diagnostics = diagnostics.lock();
                diagnostics.extend(Some(diagnostic.clone()));
            }
        }
    })
}

pub fn with_deps<R>(task_deps: TaskDepsRef<'_>, op: impl FnOnce() -> R) -> R {
    crate::tls::with_context_opt(|icx| {
        let icx = crate::tls::ImplicitCtxt { task_deps, ..icx.cloned().unwrap_or_default() };

        crate::tls::enter_context(&icx, op)
    })
}

pub fn read_deps(op: impl FnOnce(TaskDepsRef<'_>)) {
    crate::tls::with_context_opt(|icx| if let Some(icx) = icx { op(icx.task_deps) } else { return })
}

/// Get the query information from the TLS context.
#[inline(always)]
pub fn current_query_job() -> Option<QueryJobId> {
    with_context_opt(|icx| icx?.query)
}

/// Executes a job by changing the `ImplicitCtxt` to point to the
/// new query job while it executes. It returns the diagnostics
/// captured during execution and the actual result.
#[inline(always)]
pub fn start_query<R>(
    token: QueryJobId,
    diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
    compute: impl FnOnce() -> R,
) -> R {
    with_context_opt(move |current_icx| {
        let task_deps = current_icx.map_or(TaskDepsRef::Ignore, |icx| icx.task_deps);
        let layout_depth = current_icx.map_or(0, |icx| icx.layout_depth);

        // Update the `ImplicitCtxt` to point to our new query job.
        let new_icx = ImplicitCtxt { query: Some(token), diagnostics, task_deps, layout_depth };

        // Use the `ImplicitCtxt` while we execute the query.
        enter_context(&new_icx, || rustc_data_structures::stack::ensure_sufficient_stack(compute))
    })
}

pub fn with_increased_layout_depth<R>(f: impl FnOnce(usize) -> R) -> R {
    with_context_opt(|icx| {
        let layout_depth = icx.map_or(0, |icx| icx.layout_depth);

        // Update the ImplicitCtxt to increase the layout_depth
        let icx =
            ImplicitCtxt { layout_depth: layout_depth + 1, ..icx.cloned().unwrap_or_default() };

        enter_context(&icx, || f(layout_depth))
    })
}
