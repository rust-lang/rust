use crate::dep_graph::TaskDeps;
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
#[derive(Clone)]
pub struct ImplicitCtxt<'a> {
    /// The current query job, if any. This is updated by `JobOwner::start` in
    /// `ty::query::plumbing` when executing a query.
    query: Option<QueryJobId>,

    /// Where to store diagnostics for the current query job, if any.
    /// This is updated by `JobOwner::start` in `ty::query::plumbing` when executing a query.
    diagnostics: Option<&'a Lock<ThinVec<Diagnostic>>>,

    /// The current dep graph task. This is used to add dependencies to queries
    /// when executing them.
    task_deps: Option<&'a Lock<TaskDeps>>,
}

impl<'a> ImplicitCtxt<'a> {
    pub fn new() -> Self {
        ImplicitCtxt { query: None, diagnostics: None, task_deps: None }
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
pub fn enter_context<'a, F, R>(context: &ImplicitCtxt<'a>, f: F) -> R
where
    F: FnOnce(&ImplicitCtxt<'a>) -> R,
{
    set_tlv(context as *const _ as usize, || f(&context))
}

/// Allows access to the current `ImplicitCtxt` in a closure if one is available.
#[inline]
fn with_context_opt<F, R>(f: F) -> R
where
    F: for<'a> FnOnce(Option<&ImplicitCtxt<'a>>) -> R,
{
    let context = get_tlv();
    if context == 0 {
        f(None)
    } else {
        // We could get a `ImplicitCtxt` pointer from another thread.
        // Ensure that `ImplicitCtxt` is `Sync`.
        sync::assert_sync::<ImplicitCtxt<'_>>();

        unsafe { f(Some(&*(context as *const ImplicitCtxt<'_>))) }
    }
}

/// Allows access to the current `ImplicitCtxt`.
/// Panics if there is no `ImplicitCtxt` available.
#[inline]
fn with_context<F, R>(f: F) -> R
where
    F: for<'a> FnOnce(&ImplicitCtxt<'a>) -> R,
{
    with_context_opt(|opt_context| f(opt_context.expect("no ImplicitCtxt stored in tls")))
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

pub fn with_deps<OP, R>(task_deps: Option<&Lock<TaskDeps>>, op: OP) -> R
where
    OP: FnOnce() -> R,
{
    crate::tls::with_context(|icx| {
        let icx = crate::tls::ImplicitCtxt { task_deps, ..icx.clone() };

        crate::tls::enter_context(&icx, |_| op())
    })
}

pub fn read_deps<OP>(op: OP)
where
    OP: for<'a> FnOnce(Option<&'a Lock<TaskDeps>>),
{
    crate::tls::with_context_opt(|icx| {
        let icx = if let Some(icx) = icx { icx } else { return };
        op(icx.task_deps)
    })
}

/// Get the query information from the TLS context.
#[inline(always)]
pub fn current_query_job() -> Option<QueryJobId> {
    with_context(|icx| icx.query)
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
    with_context(move |current_icx| {
        // Update the `ImplicitCtxt` to point to our new query job.
        let new_icx =
            ImplicitCtxt { query: Some(token), diagnostics, task_deps: current_icx.task_deps };

        // Use the `ImplicitCtxt` while we execute the query.
        enter_context(&new_icx, |_| rustc_data_structures::stack::ensure_sufficient_stack(compute))
    })
}
