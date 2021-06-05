use crate::dep_graph::TaskDepsRef;
use crate::query::{QueryContext, QueryJobId};
use rustc_data_structures::sync::{self, Lock};
use rustc_errors::Diagnostic;

#[cfg(not(parallel_compiler))]
use std::cell::Cell;
use thin_vec::ThinVec;

#[derive(Copy, Clone)]
struct GCXPointer(*const ());

unsafe impl sync::DynSync for GCXPointer {}

/// This is the implicit state of rustc. It contains the current
/// `TyCtxt` and query. It is updated when creating a local interner or
/// executing a new query. Whenever there's a `TyCtxt` value available
/// you should also have access to an `ImplicitCtxt` through the functions
/// in this module.
#[derive(Clone)]
struct ImplicitCtxt<'a> {
    /// Pointer to the current `GlobalCtxt`.
    gcx: GCXPointer,

    /// The current query job, if any. This is updated by `JobOwner::start` in
    /// `ty::query::plumbing` when executing a query.
    query: Option<QueryJobId>,

    /// Where to store diagnostics for the current query job, if any.
    /// This is updated by `JobOwner::start` in `ty::query::plumbing` when executing a query.
    diagnostics: Option<&'a Lock<ThinVec<Diagnostic>>>,

    /// Used to prevent queries from calling too deeply.
    query_depth: usize,

    /// The current dep graph task. This is used to add dependencies to queries
    /// when executing them.
    task_deps: TaskDepsRef<'a>,
}

pub fn create_and_enter_context<GCX: sync::DynSync, R>(gcx: &GCX, f: impl FnOnce() -> R) -> R {
    let icx = ImplicitCtxt {
        gcx: GCXPointer((gcx as *const GCX).cast()),
        query: None,
        diagnostics: None,
        query_depth: 0,
        task_deps: TaskDepsRef::Ignore,
    };
    enter_context(&icx, f)
}

// Import the thread-local variable from Rayon, which is preserved for Rayon jobs.
#[cfg(parallel_compiler)]
use rayon_core::tlv::TLV;

// Otherwise define our own
#[cfg(not(parallel_compiler))]
thread_local! {
    /// A thread local variable that stores a pointer to the current `ImplicitCtxt`.
    static TLV: Cell<*const ()> = const { Cell::new(ptr::null()) };
}

#[inline]
fn erase(context: &ImplicitCtxt<'_>) -> *const () {
    context as *const _ as *const ()
}

#[inline]
unsafe fn downcast<'a>(context: *const ()) -> &'a ImplicitCtxt<'a> {
    &*(context as *const ImplicitCtxt<'a>)
}

/// Sets `context` as the new current `ImplicitCtxt` for the duration of the function `f`.
#[inline]
fn enter_context<'a, F, R>(context: &ImplicitCtxt<'a>, f: F) -> R
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
fn with_context_opt<F, R>(f: F) -> R
where
    F: for<'a> FnOnce(Option<&ImplicitCtxt<'a>>) -> R,
{
    let context = TLV.get();
    if context.is_null() {
        f(None)
    } else {
        // We could get an `ImplicitCtxt` pointer from another thread.
        // Ensure that `ImplicitCtxt` is `DynSync`.
        sync::assert_dyn_sync::<ImplicitCtxt<'_>>();

        unsafe { f(Some(downcast(context))) }
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

/// Allows access to the `TyCtxt` in the current `ImplicitCtxt`.
/// Panics if there is no `ImplicitCtxt` available.
#[inline]
pub fn with<F, R>(f: F) -> R
where
    F: for<'tcx> FnOnce(*const ()) -> R,
{
    with_context(|context| f(context.gcx.0))
}

/// Allows access to the `TyCtxt` in the current `ImplicitCtxt`.
/// The closure is passed None if there is no `ImplicitCtxt` available.
#[inline]
pub fn with_opt<F, R>(f: F) -> R
where
    F: for<'tcx> FnOnce(Option<*const ()>) -> R,
{
    with_context_opt(|opt_context| f(opt_context.map(|context| context.gcx.0)))
}

/// This is a callback from `rustc_ast` as it cannot access the implicit state
/// in `rustc_middle` otherwise. It is used to when diagnostic messages are
/// emitted and stores them in the current query, if there is one.
pub fn track_diagnostic(diagnostic: Diagnostic, f: &mut dyn FnMut(Diagnostic)) {
    with_context_opt(|icx| {
        if let Some(icx) = icx {
            if let Some(ref diagnostics) = icx.diagnostics {
                diagnostics.lock().extend(Some(diagnostic.clone()));
            }

            // Diagnostics are tracked, we can ignore the dependency.
            let icx = ImplicitCtxt { task_deps: TaskDepsRef::Ignore, ..icx.clone() };
            return enter_context(&icx, move || (*f)(diagnostic));
        }

        // In any other case, invoke diagnostics anyway.
        (*f)(diagnostic);
    })
}

/// Execute the operation with provided dependencies.
pub fn with_deps<OP, R>(task_deps: TaskDepsRef<'_>, op: OP) -> R
where
    OP: FnOnce() -> R,
{
    crate::tls::with_context(|icx| {
        let icx = crate::tls::ImplicitCtxt { task_deps, ..icx.clone() };
        crate::tls::enter_context(&icx, op)
    })
}

/// Access dependencies from current implicit context.
pub fn read_deps<OP>(op: OP)
where
    OP: FnOnce(TaskDepsRef<'_>),
{
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
pub fn start_query<QCX: QueryContext, R>(
    qcx: QCX,
    token: QueryJobId,
    depth_limit: bool,
    diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
    compute: impl FnOnce() -> R,
) -> R {
    with_context(move |current_icx| {
        if depth_limit && !qcx.recursion_limit().value_within_limit(current_icx.query_depth) {
            qcx.depth_limit_error(token);
        }

        // Update the `ImplicitCtxt` to point to our new query job.
        let new_icx = ImplicitCtxt {
            gcx: current_icx.gcx,
            query: Some(token),
            diagnostics,
            query_depth: current_icx.query_depth + depth_limit as usize,
            task_deps: current_icx.task_deps,
        };

        // Use the `ImplicitCtxt` while we execute the query.
        enter_context(&new_icx, || rustc_data_structures::stack::ensure_sufficient_stack(compute))
    })
}
