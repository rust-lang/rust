use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_session::Session;
use rustc_span::Span;

use super::QueryStackFrameExtra;
use crate::dep_graph::DepContext;
use crate::error::CycleStack;
use crate::query::plumbing::CycleError;
use crate::query::{QueryContext, QueryStackFrame};

/// Represents a span and a query key.
#[derive(Clone, Debug)]
pub struct QueryInfo<I> {
    /// The span corresponding to the reason for which this query was required.
    pub span: Span,
    pub query: QueryStackFrame<I>,
}

impl<I> QueryInfo<I> {
    pub(crate) fn lift<Qcx: QueryContext<QueryInfo = I>>(
        &self,
        qcx: Qcx,
    ) -> QueryInfo<QueryStackFrameExtra> {
        QueryInfo { span: self.span, query: self.query.lift(qcx) }
    }
}

pub type QueryMap<I> = FxHashMap<QueryJobId, QueryJobInfo<I>>;

/// A value uniquely identifying an active query job.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QueryJobId(pub NonZero<u64>);

impl QueryJobId {
    fn query<I: Clone>(self, map: &QueryMap<I>) -> QueryStackFrame<I> {
        map.get(&self).unwrap().query.clone()
    }
}

#[derive(Clone, Debug)]
pub struct QueryJobInfo<I> {
    pub query: QueryStackFrame<I>,
    pub job: QueryJob<I>,
}

/// Represents an active query job.
#[derive(Debug)]
pub struct QueryJob<I> {
    pub id: QueryJobId,

    /// The span corresponding to the reason for which this query was required.
    pub span: Span,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<QueryJobId>,

    /// The latch that is used to wait on this job.
    latch: Option<QueryLatch<I>>,
}

impl<I> Clone for QueryJob<I> {
    fn clone(&self) -> Self {
        Self { id: self.id, span: self.span, parent: self.parent, latch: self.latch.clone() }
    }
}

impl<I> QueryJob<I> {
    /// Creates a new query job.
    #[inline]
    pub fn new(id: QueryJobId, span: Span, parent: Option<QueryJobId>) -> Self {
        QueryJob { id, span, parent, latch: None }
    }

    pub(super) fn latch(&mut self) -> QueryLatch<I> {
        if self.latch.is_none() {
            self.latch = Some(QueryLatch::new());
        }
        self.latch.as_ref().unwrap().clone()
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    #[inline]
    pub fn signal_complete(self) {
        if let Some(latch) = self.latch {
            latch.set();
        }
    }
}

impl QueryJobId {
    pub(super) fn find_cycle_in_stack<I: Clone>(
        &self,
        query_map: QueryMap<I>,
        current_job: &Option<QueryJobId>,
        span: Span,
    ) -> CycleError<I> {
        // Find the waitee amongst `current_job` parents
        let mut cycle = Vec::new();
        let mut current_job = Option::clone(current_job);

        while let Some(job) = current_job {
            let info = query_map.get(&job).unwrap();
            cycle.push(QueryInfo { span: info.job.span, query: info.query.clone() });

            if job == *self {
                cycle.reverse();

                // This is the end of the cycle
                // The span entry we included was for the usage
                // of the cycle itself, and not part of the cycle
                // Replace it with the span which caused the cycle to form
                cycle[0].span = span;
                // Find out why the cycle itself was used
                let usage = info
                    .job
                    .parent
                    .as_ref()
                    .map(|parent| (info.job.span, parent.query(&query_map)));
                return CycleError { usage, cycle };
            }

            current_job = info.job.parent;
        }

        panic!("did not find a cycle")
    }

    #[cold]
    #[inline(never)]
    pub fn find_dep_kind_root<I: Clone>(&self, query_map: QueryMap<I>) -> (QueryJobInfo<I>, usize) {
        let mut depth = 1;
        let info = query_map.get(&self).unwrap();
        let dep_kind = info.query.dep_kind;
        let mut current_id = info.job.parent;
        let mut last_layout = (info.clone(), depth);

        while let Some(id) = current_id {
            let info = query_map.get(&id).unwrap();
            if info.query.dep_kind == dep_kind {
                depth += 1;
                last_layout = (info.clone(), depth);
            }
            current_id = info.job.parent;
        }
        last_layout
    }
}

#[derive(Debug)]
struct QueryWaiter<I> {
    condvar: Condvar,
    cycle: Mutex<Option<CycleError<I>>>,
}

#[derive(Debug)]
struct QueryLatchInfo<I> {
    complete: bool,
    waiters: Vec<Arc<QueryWaiter<I>>>,
}

#[derive(Debug)]
pub(super) struct QueryLatch<I> {
    info: Arc<Mutex<QueryLatchInfo<I>>>,
}

impl<I> Clone for QueryLatch<I> {
    fn clone(&self) -> Self {
        Self { info: Arc::clone(&self.info) }
    }
}

impl<I> QueryLatch<I> {
    fn new() -> Self {
        QueryLatch {
            info: Arc::new(Mutex::new(QueryLatchInfo { complete: false, waiters: Vec::new() })),
        }
    }

    /// Awaits for the query job to complete.
    pub(super) fn wait_on(&self, qcx: impl QueryContext) -> Result<(), CycleError<I>> {
        let waiter = Arc::new(QueryWaiter { cycle: Mutex::new(None), condvar: Condvar::new() });
        self.wait_on_inner(qcx, &waiter);
        // FIXME: Get rid of this lock. We have ownership of the QueryWaiter
        // although another thread may still have a Arc reference so we cannot
        // use Arc::get_mut
        let mut cycle = waiter.cycle.lock();
        match cycle.take() {
            None => Ok(()),
            Some(cycle) => Err(cycle),
        }
    }

    /// Awaits the caller on this latch by blocking the current thread.
    fn wait_on_inner(&self, qcx: impl QueryContext, waiter: &Arc<QueryWaiter<I>>) {
        let mut info = self.info.lock();
        if !info.complete {
            // We push the waiter on to the `waiters` list. It can be accessed inside
            // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
            // Both of these will remove it from the `waiters` list before resuming
            // this thread.
            info.waiters.push(Arc::clone(waiter));

            let proxy = qcx.jobserver_proxy();
            proxy.release_thread();
            waiter.condvar.wait(&mut info);
            // Release the lock before we potentially block in `acquire_thread`
            drop(info);
            proxy.acquire_thread();
        }
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        info.complete = true;
        for waiter in info.waiters.drain(..) {
            waiter.condvar.notify_one();
        }
    }
}

#[inline(never)]
#[cold]
pub fn report_cycle<'a>(
    sess: &'a Session,
    CycleError { usage, cycle: stack }: &CycleError,
) -> Diag<'a> {
    assert!(!stack.is_empty());

    let span = stack[0].query.info.default_span(stack[1 % stack.len()].span);

    let mut cycle_stack = Vec::new();

    use crate::error::StackCount;
    let stack_count = if stack.len() == 1 { StackCount::Single } else { StackCount::Multiple };

    for i in 1..stack.len() {
        let query = &stack[i].query;
        let span = query.info.default_span(stack[(i + 1) % stack.len()].span);
        cycle_stack.push(CycleStack { span, desc: query.info.description.to_owned() });
    }

    let mut cycle_usage = None;
    if let Some((span, ref query)) = *usage {
        cycle_usage = Some(crate::error::CycleUsage {
            span: query.info.default_span(span),
            usage: query.info.description.to_string(),
        });
    }

    let alias =
        if stack.iter().all(|entry| matches!(entry.query.info.def_kind, Some(DefKind::TyAlias))) {
            Some(crate::error::Alias::Ty)
        } else if stack.iter().all(|entry| entry.query.info.def_kind == Some(DefKind::TraitAlias)) {
            Some(crate::error::Alias::Trait)
        } else {
            None
        };

    let cycle_diag = crate::error::Cycle {
        span,
        cycle_stack,
        stack_bottom: stack[0].query.info.description.to_owned(),
        alias,
        cycle_usage,
        stack_count,
        note_span: (),
    };

    sess.dcx().create_err(cycle_diag)
}

pub fn print_query_stack<Qcx: QueryContext>(
    qcx: Qcx,
    mut current_query: Option<QueryJobId>,
    dcx: DiagCtxtHandle<'_>,
    limit_frames: Option<usize>,
    mut file: Option<std::fs::File>,
) -> usize {
    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `DiagCtxt` may be in a weird
    // state if it was responsible for triggering the panic.
    let mut count_printed = 0;
    let mut count_total = 0;

    // Make use of a partial query map if we fail to take locks collecting active queries.
    let query_map = match qcx.collect_active_jobs() {
        Ok(query_map) => query_map,
        Err(query_map) => query_map,
    };

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "\n\nquery stack during panic:");
    }
    while let Some(query) = current_query {
        let Some(query_info) = query_map.get(&query) else {
            break;
        };
        let query_extra = qcx.lift_query_info(&query_info.query.info);
        if Some(count_printed) < limit_frames || limit_frames.is_none() {
            // Only print to stderr as many stack frames as `num_frames` when present.
            // FIXME: needs translation
            #[allow(rustc::diagnostic_outside_of_impl)]
            #[allow(rustc::untranslatable_diagnostic)]
            dcx.struct_failure_note(format!(
                "#{} [{:?}] {}",
                count_printed, query_info.query.dep_kind, query_extra.description
            ))
            .with_span(query_info.job.span)
            .emit();
            count_printed += 1;
        }

        if let Some(ref mut file) = file {
            let _ = writeln!(
                file,
                "#{} [{}] {}",
                count_total,
                qcx.dep_context().dep_kind_info(query_info.query.dep_kind).name,
                query_extra.description
            );
        }

        current_query = query_info.job.parent;
        count_total += 1;
    }

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "end of query stack");
    }
    count_total
}
