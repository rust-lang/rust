use std::collections::{BTreeMap, hash_map};
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::num::NonZero;
use std::sync::{Arc, Weak};
use std::thread::ThreadId;
use std::{iter, ops};

use parking_lot::{Condvar, Mutex};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexmap::{self, IndexMap, IndexSet};
use rustc_data_structures::sync::BranchKey;
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_session::Session;
use rustc_span::{DUMMY_SP, Span};
use smallvec::SmallVec;

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

    fn span<I>(self, map: &QueryMap<I>) -> Span {
        map.get(&self).unwrap().job.span
    }

    fn parent<I>(self, map: &QueryMap<I>) -> Option<QueryInclusion> {
        map.get(&self).unwrap().job.parent
    }

    fn latch<I>(self, map: &QueryMap<I>) -> &Weak<QueryLatch<I>> {
        &map.get(&self).unwrap().job.latch
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
    pub parent: Option<QueryInclusion>,

    /// Id of the query's execution thread.
    pub thread_id: ThreadId,

    /// The latch that is used to wait on this job.
    latch: Weak<QueryLatch<I>>,
}

impl<I> Clone for QueryJob<I> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            span: self.span,
            parent: self.parent,
            thread_id: self.thread_id,
            latch: self.latch.clone(),
        }
    }
}

impl<I> QueryJob<I> {
    /// Creates a new query job.
    #[inline]
    pub fn new(
        id: QueryJobId,
        span: Span,
        parent: Option<QueryInclusion>,
        thread_id: ThreadId,
    ) -> Self {
        QueryJob { id, span, parent, thread_id, latch: Weak::new() }
    }

    pub fn real_depth(&self) -> usize {
        self.parent.as_ref().map_or(0, |i| i.real_depth.get())
    }

    pub(super) fn latch(&mut self) -> Arc<QueryLatch<I>> {
        if let Some(latch) = self.latch.upgrade() {
            latch
        } else {
            let latch = Arc::new(QueryLatch::new());
            self.latch = Arc::downgrade(&latch);
            latch
        }
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    #[inline]
    pub fn signal_complete(self) {
        if let Some(latch) = self.latch.upgrade() {
            latch.set();
        }
    }
}

impl QueryJobId {
    pub(super) fn find_cycle_in_stack<I: Clone>(
        &self,
        query_map: QueryMap<I>,
        mut current_job: Option<QueryJobId>,
        span: Span,
    ) -> CycleError<I> {
        // Find the waitee amongst `current_job` parents
        let mut cycle = Vec::new();

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
                    .map(|parent| (info.job.span, parent.id.query(&query_map)));
                return CycleError { usage, cycle };
            }

            current_job = info.job.parent.map(|i| i.id);
        }

        panic!("did not find a cycle")
    }

    #[cold]
    #[inline(never)]
    pub fn find_dep_kind_root<I: Clone>(&self, query_map: QueryMap<I>) -> (QueryJobInfo<I>, usize) {
        let mut depth = 1;
        let info = query_map.get(&self).unwrap();
        let dep_kind = info.query.dep_kind;
        let mut current = info.job.parent;
        let mut last_layout = (info.clone(), depth);

        while let Some(inclusion) = current {
            let info = query_map.get(&inclusion.id).unwrap();
            if info.query.dep_kind == dep_kind {
                depth += 1;
                last_layout = (info.clone(), depth);
            }
            current = info.job.parent;
        }
        last_layout
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QueryInclusion {
    pub id: QueryJobId,
    pub branch: BranchKey,
    pub real_depth: NonZero<usize>,
}

#[derive(Debug)]
struct QueryWaiter<I> {
    query: Option<QueryInclusion>,
    thread_id: ThreadId,
    condvar: Condvar,
    span: Span,
    cycle: Mutex<Option<CycleError<I>>>,
}

impl<I> QueryWaiter<I> {
    fn real_depth(&self) -> usize {
        self.query.as_ref().map_or(0, |i| i.real_depth.get())
    }
}

#[derive(Debug)]
struct QueryLatchInfo<I> {
    complete: bool,
    waiters: Vec<Arc<QueryWaiter<I>>>,
}

#[derive(Debug)]
pub(super) struct QueryLatch<I> {
    info: Mutex<QueryLatchInfo<I>>,
}

impl<I> QueryLatch<I> {
    fn new() -> Self {
        QueryLatch { info: Mutex::new(QueryLatchInfo { complete: false, waiters: Vec::new() }) }
    }

    /// Awaits for the query job to complete.
    pub(super) fn wait_on(
        &self,
        qcx: impl QueryContext,
        query: Option<QueryInclusion>,
        span: Span,
    ) -> Result<(), CycleError<I>> {
        let waiter = Arc::new(QueryWaiter {
            query,
            span,
            thread_id: std::thread::current().id(),
            cycle: Mutex::new(None),
            condvar: Condvar::new(),
        });
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

            // If this detects a deadlock and the deadlock handler wants to resume this thread
            // we have to be in the `wait` call. This is ensured by the deadlock handler
            // getting the self.info lock.
            rustc_thread_pool::mark_blocked();
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
        let registry = rustc_thread_pool::Registry::current();
        for waiter in info.waiters.drain(..) {
            rustc_thread_pool::mark_unblocked(&registry);
            waiter.condvar.notify_one();
        }
    }

    /// Removes a single waiter from the list of waiters.
    /// This is used to break query cycles.
    fn extract_waiter(&self, waiter: usize) -> Arc<QueryWaiter<I>> {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        // Remove the waiter from the list of waiters
        info.waiters.remove(waiter)
    }
}

/// A resumable waiter of a query. The usize is the index into waiters in the query's latch
type Waiter = (QueryJobId, usize);

// Deterministically pick an query from a list
fn pick_query<'a, I: Clone, T, F>(query_map: &QueryMap<I>, queries: &'a [T], f: F) -> &'a T
where
    F: Fn(&T) -> (Span, QueryJobId),
{
    // Deterministically pick an entry point
    // FIXME: Sort this instead
    queries
        .iter()
        .min_by_key(|v| {
            let (span, query) = f(v);
            let hash = query.query(query_map).hash;
            // Prefer entry points which have valid spans for nicer error messages
            // We add an integer to the tuple ensuring that entry points
            // with valid spans are picked first
            let span_cmp = if span == DUMMY_SP { 1 } else { 0 };
            (span_cmp, hash)
        })
        .unwrap()
}

/// Detects query cycles by using depth first search over all active query jobs.
/// If a query cycle is found it will break the cycle by finding an edge which
/// uses a query latch and then resuming that waiter.
/// There may be multiple cycles involved in a deadlock, so this searches
/// all active queries for cycles before finally resuming all the waiters at once.
#[allow(rustc::potential_query_instability)]
pub fn break_query_cycles<I: Clone + Debug>(
    query_map: QueryMap<I>,
    registry: &rustc_thread_pool::Registry,
) {
    #[derive(Debug)]
    struct QueryStackIntermediate {
        start: Option<QueryJobId>,
        depth: ops::RangeInclusive<usize>,
        wait: Option<QueryWait>,
    }

    impl QueryStackIntermediate {
        fn from_depth(depth: usize) -> Self {
            QueryStackIntermediate { start: None, depth: depth..=depth, wait: None }
        }

        fn update_depth(&mut self, depth: usize) {
            let (start, end) = self.depth.clone().into_inner();
            if depth < start {
                self.depth = depth..=end;
            }
            if end < depth {
                self.depth = start..=depth
            }
        }
    }

    #[derive(Debug)]
    enum QueryWait {
        /// Waits on a running query
        Waiter { waited_on: QueryJobId, waiter_idx: usize },
        /// Waits other for tasks inside of `join` or `scope`
        Direct { waited_on: Vec<QueryJobId> },
    }

    impl QueryWait {
        fn waiting_query<I>(&self, query_map: &QueryMap<I>) -> QueryJobId {
            match self {
                QueryWait::Waiter { waited_on, waiter_idx } => {
                    query_map[waited_on]
                        .job
                        .latch
                        .upgrade()
                        .unwrap()
                        .info
                        .try_lock()
                        .unwrap()
                        .waiters[*waiter_idx]
                        .query
                        .unwrap()
                        .id
                }
                QueryWait::Direct { waited_on } => query_map[&waited_on[0]].job.parent.unwrap().id,
            }
        }
    }

    let mut stacks = FxHashMap::<ThreadId, QueryStackIntermediate>::default();
    for query in query_map.values() {
        let query_depth = query.job.real_depth();
        let entry = stacks.entry(query.job.thread_id);
        let stack = match entry {
            hash_map::Entry::Vacant(entry) => {
                entry.insert(QueryStackIntermediate::from_depth(query_depth))
            }
            hash_map::Entry::Occupied(mut entry) => {
                let stack = entry.into_mut();
                stack.update_depth(query_depth);
                stack
            }
        };

        if query
            .job
            .parent
            .is_none_or(|inclusion| query_map[&inclusion.id].job.thread_id != query.job.thread_id)
        {
            // Register the thread's query stack beginning
            assert!(stack.start.is_none(), "found two active queries at a thread's begining");
            stack.start = Some(query.job.id);
        }

        let Some(latch) = query.job.latch.upgrade() else {
            continue;
        };
        let lock = latch.info.try_lock().unwrap();
        assert!(!lock.complete);
        for (waiter_idx, waiter) in lock.waiters.iter().enumerate() {
            let waiting_stack = stacks
                .entry(waiter.thread_id)
                .or_insert_with(|| QueryStackIntermediate::from_depth(waiter.real_depth() - 1));
            assert!(
                waiting_stack.wait.is_none(),
                "found two active queries a thread is waiting for"
            );
            waiting_stack.wait = Some(QueryWait::Waiter { waited_on: query.job.id, waiter_idx });
        }
    }

    // Figure out what queries leftover stacks are blocked on
    let mut root_query = None;
    let mut thread_ids: Vec<_> = stacks.keys().copied().collect();
    for thread_id in &thread_ids {
        let stack = &stacks[thread_id];
        let start = stack.start.unwrap();
        if let Some(inclusion) = query_map[&start].job.parent {
            let parent = &query_map[&inclusion.id];
            assert_eq!(inclusion.real_depth.get(), *stack.depth.start());
            let waiting_stack = stacks.get_mut(&parent.job.thread_id).unwrap();
            if *waiting_stack.depth.end() == (inclusion.real_depth.get() - 1) {
                match &mut waiting_stack.wait {
                    None => waiting_stack.wait = Some(QueryWait::Direct { waited_on: vec![start] }),
                    Some(QueryWait::Direct { waited_on }) => {
                        assert!(!waited_on.contains(&start));
                        waited_on.push(start);
                    }
                    Some(QueryWait::Waiter { .. }) => (),
                }
            }
        } else {
            assert!(root_query.is_none(), "found multiple threads without start");
            root_query = Some(start);
        }
    }

    let root_query = root_query.expect("no root query was found");

    for stack in stacks.values() {
        match stack.wait.as_ref().expect("failed to figure out what active thread is waiting") {
            QueryWait::Waiter { waited_on, waiter_idx } => {
                assert_eq!(
                    query_map[waited_on]
                        .job
                        .latch
                        .upgrade()
                        .unwrap()
                        .info
                        .try_lock()
                        .unwrap()
                        .waiters[*waiter_idx]
                        .real_depth()
                        - 1,
                    *stack.depth.end()
                )
            }
            QueryWait::Direct { waited_on } => {
                let waited_on_query = &query_map[&waited_on[0]];
                let query_inclusion = waited_on_query.job.parent.unwrap();
                let parent_id = query_inclusion.id;
                for waited_on_id in &waited_on[1..] {
                    assert_eq!(parent_id, query_map[waited_on_id].job.parent.unwrap().id);
                }
                assert_eq!(query_inclusion.real_depth.get() - 1, *stack.depth.end());
            }
        }
    }

    fn collect_branches<I>(query_id: QueryJobId, query_map: &QueryMap<I>) -> Vec<BranchKey> {
        let query = &query_map[&query_id];
        let Some(inclusion) = query.job.parent.as_ref() else { return Vec::new() };
        // Skip trivial branches
        if inclusion.branch == BranchKey::root() {
            return collect_branches(inclusion.id, query_map);
        }
        let mut out = collect_branches(inclusion.id, query_map);
        out.push(inclusion.branch);
        out
    }
    let branches: FxHashMap<_, _> = thread_ids
        .iter()
        .map(|t| {
            (
                *t,
                collect_branches(
                    stacks[t].wait.as_ref().unwrap().waiting_query(&query_map),
                    &query_map,
                ),
            )
        })
        .collect();

    thread_ids.sort_by_key(|t| branches[t].as_slice());

    let branch_enumerations: FxHashMap<_, _> =
        thread_ids.iter().enumerate().map(|(v, k)| (*k, v)).collect();

    let mut subqueries = FxHashMap::<_, BTreeMap<BranchKey, _>>::default();
    for query in query_map.values() {
        let Some(inclusion) = &query.job.parent else {
            continue;
        };
        let old = subqueries
            .entry(inclusion.id)
            .or_default()
            .insert(inclusion.branch, (query.job.id, usize::MAX));
        assert!(old.is_none());
    }

    for stack in stacks.values() {
        let &QueryWait::Waiter { waited_on, waiter_idx } = stack.wait.as_ref().unwrap() else {
            continue;
        };

        let inclusion =
            query_map[&waited_on].job.latch.upgrade().unwrap().info.try_lock().unwrap().waiters
                [waiter_idx]
                .query
                .unwrap();
        let old = subqueries
            .entry(inclusion.id)
            .or_default()
            .insert(inclusion.branch, (waited_on, waiter_idx));
        assert!(old.is_none());
    }

    let mut visited = IndexMap::new();
    let mut last_usage = None;
    let mut last_waiter_idx = usize::MAX;
    let mut current = root_query;
    while let indexmap::map::Entry::Vacant(entry) = visited.entry(current) {
        entry.insert((last_usage, last_waiter_idx));
        last_usage = Some(current);
        (current, last_waiter_idx) = *subqueries
            .get(&current)
            .unwrap_or_else(|| {
                panic!(
                    "deadlock detected as we're unable to find a query cycle to break\n\
                current query map:\n{:#?}",
                    query_map
                )
            })
            .first_key_value()
            .unwrap()
            .1;
    }
    let usage = visited[&current].0;
    let mut iter = visited.keys().rev();
    let mut cycle = Vec::new();
    loop {
        let query_id = *iter.next().unwrap();
        let query = &query_map[&query_id];
        cycle.push(QueryInfo { span: query.job.span, query: query.query.clone() });
        if query_id == current {
            break;
        }
    }

    cycle.reverse();
    let cycle_error = CycleError {
        usage: usage.map(|id| {
            let query = &query_map[&id];
            (query.job.span, query.query.clone())
        }),
        cycle,
    };

    let (waited_on, waiter_idx) = if last_waiter_idx != usize::MAX {
        (current, last_waiter_idx)
    } else {
        let (&waited_on, &(_, waiter_idx)) =
            visited.iter().rev().find(|(_, (_, waiter_idx))| *waiter_idx != usize::MAX).unwrap();
        (waited_on, waiter_idx)
    };
    let waited_on = &query_map[&waited_on];
    let latch = waited_on.job.latch.upgrade().unwrap();
    let latch_info_lock = latch.info.try_lock().unwrap();
    let waiter = &latch_info_lock.waiters[waiter_idx];
    let mut cycle_lock = waiter.cycle.try_lock().unwrap();
    assert!(cycle_lock.is_none());
    *cycle_lock = Some(cycle_error);
    rustc_thread_pool::mark_unblocked(registry);
    waiter.condvar.notify_one();

    // // Check that a cycle was found. It is possible for a deadlock to occur without
    // // a query cycle if a query which can be waited on uses Rayon to do multithreading
    // // internally. Such a query (X) may be executing on 2 threads (A and B) and A may
    // // wait using Rayon on B. Rayon may then switch to executing another query (Y)
    // // which in turn will wait on X causing a deadlock. We have a false dependency from
    // // X to Y due to Rayon waiting and a true dependency from Y to X. The algorithm here
    // // only considers the true dependency and won't detect a cycle.
    // if !found_cycle {
    //     panic!(
    //         "deadlock detected as we're unable to find a query cycle to break\n\
    //         current query map:\n{:#?}",
    //         query_map
    //     );
    // }

    // // Mark all the thread we're about to wake up as unblocked. This needs to be done before
    // // we wake the threads up as otherwise Rayon could detect a deadlock if a thread we
    // // resumed fell asleep and this thread had yet to mark the remaining threads as unblocked.
    // for _ in 0..wakelist.len() {
    //     rustc_thread_pool::mark_unblocked(registry);
    // }

    // for waiter in wakelist.into_iter() {
    //     waiter.condvar.notify_one();
    // }
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
    let query_map = match qcx.collect_active_jobs(false) {
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

        current_query = query_info.job.parent.map(|i| i.id);
        count_total += 1;
    }

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "end of query stack");
    }
    count_total
}
