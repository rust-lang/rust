//! The implementation of the query system itself. Defines the macros
//! that generate the actual methods on tcx which find and execute the
//! provider, manage the caches, and so forth.

use dep_graph::{DepNodeIndex, DepNode, DepKind, DepNodeColor};
use errors::DiagnosticBuilder;
use errors::Level;
use errors::Diagnostic;
use errors::FatalError;
use ty::tls;
use ty::{TyCtxt};
use ty::query::Query;
use ty::query::config::{QueryConfig, QueryDescription};
use ty::query::job::{QueryJob, QueryResult, QueryInfo};
use ty::item_path;

use util::common::{profq_msg, ProfileQueriesMsg, QueryMsg};

use rustc_data_structures::fx::{FxHashMap};
use rustc_data_structures::sync::{Lrc, Lock};
use std::mem;
use std::ptr;
use std::collections::hash_map::Entry;
use syntax_pos::Span;
use syntax::source_map::DUMMY_SP;

pub struct QueryCache<'tcx, D: QueryConfig<'tcx> + ?Sized> {
    pub(super) results: FxHashMap<D::Key, QueryValue<D::Value>>,
    pub(super) active: FxHashMap<D::Key, QueryResult<'tcx>>,
    #[cfg(debug_assertions)]
    pub(super) cache_hits: usize,
}

pub(super) struct QueryValue<T> {
    pub(super) value: T,
    pub(super) index: DepNodeIndex,
}

impl<T> QueryValue<T> {
    pub(super) fn new(value: T,
                      dep_node_index: DepNodeIndex)
                      -> QueryValue<T> {
        QueryValue {
            value,
            index: dep_node_index,
        }
    }
}

impl<'tcx, M: QueryConfig<'tcx>> Default for QueryCache<'tcx, M> {
    fn default() -> QueryCache<'tcx, M> {
        QueryCache {
            results: FxHashMap::default(),
            active: FxHashMap::default(),
            #[cfg(debug_assertions)]
            cache_hits: 0,
        }
    }
}

// If enabled, send a message to the profile-queries thread
macro_rules! profq_msg {
    ($tcx:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            if $tcx.sess.profile_queries() {
                profq_msg($tcx.sess, $msg)
            }
        }
    }
}

// If enabled, format a key using its debug string, which can be
// expensive to compute (in terms of time).
macro_rules! profq_query_msg {
    ($query:expr, $tcx:expr, $key:expr) => {{
        let msg = if cfg!(debug_assertions) {
            if $tcx.sess.profile_queries_and_keys() {
                Some(format!("{:?}", $key))
            } else { None }
        } else { None };
        QueryMsg {
            query: $query,
            msg,
        }
    }}
}

/// A type representing the responsibility to execute the job in the `job` field.
/// This will poison the relevant query if dropped.
pub(super) struct JobOwner<'a, 'tcx: 'a, Q: QueryDescription<'tcx> + 'a> {
    cache: &'a Lock<QueryCache<'tcx, Q>>,
    key: Q::Key,
    job: Lrc<QueryJob<'tcx>>,
}

impl<'a, 'tcx, Q: QueryDescription<'tcx>> JobOwner<'a, 'tcx, Q> {
    /// Either gets a JobOwner corresponding the query, allowing us to
    /// start executing the query, or it returns with the result of the query.
    /// If the query is executing elsewhere, this will wait for it.
    /// If the query panicked, this will silently panic.
    ///
    /// This function is inlined because that results in a noticeable speedup
    /// for some compile-time benchmarks.
    #[inline(always)]
    pub(super) fn try_get(
        tcx: TyCtxt<'a, 'tcx, '_>,
        span: Span,
        key: &Q::Key,
    ) -> TryGetJob<'a, 'tcx, Q> {
        let cache = Q::query_cache(tcx);
        loop {
            let mut lock = cache.borrow_mut();
            if let Some(value) = lock.results.get(key) {
                profq_msg!(tcx, ProfileQueriesMsg::CacheHit);
                tcx.sess.profiler(|p| {
                    p.record_query(Q::CATEGORY);
                    p.record_query_hit(Q::CATEGORY);
                });

                let result = Ok((value.value.clone(), value.index));
                #[cfg(debug_assertions)]
                {
                    lock.cache_hits += 1;
                }
                return TryGetJob::JobCompleted(result);
            }
            let job = match lock.active.entry((*key).clone()) {
                Entry::Occupied(entry) => {
                    match *entry.get() {
                        QueryResult::Started(ref job) => job.clone(),
                        QueryResult::Poisoned => FatalError.raise(),
                    }
                }
                Entry::Vacant(entry) => {
                    // No job entry for this query. Return a new one to be started later
                    return tls::with_related_context(tcx, |icx| {
                        // Create the `parent` variable before `info`. This allows LLVM
                        // to elide the move of `info`
                        let parent = icx.query.clone();
                        let info = QueryInfo {
                            span,
                            query: Q::query(key.clone()),
                        };
                        let job = Lrc::new(QueryJob::new(info, parent));
                        let owner = JobOwner {
                            cache,
                            job: job.clone(),
                            key: (*key).clone(),
                        };
                        entry.insert(QueryResult::Started(job));
                        TryGetJob::NotYetStarted(owner)
                    })
                }
            };
            mem::drop(lock);

            // If we are single-threaded we know that we have cycle error,
            // so we just turn the errror
            #[cfg(not(parallel_queries))]
            return job.cycle_error(tcx, span);

            // With parallel queries we might just have to wait on some other
            // thread
            #[cfg(parallel_queries)]
            {
                if let Err(cycle) = job.await(tcx, span) {
                    return TryGetJob::JobCompleted(Err(cycle));
                }
            }
        }
    }

    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter and forgets the JobOwner, so it won't poison the query
    #[inline(always)]
    pub(super) fn complete(self, result: &Q::Value, dep_node_index: DepNodeIndex) {
        // We can move out of `self` here because we `mem::forget` it below
        let key = unsafe { ptr::read(&self.key) };
        let job = unsafe { ptr::read(&self.job) };
        let cache = self.cache;

        // Forget ourself so our destructor won't poison the query
        mem::forget(self);

        let value = QueryValue::new(result.clone(), dep_node_index);
        {
            let mut lock = cache.borrow_mut();
            lock.active.remove(&key);
            lock.results.insert(key, value);
        }

        job.signal_complete();
    }

    /// Executes a job by changing the ImplicitCtxt to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    pub(super) fn start<'lcx, F, R>(
        &self,
        tcx: TyCtxt<'_, 'tcx, 'lcx>,
        compute: F)
    -> (R, Vec<Diagnostic>)
    where
        F: for<'b> FnOnce(TyCtxt<'b, 'tcx, 'lcx>) -> R
    {
        // The TyCtxt stored in TLS has the same global interner lifetime
        // as `tcx`, so we use `with_related_context` to relate the 'gcx lifetimes
        // when accessing the ImplicitCtxt
        let r = tls::with_related_context(tcx, move |current_icx| {
            // Update the ImplicitCtxt to point to our new query job
            let new_icx = tls::ImplicitCtxt {
                tcx: tcx.global_tcx(),
                query: Some(self.job.clone()),
                layout_depth: current_icx.layout_depth,
                task: current_icx.task,
            };

            // Use the ImplicitCtxt while we execute the query
            tls::enter_context(&new_icx, |_| {
                compute(tcx)
            })
        });

        // Extract the diagnostic from the job
        let diagnostics = mem::replace(&mut *self.job.diagnostics.lock(), Vec::new());

        (r, diagnostics)
    }
}

impl<'a, 'tcx, Q: QueryDescription<'tcx>> Drop for JobOwner<'a, 'tcx, Q> {
    #[inline(never)]
    #[cold]
    fn drop(&mut self) {
        // Poison the query so jobs waiting on it panic
        self.cache.borrow_mut().active.insert(self.key.clone(), QueryResult::Poisoned);
        // Also signal the completion of the job, so waiters
        // will continue execution
        self.job.signal_complete();
    }
}

#[derive(Clone)]
pub struct CycleError<'tcx> {
    /// The query and related span which uses the cycle
    pub(super) usage: Option<(Span, Query<'tcx>)>,
    pub(super) cycle: Vec<QueryInfo<'tcx>>,
}

/// The result of `try_get_lock`
pub(super) enum TryGetJob<'a, 'tcx: 'a, D: QueryDescription<'tcx> + 'a> {
    /// The query is not yet started. Contains a guard to the cache eventually used to start it.
    NotYetStarted(JobOwner<'a, 'tcx, D>),

    /// The query was already completed.
    /// Returns the result of the query and its dep node index
    /// if it succeeded or a cycle error if it failed
    JobCompleted(Result<(D::Value, DepNodeIndex), Box<CycleError<'tcx>>>),
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    #[inline(never)]
    #[cold]
    pub(super) fn report_cycle(
        self,
        box CycleError { usage, cycle: stack }: Box<CycleError<'gcx>>
    ) -> Box<DiagnosticBuilder<'a>>
    {
        assert!(!stack.is_empty());

        let fix_span = |span: Span, query: &Query<'gcx>| {
            self.sess.source_map().def_span(query.default_span(self, span))
        };

        // Disable naming impls with types in this path, since that
        // sometimes cycles itself, leading to extra cycle errors.
        // (And cycle errors around impls tend to occur during the
        // collect/coherence phases anyhow.)
        item_path::with_forced_impl_filename_line(|| {
            let span = fix_span(stack[1 % stack.len()].span, &stack[0].query);
            let mut err = struct_span_err!(self.sess,
                                           span,
                                           E0391,
                                           "cycle detected when {}",
                                           stack[0].query.describe(self));

            for i in 1..stack.len() {
                let query = &stack[i].query;
                let span = fix_span(stack[(i + 1) % stack.len()].span, query);
                err.span_note(span, &format!("...which requires {}...", query.describe(self)));
            }

            err.note(&format!("...which again requires {}, completing the cycle",
                              stack[0].query.describe(self)));

            if let Some((span, query)) = usage {
                err.span_note(fix_span(span, &query),
                              &format!("cycle used when {}", query.describe(self)));
            }

            return Box::new(err)
        })
    }

    pub fn try_print_query_stack() {
        eprintln!("query stack during panic:");

        tls::with_context_opt(|icx| {
            if let Some(icx) = icx {
                let mut current_query = icx.query.clone();
                let mut i = 0;

                while let Some(query) = current_query {
                    let mut db = DiagnosticBuilder::new(icx.tcx.sess.diagnostic(),
                        Level::FailureNote,
                        &format!("#{} [{}] {}",
                                 i,
                                 query.info.query.name(),
                                 query.info.query.describe(icx.tcx)));
                    db.set_span(icx.tcx.sess.source_map().def_span(query.info.span));
                    icx.tcx.sess.diagnostic().force_print_db(db);

                    current_query = query.parent.clone();
                    i += 1;
                }
            }
        });

        eprintln!("end of query stack");
    }

    /// Try to read a node index for the node dep_node.
    /// A node will have an index, when it's already been marked green, or when we can mark it
    /// green. This function will mark the current task as a reader of the specified node, when
    /// a node index can be found for that node.
    pub(super) fn try_mark_green_and_read(self, dep_node: &DepNode) -> Option<DepNodeIndex> {
        match self.dep_graph.node_color(dep_node) {
            Some(DepNodeColor::Green(dep_node_index)) => {
                self.dep_graph.read_index(dep_node_index);
                Some(dep_node_index)
            }
            Some(DepNodeColor::Red) => {
                None
            }
            None => {
                // try_mark_green (called below) will panic when full incremental
                // compilation is disabled. If that's the case, we can't try to mark nodes
                // as green anyway, so we can safely return None here.
                if !self.dep_graph.is_fully_enabled() {
                    return None;
                }
                match self.dep_graph.try_mark_green(self.global_tcx(), &dep_node) {
                    Some(dep_node_index) => {
                        debug_assert!(self.dep_graph.is_green(&dep_node));
                        self.dep_graph.read_index(dep_node_index);
                        Some(dep_node_index)
                    }
                    None => {
                        None
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn try_get_with<Q: QueryDescription<'gcx>>(
        self,
        span: Span,
        key: Q::Key)
    -> Result<Q::Value, Box<CycleError<'gcx>>>
    {
        debug!("ty::queries::{}::try_get_with(key={:?}, span={:?})",
               Q::NAME,
               key,
               span);

        profq_msg!(self,
            ProfileQueriesMsg::QueryBegin(
                span.data(),
                profq_query_msg!(Q::NAME, self, key),
            )
        );

        self.sess.profiler(|p| p.record_query(Q::CATEGORY));

        let job = match JobOwner::try_get(self, span, &key) {
            TryGetJob::NotYetStarted(job) => job,
            TryGetJob::JobCompleted(result) => {
                return result.map(|(v, index)| {
                    self.sess.profiler(|p| p.record_query_hit(Q::CATEGORY));
                    self.dep_graph.read_index(index);
                    v
                })
            }
        };

        // Fast path for when incr. comp. is off. `to_dep_node` is
        // expensive for some DepKinds.
        if !self.dep_graph.is_fully_enabled() {
            let null_dep_node = DepNode::new_no_params(::dep_graph::DepKind::Null);
            return self.force_query_with_job::<Q>(key, job, null_dep_node).map(|(v, _)| v);
        }

        let dep_node = Q::to_dep_node(self, &key);

        if dep_node.kind.is_anon() {
            profq_msg!(self, ProfileQueriesMsg::ProviderBegin);
            self.sess.profiler(|p| p.start_activity(Q::CATEGORY));

            let res = job.start(self, |tcx| {
                tcx.dep_graph.with_anon_task(dep_node.kind, || {
                    Q::compute(tcx.global_tcx(), key)
                })
            });

            self.sess.profiler(|p| p.end_activity(Q::CATEGORY));
            profq_msg!(self, ProfileQueriesMsg::ProviderEnd);
            let ((result, dep_node_index), diagnostics) = res;

            self.dep_graph.read_index(dep_node_index);

            self.queries.on_disk_cache
                .store_diagnostics_for_anon_node(dep_node_index, diagnostics);

            job.complete(&result, dep_node_index);

            return Ok(result);
        }

        if !dep_node.kind.is_input() {
            if let Some(dep_node_index) = self.try_mark_green_and_read(&dep_node) {
                profq_msg!(self, ProfileQueriesMsg::CacheHit);
                self.sess.profiler(|p| p.record_query_hit(Q::CATEGORY));

                return self.load_from_disk_and_cache_in_memory::<Q>(key,
                                                                    job,
                                                                    dep_node_index,
                                                                    &dep_node)
            }
        }

        match self.force_query_with_job::<Q>(key, job, dep_node) {
            Ok((result, dep_node_index)) => {
                self.dep_graph.read_index(dep_node_index);
                Ok(result)
            }
            Err(e) => Err(e)
        }
    }

    fn load_from_disk_and_cache_in_memory<Q: QueryDescription<'gcx>>(
        self,
        key: Q::Key,
        job: JobOwner<'a, 'gcx, Q>,
        dep_node_index: DepNodeIndex,
        dep_node: &DepNode
    ) -> Result<Q::Value, Box<CycleError<'gcx>>>
    {
        // Note this function can be called concurrently from the same query
        // We must ensure that this is handled correctly

        debug_assert!(self.dep_graph.is_green(dep_node));

        // First we try to load the result from the on-disk cache
        let result = if Q::cache_on_disk(key.clone()) &&
                        self.sess.opts.debugging_opts.incremental_queries {
            let prev_dep_node_index =
                self.dep_graph.prev_dep_node_index_of(dep_node);
            let result = Q::try_load_from_disk(self.global_tcx(),
                                               prev_dep_node_index);

            // We always expect to find a cached result for things that
            // can be forced from DepNode.
            debug_assert!(!dep_node.kind.can_reconstruct_query_key() ||
                          result.is_some(),
                          "Missing on-disk cache entry for {:?}",
                          dep_node);
            result
        } else {
            // Some things are never cached on disk.
            None
        };

        let result = if let Some(result) = result {
            result
        } else {
            // We could not load a result from the on-disk cache, so
            // recompute.

            // The diagnostics for this query have already been
            // promoted to the current session during
            // try_mark_green(), so we can ignore them here.
            let (result, _) = job.start(self, |tcx| {
                // The dep-graph for this computation is already in
                // place
                tcx.dep_graph.with_ignore(|| {
                    Q::compute(tcx, key)
                })
            });
            result
        };

        // If -Zincremental-verify-ich is specified, re-hash results from
        // the cache and make sure that they have the expected fingerprint.
        if unlikely!(self.sess.opts.debugging_opts.incremental_verify_ich) {
            self.incremental_verify_ich::<Q>(&result, dep_node, dep_node_index);
        }

        if unlikely!(self.sess.opts.debugging_opts.query_dep_graph) {
            self.dep_graph.mark_loaded_from_cache(dep_node_index, true);
        }

        job.complete(&result, dep_node_index);

        Ok(result)
    }

    #[inline(never)]
    #[cold]
    fn incremental_verify_ich<Q: QueryDescription<'gcx>>(
        self,
        result: &Q::Value,
        dep_node: &DepNode,
        dep_node_index: DepNodeIndex,
    ) {
        use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
        use ich::Fingerprint;

        assert!(Some(self.dep_graph.fingerprint_of(dep_node_index)) ==
                self.dep_graph.prev_fingerprint_of(dep_node),
                "Fingerprint for green query instance not loaded \
                    from cache: {:?}", dep_node);

        debug!("BEGIN verify_ich({:?})", dep_node);
        let mut hcx = self.create_stable_hashing_context();
        let mut hasher = StableHasher::new();

        result.hash_stable(&mut hcx, &mut hasher);

        let new_hash: Fingerprint = hasher.finish();
        debug!("END verify_ich({:?})", dep_node);

        let old_hash = self.dep_graph.fingerprint_of(dep_node_index);

        assert!(new_hash == old_hash, "Found unstable fingerprints \
            for {:?}", dep_node);
    }

    fn force_query_with_job<Q: QueryDescription<'gcx>>(
        self,
        key: Q::Key,
        job: JobOwner<'_, 'gcx, Q>,
        dep_node: DepNode)
    -> Result<(Q::Value, DepNodeIndex), Box<CycleError<'gcx>>> {
        // If the following assertion triggers, it can have two reasons:
        // 1. Something is wrong with DepNode creation, either here or
        //    in DepGraph::try_mark_green()
        // 2. Two distinct query keys get mapped to the same DepNode
        //    (see for example #48923)
        assert!(!self.dep_graph.dep_node_exists(&dep_node),
                "Forcing query with already existing DepNode.\n\
                 - query-key: {:?}\n\
                 - dep-node: {:?}",
                key, dep_node);

        profq_msg!(self, ProfileQueriesMsg::ProviderBegin);
        self.sess.profiler(|p| {
            p.start_activity(Q::CATEGORY);
            p.record_query(Q::CATEGORY);
        });

        let res = job.start(self, |tcx| {
            if dep_node.kind.is_eval_always() {
                tcx.dep_graph.with_eval_always_task(dep_node,
                                                    tcx,
                                                    key,
                                                    Q::compute)
            } else {
                tcx.dep_graph.with_task(dep_node,
                                        tcx,
                                        key,
                                        Q::compute)
            }
        });

        self.sess.profiler(|p| p.end_activity(Q::CATEGORY));
        profq_msg!(self, ProfileQueriesMsg::ProviderEnd);

        let ((result, dep_node_index), diagnostics) = res;

        if unlikely!(self.sess.opts.debugging_opts.query_dep_graph) {
            self.dep_graph.mark_loaded_from_cache(dep_node_index, false);
        }

        if dep_node.kind != ::dep_graph::DepKind::Null {
            self.queries.on_disk_cache
                .store_diagnostics(dep_node_index, diagnostics);
        }

        job.complete(&result, dep_node_index);

        Ok((result, dep_node_index))
    }

    /// Ensure that either this query has all green inputs or been executed.
    /// Executing query::ensure(D) is considered a read of the dep-node D.
    ///
    /// This function is particularly useful when executing passes for their
    /// side-effects -- e.g., in order to report errors for erroneous programs.
    ///
    /// Note: The optimization is only available during incr. comp.
    pub(super) fn ensure_query<Q: QueryDescription<'gcx>>(self, key: Q::Key) -> () {
        let dep_node = Q::to_dep_node(self, &key);

        // Ensuring an "input" or anonymous query makes no sense
        assert!(!dep_node.kind.is_anon());
        assert!(!dep_node.kind.is_input());
        if self.try_mark_green_and_read(&dep_node).is_none() {
            // A None return from `try_mark_green_and_read` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.

            self.sess.profiler(|p| {
                p.start_activity(Q::CATEGORY);
                p.record_query(Q::CATEGORY);
            });

            let _ = self.get_query::<Q>(DUMMY_SP, key);

            self.sess.profiler(|p| p.end_activity(Q::CATEGORY));
        }
    }

    #[allow(dead_code)]
    fn force_query<Q: QueryDescription<'gcx>>(
        self,
        key: Q::Key,
        span: Span,
        dep_node: DepNode
    ) {
        profq_msg!(
            self,
            ProfileQueriesMsg::QueryBegin(span.data(), profq_query_msg!(Q::NAME, self, key))
        );

        // We may be concurrently trying both execute and force a query
        // Ensure that only one of them runs the query
        let job = match JobOwner::try_get(self, span, &key) {
            TryGetJob::NotYetStarted(job) => job,
            TryGetJob::JobCompleted(_) => return,
        };
        if let Err(e) = self.force_query_with_job::<Q>(key, job, dep_node) {
            self.report_cycle(e).emit();
        }
    }

    pub(super) fn try_get_query<Q: QueryDescription<'gcx>>(
        self,
        span: Span,
        key: Q::Key,
    ) -> Result<Q::Value, Box<DiagnosticBuilder<'a>>> {
        match self.try_get_with::<Q>(span, key) {
            Ok(e) => Ok(e),
            Err(e) => Err(self.report_cycle(e)),
        }
    }

    // FIXME: Try uninlining this
    #[inline(always)]
    pub(super) fn get_query<Q: QueryDescription<'gcx>>(
        self,
        span: Span,
        key: Q::Key,
    ) -> Q::Value {
        self.try_get_with::<Q>(span, key).unwrap_or_else(|e| {
            self.emit_error::<Q>(e)
        })
    }

    #[inline(never)]
    #[cold]
    fn emit_error<Q: QueryDescription<'gcx>>(
        self,
        e: Box<CycleError<'gcx>>,
    ) -> Q::Value {
        self.report_cycle(e).emit();
        Q::handle_cycle_error(self)
    }
}

macro_rules! handle_cycle_error {
    ([][$this: expr]) => {{
        Value::from_cycle_error($this.global_tcx())
    }};
    ([fatal_cycle$(, $modifiers:ident)*][$this:expr]) => {{
        $this.sess.abort_if_errors();
        unreachable!();
    }};
    ([$other:ident$(, $modifiers:ident)*][$($args:tt)*]) => {
        handle_cycle_error!([$($modifiers),*][$($args)*])
    };
}

macro_rules! define_queries {
    (<$tcx:tt> $($category:tt {
        $($(#[$attr:meta])* [$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $V:ty,)*
    },)*) => {
        define_queries_inner! { <$tcx>
            $($( $(#[$attr])* category<$category> [$($modifiers)*] fn $name: $node($K) -> $V,)*)*
        }
    }
}

macro_rules! define_queries_inner {
    (<$tcx:tt>
     $($(#[$attr:meta])* category<$category:tt>
        [$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $V:ty,)*) => {

        use std::mem;
        #[cfg(parallel_queries)]
        use ty::query::job::QueryResult;
        use rustc_data_structures::sync::Lock;
        use {
            rustc_data_structures::stable_hasher::HashStable,
            rustc_data_structures::stable_hasher::StableHasherResult,
            rustc_data_structures::stable_hasher::StableHasher,
            ich::StableHashingContext
        };
        use util::profiling::ProfileCategory;

        define_queries_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$($attr)*] [$name]))*)
        }

        impl<$tcx> Queries<$tcx> {
            pub fn new(
                providers: IndexVec<CrateNum, Providers<$tcx>>,
                fallback_extern_providers: Providers<$tcx>,
                on_disk_cache: OnDiskCache<'tcx>,
            ) -> Self {
                Queries {
                    providers,
                    fallback_extern_providers: Box::new(fallback_extern_providers),
                    on_disk_cache,
                    $($name: Default::default()),*
                }
            }

            #[cfg(parallel_queries)]
            pub fn collect_active_jobs(&self) -> Vec<Lrc<QueryJob<$tcx>>> {
                let mut jobs = Vec::new();

                // We use try_lock here since we are only called from the
                // deadlock handler, and this shouldn't be locked
                $(
                    jobs.extend(
                        self.$name.try_lock().unwrap().active.values().filter_map(|v|
                            if let QueryResult::Started(ref job) = *v {
                                Some(job.clone())
                            } else {
                                None
                            }
                        )
                    );
                )*

                jobs
            }

            pub fn print_stats(&self) {
                let mut queries = Vec::new();

                #[derive(Clone)]
                struct QueryStats {
                    name: &'static str,
                    cache_hits: usize,
                    key_size: usize,
                    key_type: &'static str,
                    value_size: usize,
                    value_type: &'static str,
                    entry_count: usize,
                }

                fn stats<'tcx, Q: QueryConfig<'tcx>>(
                    name: &'static str,
                    map: &QueryCache<'tcx, Q>
                ) -> QueryStats {
                    QueryStats {
                        name,
                        #[cfg(debug_assertions)]
                        cache_hits: map.cache_hits,
                        #[cfg(not(debug_assertions))]
                        cache_hits: 0,
                        key_size: mem::size_of::<Q::Key>(),
                        key_type: unsafe { type_name::<Q::Key>() },
                        value_size: mem::size_of::<Q::Value>(),
                        value_type: unsafe { type_name::<Q::Value>() },
                        entry_count: map.results.len(),
                    }
                }

                $(
                    queries.push(stats::<queries::$name<'_>>(
                        stringify!($name),
                        &*self.$name.lock()
                    ));
                )*

                if cfg!(debug_assertions) {
                    let hits: usize = queries.iter().map(|s| s.cache_hits).sum();
                    let results: usize = queries.iter().map(|s| s.entry_count).sum();
                    println!("\nQuery cache hit rate: {}", hits as f64 / (hits + results) as f64);
                }

                let mut query_key_sizes = queries.clone();
                query_key_sizes.sort_by_key(|q| q.key_size);
                println!("\nLarge query keys:");
                for q in query_key_sizes.iter().rev()
                                        .filter(|q| q.key_size > 8) {
                    println!(
                        "   {} - {} x {} - {}",
                        q.name,
                        q.key_size,
                        q.entry_count,
                        q.key_type
                    );
                }

                let mut query_value_sizes = queries.clone();
                query_value_sizes.sort_by_key(|q| q.value_size);
                println!("\nLarge query values:");
                for q in query_value_sizes.iter().rev()
                                          .filter(|q| q.value_size > 8) {
                    println!(
                        "   {} - {} x {} - {}",
                        q.name,
                        q.value_size,
                        q.entry_count,
                        q.value_type
                    );
                }

                if cfg!(debug_assertions) {
                    let mut query_cache_hits = queries.clone();
                    query_cache_hits.sort_by_key(|q| q.cache_hits);
                    println!("\nQuery cache hits:");
                    for q in query_cache_hits.iter().rev() {
                        println!(
                            "   {} - {} ({}%)",
                            q.name,
                            q.cache_hits,
                            q.cache_hits as f64 / (q.cache_hits + q.entry_count) as f64
                        );
                    }
                }

                let mut query_value_count = queries.clone();
                query_value_count.sort_by_key(|q| q.entry_count);
                println!("\nQuery value count:");
                for q in query_value_count.iter().rev() {
                    println!("   {} - {}", q.name, q.entry_count);
                }
            }
        }

        #[allow(nonstandard_style)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum Query<$tcx> {
            $($(#[$attr])* $name($K)),*
        }

        impl<$tcx> Query<$tcx> {
            pub fn name(&self) -> &'static str {
                match *self {
                    $(Query::$name(_) => stringify!($name),)*
                }
            }

            pub fn describe(&self, tcx: TyCtxt<'_, '_, '_>) -> Cow<'static, str> {
                let (r, name) = match *self {
                    $(Query::$name(key) => {
                        (queries::$name::describe(tcx, key), stringify!($name))
                    })*
                };
                if tcx.sess.verbose() {
                    format!("{} [{}]", r, name).into()
                } else {
                    r
                }
            }

            // FIXME(eddyb) Get more valid Span's on queries.
            pub fn default_span(&self, tcx: TyCtxt<'_, $tcx, '_>, span: Span) -> Span {
                if !span.is_dummy() {
                    return span;
                }
                // The def_span query is used to calculate default_span,
                // so exit to avoid infinite recursion
                if let Query::def_span(..) = *self {
                    return span
                }
                match *self {
                    $(Query::$name(key) => key.default_span(tcx),)*
                }
            }
        }

        impl<'a, $tcx> HashStable<StableHashingContext<'a>> for Query<$tcx> {
            fn hash_stable<W: StableHasherResult>(&self,
                                                hcx: &mut StableHashingContext<'a>,
                                                hasher: &mut StableHasher<W>) {
                mem::discriminant(self).hash_stable(hcx, hasher);
                match *self {
                    $(Query::$name(key) => key.hash_stable(hcx, hasher),)*
                }
            }
        }

        pub mod queries {
            use std::marker::PhantomData;

            $(#[allow(nonstandard_style)]
            pub struct $name<$tcx> {
                data: PhantomData<&$tcx ()>
            })*
        }

        // This module and the functions in it exist only to provide a
        // predictable symbol name prefix for query providers. This is helpful
        // for analyzing queries in profilers.
        pub(super) mod __query_compute {
            $(#[inline(never)]
            pub fn $name<F: FnOnce() -> R, R>(f: F) -> R {
                f()
            })*
        }

        $(impl<$tcx> QueryConfig<$tcx> for queries::$name<$tcx> {
            type Key = $K;
            type Value = $V;

            const NAME: &'static str = stringify!($name);
            const CATEGORY: ProfileCategory = $category;
        }

        impl<$tcx> QueryAccessors<$tcx> for queries::$name<$tcx> {
            #[inline(always)]
            fn query(key: Self::Key) -> Query<'tcx> {
                Query::$name(key)
            }

            #[inline(always)]
            fn query_cache<'a>(tcx: TyCtxt<'a, $tcx, '_>) -> &'a Lock<QueryCache<$tcx, Self>> {
                &tcx.queries.$name
            }

            #[allow(unused)]
            #[inline(always)]
            fn to_dep_node(tcx: TyCtxt<'_, $tcx, '_>, key: &Self::Key) -> DepNode {
                use dep_graph::DepConstructor::*;

                DepNode::new(tcx, $node(*key))
            }

            #[inline]
            fn compute(tcx: TyCtxt<'_, 'tcx, '_>, key: Self::Key) -> Self::Value {
                __query_compute::$name(move || {
                    let provider = tcx.queries.providers.get(key.query_crate())
                        // HACK(eddyb) it's possible crates may be loaded after
                        // the query engine is created, and because crate loading
                        // is not yet integrated with the query engine, such crates
                        // would be be missing appropriate entries in `providers`.
                        .unwrap_or(&tcx.queries.fallback_extern_providers)
                        .$name;
                    provider(tcx.global_tcx(), key)
                })
            }

            fn handle_cycle_error(tcx: TyCtxt<'_, 'tcx, '_>) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx])
            }
        }

        impl<'a, $tcx, 'lcx> queries::$name<$tcx> {
            /// Ensure that either this query has all green inputs or been executed.
            /// Executing query::ensure(D) is considered a read of the dep-node D.
            ///
            /// This function is particularly useful when executing passes for their
            /// side-effects -- e.g., in order to report errors for erroneous programs.
            ///
            /// Note: The optimization is only available during incr. comp.
            pub fn ensure(tcx: TyCtxt<'a, $tcx, 'lcx>, key: $K) -> () {
                tcx.ensure_query::<queries::$name<'_>>(key);
            }
        })*

        #[derive(Copy, Clone)]
        pub struct TyCtxtAt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
            pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
            pub span: Span,
        }

        impl<'a, 'gcx, 'tcx> Deref for TyCtxtAt<'a, 'gcx, 'tcx> {
            type Target = TyCtxt<'a, 'gcx, 'tcx>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.tcx
            }
        }

        impl<'a, $tcx, 'lcx> TyCtxt<'a, $tcx, 'lcx> {
            /// Return a transparent wrapper for `TyCtxt` which uses
            /// `span` as the location of queries performed through it.
            #[inline(always)]
            pub fn at(self, span: Span) -> TyCtxtAt<'a, $tcx, 'lcx> {
                TyCtxtAt {
                    tcx: self,
                    span
                }
            }

            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: $K) -> $V {
                self.at(DUMMY_SP).$name(key)
            })*
        }

        impl<'a, $tcx, 'lcx> TyCtxtAt<'a, $tcx, 'lcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: $K) -> $V {
                self.tcx.get_query::<queries::$name<'_>>(self.span, key)
            })*
        }

        define_provider_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$name] [$K] [$V]))*)
        }

        impl<$tcx> Copy for Providers<$tcx> {}
        impl<$tcx> Clone for Providers<$tcx> {
            fn clone(&self) -> Self { *self }
        }
    }
}

macro_rules! define_queries_struct {
    (tcx: $tcx:tt,
     input: ($(([$($modifiers:tt)*] [$($attr:tt)*] [$name:ident]))*)) => {
        pub struct Queries<$tcx> {
            /// This provides access to the incr. comp. on-disk cache for query results.
            /// Do not access this directly. It is only meant to be used by
            /// `DepGraph::try_mark_green()` and the query infrastructure.
            pub(crate) on_disk_cache: OnDiskCache<'tcx>,

            providers: IndexVec<CrateNum, Providers<$tcx>>,
            fallback_extern_providers: Box<Providers<$tcx>>,

            $($(#[$attr])*  $name: Lock<QueryCache<$tcx, queries::$name<$tcx>>>,)*
        }
    };
}

macro_rules! define_provider_struct {
    (tcx: $tcx:tt,
     input: ($(([$($modifiers:tt)*] [$name:ident] [$K:ty] [$R:ty]))*)) => {
        pub struct Providers<$tcx> {
            $(pub $name: for<'a> fn(TyCtxt<'a, $tcx, $tcx>, $K) -> $R,)*
        }

        impl<$tcx> Default for Providers<$tcx> {
            fn default() -> Self {
                $(fn $name<'a, $tcx>(_: TyCtxt<'a, $tcx, $tcx>, key: $K) -> $R {
                    bug!("tcx.{}({:?}) unsupported by its crate",
                         stringify!($name), key);
                })*
                Providers { $($name),* }
            }
        }
    };
}


/// The red/green evaluation system will try to mark a specific DepNode in the
/// dependency graph as green by recursively trying to mark the dependencies of
/// that DepNode as green. While doing so, it will sometimes encounter a DepNode
/// where we don't know if it is red or green and we therefore actually have
/// to recompute its value in order to find out. Since the only piece of
/// information that we have at that point is the DepNode we are trying to
/// re-evaluate, we need some way to re-run a query from just that. This is what
/// `force_from_dep_node()` implements.
///
/// In the general case, a DepNode consists of a DepKind and an opaque
/// GUID/fingerprint that will uniquely identify the node. This GUID/fingerprint
/// is usually constructed by computing a stable hash of the query-key that the
/// DepNode corresponds to. Consequently, it is not in general possible to go
/// back from hash to query-key (since hash functions are not reversible). For
/// this reason `force_from_dep_node()` is expected to fail from time to time
/// because we just cannot find out, from the DepNode alone, what the
/// corresponding query-key is and therefore cannot re-run the query.
///
/// The system deals with this case letting `try_mark_green` fail which forces
/// the root query to be re-evaluated.
///
/// Now, if force_from_dep_node() would always fail, it would be pretty useless.
/// Fortunately, we can use some contextual information that will allow us to
/// reconstruct query-keys for certain kinds of DepNodes. In particular, we
/// enforce by construction that the GUID/fingerprint of certain DepNodes is a
/// valid DefPathHash. Since we also always build a huge table that maps every
/// DefPathHash in the current codebase to the corresponding DefId, we have
/// everything we need to re-run the query.
///
/// Take the `mir_validated` query as an example. Like many other queries, it
/// just has a single parameter: the DefId of the item it will compute the
/// validated MIR for. Now, when we call `force_from_dep_node()` on a dep-node
/// with kind `MirValidated`, we know that the GUID/fingerprint of the dep-node
/// is actually a DefPathHash, and can therefore just look up the corresponding
/// DefId in `tcx.def_path_hash_to_def_id`.
///
/// When you implement a new query, it will likely have a corresponding new
/// DepKind, and you'll have to support it here in `force_from_dep_node()`. As
/// a rule of thumb, if your query takes a DefId or DefIndex as sole parameter,
/// then `force_from_dep_node()` should not fail for it. Otherwise, you can just
/// add it to the "We don't have enough information to reconstruct..." group in
/// the match below.
pub fn force_from_dep_node<'a, 'gcx, 'lcx>(tcx: TyCtxt<'a, 'gcx, 'lcx>,
                                           dep_node: &DepNode)
                                           -> bool {
    use hir::def_id::LOCAL_CRATE;

    // We must avoid ever having to call force_from_dep_node() for a
    // DepNode::CodegenUnit:
    // Since we cannot reconstruct the query key of a DepNode::CodegenUnit, we
    // would always end up having to evaluate the first caller of the
    // `codegen_unit` query that *is* reconstructible. This might very well be
    // the `compile_codegen_unit` query, thus re-codegenning the whole CGU just
    // to re-trigger calling the `codegen_unit` query with the right key. At
    // that point we would already have re-done all the work we are trying to
    // avoid doing in the first place.
    // The solution is simple: Just explicitly call the `codegen_unit` query for
    // each CGU, right after partitioning. This way `try_mark_green` will always
    // hit the cache instead of having to go through `force_from_dep_node`.
    // This assertion makes sure, we actually keep applying the solution above.
    debug_assert!(dep_node.kind != DepKind::CodegenUnit,
                  "calling force_from_dep_node() on DepKind::CodegenUnit");

    if !dep_node.kind.can_reconstruct_query_key() {
        return false
    }

    macro_rules! def_id {
        () => {
            if let Some(def_id) = dep_node.extract_def_id(tcx) {
                def_id
            } else {
                // return from the whole function
                return false
            }
        }
    };

    macro_rules! krate {
        () => { (def_id!()).krate }
    };

    macro_rules! force {
        ($query:ident, $key:expr) => {
            {
                tcx.force_query::<::ty::query::queries::$query<'_>>($key, DUMMY_SP, *dep_node);
            }
        }
    };

    // FIXME(#45015): We should try move this boilerplate code into a macro
    //                somehow.
    match dep_node.kind {
        // These are inputs that are expected to be pre-allocated and that
        // should therefore always be red or green already
        DepKind::AllLocalTraitImpls |
        DepKind::Krate |
        DepKind::CrateMetadata |
        DepKind::HirBody |
        DepKind::Hir |

        // This are anonymous nodes
        DepKind::TraitSelect |

        // We don't have enough information to reconstruct the query key of
        // these
        DepKind::IsCopy |
        DepKind::IsSized |
        DepKind::IsFreeze |
        DepKind::NeedsDrop |
        DepKind::Layout |
        DepKind::ConstEval |
        DepKind::ConstEvalRaw |
        DepKind::InstanceSymbolName |
        DepKind::MirShim |
        DepKind::BorrowCheckKrate |
        DepKind::Specializes |
        DepKind::ImplementationsOfTrait |
        DepKind::TypeParamPredicates |
        DepKind::CodegenUnit |
        DepKind::CompileCodegenUnit |
        DepKind::FulfillObligation |
        DepKind::VtableMethods |
        DepKind::EraseRegionsTy |
        DepKind::NormalizeProjectionTy |
        DepKind::NormalizeTyAfterErasingRegions |
        DepKind::ImpliedOutlivesBounds |
        DepKind::DropckOutlives |
        DepKind::EvaluateObligation |
        DepKind::EvaluateGoal |
        DepKind::TypeOpAscribeUserType |
        DepKind::TypeOpEq |
        DepKind::TypeOpSubtype |
        DepKind::TypeOpProvePredicate |
        DepKind::TypeOpNormalizeTy |
        DepKind::TypeOpNormalizePredicate |
        DepKind::TypeOpNormalizePolyFnSig |
        DepKind::TypeOpNormalizeFnSig |
        DepKind::SubstituteNormalizeAndTestPredicates |
        DepKind::MethodAutoderefSteps |
        DepKind::InstanceDefSizeEstimate |
        DepKind::ProgramClausesForEnv |

        // This one should never occur in this context
        DepKind::Null => {
            bug!("force_from_dep_node() - Encountered {:?}", dep_node)
        }

        // These are not queries
        DepKind::CoherenceCheckTrait |
        DepKind::ItemVarianceConstraints => {
            return false
        }

        DepKind::RegionScopeTree => { force!(region_scope_tree, def_id!()); }

        DepKind::Coherence => { force!(crate_inherent_impls, LOCAL_CRATE); }
        DepKind::CoherenceInherentImplOverlapCheck => {
            force!(crate_inherent_impls_overlap_check, LOCAL_CRATE)
        },
        DepKind::PrivacyAccessLevels => { force!(privacy_access_levels, LOCAL_CRATE); }
        DepKind::MirBuilt => { force!(mir_built, def_id!()); }
        DepKind::MirConstQualif => { force!(mir_const_qualif, def_id!()); }
        DepKind::MirConst => { force!(mir_const, def_id!()); }
        DepKind::MirValidated => { force!(mir_validated, def_id!()); }
        DepKind::MirOptimized => { force!(optimized_mir, def_id!()); }

        DepKind::BorrowCheck => { force!(borrowck, def_id!()); }
        DepKind::MirBorrowCheck => { force!(mir_borrowck, def_id!()); }
        DepKind::UnsafetyCheckResult => { force!(unsafety_check_result, def_id!()); }
        DepKind::UnsafeDeriveOnReprPacked => { force!(unsafe_derive_on_repr_packed, def_id!()); }
        DepKind::Reachability => { force!(reachable_set, LOCAL_CRATE); }
        DepKind::MirKeys => { force!(mir_keys, LOCAL_CRATE); }
        DepKind::CrateVariances => { force!(crate_variances, LOCAL_CRATE); }
        DepKind::AssociatedItems => { force!(associated_item, def_id!()); }
        DepKind::TypeOfItem => { force!(type_of, def_id!()); }
        DepKind::GenericsOfItem => { force!(generics_of, def_id!()); }
        DepKind::PredicatesOfItem => { force!(predicates_of, def_id!()); }
        DepKind::PredicatesDefinedOnItem => { force!(predicates_defined_on, def_id!()); }
        DepKind::ExplicitPredicatesOfItem => { force!(explicit_predicates_of, def_id!()); }
        DepKind::InferredOutlivesOf => { force!(inferred_outlives_of, def_id!()); }
        DepKind::InferredOutlivesCrate => { force!(inferred_outlives_crate, LOCAL_CRATE); }
        DepKind::SuperPredicatesOfItem => { force!(super_predicates_of, def_id!()); }
        DepKind::TraitDefOfItem => { force!(trait_def, def_id!()); }
        DepKind::AdtDefOfItem => { force!(adt_def, def_id!()); }
        DepKind::ImplTraitRef => { force!(impl_trait_ref, def_id!()); }
        DepKind::ImplPolarity => { force!(impl_polarity, def_id!()); }
        DepKind::Issue33140SelfTy => { force!(issue33140_self_ty, def_id!()); }
        DepKind::FnSignature => { force!(fn_sig, def_id!()); }
        DepKind::CoerceUnsizedInfo => { force!(coerce_unsized_info, def_id!()); }
        DepKind::ItemVariances => { force!(variances_of, def_id!()); }
        DepKind::IsConstFn => { force!(is_const_fn_raw, def_id!()); }
        DepKind::IsPromotableConstFn => { force!(is_promotable_const_fn, def_id!()); }
        DepKind::IsForeignItem => { force!(is_foreign_item, def_id!()); }
        DepKind::SizedConstraint => { force!(adt_sized_constraint, def_id!()); }
        DepKind::DtorckConstraint => { force!(adt_dtorck_constraint, def_id!()); }
        DepKind::AdtDestructor => { force!(adt_destructor, def_id!()); }
        DepKind::AssociatedItemDefIds => { force!(associated_item_def_ids, def_id!()); }
        DepKind::InherentImpls => { force!(inherent_impls, def_id!()); }
        DepKind::TypeckBodiesKrate => { force!(typeck_item_bodies, LOCAL_CRATE); }
        DepKind::TypeckTables => { force!(typeck_tables_of, def_id!()); }
        DepKind::UsedTraitImports => { force!(used_trait_imports, def_id!()); }
        DepKind::HasTypeckTables => { force!(has_typeck_tables, def_id!()); }
        DepKind::SymbolName => { force!(def_symbol_name, def_id!()); }
        DepKind::SpecializationGraph => { force!(specialization_graph_of, def_id!()); }
        DepKind::ObjectSafety => { force!(is_object_safe, def_id!()); }
        DepKind::TraitImpls => { force!(trait_impls_of, def_id!()); }
        DepKind::CheckMatch => { force!(check_match, def_id!()); }

        DepKind::ParamEnv => { force!(param_env, def_id!()); }
        DepKind::Environment => { force!(environment, def_id!()); }
        DepKind::DescribeDef => { force!(describe_def, def_id!()); }
        DepKind::DefSpan => { force!(def_span, def_id!()); }
        DepKind::LookupStability => { force!(lookup_stability, def_id!()); }
        DepKind::LookupDeprecationEntry => {
            force!(lookup_deprecation_entry, def_id!());
        }
        DepKind::ConstIsRvaluePromotableToStatic => {
            force!(const_is_rvalue_promotable_to_static, def_id!());
        }
        DepKind::RvaluePromotableMap => { force!(rvalue_promotable_map, def_id!()); }
        DepKind::ImplParent => { force!(impl_parent, def_id!()); }
        DepKind::TraitOfItem => { force!(trait_of_item, def_id!()); }
        DepKind::IsReachableNonGeneric => { force!(is_reachable_non_generic, def_id!()); }
        DepKind::IsUnreachableLocalDefinition => {
            force!(is_unreachable_local_definition, def_id!());
        }
        DepKind::IsMirAvailable => { force!(is_mir_available, def_id!()); }
        DepKind::ItemAttrs => { force!(item_attrs, def_id!()); }
        DepKind::CodegenFnAttrs => { force!(codegen_fn_attrs, def_id!()); }
        DepKind::FnArgNames => { force!(fn_arg_names, def_id!()); }
        DepKind::RenderedConst => { force!(rendered_const, def_id!()); }
        DepKind::DylibDepFormats => { force!(dylib_dependency_formats, krate!()); }
        DepKind::IsPanicRuntime => { force!(is_panic_runtime, krate!()); }
        DepKind::IsCompilerBuiltins => { force!(is_compiler_builtins, krate!()); }
        DepKind::HasGlobalAllocator => { force!(has_global_allocator, krate!()); }
        DepKind::HasPanicHandler => { force!(has_panic_handler, krate!()); }
        DepKind::ExternCrate => { force!(extern_crate, def_id!()); }
        DepKind::LintLevels => { force!(lint_levels, LOCAL_CRATE); }
        DepKind::InScopeTraits => { force!(in_scope_traits_map, def_id!().index); }
        DepKind::ModuleExports => { force!(module_exports, def_id!()); }
        DepKind::IsSanitizerRuntime => { force!(is_sanitizer_runtime, krate!()); }
        DepKind::IsProfilerRuntime => { force!(is_profiler_runtime, krate!()); }
        DepKind::GetPanicStrategy => { force!(panic_strategy, krate!()); }
        DepKind::IsNoBuiltins => { force!(is_no_builtins, krate!()); }
        DepKind::ImplDefaultness => { force!(impl_defaultness, def_id!()); }
        DepKind::CheckItemWellFormed => { force!(check_item_well_formed, def_id!()); }
        DepKind::CheckTraitItemWellFormed => { force!(check_trait_item_well_formed, def_id!()); }
        DepKind::CheckImplItemWellFormed => { force!(check_impl_item_well_formed, def_id!()); }
        DepKind::ReachableNonGenerics => { force!(reachable_non_generics, krate!()); }
        DepKind::NativeLibraries => { force!(native_libraries, krate!()); }
        DepKind::PluginRegistrarFn => { force!(plugin_registrar_fn, krate!()); }
        DepKind::ProcMacroDeclsStatic => { force!(proc_macro_decls_static, krate!()); }
        DepKind::CrateDisambiguator => { force!(crate_disambiguator, krate!()); }
        DepKind::CrateHash => { force!(crate_hash, krate!()); }
        DepKind::OriginalCrateName => { force!(original_crate_name, krate!()); }
        DepKind::ExtraFileName => { force!(extra_filename, krate!()); }

        DepKind::AllTraitImplementations => {
            force!(all_trait_implementations, krate!());
        }

        DepKind::DllimportForeignItems => {
            force!(dllimport_foreign_items, krate!());
        }
        DepKind::IsDllimportForeignItem => {
            force!(is_dllimport_foreign_item, def_id!());
        }
        DepKind::IsStaticallyIncludedForeignItem => {
            force!(is_statically_included_foreign_item, def_id!());
        }
        DepKind::NativeLibraryKind => { force!(native_library_kind, def_id!()); }
        DepKind::LinkArgs => { force!(link_args, LOCAL_CRATE); }

        DepKind::ResolveLifetimes => { force!(resolve_lifetimes, krate!()); }
        DepKind::NamedRegion => { force!(named_region_map, def_id!().index); }
        DepKind::IsLateBound => { force!(is_late_bound_map, def_id!().index); }
        DepKind::ObjectLifetimeDefaults => {
            force!(object_lifetime_defaults_map, def_id!().index);
        }

        DepKind::Visibility => { force!(visibility, def_id!()); }
        DepKind::DepKind => { force!(dep_kind, krate!()); }
        DepKind::CrateName => { force!(crate_name, krate!()); }
        DepKind::ItemChildren => { force!(item_children, def_id!()); }
        DepKind::ExternModStmtCnum => { force!(extern_mod_stmt_cnum, def_id!()); }
        DepKind::GetLibFeatures => { force!(get_lib_features, LOCAL_CRATE); }
        DepKind::DefinedLibFeatures => { force!(defined_lib_features, krate!()); }
        DepKind::GetLangItems => { force!(get_lang_items, LOCAL_CRATE); }
        DepKind::DefinedLangItems => { force!(defined_lang_items, krate!()); }
        DepKind::MissingLangItems => { force!(missing_lang_items, krate!()); }
        DepKind::VisibleParentMap => { force!(visible_parent_map, LOCAL_CRATE); }
        DepKind::MissingExternCrateItem => {
            force!(missing_extern_crate_item, krate!());
        }
        DepKind::UsedCrateSource => { force!(used_crate_source, krate!()); }
        DepKind::PostorderCnums => { force!(postorder_cnums, LOCAL_CRATE); }

        DepKind::Freevars => { force!(freevars, def_id!()); }
        DepKind::MaybeUnusedTraitImport => {
            force!(maybe_unused_trait_import, def_id!());
        }
        DepKind::MaybeUnusedExternCrates => { force!(maybe_unused_extern_crates, LOCAL_CRATE); }
        DepKind::StabilityIndex => { force!(stability_index, LOCAL_CRATE); }
        DepKind::AllTraits => { force!(all_traits, LOCAL_CRATE); }
        DepKind::AllCrateNums => { force!(all_crate_nums, LOCAL_CRATE); }
        DepKind::ExportedSymbols => { force!(exported_symbols, krate!()); }
        DepKind::CollectAndPartitionMonoItems => {
            force!(collect_and_partition_mono_items, LOCAL_CRATE);
        }
        DepKind::IsCodegenedItem => { force!(is_codegened_item, def_id!()); }
        DepKind::OutputFilenames => { force!(output_filenames, LOCAL_CRATE); }

        DepKind::TargetFeaturesWhitelist => { force!(target_features_whitelist, LOCAL_CRATE); }

        DepKind::Features => { force!(features_query, LOCAL_CRATE); }

        DepKind::ProgramClausesFor => { force!(program_clauses_for, def_id!()); }
        DepKind::WasmImportModuleMap => { force!(wasm_import_module_map, krate!()); }
        DepKind::ForeignModules => { force!(foreign_modules, krate!()); }

        DepKind::UpstreamMonomorphizations => {
            force!(upstream_monomorphizations, krate!());
        }
        DepKind::UpstreamMonomorphizationsFor => {
            force!(upstream_monomorphizations_for, def_id!());
        }
    }

    true
}


// FIXME(#45015): Another piece of boilerplate code that could be generated in
//                a combined define_dep_nodes!()/define_queries!() macro.
macro_rules! impl_load_from_cache {
    ($($dep_kind:ident => $query_name:ident,)*) => {
        impl DepNode {
            // Check whether the query invocation corresponding to the given
            // DepNode is eligible for on-disk-caching.
            pub fn cache_on_disk(&self, tcx: TyCtxt<'_, '_, '_>) -> bool {
                use ty::query::queries;
                use ty::query::QueryDescription;

                match self.kind {
                    $(DepKind::$dep_kind => {
                        let def_id = self.extract_def_id(tcx).unwrap();
                        queries::$query_name::cache_on_disk(def_id)
                    })*
                    _ => false
                }
            }

            // This is method will execute the query corresponding to the given
            // DepNode. It is only expected to work for DepNodes where the
            // above `cache_on_disk` methods returns true.
            // Also, as a sanity check, it expects that the corresponding query
            // invocation has been marked as green already.
            pub fn load_from_on_disk_cache(&self, tcx: TyCtxt<'_, '_, '_>) {
                match self.kind {
                    $(DepKind::$dep_kind => {
                        debug_assert!(tcx.dep_graph
                                         .node_color(self)
                                         .map(|c| c.is_green())
                                         .unwrap_or(false));

                        let def_id = self.extract_def_id(tcx).unwrap();
                        let _ = tcx.$query_name(def_id);
                    })*
                    _ => {
                        bug!()
                    }
                }
            }
        }
    }
}

impl_load_from_cache!(
    TypeckTables => typeck_tables_of,
    MirOptimized => optimized_mir,
    UnsafetyCheckResult => unsafety_check_result,
    BorrowCheck => borrowck,
    MirBorrowCheck => mir_borrowck,
    MirConstQualif => mir_const_qualif,
    SymbolName => def_symbol_name,
    ConstIsRvaluePromotableToStatic => const_is_rvalue_promotable_to_static,
    CheckMatch => check_match,
    TypeOfItem => type_of,
    GenericsOfItem => generics_of,
    PredicatesOfItem => predicates_of,
    UsedTraitImports => used_trait_imports,
    CodegenFnAttrs => codegen_fn_attrs,
    SpecializationGraph => specialization_graph_of,
);
