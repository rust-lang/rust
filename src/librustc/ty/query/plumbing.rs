//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::{DepNodeIndex, DepNode, DepKind, SerializedDepNodeIndex};
use crate::ty::tls;
use crate::ty::{self, TyCtxt};
use crate::ty::query::Query;
use crate::ty::query::config::{QueryConfig, QueryDescription};
use crate::ty::query::job::{QueryJob, QueryResult, QueryInfo};

use crate::util::common::{profq_msg, ProfileQueriesMsg, QueryMsg};

use errors::DiagnosticBuilder;
use errors::Level;
use errors::Diagnostic;
use errors::FatalError;
use rustc_data_structures::fx::{FxHashMap};
use rustc_data_structures::sync::{Lrc, Lock};
use rustc_data_structures::thin_vec::ThinVec;
#[cfg(not(parallel_compiler))]
use rustc_data_structures::cold_path;
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
pub(super) struct JobOwner<'a, 'tcx, Q: QueryDescription<'tcx>> {
    cache: &'a Lock<QueryCache<'tcx, Q>>,
    key: Q::Key,
    job: Lrc<QueryJob<'tcx>>,
}

impl<'a, 'tcx, Q: QueryDescription<'tcx>> JobOwner<'a, 'tcx, Q> {
    /// Either gets a `JobOwner` corresponding the query, allowing us to
    /// start executing the query, or it returns with the result of the query.
    /// If the query is executing elsewhere, this will wait for it.
    /// If the query panicked, this will silently panic.
    ///
    /// This function is inlined because that results in a noticeable speed-up
    /// for some compile-time benchmarks.
    #[inline(always)]
    pub(super) fn try_get(tcx: TyCtxt<'tcx>, span: Span, key: &Q::Key) -> TryGetJob<'a, 'tcx, Q> {
        let cache = Q::query_cache(tcx);
        loop {
            let mut lock = cache.borrow_mut();
            if let Some(value) = lock.results.get(key) {
                profq_msg!(tcx, ProfileQueriesMsg::CacheHit);
                tcx.sess.profiler(|p| p.record_query_hit(Q::NAME));
                let result = (value.value.clone(), value.index);
                #[cfg(debug_assertions)]
                {
                    lock.cache_hits += 1;
                }
                return TryGetJob::JobCompleted(result);
            }
            let job = match lock.active.entry((*key).clone()) {
                Entry::Occupied(entry) => {
                    match *entry.get() {
                        QueryResult::Started(ref job) => {
                            // For parallel queries, we'll block and wait until the query running
                            // in another thread has completed. Record how long we wait in the
                            // self-profiler.
                            #[cfg(parallel_compiler)]
                            tcx.sess.profiler(|p| p.query_blocked_start(Q::NAME));

                            job.clone()
                        },
                        QueryResult::Poisoned => FatalError.raise(),
                    }
                }
                Entry::Vacant(entry) => {
                    // No job entry for this query. Return a new one to be started later.
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
            // so we just return the error.
            #[cfg(not(parallel_compiler))]
            return TryGetJob::Cycle(cold_path(|| {
                Q::handle_cycle_error(tcx, job.find_cycle_in_stack(tcx, span))
            }));

            // With parallel queries we might just have to wait on some other
            // thread.
            #[cfg(parallel_compiler)]
            {
                let result = job.r#await(tcx, span);
                tcx.sess.profiler(|p| p.query_blocked_end(Q::NAME));

                if let Err(cycle) = result {
                    return TryGetJob::Cycle(Q::handle_cycle_error(tcx, cycle));
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
}

#[inline(always)]
fn with_diagnostics<F, R>(f: F) -> (R, ThinVec<Diagnostic>)
where
    F: FnOnce(Option<&Lock<ThinVec<Diagnostic>>>) -> R
{
    let diagnostics = Lock::new(ThinVec::new());
    let result = f(Some(&diagnostics));
    (result, diagnostics.into_inner())
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
pub(super) enum TryGetJob<'a, 'tcx, D: QueryDescription<'tcx>> {
    /// The query is not yet started. Contains a guard to the cache eventually used to start it.
    NotYetStarted(JobOwner<'a, 'tcx, D>),

    /// The query was already completed.
    /// Returns the result of the query and its dep node index
    /// if it succeeded or a cycle error if it failed
    JobCompleted((D::Value, DepNodeIndex)),

    /// Trying to execute the query resulted in a cycle.
    Cycle(D::Value),
}

impl<'tcx> TyCtxt<'tcx> {
    /// Executes a job by changing the ImplicitCtxt to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    #[inline(always)]
    pub(super) fn start_query<F, R>(
        self,
        job: Lrc<QueryJob<'tcx>>,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: F,
    ) -> R
    where
        F: FnOnce(TyCtxt<'tcx>) -> R,
    {
        // The TyCtxt stored in TLS has the same global interner lifetime
        // as `self`, so we use `with_related_context` to relate the 'tcx lifetimes
        // when accessing the ImplicitCtxt
        tls::with_related_context(self, move |current_icx| {
            // Update the ImplicitCtxt to point to our new query job
            let new_icx = tls::ImplicitCtxt {
                tcx: self.global_tcx(),
                query: Some(job),
                diagnostics,
                layout_depth: current_icx.layout_depth,
                task_deps: current_icx.task_deps,
            };

            // Use the ImplicitCtxt while we execute the query
            tls::enter_context(&new_icx, |_| {
                compute(self.global_tcx())
            })
        })
    }

    #[inline(never)]
    #[cold]
    pub(super) fn report_cycle(
        self,
        CycleError { usage, cycle: stack }: CycleError<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        assert!(!stack.is_empty());

        let fix_span = |span: Span, query: &Query<'tcx>| {
            self.sess.source_map().def_span(query.default_span(self, span))
        };

        // Disable naming impls with types in this path, since that
        // sometimes cycles itself, leading to extra cycle errors.
        // (And cycle errors around impls tend to occur during the
        // collect/coherence phases anyhow.)
        ty::print::with_forced_impl_filename_line(|| {
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

            err
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

    #[inline(never)]
    pub(super) fn get_query<Q: QueryDescription<'tcx>>(self, span: Span, key: Q::Key) -> Q::Value {
        debug!("ty::query::get_query<{}>(key={:?}, span={:?})",
               Q::NAME.as_str(),
               key,
               span);

        profq_msg!(self,
            ProfileQueriesMsg::QueryBegin(
                span.data(),
                profq_query_msg!(Q::NAME.as_str(), self, key),
            )
        );

        let job = match JobOwner::try_get(self, span, &key) {
            TryGetJob::NotYetStarted(job) => job,
            TryGetJob::Cycle(result) => return result,
            TryGetJob::JobCompleted((v, index)) => {
                self.dep_graph.read_index(index);
                return v
            }
        };

        // Fast path for when incr. comp. is off. `to_dep_node` is
        // expensive for some DepKinds.
        if !self.dep_graph.is_fully_enabled() {
            let null_dep_node = DepNode::new_no_params(crate::dep_graph::DepKind::Null);
            return self.force_query_with_job::<Q>(key, job, null_dep_node).0;
        }

        if Q::ANON {
            profq_msg!(self, ProfileQueriesMsg::ProviderBegin);
            self.sess.profiler(|p| p.start_query(Q::NAME));

            let ((result, dep_node_index), diagnostics) = with_diagnostics(|diagnostics| {
                self.start_query(job.job.clone(), diagnostics, |tcx| {
                    tcx.dep_graph.with_anon_task(Q::dep_kind(), || {
                        Q::compute(tcx.global_tcx(), key)
                    })
                })
            });

            self.sess.profiler(|p| p.end_query(Q::NAME));
            profq_msg!(self, ProfileQueriesMsg::ProviderEnd);

            self.dep_graph.read_index(dep_node_index);

            if unlikely!(!diagnostics.is_empty()) {
                self.queries.on_disk_cache
                    .store_diagnostics_for_anon_node(dep_node_index, diagnostics);
            }

            job.complete(&result, dep_node_index);

            return result;
        }

        let dep_node = Q::to_dep_node(self, &key);

        if !Q::EVAL_ALWAYS {
            // The diagnostics for this query will be
            // promoted to the current session during
            // try_mark_green(), so we can ignore them here.
            let loaded = self.start_query(job.job.clone(), None, |tcx| {
                let marked = tcx.dep_graph.try_mark_green_and_read(tcx, &dep_node);
                marked.map(|(prev_dep_node_index, dep_node_index)| {
                    (tcx.load_from_disk_and_cache_in_memory::<Q>(
                        key.clone(),
                        prev_dep_node_index,
                        dep_node_index,
                        &dep_node
                    ), dep_node_index)
                })
            });
            if let Some((result, dep_node_index)) = loaded {
                job.complete(&result, dep_node_index);
                return result;
            }
        }

        let (result, dep_node_index) = self.force_query_with_job::<Q>(key, job, dep_node);
        self.dep_graph.read_index(dep_node_index);
        result
    }

    fn load_from_disk_and_cache_in_memory<Q: QueryDescription<'tcx>>(
        self,
        key: Q::Key,
        prev_dep_node_index: SerializedDepNodeIndex,
        dep_node_index: DepNodeIndex,
        dep_node: &DepNode,
    ) -> Q::Value {
        // Note this function can be called concurrently from the same query
        // We must ensure that this is handled correctly

        debug_assert!(self.dep_graph.is_green(dep_node));

        // First we try to load the result from the on-disk cache
        let result = if Q::cache_on_disk(self.global_tcx(), key.clone(), None) &&
                        self.sess.opts.debugging_opts.incremental_queries {
            self.sess.profiler(|p| p.incremental_load_result_start(Q::NAME));
            let result = Q::try_load_from_disk(self.global_tcx(), prev_dep_node_index);
            self.sess.profiler(|p| p.incremental_load_result_end(Q::NAME));

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
            profq_msg!(self, ProfileQueriesMsg::CacheHit);
            self.sess.profiler(|p| p.record_query_hit(Q::NAME));

            result
        } else {
            // We could not load a result from the on-disk cache, so
            // recompute.

            self.sess.profiler(|p| p.start_query(Q::NAME));

            // The dep-graph for this computation is already in
            // place
            let result = self.dep_graph.with_ignore(|| {
                Q::compute(self, key)
            });

            self.sess.profiler(|p| p.end_query(Q::NAME));
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

        result
    }

    #[inline(never)]
    #[cold]
    fn incremental_verify_ich<Q: QueryDescription<'tcx>>(
        self,
        result: &Q::Value,
        dep_node: &DepNode,
        dep_node_index: DepNodeIndex,
    ) {
        use crate::ich::Fingerprint;

        assert!(Some(self.dep_graph.fingerprint_of(dep_node_index)) ==
                self.dep_graph.prev_fingerprint_of(dep_node),
                "Fingerprint for green query instance not loaded \
                    from cache: {:?}", dep_node);

        debug!("BEGIN verify_ich({:?})", dep_node);
        let mut hcx = self.create_stable_hashing_context();

        let new_hash = Q::hash_result(&mut hcx, result).unwrap_or(Fingerprint::ZERO);
        debug!("END verify_ich({:?})", dep_node);

        let old_hash = self.dep_graph.fingerprint_of(dep_node_index);

        assert!(new_hash == old_hash, "Found unstable fingerprints \
            for {:?}", dep_node);
    }

    #[inline(always)]
    fn force_query_with_job<Q: QueryDescription<'tcx>>(
        self,
        key: Q::Key,
        job: JobOwner<'_, 'tcx, Q>,
        dep_node: DepNode,
    ) -> (Q::Value, DepNodeIndex) {
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
        self.sess.profiler(|p| p.start_query(Q::NAME));

        let ((result, dep_node_index), diagnostics) = with_diagnostics(|diagnostics| {
            self.start_query(job.job.clone(), diagnostics, |tcx| {
                if Q::EVAL_ALWAYS {
                    tcx.dep_graph.with_eval_always_task(dep_node,
                                                        tcx,
                                                        key,
                                                        Q::compute,
                                                        Q::hash_result)
                } else {
                    tcx.dep_graph.with_task(dep_node,
                                            tcx,
                                            key,
                                            Q::compute,
                                            Q::hash_result)
                }
            })
        });

        self.sess.profiler(|p| p.end_query(Q::NAME));
        profq_msg!(self, ProfileQueriesMsg::ProviderEnd);

        if unlikely!(self.sess.opts.debugging_opts.query_dep_graph) {
            self.dep_graph.mark_loaded_from_cache(dep_node_index, false);
        }

        if unlikely!(!diagnostics.is_empty()) {
            if dep_node.kind != crate::dep_graph::DepKind::Null {
                self.queries.on_disk_cache
                    .store_diagnostics(dep_node_index, diagnostics);
            }
        }

        job.complete(&result, dep_node_index);

        (result, dep_node_index)
    }

    /// Ensure that either this query has all green inputs or been executed.
    /// Executing query::ensure(D) is considered a read of the dep-node D.
    ///
    /// This function is particularly useful when executing passes for their
    /// side-effects -- e.g., in order to report errors for erroneous programs.
    ///
    /// Note: The optimization is only available during incr. comp.
    pub(super) fn ensure_query<Q: QueryDescription<'tcx>>(self, key: Q::Key) -> () {
        if Q::EVAL_ALWAYS {
            let _ = self.get_query::<Q>(DUMMY_SP, key);
            return;
        }

        // Ensuring an anonymous query makes no sense
        assert!(!Q::ANON);

        let dep_node = Q::to_dep_node(self, &key);

        if self.dep_graph.try_mark_green_and_read(self, &dep_node).is_none() {
            // A None return from `try_mark_green_and_read` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.

            let _ = self.get_query::<Q>(DUMMY_SP, key);
        } else {
            profq_msg!(self, ProfileQueriesMsg::CacheHit);
            self.sess.profiler(|p| p.record_query_hit(Q::NAME));
        }
    }

    #[allow(dead_code)]
    fn force_query<Q: QueryDescription<'tcx>>(self, key: Q::Key, span: Span, dep_node: DepNode) {
        profq_msg!(
            self,
            ProfileQueriesMsg::QueryBegin(span.data(),
                                          profq_query_msg!(Q::NAME.as_str(), self, key))
        );

        // We may be concurrently trying both execute and force a query.
        // Ensure that only one of them runs the query.
        let job = match JobOwner::try_get(self, span, &key) {
            TryGetJob::NotYetStarted(job) => job,
            TryGetJob::Cycle(_) |
            TryGetJob::JobCompleted(_) => {
                return
            }
        };
        self.force_query_with_job::<Q>(key, job, dep_node);
    }
}

macro_rules! handle_cycle_error {
    ([][$tcx: expr, $error:expr]) => {{
        $tcx.report_cycle($error).emit();
        Value::from_cycle_error($tcx.global_tcx())
    }};
    ([fatal_cycle$(, $modifiers:ident)*][$tcx:expr, $error:expr]) => {{
        $tcx.report_cycle($error).emit();
        $tcx.sess.abort_if_errors();
        unreachable!()
    }};
    ([cycle_delay_bug$(, $modifiers:ident)*][$tcx:expr, $error:expr]) => {{
        $tcx.report_cycle($error).delay_as_bug();
        Value::from_cycle_error($tcx.global_tcx())
    }};
    ([$other:ident$(, $modifiers:ident)*][$($args:tt)*]) => {
        handle_cycle_error!([$($modifiers),*][$($args)*])
    };
}

macro_rules! is_anon {
    ([]) => {{
        false
    }};
    ([anon$(, $modifiers:ident)*]) => {{
        true
    }};
    ([$other:ident$(, $modifiers:ident)*]) => {
        is_anon!([$($modifiers),*])
    };
}

macro_rules! is_eval_always {
    ([]) => {{
        false
    }};
    ([eval_always$(, $modifiers:ident)*]) => {{
        true
    }};
    ([$other:ident$(, $modifiers:ident)*]) => {
        is_eval_always!([$($modifiers),*])
    };
}

macro_rules! hash_result {
    ([][$hcx:expr, $result:expr]) => {{
        dep_graph::hash_result($hcx, &$result)
    }};
    ([no_hash$(, $modifiers:ident)*][$hcx:expr, $result:expr]) => {{
        None
    }};
    ([$other:ident$(, $modifiers:ident)*][$($args:tt)*]) => {
        hash_result!([$($modifiers),*][$($args)*])
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
        #[cfg(parallel_compiler)]
        use ty::query::job::QueryResult;
        use rustc_data_structures::sync::Lock;
        use crate::{
            rustc_data_structures::stable_hasher::HashStable,
            rustc_data_structures::stable_hasher::StableHasherResult,
            rustc_data_structures::stable_hasher::StableHasher,
            ich::StableHashingContext
        };
        use crate::util::profiling::ProfileCategory;

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

            #[cfg(parallel_compiler)]
            pub fn collect_active_jobs(&self) -> Vec<Lrc<QueryJob<$tcx>>> {
                let mut jobs = Vec::new();

                // We use try_lock here since we are only called from the
                // deadlock handler, and this shouldn't be locked.
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
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum QueryName {
            $($name),*
        }

        impl QueryName {
            pub fn register_with_profiler(profiler: &crate::util::profiling::SelfProfiler) {
                $(profiler.register_query_name(QueryName::$name);)*
            }

            pub fn as_str(&self) -> &'static str {
                match self {
                    $(QueryName::$name => stringify!($name),)*
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

            pub fn describe(&self, tcx: TyCtxt<'_>) -> Cow<'static, str> {
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
            pub fn default_span(&self, tcx: TyCtxt<$tcx>, span: Span) -> Span {
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

            pub fn query_name(&self) -> QueryName {
                match self {
                    $(Query::$name(_) => QueryName::$name,)*
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

            const NAME: QueryName = QueryName::$name;
            const CATEGORY: ProfileCategory = $category;
        }

        impl<$tcx> QueryAccessors<$tcx> for queries::$name<$tcx> {
            const ANON: bool = is_anon!([$($modifiers)*]);
            const EVAL_ALWAYS: bool = is_eval_always!([$($modifiers)*]);

            #[inline(always)]
            fn query(key: Self::Key) -> Query<'tcx> {
                Query::$name(key)
            }

            #[inline(always)]
            fn query_cache<'a>(tcx: TyCtxt<$tcx>) -> &'a Lock<QueryCache<$tcx, Self>> {
                &tcx.queries.$name
            }

            #[allow(unused)]
            #[inline(always)]
            fn to_dep_node(tcx: TyCtxt<$tcx>, key: &Self::Key) -> DepNode {
                use crate::dep_graph::DepConstructor::*;

                DepNode::new(tcx, $node(*key))
            }

            #[inline(always)]
            fn dep_kind() -> dep_graph::DepKind {
                dep_graph::DepKind::$node
            }

            #[inline]
            fn compute(tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
                __query_compute::$name(move || {
                    let provider = tcx.queries.providers.get(key.query_crate())
                        // HACK(eddyb) it's possible crates may be loaded after
                        // the query engine is created, and because crate loading
                        // is not yet integrated with the query engine, such crates
                        // would be missing appropriate entries in `providers`.
                        .unwrap_or(&tcx.queries.fallback_extern_providers)
                        .$name;
                    provider(tcx.global_tcx(), key)
                })
            }

            fn hash_result(
                _hcx: &mut StableHashingContext<'_>,
                _result: &Self::Value
            ) -> Option<Fingerprint> {
                hash_result!([$($modifiers)*][_hcx, _result])
            }

            fn handle_cycle_error(
                tcx: TyCtxt<'tcx>,
                error: CycleError<'tcx>
            ) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx, error])
            }
        })*

        #[derive(Copy, Clone)]
        pub struct TyCtxtEnsure<'tcx> {
            pub tcx: TyCtxt<'tcx>,
        }

        impl TyCtxtEnsure<$tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: $K) {
                self.tcx.ensure_query::<queries::$name<'_>>(key)
            })*
        }

        #[derive(Copy, Clone)]
        pub struct TyCtxtAt<'tcx> {
            pub tcx: TyCtxt<'tcx>,
            pub span: Span,
        }

        impl Deref for TyCtxtAt<'tcx> {
            type Target = TyCtxt<'tcx>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.tcx
            }
        }

        impl TyCtxt<$tcx> {
            /// Returns a transparent wrapper for `TyCtxt`, which ensures queries
            /// are executed instead of just returing their results.
            #[inline(always)]
            pub fn ensure(self) -> TyCtxtEnsure<$tcx> {
                TyCtxtEnsure {
                    tcx: self,
                }
            }

            /// Returns a transparent wrapper for `TyCtxt` which uses
            /// `span` as the location of queries performed through it.
            #[inline(always)]
            pub fn at(self, span: Span) -> TyCtxtAt<$tcx> {
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

        impl TyCtxtAt<$tcx> {
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
            /// This provides access to the incrimental comilation on-disk cache for query results.
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
            $(pub $name: fn(TyCtxt<$tcx>, $K) -> $R,)*
        }

        impl<$tcx> Default for Providers<$tcx> {
            fn default() -> Self {
                $(fn $name<$tcx>(_: TyCtxt<$tcx>, key: $K) -> $R {
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
/// reconstruct query-keys for certain kinds of `DepNode`s. In particular, we
/// enforce by construction that the GUID/fingerprint of certain `DepNode`s is a
/// valid `DefPathHash`. Since we also always build a huge table that maps every
/// `DefPathHash` in the current codebase to the corresponding `DefId`, we have
/// everything we need to re-run the query.
///
/// Take the `mir_validated` query as an example. Like many other queries, it
/// just has a single parameter: the `DefId` of the item it will compute the
/// validated MIR for. Now, when we call `force_from_dep_node()` on a `DepNode`
/// with kind `MirValidated`, we know that the GUID/fingerprint of the `DepNode`
/// is actually a `DefPathHash`, and can therefore just look up the corresponding
/// `DefId` in `tcx.def_path_hash_to_def_id`.
///
/// When you implement a new query, it will likely have a corresponding new
/// `DepKind`, and you'll have to support it here in `force_from_dep_node()`. As
/// a rule of thumb, if your query takes a `DefId` or `DefIndex` as sole parameter,
/// then `force_from_dep_node()` should not fail for it. Otherwise, you can just
/// add it to the "We don't have enough information to reconstruct..." group in
/// the match below.
pub fn force_from_dep_node(tcx: TyCtxt<'_>, dep_node: &DepNode) -> bool {
    use crate::dep_graph::RecoverKey;

    // We must avoid ever having to call force_from_dep_node() for a
    // DepNode::codegen_unit:
    // Since we cannot reconstruct the query key of a DepNode::codegen_unit, we
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
    debug_assert!(dep_node.kind != DepKind::codegen_unit,
                  "calling force_from_dep_node() on DepKind::codegen_unit");

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

    macro_rules! force_ex {
        ($tcx:expr, $query:ident, $key:expr) => {
            {
                $tcx.force_query::<crate::ty::query::queries::$query<'_>>(
                    $key,
                    DUMMY_SP,
                    *dep_node
                );
            }
        }
    };

    macro_rules! force {
        ($query:ident, $key:expr) => { force_ex!(tcx, $query, $key) }
    };

    rustc_dep_node_force!([dep_node, tcx]
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
        DepKind::CompileCodegenUnit => {
            bug!("force_from_dep_node() - Encountered {:?}", dep_node)
        }

        DepKind::Analysis => { force!(analysis, krate!()); }
    );

    true
}
