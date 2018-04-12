// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The implementation of the query system itself. Defines the macros
//! that generate the actual methods on tcx which find and execute the
//! provider, manage the caches, and so forth.

use dep_graph::{DepNodeIndex, DepNode, DepKind, DepNodeColor};
use errors::DiagnosticBuilder;
use errors::Level;
use ty::tls;
use ty::{TyCtxt};
use ty::maps::config::QueryDescription;
use ty::maps::job::{QueryResult, QueryInfo};
use ty::item_path;

use rustc_data_structures::fx::{FxHashMap};
use rustc_data_structures::sync::LockGuard;
use std::marker::PhantomData;
use syntax_pos::Span;

pub(super) struct QueryMap<'tcx, D: QueryDescription<'tcx>> {
    phantom: PhantomData<(D, &'tcx ())>,
    pub(super) map: FxHashMap<D::Key, QueryResult<'tcx, QueryValue<D::Value>>>,
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

impl<'tcx, M: QueryDescription<'tcx>> QueryMap<'tcx, M> {
    pub(super) fn new() -> QueryMap<'tcx, M> {
        QueryMap {
            phantom: PhantomData,
            map: FxHashMap(),
        }
    }
}

pub(super) trait GetCacheInternal<'tcx>: QueryDescription<'tcx> + Sized {
    fn get_cache_internal<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>)
                              -> LockGuard<'a, QueryMap<'tcx, Self>>;
}

#[derive(Clone)]
pub(super) struct CycleError<'tcx> {
    pub(super) span: Span,
    pub(super) cycle: Vec<QueryInfo<'tcx>>,
}

/// The result of `try_get_lock`
pub(super) enum TryGetLock<'a, 'tcx: 'a, T, D: QueryDescription<'tcx> + 'a> {
    /// The query is not yet started. Contains a guard to the map eventually used to start it.
    NotYetStarted(LockGuard<'a, QueryMap<'tcx, D>>),

    /// The query was already completed.
    /// Returns the result of the query and its dep node index
    /// if it succeeded or a cycle error if it failed
    JobCompleted(Result<(T, DepNodeIndex), CycleError<'tcx>>),
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub(super) fn report_cycle(self, CycleError { span, cycle: stack }: CycleError)
        -> DiagnosticBuilder<'a>
    {
        assert!(!stack.is_empty());

        // Disable naming impls with types in this path, since that
        // sometimes cycles itself, leading to extra cycle errors.
        // (And cycle errors around impls tend to occur during the
        // collect/coherence phases anyhow.)
        item_path::with_forced_impl_filename_line(|| {
            let span = self.sess.codemap().def_span(span);
            let mut err =
                struct_span_err!(self.sess, span, E0391,
                                 "cyclic dependency detected");
            err.span_label(span, "cyclic reference");

            err.span_note(self.sess.codemap().def_span(stack[0].span),
                          &format!("the cycle begins when {}...", stack[0].query.describe(self)));

            for &QueryInfo { span, ref query, .. } in &stack[1..] {
                err.span_note(self.sess.codemap().def_span(span),
                              &format!("...which then requires {}...", query.describe(self)));
            }

            err.note(&format!("...which then again requires {}, completing the cycle.",
                              stack[0].query.describe(self)));

            return err
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
                    db.set_span(icx.tcx.sess.codemap().def_span(query.info.span));
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
    /// the a node index can be found for that node.
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
macro_rules! profq_key {
    ($tcx:expr, $key:expr) => {
        if cfg!(debug_assertions) {
            if $tcx.sess.profile_queries_and_keys() {
                Some(format!("{:?}", $key))
            } else { None }
        } else { None }
    }
}

macro_rules! handle_cycle_error {
    ([][$this: expr]) => {{
        Value::from_cycle_error($this.global_tcx())
    }};
    ([fatal_cycle$(, $modifiers:ident)*][$this:expr]) => {{
        $this.tcx.sess.abort_if_errors();
        unreachable!();
    }};
    ([$other:ident$(, $modifiers:ident)*][$($args:tt)*]) => {
        handle_cycle_error!([$($modifiers),*][$($args)*])
    };
}

macro_rules! define_maps {
    (<$tcx:tt>
     $($(#[$attr:meta])*
       [$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $V:ty,)*) => {

        use dep_graph::DepNodeIndex;
        use std::mem;
        use errors::Diagnostic;
        use errors::FatalError;
        use rustc_data_structures::sync::{Lock, LockGuard};
        use rustc_data_structures::OnDrop;

        define_map_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$($attr)*] [$name]))*)
        }

        impl<$tcx> Maps<$tcx> {
            pub fn new(providers: IndexVec<CrateNum, Providers<$tcx>>)
                       -> Self {
                Maps {
                    providers,
                    $($name: Lock::new(QueryMap::new())),*
                }
            }
        }

        #[allow(bad_style)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum Query<$tcx> {
            $($(#[$attr])* $name($K)),*
        }

        #[allow(bad_style)]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum QueryMsg {
            $($name(Option<String>)),*
        }

        impl<$tcx> Query<$tcx> {
            pub fn name(&self) -> &'static str {
                match *self {
                    $(Query::$name(_) => stringify!($name),)*
                }
            }

            pub fn describe(&self, tcx: TyCtxt) -> String {
                let (r, name) = match *self {
                    $(Query::$name(key) => {
                        (queries::$name::describe(tcx, key), stringify!($name))
                    })*
                };
                if tcx.sess.verbose() {
                    format!("{} [{}]", r, name)
                } else {
                    r
                }
            }
        }

        pub mod queries {
            use std::marker::PhantomData;

            $(#[allow(bad_style)]
            pub struct $name<$tcx> {
                data: PhantomData<&$tcx ()>
            })*
        }

        $(impl<$tcx> QueryConfig for queries::$name<$tcx> {
            type Key = $K;
            type Value = $V;
        }

        impl<$tcx> GetCacheInternal<$tcx> for queries::$name<$tcx> {
            fn get_cache_internal<'a>(tcx: TyCtxt<'a, $tcx, $tcx>)
                                      -> LockGuard<'a, QueryMap<$tcx, Self>> {
                tcx.maps.$name.borrow()
            }
        }

        impl<'a, $tcx, 'lcx> queries::$name<$tcx> {

            #[allow(unused)]
            fn to_dep_node(tcx: TyCtxt<'a, $tcx, 'lcx>, key: &$K) -> DepNode {
                use dep_graph::DepConstructor::*;

                DepNode::new(tcx, $node(*key))
            }

            /// Either get the lock of the query map, allowing us to
            /// start executing the query, or it returns with the result of the query.
            /// If the query already executed and panicked, this will fatal error / silently panic
            fn try_get_lock(
                tcx: TyCtxt<'a, $tcx, 'lcx>,
                mut span: Span,
                key: &$K
            ) -> TryGetLock<'a, $tcx, $V, Self>
            {
                loop {
                    let lock = tcx.maps.$name.borrow_mut();
                    let job = if let Some(value) = lock.map.get(key) {
                        match *value {
                            QueryResult::Started(ref job) => Some(job.clone()),
                            QueryResult::Complete(ref value) => {
                                profq_msg!(tcx, ProfileQueriesMsg::CacheHit);
                                let result = Ok(((&value.value).clone(), value.index));
                                return TryGetLock::JobCompleted(result);
                            },
                            QueryResult::Poisoned => FatalError.raise(),
                        }
                    } else {
                        None
                    };
                    let job = if let Some(job) = job {
                        job
                    } else {
                        return TryGetLock::NotYetStarted(lock);
                    };
                    mem::drop(lock);

                    // This just matches the behavior of `try_get_with` so the span when
                    // we await matches the span we would use when executing.
                    // See the FIXME there.
                    if span == DUMMY_SP && stringify!($name) != "def_span" {
                        span = key.default_span(tcx);
                    }

                    if let Err(cycle) = job.await(tcx, span) {
                        return TryGetLock::JobCompleted(Err(cycle));
                    }
                }
            }

            fn try_get_with(tcx: TyCtxt<'a, $tcx, 'lcx>,
                            mut span: Span,
                            key: $K)
                            -> Result<$V, CycleError<$tcx>>
            {
                debug!("ty::queries::{}::try_get_with(key={:?}, span={:?})",
                       stringify!($name),
                       key,
                       span);

                profq_msg!(tcx,
                    ProfileQueriesMsg::QueryBegin(
                        span.data(),
                        QueryMsg::$name(profq_key!(tcx, key))
                    )
                );

                /// Get the lock used to start the query or
                /// return the result of the completed query
                macro_rules! get_lock_or_return {
                    () => {{
                        match Self::try_get_lock(tcx, span, &key) {
                            TryGetLock::NotYetStarted(lock) => lock,
                            TryGetLock::JobCompleted(result) => {
                                return result.map(|(v, index)| {
                                    tcx.dep_graph.read_index(index);
                                    v
                                })
                            }
                        }
                    }}
                }

                let mut lock = get_lock_or_return!();

                // FIXME(eddyb) Get more valid Span's on queries.
                // def_span guard is necessary to prevent a recursive loop,
                // default_span calls def_span query internally.
                if span == DUMMY_SP && stringify!($name) != "def_span" {
                    // This might deadlock if we hold the map lock since we might be
                    // waiting for the def_span query and switch to some other fiber
                    // So we drop the lock here and reacquire it
                    mem::drop(lock);
                    span = key.default_span(tcx);
                    lock = get_lock_or_return!();
                }

                // Fast path for when incr. comp. is off. `to_dep_node` is
                // expensive for some DepKinds.
                if !tcx.dep_graph.is_fully_enabled() {
                    let null_dep_node = DepNode::new_no_params(::dep_graph::DepKind::Null);
                    return Self::force_with_lock(tcx, key, span, lock, null_dep_node)
                                .map(|(v, _)| v);
                }

                let dep_node = Self::to_dep_node(tcx, &key);

                if dep_node.kind.is_anon() {
                    profq_msg!(tcx, ProfileQueriesMsg::ProviderBegin);

                    let res = Self::start_job(tcx, span, key, lock, |tcx| {
                        tcx.dep_graph.with_anon_task(dep_node.kind, || {
                            Self::compute_result(tcx.global_tcx(), key)
                        })
                    })?;

                    profq_msg!(tcx, ProfileQueriesMsg::ProviderEnd);
                    let (((result, dep_node_index), diagnostics), job) = res;

                    tcx.dep_graph.read_index(dep_node_index);

                    tcx.on_disk_query_result_cache
                       .store_diagnostics_for_anon_node(dep_node_index, diagnostics);

                    let value = QueryValue::new(Clone::clone(&result), dep_node_index);

                    tcx.maps
                       .$name
                       .borrow_mut()
                       .map
                       .insert(key, QueryResult::Complete(value));

                    job.signal_complete();

                    return Ok(result);
                }

                if !dep_node.kind.is_input() {
                    // try_mark_green_and_read may force queries. So we must drop our lock here
                    mem::drop(lock);
                    if let Some(dep_node_index) = tcx.try_mark_green_and_read(&dep_node) {
                        profq_msg!(tcx, ProfileQueriesMsg::CacheHit);
                        return Self::load_from_disk_and_cache_in_memory(tcx,
                                                                        key,
                                                                        span,
                                                                        dep_node_index,
                                                                        &dep_node)
                    }
                    lock = get_lock_or_return!();
                }

                match Self::force_with_lock(tcx, key, span, lock, dep_node) {
                    Ok((result, dep_node_index)) => {
                        tcx.dep_graph.read_index(dep_node_index);
                        Ok(result)
                    }
                    Err(e) => Err(e)
                }
            }

            /// Ensure that either this query has all green inputs or been executed.
            /// Executing query::ensure(D) is considered a read of the dep-node D.
            ///
            /// This function is particularly useful when executing passes for their
            /// side-effects -- e.g., in order to report errors for erroneous programs.
            ///
            /// Note: The optimization is only available during incr. comp.
            pub fn ensure(tcx: TyCtxt<'a, $tcx, 'lcx>, key: $K) -> () {
                let dep_node = Self::to_dep_node(tcx, &key);

                // Ensuring an "input" or anonymous query makes no sense
                assert!(!dep_node.kind.is_anon());
                assert!(!dep_node.kind.is_input());
                if tcx.try_mark_green_and_read(&dep_node).is_none() {
                    // A None return from `try_mark_green_and_read` means that this is either
                    // a new dep node or that the dep node has already been marked red.
                    // Either way, we can't call `dep_graph.read()` as we don't have the
                    // DepNodeIndex. We must invoke the query itself. The performance cost
                    // this introduces should be negligible as we'll immediately hit the
                    // in-memory cache, or another query down the line will.
                    let _ = tcx.$name(key);
                }
            }

            /// Creates a job for the query and updates the query map indicating that it started.
            /// Then it changes ImplicitCtxt to point to the new query job while it executes.
            /// If the query panics, this updates the query map to indicate so.
            fn start_job<F, R>(tcx: TyCtxt<'_, $tcx, 'lcx>,
                               span: Span,
                               key: $K,
                               mut map: LockGuard<'_, QueryMap<$tcx, Self>>,
                               compute: F)
                -> Result<((R, Vec<Diagnostic>), Lrc<QueryJob<$tcx>>), CycleError<$tcx>>
                where F: for<'b> FnOnce(TyCtxt<'b, $tcx, 'lcx>) -> R
            {
                let query = Query::$name(Clone::clone(&key));

                let entry = QueryInfo {
                    span,
                    query,
                };

                // The TyCtxt stored in TLS has the same global interner lifetime
                // as `tcx`, so we use `with_related_context` to relate the 'gcx lifetimes
                // when accessing the ImplicitCtxt
                let (r, job) = ty::tls::with_related_context(tcx, move |icx| {
                    let job = Lrc::new(QueryJob::new(entry, icx.query.clone()));

                    // Store the job in the query map and drop the lock to allow
                    // others to wait it
                    map.map.entry(key).or_insert(QueryResult::Started(job.clone()));
                    mem::drop(map);

                    let r = {
                        let on_drop = OnDrop(|| {
                            // Poison the query so jobs waiting on it panic
                            tcx.maps
                            .$name
                            .borrow_mut()
                            .map
                            .insert(key, QueryResult::Poisoned);
                            // Also signal the completion of the job, so waiters
                            // will continue execution
                            job.signal_complete();
                        });

                        // Update the ImplicitCtxt to point to our new query job
                        let icx = ty::tls::ImplicitCtxt {
                            tcx,
                            query: Some(job.clone()),
                        };

                        // Use the ImplicitCtxt while we execute the query
                        let r = ty::tls::enter_context(&icx, |icx| {
                            compute(icx.tcx)
                        });

                        mem::forget(on_drop);

                        r
                    };

                    (r, job)
                });

                // Extract the diagnostic from the job
                let diagnostics: Vec<_> = mem::replace(&mut *job.diagnostics.lock(), Vec::new());

                Ok(((r, diagnostics), job))
            }

            fn compute_result(tcx: TyCtxt<'a, $tcx, 'lcx>, key: $K) -> $V {
                let provider = tcx.maps.providers[key.map_crate()].$name;
                provider(tcx.global_tcx(), key)
            }

            fn load_from_disk_and_cache_in_memory(tcx: TyCtxt<'a, $tcx, 'lcx>,
                                                  key: $K,
                                                  span: Span,
                                                  dep_node_index: DepNodeIndex,
                                                  dep_node: &DepNode)
                                                  -> Result<$V, CycleError<$tcx>>
            {
                // Note this function can be called concurrently from the same query
                // We must ensure that this is handled correctly

                debug_assert!(tcx.dep_graph.is_green(dep_node));

                // First we try to load the result from the on-disk cache
                let result = if Self::cache_on_disk(key) &&
                                tcx.sess.opts.debugging_opts.incremental_queries {
                    let prev_dep_node_index =
                        tcx.dep_graph.prev_dep_node_index_of(dep_node);
                    let result = Self::try_load_from_disk(tcx.global_tcx(),
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

                let (result, job) = if let Some(result) = result {
                    (result, None)
                } else {
                    // We could not load a result from the on-disk cache, so
                    // recompute.

                    // The diagnostics for this query have already been
                    // promoted to the current session during
                    // try_mark_green(), so we can ignore them here.
                    let ((result, _), job) = Self::start_job(tcx,
                                                             span,
                                                             key,
                                                             tcx.maps.$name.borrow_mut(),
                                                             |tcx| {
                        // The dep-graph for this computation is already in
                        // place
                        tcx.dep_graph.with_ignore(|| {
                            Self::compute_result(tcx, key)
                        })
                    })?;
                    (result, Some(job))
                };

                // If -Zincremental-verify-ich is specified, re-hash results from
                // the cache and make sure that they have the expected fingerprint.
                if tcx.sess.opts.debugging_opts.incremental_verify_ich {
                    use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
                    use ich::Fingerprint;

                    assert!(Some(tcx.dep_graph.fingerprint_of(dep_node_index)) ==
                            tcx.dep_graph.prev_fingerprint_of(dep_node),
                            "Fingerprint for green query instance not loaded \
                             from cache: {:?}", dep_node);

                    debug!("BEGIN verify_ich({:?})", dep_node);
                    let mut hcx = tcx.create_stable_hashing_context();
                    let mut hasher = StableHasher::new();

                    result.hash_stable(&mut hcx, &mut hasher);

                    let new_hash: Fingerprint = hasher.finish();
                    debug!("END verify_ich({:?})", dep_node);

                    let old_hash = tcx.dep_graph.fingerprint_of(dep_node_index);

                    assert!(new_hash == old_hash, "Found unstable fingerprints \
                        for {:?}", dep_node);
                }

                if tcx.sess.opts.debugging_opts.query_dep_graph {
                    tcx.dep_graph.mark_loaded_from_cache(dep_node_index, true);
                }

                let value = QueryValue::new(Clone::clone(&result), dep_node_index);

                tcx.maps
                   .$name
                   .borrow_mut()
                   .map
                   .insert(key, QueryResult::Complete(value));

                job.map(|j| j.signal_complete());

                Ok(result)
            }

            #[allow(dead_code)]
            fn force(tcx: TyCtxt<'a, $tcx, 'lcx>,
                     key: $K,
                     span: Span,
                     dep_node: DepNode)
                     -> Result<($V, DepNodeIndex), CycleError<$tcx>> {
                // We may be concurrently trying both execute and force a query
                // Ensure that only one of them runs the query
                let lock = match Self::try_get_lock(tcx, span, &key) {
                    TryGetLock::NotYetStarted(lock) => lock,
                    TryGetLock::JobCompleted(result) => return result,
                };
                Self::force_with_lock(tcx,
                                      key,
                                      span,
                                      lock,
                                      dep_node)
            }

            fn force_with_lock(tcx: TyCtxt<'a, $tcx, 'lcx>,
                               key: $K,
                               span: Span,
                               map: LockGuard<'_, QueryMap<$tcx, Self>>,
                               dep_node: DepNode)
                               -> Result<($V, DepNodeIndex), CycleError<$tcx>> {
                // If the following assertion triggers, it can have two reasons:
                // 1. Something is wrong with DepNode creation, either here or
                //    in DepGraph::try_mark_green()
                // 2. Two distinct query keys get mapped to the same DepNode
                //    (see for example #48923)
                assert!(!tcx.dep_graph.dep_node_exists(&dep_node),
                        "Forcing query with already existing DepNode.\n\
                          - query-key: {:?}\n\
                          - dep-node: {:?}",
                        key, dep_node);

                profq_msg!(tcx, ProfileQueriesMsg::ProviderBegin);
                let res = Self::start_job(tcx,
                                          span,
                                          key,
                                          map,
                                          |tcx| {
                    if dep_node.kind.is_eval_always() {
                        tcx.dep_graph.with_eval_always_task(dep_node,
                                                            tcx,
                                                            key,
                                                            Self::compute_result)
                    } else {
                        tcx.dep_graph.with_task(dep_node,
                                                tcx,
                                                key,
                                                Self::compute_result)
                    }
                })?;
                profq_msg!(tcx, ProfileQueriesMsg::ProviderEnd);

                let (((result, dep_node_index), diagnostics), job) = res;

                if tcx.sess.opts.debugging_opts.query_dep_graph {
                    tcx.dep_graph.mark_loaded_from_cache(dep_node_index, false);
                }

                if dep_node.kind != ::dep_graph::DepKind::Null {
                    tcx.on_disk_query_result_cache
                       .store_diagnostics(dep_node_index, diagnostics);
                }

                let value = QueryValue::new(Clone::clone(&result), dep_node_index);

                tcx.maps
                   .$name
                   .borrow_mut()
                   .map
                   .insert(key, QueryResult::Complete(value));

                let job: Lrc<QueryJob> = job;

                job.signal_complete();

                Ok((result, dep_node_index))
            }

            pub fn try_get(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K)
                           -> Result<$V, DiagnosticBuilder<'a>> {
                match Self::try_get_with(tcx, span, key) {
                    Ok(e) => Ok(e),
                    Err(e) => Err(tcx.report_cycle(e)),
                }
            }
        })*

        #[derive(Copy, Clone)]
        pub struct TyCtxtAt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
            pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
            pub span: Span,
        }

        impl<'a, 'gcx, 'tcx> Deref for TyCtxtAt<'a, 'gcx, 'tcx> {
            type Target = TyCtxt<'a, 'gcx, 'tcx>;
            fn deref(&self) -> &Self::Target {
                &self.tcx
            }
        }

        impl<'a, $tcx, 'lcx> TyCtxt<'a, $tcx, 'lcx> {
            /// Return a transparent wrapper for `TyCtxt` which uses
            /// `span` as the location of queries performed through it.
            pub fn at(self, span: Span) -> TyCtxtAt<'a, $tcx, 'lcx> {
                TyCtxtAt {
                    tcx: self,
                    span
                }
            }

            $($(#[$attr])*
            pub fn $name(self, key: $K) -> $V {
                self.at(DUMMY_SP).$name(key)
            })*
        }

        impl<'a, $tcx, 'lcx> TyCtxtAt<'a, $tcx, 'lcx> {
            $($(#[$attr])*
            pub fn $name(self, key: $K) -> $V {
                queries::$name::try_get(self.tcx, self.span, key).unwrap_or_else(|mut e| {
                    e.emit();
                    handle_cycle_error!([$($modifiers)*][self])
                })
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

macro_rules! define_map_struct {
    (tcx: $tcx:tt,
     input: ($(([$($modifiers:tt)*] [$($attr:tt)*] [$name:ident]))*)) => {
        pub struct Maps<$tcx> {
            providers: IndexVec<CrateNum, Providers<$tcx>>,
            $($(#[$attr])*  $name: Lock<QueryMap<$tcx, queries::$name<$tcx>>>,)*
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
                    bug!("tcx.maps.{}({:?}) unsupported by its crate",
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
    use ty::maps::keys::Key;
    use hir::def_id::LOCAL_CRATE;

    // We must avoid ever having to call force_from_dep_node() for a
    // DepNode::CodegenUnit:
    // Since we cannot reconstruct the query key of a DepNode::CodegenUnit, we
    // would always end up having to evaluate the first caller of the
    // `codegen_unit` query that *is* reconstructible. This might very well be
    // the `compile_codegen_unit` query, thus re-translating the whole CGU just
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
                use $crate::util::common::{ProfileQueriesMsg, profq_msg};

                // FIXME(eddyb) Get more valid Span's on queries.
                // def_span guard is necessary to prevent a recursive loop,
                // default_span calls def_span query internally.
                let span = if stringify!($query) != "def_span" {
                    $key.default_span(tcx)
                } else {
                    ::syntax_pos::DUMMY_SP
                };

                profq_msg!(tcx,
                    ProfileQueriesMsg::QueryBegin(
                        span.data(),
                        ::ty::maps::QueryMsg::$query(profq_key!(tcx, $key))
                    )
                );

                match ::ty::maps::queries::$query::force(tcx, $key, span, *dep_node) {
                    Ok(_) => {},
                    Err(e) => {
                        tcx.report_cycle(e).emit();
                    }
                }
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
        DepKind::DropckOutlives |
        DepKind::SubstituteNormalizeAndTestPredicates |
        DepKind::InstanceDefSizeEstimate |

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
        DepKind::InferredOutlivesOf => { force!(inferred_outlives_of, def_id!()); }
        DepKind::SuperPredicatesOfItem => { force!(super_predicates_of, def_id!()); }
        DepKind::TraitDefOfItem => { force!(trait_def, def_id!()); }
        DepKind::AdtDefOfItem => { force!(adt_def, def_id!()); }
        DepKind::ImplTraitRef => { force!(impl_trait_ref, def_id!()); }
        DepKind::ImplPolarity => { force!(impl_polarity, def_id!()); }
        DepKind::FnSignature => { force!(fn_sig, def_id!()); }
        DepKind::CoerceUnsizedInfo => { force!(coerce_unsized_info, def_id!()); }
        DepKind::ItemVariances => { force!(variances_of, def_id!()); }
        DepKind::IsConstFn => { force!(is_const_fn, def_id!()); }
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
        DepKind::DescribeDef => { force!(describe_def, def_id!()); }
        DepKind::DefSpan => { force!(def_span, def_id!()); }
        DepKind::LookupStability => { force!(lookup_stability, def_id!()); }
        DepKind::LookupDeprecationEntry => {
            force!(lookup_deprecation_entry, def_id!());
        }
        DepKind::ItemBodyNestedBodies => { force!(item_body_nested_bodies, def_id!()); }
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
        DepKind::TransFnAttrs => { force!(trans_fn_attrs, def_id!()); }
        DepKind::FnArgNames => { force!(fn_arg_names, def_id!()); }
        DepKind::DylibDepFormats => { force!(dylib_dependency_formats, krate!()); }
        DepKind::IsPanicRuntime => { force!(is_panic_runtime, krate!()); }
        DepKind::IsCompilerBuiltins => { force!(is_compiler_builtins, krate!()); }
        DepKind::HasGlobalAllocator => { force!(has_global_allocator, krate!()); }
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
        DepKind::DeriveRegistrarFn => { force!(derive_registrar_fn, krate!()); }
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
        DepKind::GetLangItems => { force!(get_lang_items, LOCAL_CRATE); }
        DepKind::DefinedLangItems => { force!(defined_lang_items, krate!()); }
        DepKind::MissingLangItems => { force!(missing_lang_items, krate!()); }
        DepKind::ExternConstBody => { force!(extern_const_body, def_id!()); }
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
        DepKind::CollectAndPartitionTranslationItems => {
            force!(collect_and_partition_translation_items, LOCAL_CRATE);
        }
        DepKind::IsTranslatedItem => { force!(is_translated_item, def_id!()); }
        DepKind::OutputFilenames => { force!(output_filenames, LOCAL_CRATE); }

        DepKind::TargetFeaturesWhitelist => { force!(target_features_whitelist, LOCAL_CRATE); }

        DepKind::Features => { force!(features_query, LOCAL_CRATE); }

        DepKind::ProgramClausesFor => { force!(program_clauses_for, def_id!()); }
        DepKind::WasmCustomSections => { force!(wasm_custom_sections, krate!()); }
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
//                a combined define_dep_nodes!()/define_maps!() macro.
macro_rules! impl_load_from_cache {
    ($($dep_kind:ident => $query_name:ident,)*) => {
        impl DepNode {
            // Check whether the query invocation corresponding to the given
            // DepNode is eligible for on-disk-caching.
            pub fn cache_on_disk(&self, tcx: TyCtxt) -> bool {
                use ty::maps::queries;
                use ty::maps::QueryDescription;

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
            pub fn load_from_on_disk_cache(&self, tcx: TyCtxt) {
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
    TransFnAttrs => trans_fn_attrs,
    SpecializationGraph => specialization_graph_of,
);
