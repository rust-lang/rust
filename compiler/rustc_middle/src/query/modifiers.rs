//! This contains documentation which is linked from query modifiers used in the `rustc_queries!` proc macro.
//!
//! The dummy items in this module are used to enable hover documentation for
//! modifier names in the query list, and to allow find-all-references to list
//! all queries that use a particular modifier.
#![allow(unused, non_camel_case_types)]

// tidy-alphabetical-start
//
/// # `arena_cache` query modifier
///
/// Query return values must impl `Copy` and be small, but some queries must return values that
/// doesn't meet those criteria. Queries marked with this modifier have their values allocated in
/// an arena and the query returns a reference to the value. There are two cases.
/// - If the provider function returns `T` then the query will return `&'tcx T`.
/// - If the provider function returns `Option<T>` then the query will return `Option<&'tcx T>`.
///
/// The query plumbing takes care of the arenas and the type manipulations.
pub(crate) struct arena_cache;

/// # `cache_on_disk` query modifier
///
/// The query's return values are cached to disk, and can be loaded by subsequent
/// sessions if the corresponding dep node is green.
///
/// If the [`separate_provide_extern`] modifier is also present, values will only
/// be cached to disk for "local" keys, because values for external crates should
/// be loadable from crate metadata instead.
pub(crate) struct cache_on_disk;

/// # `depth_limit` query modifier
///
/// Impose a recursion call depth limit on the query to prevent stack overflow.
pub(crate) struct depth_limit;

/// # `desc { ... }` query modifier
///
/// The human-readable description of the query, for diagnostics and profiling. Required for every
/// query. The block should contain a `format!`-style string literal followed by optional
/// arguments. The query key identifier is available for use within the block, as is `tcx`.
pub(crate) struct desc;

/// # `eval_always` query modifier
///
/// Queries with this modifier do not track their dependencies, and are treated as always having a
/// red (dirty) dependency instead. This is necessary for queries that interact with state that
/// isn't tracked by the query system.
///
/// It can also improve performance for queries that are so likely to be dirtied by any change that
/// it's not worth tracking their actual dependencies at all.
///
/// As with all queries, the return value is still cached in memory for the rest of the compiler
/// session.
pub(crate) struct eval_always;

/// # `feedable` query modifier
///
/// Generate a `feed` method to set the query's value from another query.
pub(crate) struct feedable;

/// # `handle_cycle_error` query modifier
///
/// The default behaviour for a query cycle is to emit a cycle error and halt
/// compilation. Queries with this modifier will instead use a custom handler,
/// which must be provided at `rustc_query_impl::handle_cycle_error::$name`,
/// where `$name` is the query name.
pub(crate) struct handle_cycle_error;

/// # `no_force` query modifier
///
/// Dep nodes of queries with this modifier will never be "forced" when trying
/// to mark their dependents green, even if their key is recoverable from the
/// key fingerprint.
///
/// Used by some queries with custom cycle-handlers to ensure that if a cycle
/// occurs, all of the relevant query calls will be on the query stack.
pub(crate) struct no_force;

/// # `no_hash` query modifier
///
/// Do not hash the query's return value for incremental compilation. If the value needs to be
/// recomputed, always mark its node as red (dirty).
pub(crate) struct no_hash;

/// # `separate_provide_extern` query modifier
///
/// Use separate query provider functions for local and extern crates.
///
/// Also affects the [`cache_on_disk`] modifier.
pub(crate) struct separate_provide_extern;

// tidy-alphabetical-end
