//! This contains documentation which is linked from query modifiers used in the `rustc_queries!` proc macro.
#![allow(unused, non_camel_case_types)]
// FIXME: Update and clarify documentation for these modifiers.

/// # `desc` query modifier
///
/// The description of the query. This modifier is required on every query.
pub struct desc;

/// # `arena_cache` query modifier
///
/// Use this type for the in-memory cache.
pub struct arena_cache;

/// # `cache_on_disk_if` query modifier
///
/// Cache the query to disk if the `Block` returns true.
pub struct cache_on_disk_if;

/// # `cycle_fatal` query modifier
///
/// A cycle error for this query aborting the compilation with a fatal error.
pub struct cycle_fatal;

/// # `cycle_delay_bug` query modifier
///
/// A cycle error results in a delay_bug call
pub struct cycle_delay_bug;

/// # `cycle_stash` query modifier
///
/// A cycle error results in a stashed cycle error that can be unstashed and canceled later
pub struct cycle_stash;

/// # `no_hash` query modifier
///
/// Don't hash the result, instead just mark a query red if it runs
pub struct no_hash;

/// # `anon` query modifier
///
/// Generate a dep node based on the dependencies of the query
pub struct anon;

/// # `eval_always` query modifier
///
/// Always evaluate the query, ignoring its dependencies
pub struct eval_always;

/// # `depth_limit` query modifier
///
/// Whether the query has a call depth limit
pub struct depth_limit;

/// # `separate_provide_extern` query modifier
///
/// Use a separate query provider for local and extern crates
pub struct separate_provide_extern;

/// # `feedable` query modifier
///
/// Generate a `feed` method to set the query's value from another query.
pub struct feedable;

/// # `return_result_from_ensure_ok` query modifier
///
/// When this query is called via `tcx.ensure_ok()`, it returns
/// `Result<(), ErrorGuaranteed>` instead of `()`. If the query needs to
/// be executed, and that execution returns an error, the error result is
/// returned to the caller.
///
/// If execution is skipped, a synthetic `Ok(())` is returned, on the
/// assumption that a query with all-green inputs must have succeeded.
///
/// Can only be applied to queries with a return value of
/// `Result<_, ErrorGuaranteed>`.
pub struct return_result_from_ensure_ok;
