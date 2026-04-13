use rustc_middle::queries::TaggedQueryKey;
use rustc_middle::query::erase::Erased;
use rustc_middle::query::{QueryMode, QueryVTable};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::query::impl_::GetQueryVTable;

macro_rules! define_queries {
    (
        // Note: `$K` and `$V` are unused but present so this can be called by
        // `rustc_with_all_queries`.
        queries {
            $(
                $(#[$attr:meta])*
                fn $name:ident($K:ty) -> $V:ty
                {
                    // Search for (QMODLIST) to find all occurrences of this query modifier list.
                    arena_cache: $arena_cache:literal,
                    cache_on_disk: $cache_on_disk:literal,
                    depth_limit: $depth_limit:literal,
                    desc: $desc:expr,
                    eval_always: $eval_always:literal,
                    feedable: $feedable:literal,
                    handle_cycle_error: $handle_cycle_error:literal,
                    no_force: $no_force:literal,
                    no_hash: $no_hash:literal,
                    returns_error_guaranteed: $returns_error_guaranteed:literal,
                    separate_provide_extern: $separate_provide_extern:literal,
                }
            )*
        }
        // Non-queries are unused here.
        non_queries { $($_:tt)* }
    ) => {
        // This macro expects to be expanded into `crate::query::impl_::query_impl`, which is this file.
        $(
            pub(crate) mod $name {
                use super::*;

                // It seems to be important that every query has its own monomorphic
                // copy of `execute_query_incr` and `execute_query_non_incr`.
                // Trying to inline these wrapper functions into their generic
                // "inner" helpers tends to break `tests/run-make/short-ice`.

                pub(crate) mod execute_query_incr {
                    use super::*;
                    use rustc_middle::queries::$name::{Key, Value};

                    // Adding `__rust_end_short_backtrace` marker to backtraces so that we emit the frames
                    // when `RUST_BACKTRACE=1`, add a new mod with `$name` here is to allow duplicate naming
                    #[inline(never)]
                    pub(crate) fn __rust_end_short_backtrace<'tcx>(
                        tcx: TyCtxt<'tcx>,
                        span: Span,
                        key: Key<'tcx>,
                        mode: QueryMode,
                    ) -> Option<Erased<Value<'tcx>>> {
                        #[cfg(debug_assertions)]
                        let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();
                        crate::query::impl_::execution::execute_query_incr_inner(
                            &tcx.query_system.query_vtables.$name,
                            tcx,
                            span,
                            key,
                            mode
                        )
                    }
                }

                pub(crate) mod execute_query_non_incr {
                    use super::*;
                    use rustc_middle::queries::$name::{Key, Value};

                    #[inline(never)]
                    pub(crate) fn __rust_end_short_backtrace<'tcx>(
                        tcx: TyCtxt<'tcx>,
                        span: Span,
                        key: Key<'tcx>,
                        __mode: QueryMode,
                    ) -> Option<Erased<Value<'tcx>>> {
                        Some(crate::query::impl_::execution::execute_query_non_incr_inner(
                            &tcx.query_system.query_vtables.$name,
                            tcx,
                            span,
                            key,
                        ))
                    }
                }

                pub(crate) fn make_query_vtable<'tcx>()
                    -> QueryVTable<'tcx, rustc_middle::queries::$name::Cache<'tcx>, rustc_middle::queries::$name::Helper>
                {
                    QueryVTable {
                        name: stringify!($name),
                        eval_always: $eval_always,
                        depth_limit: $depth_limit,
                        feedable: $feedable,
                        dep_kind: rustc_middle::dep_graph::DepKind::$name,
                        state: Default::default(),
                        cache: Default::default(),

                        helper: Default::default(),

                        #[cfg($handle_cycle_error)]
                        handle_cycle_error_fn: |tcx, key, cycle, err| {
                            use rustc_middle::query::erase::erase_val;

                            erase_val($crate::query::impl_::handle_cycle_error::$name(tcx, key, cycle, err))
                        },
                        #[cfg(not($handle_cycle_error))]
                        handle_cycle_error_fn: |_tcx, _key, _cycle, err| {
                            $crate::query::impl_::handle_cycle_error::default(err)
                        },

                        create_tagged_key: TaggedQueryKey::$name,
                    }
                }

                /// Marker type that implements [`GetQueryVTable`] for this query.
                pub(crate) enum VTableGetter {}

                impl<'tcx> GetQueryVTable<'tcx> for VTableGetter {
                    type Cache = rustc_middle::queries::$name::Cache<'tcx>;
                    type Helper = rustc_middle::queries::$name::Helper;

                    #[inline(always)]
                    fn query_vtable(tcx: TyCtxt<'tcx>) -> &'tcx QueryVTable<'tcx, Self::Cache, Self::Helper> {
                        &tcx.query_system.query_vtables.$name
                    }
                }
            }
        )*

        pub(crate) fn make_query_vtables<'tcx>()
            -> rustc_middle::queries::QueryVTables<'tcx>
        {
            rustc_middle::queries::QueryVTables {
                $(
                    $name: crate::query::impl_::query_impl::$name::make_query_vtable(),
                )*
            }
        }

        /// Given a filter condition (e.g. `ALL` or `CACHE_ON_DISK`), a `tcx`,
        /// and a closure expression that accepts `&QueryVTable`, this macro
        /// calls that closure with each query vtable that satisfies the filter
        /// condition.
        ///
        /// This needs to be a macro, because the vtables can have different
        /// key/value/cache types for different queries.
        ///
        /// This macro's argument syntax is specifically intended to look like
        /// plain Rust code, so that `for_each_query_vtable!(..)` calls will be
        /// formatted by rustfmt.
        ///
        /// To avoid too much nested-macro complication, filter conditions are
        /// implemented by hand as needed.
        macro_rules! for_each_query_vtable {
            // Call with all queries.
            (ALL, $tcx:expr, $closure:expr) => {{
                let tcx: rustc_middle::ty::TyCtxt<'_> = $tcx;
                $(
                    let query: &rustc_middle::query::QueryVTable<'_, _, _> =
                        &tcx.query_system.query_vtables.$name;
                    $closure(query);
                )*
            }};

            // Only call with queries that can potentially cache to disk.
            //
            // This allows the use of trait bounds that only need to be satisfied
            // by the subset of queries that actually cache to disk.
            (CACHE_ON_DISK, $tcx:expr, $closure:expr) => {{
                let tcx: rustc_middle::ty::TyCtxt<'_> = $tcx;
                $(
                    #[cfg($cache_on_disk)]
                    {
                        let query: &rustc_middle::query::QueryVTable<'_, _, _> =
                            &tcx.query_system.query_vtables.$name;
                        $closure(query);
                    }
                )*
            }}
        }

        pub(crate) use for_each_query_vtable;
    }
}

rustc_middle::queries::rustc_with_all_queries! { define_queries! }
