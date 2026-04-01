use rustc_middle::queries::TaggedQueryKey;
use rustc_middle::query::erase::{self, Erased};
use rustc_middle::query::{AsLocalQueryKey, QueryMode, QueryVTable};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::GetQueryVTable;

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
                    eval_always: $eval_always:literal,
                    feedable: $feedable:literal,
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
        // This macro expects to be expanded into `crate::query_impl`, which is this file.
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
                        crate::execution::execute_query_incr_inner(
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
                        Some(crate::execution::execute_query_non_incr_inner(
                            &tcx.query_system.query_vtables.$name,
                            tcx,
                            span,
                            key,
                        ))
                    }
                }

                /// Defines an `invoke_provider` function that calls the query's provider,
                /// to be used as a function pointer in the query's vtable.
                ///
                /// To mark a short-backtrace boundary, the function's actual name
                /// (after demangling) must be `__rust_begin_short_backtrace`.
                mod invoke_provider_fn {
                    use super::*;
                    use rustc_middle::queries::$name::{Key, Value, provided_to_erased};

                    #[inline(never)]
                    pub(crate) fn __rust_begin_short_backtrace<'tcx>(
                        tcx: TyCtxt<'tcx>,
                        key: Key<'tcx>,
                    ) -> Erased<Value<'tcx>> {
                        #[cfg(debug_assertions)]
                        let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();

                        // Call the actual provider function for this query.

                        #[cfg($separate_provide_extern)]
                        let provided_value = if let Some(local_key) = key.as_local_key() {
                            (tcx.query_system.local_providers.$name)(tcx, local_key)
                        } else {
                            (tcx.query_system.extern_providers.$name)(tcx, key)
                        };

                        #[cfg(not($separate_provide_extern))]
                        let provided_value = (tcx.query_system.local_providers.$name)(tcx, key);

                        rustc_middle::ty::print::with_reduced_queries!({
                            tracing::trace!(?provided_value);
                        });

                        // Erase the returned value, because `QueryVTable` uses erased values.
                        // For queries with `arena_cache`, this also arena-allocates the value.
                        provided_to_erased(tcx, provided_value)
                    }
                }

                fn will_cache_on_disk_for_key<'tcx>(
                    _key: rustc_middle::queries::$name::Key<'tcx>,
                ) -> bool {
                    cfg_select! {
                        // If a query has both `cache_on_disk` and `separate_provide_extern`, only
                        // disk-cache values for "local" keys, i.e. things in the current crate.
                        all($cache_on_disk, $separate_provide_extern) => {
                            AsLocalQueryKey::as_local_key(&_key).is_some()
                        }
                        all($cache_on_disk, not($separate_provide_extern)) => true,
                        not($cache_on_disk) => false,
                    }
                }

                pub(crate) fn make_query_vtable<'tcx>(incremental: bool)
                    -> QueryVTable<'tcx, rustc_middle::queries::$name::Cache<'tcx>>
                {
                    use rustc_middle::queries::$name::Value;

                    QueryVTable {
                        name: stringify!($name),
                        eval_always: $eval_always,
                        depth_limit: $depth_limit,
                        feedable: $feedable,
                        dep_kind: rustc_middle::dep_graph::DepKind::$name,
                        state: Default::default(),
                        cache: Default::default(),

                        invoke_provider_fn: self::invoke_provider_fn::__rust_begin_short_backtrace,

                        will_cache_on_disk_for_key_fn:
                            $crate::query_impl::$name::will_cache_on_disk_for_key,

                        #[cfg($cache_on_disk)]
                        try_load_from_disk_fn: |tcx, key, prev_index, index| {
                            use rustc_middle::queries::$name::{ProvidedValue, provided_to_erased};

                            // Check the cache-on-disk condition for this key.
                            if !$crate::query_impl::$name::will_cache_on_disk_for_key(key) {
                                return None;
                            }

                            let loaded_value: ProvidedValue<'tcx> =
                                $crate::plumbing::try_load_from_disk(tcx, prev_index, index)?;

                            // Arena-alloc the value if appropriate, and erase it.
                            Some(provided_to_erased(tcx, loaded_value))
                        },
                        #[cfg(not($cache_on_disk))]
                        try_load_from_disk_fn: |_tcx, _key, _prev_index, _index| None,

                        // The default just emits `err` and then aborts.
                        // `handle_cycle_error::specialize_query_vtables` overwrites this default
                        // for certain queries.
                        handle_cycle_error_fn: |_tcx, _key, _cycle, err| {
                            $crate::handle_cycle_error::default(err)
                        },

                        #[cfg($no_hash)]
                        hash_value_fn: None,
                        #[cfg(not($no_hash))]
                        hash_value_fn: Some(|hcx, erased_value: &erase::Erased<Value<'tcx>>| {
                            let value = erase::restore_val(*erased_value);
                            rustc_middle::dep_graph::hash_result(hcx, &value)
                        }),

                        format_value: |erased_value: &erase::Erased<Value<'tcx>>| {
                            format!("{:?}", erase::restore_val(*erased_value))
                        },
                        create_tagged_key: TaggedQueryKey::$name,
                        execute_query_fn: if incremental {
                            crate::query_impl::$name::execute_query_incr::__rust_end_short_backtrace
                        } else {
                            crate::query_impl::$name::execute_query_non_incr::__rust_end_short_backtrace
                        },
                    }
                }

                /// Marker type that implements [`GetQueryVTable`] for this query.
                pub(crate) enum VTableGetter {}

                impl<'tcx> GetQueryVTable<'tcx> for VTableGetter {
                    type Cache = rustc_middle::queries::$name::Cache<'tcx>;

                    #[inline(always)]
                    fn query_vtable(tcx: TyCtxt<'tcx>) -> &'tcx QueryVTable<'tcx, Self::Cache> {
                        &tcx.query_system.query_vtables.$name
                    }
                }
            }
        )*

        pub(crate) fn make_query_vtables<'tcx>(incremental: bool)
            -> rustc_middle::queries::QueryVTables<'tcx>
        {
            rustc_middle::queries::QueryVTables {
                $(
                    $name: crate::query_impl::$name::make_query_vtable(incremental),
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
                    let query: &rustc_middle::query::QueryVTable<'_, _> =
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
                        let query: &rustc_middle::query::QueryVTable<'_, _> =
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
