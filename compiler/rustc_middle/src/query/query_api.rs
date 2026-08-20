macro_rules! maybe_into_query_key {
    (DefId) => { impl $crate::query::IntoQueryKey<DefId> };
    (LocalDefId) => { impl $crate::query::IntoQueryKey<LocalDefId> };
    ($K:ty) => { $K };
}

macro_rules! define_query_api {
    (
        // You might expect the key to be `$K:ty`, but it needs to be `$($K:tt)*` so that
        // `maybe_into_query_key!` can match on specific type names.
        queries {
            $(
                $(#[$attr:meta])*
                fn $name:ident($($K:tt)*) -> $V:ty
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
        $(
            pub mod $name {
                use super::*;
                use $crate::query::erase::{self, Erased};

                pub type Key<'tcx> = $($K)*;
                pub type Value<'tcx> = $V;

                /// Key type used by provider functions in `local_providers`.
                /// This query has the `separate_provide_extern` modifier.
                #[cfg($separate_provide_extern)]
                pub type LocalKey<'tcx> =
                    <Key<'tcx> as $crate::query::QueryKey>::LocalQueryKey;
                /// Key type used by provider functions in `local_providers`.
                #[cfg(not($separate_provide_extern))]
                pub type LocalKey<'tcx> = Key<'tcx>;

                /// Type returned from query providers and loaded from disk-cache.
                #[cfg($arena_cache)]
                pub type ProvidedValue<'tcx> =
                    <Value<'tcx> as $crate::query::arena_cached::ArenaCached<'tcx>>::Provided;
                /// Type returned from query providers and loaded from disk-cache.
                #[cfg(not($arena_cache))]
                pub type ProvidedValue<'tcx> = Value<'tcx>;

                pub type Cache<'tcx> =
                    <Key<'tcx> as $crate::query::QueryKey>::Cache<Erased<Value<'tcx>>>;

                /// This helper function takes a value returned by the query provider
                /// (or loaded from disk, or supplied by query feeding), allocates
                /// it in an arena if requested by the `arena_cache` modifier, and
                /// then returns an erased copy of it.
                #[inline(always)]
                pub fn provided_to_erased<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    provided_value: ProvidedValue<'tcx>,
                ) -> Erased<Value<'tcx>> {
                    // For queries with the `arena_cache` modifier, store the
                    // provided value in an arena and get a reference to it.
                    #[cfg($arena_cache)]
                    let value: Value<'tcx> = {
                        use $crate::query::arena_cached::ArenaCached;
                        <Value<'tcx> as ArenaCached>::alloc_in_arena(
                            tcx,
                            &tcx.query_system.arenas.$name,
                            provided_value,
                        )
                    };

                    // Otherwise, the provided value is the value (and `tcx` is unused).
                    #[cfg(not($arena_cache))]
                    let value: Value<'tcx> = {
                        let _ = tcx;
                        provided_value
                    };

                    erase::erase_val(value)
                }

                // Ensure that keys grow no larger than 88 bytes by accident.
                // Increase this limit if necessary, but do try to keep the size low if possible
                #[cfg(target_pointer_width = "64")]
                const _: () = {
                    if size_of::<Key<'static>>() > 88 {
                        panic!("{}", concat!(
                            "the query `",
                            stringify!($name),
                            "` has a key type `",
                            stringify!($($K)*),
                            "` that is too large"
                        ));
                    }
                };

                // Ensure that values grow no larger than 64 bytes by accident.
                // Increase this limit if necessary, but do try to keep the size low if possible
                #[cfg(target_pointer_width = "64")]
                #[cfg(not(feature = "rustc_randomized_layouts"))]
                const _: () = {
                    if size_of::<Value<'static>>() > 64 {
                        panic!("{}", concat!(
                            "the query `",
                            stringify!($name),
                            "` has a value type `",
                            stringify!($V),
                            "` that is too large"
                        ));
                    }
                };
            }
        )*

        /// Identifies a query by kind and key. This is in contrast to `QueryJobId` which is just a
        /// number.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug)]
        pub enum TaggedQueryKey<'tcx> {
            $(
                $name($name::Key<'tcx>),
            )*
        }

        impl<'tcx> TaggedQueryKey<'tcx> {
            /// Returns the name of the query this key is tagged with.
            ///
            /// This is useful for error/debug output, but don't use it to check for
            /// specific query names. Instead, match on the `TaggedQueryKey` variant.
            pub fn query_name(&self) -> &'static str {
                match self {
                    $(
                        TaggedQueryKey::$name(_) => stringify!($name),
                    )*
                }
            }

            /// Formats a human-readable description of this query and its key, as
            /// specified by the `desc` query modifier.
            ///
            /// Used when reporting query cycle errors and similar problems.
            pub fn description(&self, tcx: TyCtxt<'tcx>) -> String {
                let (name, description) = ty::print::with_no_queries!(match self {
                    $(
                        TaggedQueryKey::$name(key) => (stringify!($name), ($desc)(tcx, *key)),
                    )*
                });
                if tcx.sess.verbose_internals() {
                    format!("{description} [{name:?}]")
                } else {
                    description
                }
            }

            /// Calls `self.description` or returns a fallback if there was a fatal error
            pub fn catch_description(&self, tcx: TyCtxt<'tcx>) -> String {
                catch_fatal_errors(|| self.description(tcx)).unwrap_or_else(|_| format!("<error describing {}>", self.query_name()))
            }

            /// Returns the default span for this query if `span` is a dummy span.
            pub fn default_span(&self, tcx: TyCtxt<'tcx>, span: Span) -> Span {
                if !span.is_dummy() {
                    return span
                }
                if let TaggedQueryKey::def_span(..) = self {
                    // The `def_span` query is used to calculate `default_span`,
                    // so exit to avoid infinite recursion.
                    return DUMMY_SP
                }
                match self {
                    $(
                        TaggedQueryKey::$name(key) =>
                            $crate::query::QueryKey::default_span(key, tcx),
                    )*
                }
            }

            /// Calls `self.default_span` or returns `DUMMY_SP` if there was a fatal error
            pub fn catch_default_span(&self, tcx: TyCtxt<'tcx>, span: Span) -> Span {
                catch_fatal_errors(|| self.default_span(tcx, span)).unwrap_or(DUMMY_SP)
            }
        }

        /// Holds a `QueryVTable` for each query.
        pub struct QueryVTables<'tcx> {
            $(
                pub $name: $crate::query::QueryVTable<'tcx, $name::Cache<'tcx>>,
            )*
        }

        /// Holds per-query arenas for queries with the `arena_cache` modifier.
        #[derive(Default)]
        pub struct QueryArenas<'tcx> {
            $(
                // Use the `ArenaCached` helper trait to determine the arena's value type.
                #[cfg($arena_cache)]
                pub $name: TypedArena<
                    <$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Allocated,
                >,
            )*
        }

        pub struct Providers {
            $(
                /// This is the provider for the query. Use `Find references` on this to
                /// navigate between the provider assignment and the query definition.
                pub $name: for<'tcx> fn(
                    TyCtxt<'tcx>,
                    $name::LocalKey<'tcx>,
                ) -> $name::ProvidedValue<'tcx>,
            )*
        }

        pub struct ExternProviders {
            $(
                #[cfg($separate_provide_extern)]
                pub $name: for<'tcx> fn(
                    TyCtxt<'tcx>,
                    $name::Key<'tcx>,
                ) -> $name::ProvidedValue<'tcx>,
            )*
        }

        impl Default for Providers {
            fn default() -> Self {
                Providers {
                    $(
                        $name: |_, key| {
                            $crate::query::query_api::default_query(stringify!($name), &key)
                        },
                    )*
                }
            }
        }

        impl Default for ExternProviders {
            fn default() -> Self {
                ExternProviders {
                    $(
                        #[cfg($separate_provide_extern)]
                        $name: |_, key| $crate::query::query_api::default_extern_query(
                            stringify!($name),
                            &key,
                        ),
                    )*
                }
            }
        }

        impl Copy for Providers {}
        impl Clone for Providers {
            fn clone(&self) -> Self { *self }
        }

        impl Copy for ExternProviders {}
        impl Clone for ExternProviders {
            fn clone(&self) -> Self { *self }
        }

        impl<'tcx> TyCtxt<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                #[must_use]
                pub fn $name(self, key: maybe_into_query_key!($($K)*)) -> $V {
                    self.at(DUMMY_SP).$name(key)
                }
            )*
        }

        impl<'tcx> $crate::query::TyCtxtAt<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, key: maybe_into_query_key!($($K)*)) -> $V {
                    $crate::query::erase::restore_val::<$V>($crate::query::calls::query_get_at(
                        self.tcx,
                        self.span,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryKey::into_query_key(key),
                    ))
                }
            )*
        }

        impl<'tcx> $crate::query::TyCtxtEnsureOk<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, key: maybe_into_query_key!($($K)*)) {
                    $crate::query::calls::query_ensure_ok(
                        self.tcx,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryKey::into_query_key(key),
                    )
                }
            )*
        }

        // Only defined when the `returns_error_guaranteed` modifier is present.
        impl<'tcx> $crate::query::TyCtxtEnsureResult<'tcx> {
            $(
                #[cfg($returns_error_guaranteed)]
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(
                    self,
                    key: maybe_into_query_key!($($K)*),
                ) -> Result<(), rustc_errors::ErrorGuaranteed> {
                    $crate::query::calls::query_ensure_result(
                        self.tcx,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryKey::into_query_key(key),
                    )
                }
            )*
        }

        impl<'tcx> $crate::query::TyCtxtEnsureDone<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, key: maybe_into_query_key!($($K)*)) {
                    // This has the same implementation as `tcx.$query(..)` as it isn't currently
                    // beneficial to have an optimized variant due to how promotion works.
                    let _ = self.tcx.$name(key);
                }
            )*
        }

        $(
            // Only defined when the `feedable` modifier is present.
            #[cfg($feedable)]
            impl<'tcx, K: $crate::query::IntoQueryKey<$name::Key<'tcx>> + Copy>
                TyCtxtFeed<'tcx, K>
            {
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, value: $name::ProvidedValue<'tcx>) {
                    $crate::query::calls::query_feed(
                        self.tcx,
                        &self.tcx.query_system.query_vtables.$name,
                        self.key().into_query_key(),
                        $name::provided_to_erased(self.tcx, value),
                    );
                }
            }
        )*
    };
}

// Re-export `macro_rules!` macros as normal items, so that they can be imported normally.
pub(crate) use define_query_api;
pub(crate) use maybe_into_query_key;

#[cold]
pub(crate) fn default_query(name: &str, key: &dyn std::fmt::Debug) -> ! {
    bug!(
        "`tcx.{name}({key:?})` is not supported for this key;\n\
        hint: Queries can be either made to the local crate, or the external crate. \
        This error means you tried to use it for one that's not supported.\n\
        If that's not the case, {name} was likely never assigned to a provider function.\n",
    )
}

#[cold]
pub(crate) fn default_extern_query(name: &str, key: &dyn std::fmt::Debug) -> ! {
    bug!(
        "`tcx.{name}({key:?})` unsupported by its crate; \
         perhaps the `{name}` query was never assigned a provider function",
    )
}
