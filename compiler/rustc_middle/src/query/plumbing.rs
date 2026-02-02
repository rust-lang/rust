/// Calls either `query_ensure` or `query_ensure_error_guaranteed`, depending
/// on whether the list of modifiers contains `return_result_from_ensure_ok`.
macro_rules! query_ensure_select {
    ([]$($args:tt)*) => {
        crate::query::inner::query_ensure($($args)*)
    };
    ([(return_result_from_ensure_ok) $($rest:tt)*]$($args:tt)*) => {
        crate::query::inner::query_ensure_error_guaranteed($($args)*)
    };
    ([$other:tt $($modifiers:tt)*]$($args:tt)*) => {
        query_ensure_select!([$($modifiers)*]$($args)*)
    };
}

macro_rules! query_helper_param_ty {
    (DefId) => { impl IntoQueryParam<DefId> };
    (LocalDefId) => { impl IntoQueryParam<LocalDefId> };
    ($K:ty) => { $K };
}

macro_rules! query_if_arena {
    ([] $arena:tt $no_arena:tt) => {
        $no_arena
    };
    ([(arena_cache) $($rest:tt)*] $arena:tt $no_arena:tt) => {
        $arena
    };
    ([$other:tt $($modifiers:tt)*]$($args:tt)*) => {
        query_if_arena!([$($modifiers)*]$($args)*)
    };
}

/// If `separate_provide_extern`, then the key can be projected to its
/// local key via `<$K as AsLocalKey>::LocalKey`.
macro_rules! local_key_if_separate_extern {
    ([] $($K:tt)*) => {
        $($K)*
    };
    ([(separate_provide_extern) $($rest:tt)*] $($K:tt)*) => {
        <$($K)* as AsLocalKey>::LocalKey
    };
    ([$other:tt $($modifiers:tt)*] $($K:tt)*) => {
        local_key_if_separate_extern!([$($modifiers)*] $($K)*)
    };
}

macro_rules! separate_provide_extern_decl {
    ([][$name:ident]) => {
        ()
    };
    ([(separate_provide_extern) $($rest:tt)*][$name:ident]) => {
        for<'tcx> fn(
            TyCtxt<'tcx>,
            queries::$name::Key<'tcx>,
        ) -> queries::$name::ProvidedValue<'tcx>
    };
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        separate_provide_extern_decl!([$($modifiers)*][$($args)*])
    };
}

macro_rules! ensure_ok_result {
    ( [] ) => {
        ()
    };
    ( [(return_result_from_ensure_ok) $($rest:tt)*] ) => {
        Result<(), ErrorGuaranteed>
    };
    ( [$other:tt $($modifiers:tt)*] ) => {
        ensure_ok_result!( [$($modifiers)*] )
    };
}

macro_rules! separate_provide_extern_default {
    ([][$name:ident]) => {
        ()
    };
    ([(separate_provide_extern) $($rest:tt)*][$name:ident]) => {
        |_, key| $crate::query::plumbing::default_extern_query(stringify!($name), &key)
    };
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        separate_provide_extern_default!([$($modifiers)*][$($args)*])
    };
}

macro_rules! define_callbacks {
    (
        $(
            $(#[$attr:meta])*
            [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,
        )*
    ) => {

        #[allow(unused_lifetimes)]
        pub mod queries {
            $(pub mod $name {
                use super::super::*;
                use $crate::query::erase::{self, Erased};

                pub type Key<'tcx> = $($K)*;
                pub type Value<'tcx> = $V;

                pub type LocalKey<'tcx> = local_key_if_separate_extern!([$($modifiers)*] $($K)*);

                /// This type alias specifies the type returned from query providers and the type
                /// used for decoding. For regular queries this is the declared returned type `V`,
                /// but `arena_cache` will use `<V as ArenaCached>::Provided` instead.
                pub type ProvidedValue<'tcx> = query_if_arena!(
                    [$($modifiers)*]
                    (<$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Provided)
                    ($V)
                );

                /// This function takes `ProvidedValue` and converts it to an erased `Value` by
                /// allocating it on an arena if the query has the `arena_cache` modifier. The
                /// value is then erased and returned. This will happen when computing the query
                /// using a provider or decoding a stored result.
                #[inline(always)]
                pub fn provided_to_erased<'tcx>(
                    _tcx: TyCtxt<'tcx>,
                    provided_value: ProvidedValue<'tcx>,
                ) -> Erased<Value<'tcx>> {
                    // Store the provided value in an arena and get a reference
                    // to it, for queries with `arena_cache`.
                    let value: Value<'tcx> = query_if_arena!([$($modifiers)*]
                        {
                            use $crate::query::arena_cached::ArenaCached;

                            if mem::needs_drop::<<$V as ArenaCached<'tcx>>::Allocated>() {
                                <$V as ArenaCached>::alloc_in_arena(
                                    |v| _tcx.query_system.arenas.$name.alloc(v),
                                    provided_value,
                                )
                            } else {
                                <$V as ArenaCached>::alloc_in_arena(
                                    |v| _tcx.arena.dropless.alloc(v),
                                    provided_value,
                                )
                            }
                        }
                        // Otherwise, the provided value is the value.
                        (provided_value)
                    );
                    erase::erase_val(value)
                }

                pub type Storage<'tcx> = <$($K)* as keys::Key>::Cache<Erased<$V>>;

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
            })*
        }

        /// Holds per-query arenas for queries with the `arena_cache` modifier.
        #[derive(Default)]
        pub struct QueryArenas<'tcx> {
            $(
                $(#[$attr])*
                pub $name: query_if_arena!([$($modifiers)*]
                    // Use the `ArenaCached` helper trait to determine the arena's value type.
                    (TypedArena<<$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Allocated>)
                    // No arena for this query, so the field type is `()`.
                    ()
                ),
            )*
        }

        #[derive(Default)]
        pub struct QueryCaches<'tcx> {
            $($(#[$attr])* pub $name: queries::$name::Storage<'tcx>,)*
        }

        impl<'tcx> TyCtxtEnsureOk<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(
                self,
                key: query_helper_param_ty!($($K)*),
            ) -> ensure_ok_result!([$($modifiers)*]) {
                query_ensure_select!(
                    [$($modifiers)*]
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    key.into_query_param(),
                    false,
                )
            })*
        }

        impl<'tcx> TyCtxtEnsureDone<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                crate::query::inner::query_ensure(
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    key.into_query_param(),
                    true,
                );
            })*
        }

        impl<'tcx> TyCtxt<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            #[must_use]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V
            {
                self.at(DUMMY_SP).$name(key)
            })*
        }

        impl<'tcx> TyCtxtAt<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V
            {
                use $crate::query::{erase, inner};

                erase::restore_val::<$V>(inner::query_get_at(
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    self.span,
                    key.into_query_param(),
                ))
            })*
        }

        /// Holds a `QueryVTable` for each query.
        ///
        /// ("Per" just makes this pluralized name more visually distinct.)
        pub struct PerQueryVTables<'tcx> {
            $(
                pub $name: ::rustc_middle::query::plumbing::QueryVTable<'tcx, queries::$name::Storage<'tcx>>,
            )*
        }

        #[derive(Default)]
        pub struct QueryStates<'tcx> {
            $(
                pub $name: QueryState<'tcx, $($K)*>,
            )*
        }

        pub struct Providers {
            $(pub $name: for<'tcx> fn(
                TyCtxt<'tcx>,
                queries::$name::LocalKey<'tcx>,
            ) -> queries::$name::ProvidedValue<'tcx>,)*
        }

        pub struct ExternProviders {
            $(pub $name: separate_provide_extern_decl!([$($modifiers)*][$name]),)*
        }

        impl Default for Providers {
            fn default() -> Self {
                Providers {
                    $($name: |_, key| $crate::query::plumbing::default_query(stringify!($name), &key)),*
                }
            }
        }

        impl Default for ExternProviders {
            fn default() -> Self {
                ExternProviders {
                    $($name: separate_provide_extern_default!([$($modifiers)*][$name]),)*
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

        pub struct QueryEngine {
            $(pub $name: for<'tcx> fn(
                TyCtxt<'tcx>,
                Span,
                queries::$name::Key<'tcx>,
                QueryMode,
            ) -> Option<$crate::query::erase::Erased<$V>>,)*
        }
    };
}

macro_rules! hash_result {
    ([]) => {{
        Some(dep_graph::hash_result)
    }};
    ([(no_hash) $($rest:tt)*]) => {{
        None
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        hash_result!([$($modifiers)*])
    };
}

macro_rules! define_feedable {
    ($($(#[$attr:meta])* [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {
        $(impl<'tcx, K: IntoQueryParam<$($K)*> + Copy> TyCtxtFeed<'tcx, K> {
            $(#[$attr])*
            #[inline(always)]
            pub fn $name(self, value: queries::$name::ProvidedValue<'tcx>) {
                let key = self.key().into_query_param();

                let tcx = self.tcx;
                let erased = queries::$name::provided_to_erased(tcx, value);
                let cache = &tcx.query_system.caches.$name;

                let dep_kind: dep_graph::DepKind = dep_graph::dep_kinds::$name;
                let hasher: Option<fn(&mut StableHashingContext<'_>, &_) -> _> = hash_result!([$($modifiers)*]);

                $crate::query::inner::query_feed(
                    tcx,
                    dep_kind,
                    hasher,
                    cache,
                    key,
                    erased,
                );
            }
        })*
    }
}

// Each of these queries corresponds to a function pointer field in the
// `Providers` struct for requesting a value of that type, and a method
// on `tcx: TyCtxt` (and `tcx.at(span)`) for doing that request in a way
// which memoizes and does dep-graph tracking, wrapping around the actual
// `Providers` that the driver creates (using several `rustc_*` crates).
//
// The result type of each query must implement `Clone`, and additionally
// `ty::query::values::Value`, which produces an appropriate placeholder
// (error) value if the query resulted in a query cycle.
// Queries marked with `cycle_fatal` do not need the latter implementation,
// as they will raise an fatal error on query cycles instead.
