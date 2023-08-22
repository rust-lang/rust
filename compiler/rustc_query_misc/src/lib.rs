//! This crate provides miscellaneous functions for use in rustc_query_impl crate to avoid
//! code generation for it to grow too large.

#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![allow(internal_features)]

#[macro_use]
extern crate rustc_middle;

use crate::rustc_middle::ty::TyEncoder;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_index::Idx;
use rustc_middle::dep_graph;
use rustc_middle::query::cached;
use rustc_middle::query::erase::{erase, restore, Erase, EraseType};
use rustc_middle::query::on_disk_cache::AbsoluteBytePos;
use rustc_middle::query::on_disk_cache::CacheDecoder;
use rustc_middle::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex};
use rustc_middle::query::plumbing::QueryKeyStringCache;
use rustc_middle::query::queries;
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::DepNodeIndex;
use rustc_query_system::dep_graph::SerializedDepNodeIndex;
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{QueryCache, QueryInfo, QueryState};
use rustc_query_system::Value;
use rustc_serialize::Decodable;
use rustc_serialize::Encodable;
use rustc_span::ErrorGuaranteed;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

macro_rules! item_if_hashable {
    ([] { $($tokens:tt)* }) => {
        $($tokens)*};
    ([(no_hash) $($rest:tt)*] $tokens:tt) => {};
    ([$other:tt $($modifiers:tt)*] $tokens:tt) => {
        item_if_hashable! { [$($modifiers)*] $tokens }
    };
}

macro_rules! if_can_cache {
    ([]$yes:tt $no:tt) => {{
        $no
    }};
    ([(cache) $($rest:tt)*]$yes:tt $no:tt) => {{
        $yes
    }};
    ([$other:tt $($modifiers:tt)*]$yes:tt $no:tt) => {
        if_can_cache!([$($modifiers)*]$yes $no)
    };
}

macro_rules! item_if_can_cache {
    ([] $tokens:tt) => {};
    ([(cache) $($rest:tt)*] { $($tokens:tt)* }) => {
        $($tokens)*
    };
    ([$other:tt $($modifiers:tt)*] $tokens:tt) => {
        item_if_can_cache! { [$($modifiers)*] $tokens }
    };
}

macro_rules! expand_if_can_cache {
    ([], $tokens:expr) => {{
        None
    }};
    ([(cache) $($rest:tt)*], $tokens:expr) => {{
        Some($tokens)
    }};
    ([$other:tt $($modifiers:tt)*], $tokens:expr) => {
        expand_if_can_cache!([$($modifiers)*], $tokens)
    };
}

// NOTE: `$V` isn't used here, but we still need to match on it so it can be passed to other macros
// invoked by `rustc_query_append`.
macro_rules! query_functions {
    (
     $($(#[$attr:meta])*
        [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {

        pub mod query_utils { $(pub mod $name {
            // Actually used
            #[allow(unused_imports)]
            use super::super::*;

            pub fn try_load_from_disk<'tcx>(
                _tcx: TyCtxt<'tcx>,
                _key: &queries::$name::Key<'tcx>,
                _prev_index: SerializedDepNodeIndex,
                _index: DepNodeIndex,
            ) -> Option<Erase<queries::$name::Value<'tcx>>> {
                if_can_cache!([$($modifiers)*] {
                    if cached::$name(_tcx, _key) {
                        let value = try_load_from_disk_impl::<queries::$name::ProvidedValue<'tcx>>(
                            _tcx,
                            _prev_index,
                            _index,
                        );
                        value.map(|value| queries::$name::provided_to_erased(_tcx, value))
                    } else {
                        None
                    }
                } {
                    None
                })
            }

            pub fn loadable_from_disk<'tcx>(
                _tcx: TyCtxt<'tcx>,
                _key: &queries::$name::Key<'tcx>,
                _index: SerializedDepNodeIndex,
            ) -> bool {
                if_can_cache!([$($modifiers)*] {
                    cached::$name(_tcx, _key) && loadable_from_disk_impl(_tcx, _index)
                } {
                    false
                })
            }

            pub fn value_from_cycle_error<'tcx>(
                tcx: TyCtxt<'tcx>,
                cycle: &[QueryInfo],
                guar: ErrorGuaranteed,
            ) -> Erase<queries::$name::Value<'tcx>> {
                let result: queries::$name::Value<'tcx> = Value::from_cycle_error(tcx, cycle, guar);
                erase(result)
            }

            pub fn format_value<'tcx>(value: &Erase<queries::$name::Value<'tcx>>) -> String {
                format!("{:?}", restore::<queries::$name::Value<'tcx>>(*value))
            }

            pub fn alloc_self_profile_query_strings<'tcx>(tcx: TyCtxt<'tcx>, string_cache: &mut QueryKeyStringCache) {
                profiling_support::alloc_self_profile_query_strings_for_query_cache(
                    tcx,
                    stringify!($name),
                    &tcx.query_system.caches.$name,
                    string_cache,
                )
            }

            item_if_hashable! { [$($modifiers)*] {
                pub fn hash_result<'tcx>(
                    hcx: &mut StableHashingContext<'_>,
                    value: &Erase<queries::$name::Value<'tcx>>
                ) -> Fingerprint {
                    dep_graph::hash_result(hcx, &restore::<queries::$name::Value<'tcx>>(*value))
                }
            }}

            item_if_can_cache! { [$($modifiers)*] {
                #[inline(never)]
                pub(crate) fn encode_query_results<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    encoder: &mut CacheEncoder<'_, 'tcx>,
                    query_result_index: &mut EncodedDepNodeIndex
                ) {
                    encode_query_results_impl::<_, queries::$name::Value<'tcx>>(
                        tcx,
                        stringify!($name),
                        cached::$name,
                        &tcx.query_system.caches.$name,
                        &tcx.query_system.states.$name,
                        encoder,
                        query_result_index,
                    )
                }
            }}
        })*}

        // These arrays are used for iteration and can't be indexed by `DepKind`.

        const ALLOC_SELF_PROFILE_QUERY_STRINGS: &[
            for<'tcx> fn(TyCtxt<'tcx>, &mut QueryKeyStringCache)
        ] = &[$(query_utils::$name::alloc_self_profile_query_strings),*];

        const ENCODE_QUERY_RESULTS: &[
            Option<for<'tcx> fn(
                TyCtxt<'tcx>,
                &mut CacheEncoder<'_, 'tcx>,
                &mut EncodedDepNodeIndex)
            >
        ] = &[$(expand_if_can_cache!([$($modifiers)*], query_utils::$name::encode_query_results)),*];
    }
}

rustc_query_append! { query_functions! }

fn loadable_from_disk_impl<'tcx>(tcx: TyCtxt<'tcx>, id: SerializedDepNodeIndex) -> bool {
    if let Some(cache) = tcx.query_system.on_disk_cache.as_ref() {
        cache.loadable_from_disk(id)
    } else {
        false
    }
}

fn try_load_from_disk_impl<'tcx, V>(
    tcx: TyCtxt<'tcx>,
    prev_index: SerializedDepNodeIndex,
    index: DepNodeIndex,
) -> Option<V>
where
    V: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
{
    let on_disk_cache = tcx.query_system.on_disk_cache.as_ref()?;

    let prof_timer = tcx.prof.incr_cache_loading();

    // The call to `with_query_deserialization` enforces that no new `DepNodes`
    // are created during deserialization. See the docs of that method for more
    // details.
    let value = tcx
        .dep_graph
        .with_query_deserialization(|| on_disk_cache.try_load_query_result(tcx, prev_index));

    prof_timer.finish_with_query_invocation_id(index.into());

    value
}

#[inline(always)]
fn encode_query_results_impl<'tcx, C: QueryCache<Value = Erase<V>>, V>(
    tcx: TyCtxt<'tcx>,
    name: &'static str,
    cache_on_disk: impl Fn(TyCtxt<'tcx>, &C::Key) -> bool,
    cache: &C,
    state: &QueryState<C::Key>,
    encoder: &mut CacheEncoder<'_, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) where
    V: EraseType + for<'a> Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer = tcx.prof.generic_activity_with_arg("encode_query_results_for", name);

    assert!(state.all_inactive());
    cache.iter(&mut |key, value, dep_node| {
        if cache_on_disk(tcx, &key) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            encoder.encode_tagged(dep_node, &restore::<V>(*value));
        }
    });
}

pub fn encode_all_query_results<'tcx>(
    tcx: TyCtxt<'tcx>,
    encoder: &mut CacheEncoder<'_, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) {
    for encode in ENCODE_QUERY_RESULTS.iter().copied().flatten() {
        encode(tcx, encoder, query_result_index);
    }
}
