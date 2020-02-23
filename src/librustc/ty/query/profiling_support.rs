use crate::hir::map::definitions::DefPathData;
use crate::ty::context::TyCtxt;
use crate::ty::query::config::QueryAccessors;
use crate::ty::query::plumbing::QueryState;
use measureme::{StringComponent, StringId};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::profiling::SelfProfiler;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX, LOCAL_CRATE};
use std::fmt::Debug;
use std::io::Write;

pub struct QueryKeyStringCache {
    def_id_cache: FxHashMap<DefId, StringId>,
}

impl QueryKeyStringCache {
    pub fn new() -> QueryKeyStringCache {
        QueryKeyStringCache { def_id_cache: Default::default() }
    }
}

pub struct QueryKeyStringBuilder<'p, 'c, 'tcx> {
    profiler: &'p SelfProfiler,
    tcx: TyCtxt<'tcx>,
    string_cache: &'c mut QueryKeyStringCache,
}

impl<'p, 'c, 'tcx> QueryKeyStringBuilder<'p, 'c, 'tcx> {
    pub fn new(
        profiler: &'p SelfProfiler,
        tcx: TyCtxt<'tcx>,
        string_cache: &'c mut QueryKeyStringCache,
    ) -> QueryKeyStringBuilder<'p, 'c, 'tcx> {
        QueryKeyStringBuilder { profiler, tcx, string_cache }
    }

    // The current implementation is rather crude. In the future it might be a
    // good idea to base this on `ty::print` in order to get nicer and more
    // efficient query keys.
    fn def_id_to_string_id(&mut self, def_id: DefId) -> StringId {
        if let Some(&string_id) = self.string_cache.def_id_cache.get(&def_id) {
            return string_id;
        }

        let def_key = self.tcx.def_key(def_id);

        let (parent_string_id, start_index) = match def_key.parent {
            Some(parent_index) => {
                let parent_def_id = DefId { index: parent_index, krate: def_id.krate };

                (self.def_id_to_string_id(parent_def_id), 0)
            }
            None => (StringId::INVALID, 2),
        };

        let dis_buffer = &mut [0u8; 16];
        let name;
        let dis;
        let end_index;

        match def_key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                name = self.tcx.original_crate_name(def_id.krate).as_str();
                dis = "";
                end_index = 3;
            }
            other => {
                name = other.as_symbol().as_str();
                if def_key.disambiguated_data.disambiguator == 0 {
                    dis = "";
                    end_index = 3;
                } else {
                    write!(&mut dis_buffer[..], "[{}]", def_key.disambiguated_data.disambiguator)
                        .unwrap();
                    let end_of_dis = dis_buffer.iter().position(|&c| c == b']').unwrap();
                    dis = std::str::from_utf8(&dis_buffer[..end_of_dis + 1]).unwrap();
                    end_index = 4;
                }
            }
        }

        let components = [
            StringComponent::Ref(parent_string_id),
            StringComponent::Value("::"),
            StringComponent::Value(&name[..]),
            StringComponent::Value(dis),
        ];

        let string_id = self.profiler.alloc_string(&components[start_index..end_index]);

        self.string_cache.def_id_cache.insert(def_id, string_id);

        string_id
    }
}

pub trait IntoSelfProfilingString {
    fn to_self_profile_string(&self, builder: &mut QueryKeyStringBuilder<'_, '_, '_>) -> StringId;
}

// The default implementation of `IntoSelfProfilingString` just uses `Debug`
// which is slow and causes lots of duplication of string data.
// The specialized impls below take care of making the `DefId` case more
// efficient.
impl<T: Debug> IntoSelfProfilingString for T {
    default fn to_self_profile_string(
        &self,
        builder: &mut QueryKeyStringBuilder<'_, '_, '_>,
    ) -> StringId {
        let s = format!("{:?}", self);
        builder.profiler.alloc_string(&s[..])
    }
}

impl IntoSelfProfilingString for DefId {
    fn to_self_profile_string(&self, builder: &mut QueryKeyStringBuilder<'_, '_, '_>) -> StringId {
        builder.def_id_to_string_id(*self)
    }
}

impl IntoSelfProfilingString for CrateNum {
    fn to_self_profile_string(&self, builder: &mut QueryKeyStringBuilder<'_, '_, '_>) -> StringId {
        builder.def_id_to_string_id(DefId { krate: *self, index: CRATE_DEF_INDEX })
    }
}

impl IntoSelfProfilingString for DefIndex {
    fn to_self_profile_string(&self, builder: &mut QueryKeyStringBuilder<'_, '_, '_>) -> StringId {
        builder.def_id_to_string_id(DefId { krate: LOCAL_CRATE, index: *self })
    }
}

impl<T0, T1> IntoSelfProfilingString for (T0, T1)
where
    T0: IntoSelfProfilingString + Debug,
    T1: IntoSelfProfilingString + Debug,
{
    default fn to_self_profile_string(
        &self,
        builder: &mut QueryKeyStringBuilder<'_, '_, '_>,
    ) -> StringId {
        let val0 = self.0.to_self_profile_string(builder);
        let val1 = self.1.to_self_profile_string(builder);

        let components = &[
            StringComponent::Value("("),
            StringComponent::Ref(val0),
            StringComponent::Value(","),
            StringComponent::Ref(val1),
            StringComponent::Value(")"),
        ];

        builder.profiler.alloc_string(components)
    }
}

/// Allocate the self-profiling query strings for a single query cache. This
/// method is called from `alloc_self_profile_query_strings` which knows all
/// the queries via macro magic.
pub(super) fn alloc_self_profile_query_strings_for_query_cache<'tcx, Q>(
    tcx: TyCtxt<'tcx>,
    query_name: &'static str,
    query_state: &QueryState<'tcx, Q>,
    string_cache: &mut QueryKeyStringCache,
) where
    Q: QueryAccessors<'tcx>,
{
    tcx.prof.with_profiler(|profiler| {
        let event_id_builder = profiler.event_id_builder();

        // Walk the entire query cache and allocate the appropriate
        // string representations. Each cache entry is uniquely
        // identified by its dep_node_index.
        if profiler.query_key_recording_enabled() {
            let mut query_string_builder = QueryKeyStringBuilder::new(profiler, tcx, string_cache);

            let query_name = profiler.get_or_alloc_cached_string(query_name);

            // Since building the string representation of query keys might
            // need to invoke queries itself, we cannot keep the query caches
            // locked while doing so. Instead we copy out the
            // `(query_key, dep_node_index)` pairs and release the lock again.
            let query_keys_and_indices: Vec<_> = query_state
                .iter_results(|results| results.map(|(k, _, i)| (k.clone(), i)).collect());

            // Now actually allocate the strings. If allocating the strings
            // generates new entries in the query cache, we'll miss them but
            // we don't actually care.
            for (query_key, dep_node_index) in query_keys_and_indices {
                // Translate the DepNodeIndex into a QueryInvocationId
                let query_invocation_id = dep_node_index.into();

                // Create the string version of the query-key
                let query_key = query_key.to_self_profile_string(&mut query_string_builder);
                let event_id = event_id_builder.from_label_and_arg(query_name, query_key);

                // Doing this in bulk might be a good idea:
                profiler.map_query_invocation_id_to_string(
                    query_invocation_id,
                    event_id.to_string_id(),
                );
            }
        } else {
            // In this branch we don't allocate query keys
            let query_name = profiler.get_or_alloc_cached_string(query_name);
            let event_id = event_id_builder.from_label(query_name).to_string_id();

            query_state.iter_results(|results| {
                let query_invocation_ids: Vec<_> = results.map(|v| v.2.into()).collect();

                profiler.bulk_map_query_invocation_id_to_single_string(
                    query_invocation_ids.into_iter(),
                    event_id,
                );
            });
        }
    });
}
