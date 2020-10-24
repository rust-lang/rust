use crate::ty::query::queries;
use crate::ty::TyCtxt;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_query_system::query::{QueryAccessors, QueryCache, QueryContext, QueryState};

use std::any::type_name;
use std::hash::Hash;
use std::mem;
#[cfg(debug_assertions)]
use std::sync::atomic::Ordering;

trait KeyStats {
    fn key_stats(&self, stats: &mut QueryStats);
}

impl<T> KeyStats for T {
    default fn key_stats(&self, _: &mut QueryStats) {}
}

impl KeyStats for DefId {
    fn key_stats(&self, stats: &mut QueryStats) {
        if self.krate == LOCAL_CRATE {
            stats.local_def_id_keys = Some(stats.local_def_id_keys.unwrap_or(0) + 1);
        }
    }
}

#[derive(Clone)]
struct QueryStats {
    name: &'static str,
    cache_hits: usize,
    key_size: usize,
    key_type: &'static str,
    value_size: usize,
    value_type: &'static str,
    entry_count: usize,
    local_def_id_keys: Option<usize>,
}

fn stats<D, Q, C>(name: &'static str, map: &QueryState<D, Q, C>) -> QueryStats
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    C: QueryCache,
{
    let mut stats = QueryStats {
        name,
        #[cfg(debug_assertions)]
        cache_hits: map.cache_hits.load(Ordering::Relaxed),
        #[cfg(not(debug_assertions))]
        cache_hits: 0,
        key_size: mem::size_of::<C::Key>(),
        key_type: type_name::<C::Key>(),
        value_size: mem::size_of::<C::Value>(),
        value_type: type_name::<C::Value>(),
        entry_count: map.iter_results(|results| results.count()),
        local_def_id_keys: None,
    };
    map.iter_results(|results| {
        for (key, _, _) in results {
            key.key_stats(&mut stats)
        }
    });
    stats
}

pub fn print_stats(tcx: TyCtxt<'_>) {
    let queries = query_stats(tcx);

    if cfg!(debug_assertions) {
        let hits: usize = queries.iter().map(|s| s.cache_hits).sum();
        let results: usize = queries.iter().map(|s| s.entry_count).sum();
        println!("\nQuery cache hit rate: {}", hits as f64 / (hits + results) as f64);
    }

    let mut query_key_sizes = queries.clone();
    query_key_sizes.sort_by_key(|q| q.key_size);
    println!("\nLarge query keys:");
    for q in query_key_sizes.iter().rev().filter(|q| q.key_size > 8) {
        println!("   {} - {} x {} - {}", q.name, q.key_size, q.entry_count, q.key_type);
    }

    let mut query_value_sizes = queries.clone();
    query_value_sizes.sort_by_key(|q| q.value_size);
    println!("\nLarge query values:");
    for q in query_value_sizes.iter().rev().filter(|q| q.value_size > 8) {
        println!("   {} - {} x {} - {}", q.name, q.value_size, q.entry_count, q.value_type);
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

    let mut def_id_density: Vec<_> =
        queries.iter().filter(|q| q.local_def_id_keys.is_some()).collect();
    def_id_density.sort_by_key(|q| q.local_def_id_keys.unwrap());
    println!("\nLocal DefId density:");
    let total = tcx.hir().definitions().def_index_count() as f64;
    for q in def_id_density.iter().rev() {
        let local = q.local_def_id_keys.unwrap();
        println!("   {} - {} = ({}%)", q.name, local, (local as f64 * 100.0) / total);
    }
}

macro_rules! print_stats {
    (<$tcx:tt>
        $($(#[$attr:meta])* [$($modifiers:tt)*] fn $name:ident($K:ty) -> $V:ty,)*
    ) => {
        fn query_stats(tcx: TyCtxt<'_>) -> Vec<QueryStats> {
            let mut queries = Vec::new();

            $(
                queries.push(stats::<
                    crate::dep_graph::DepKind,
                    <TyCtxt<'_> as QueryContext>::Query,
                    <queries::$name<'_> as QueryAccessors<TyCtxt<'_>>>::Cache,
                >(
                    stringify!($name),
                    &tcx.queries.$name,
                ));
            )*

            queries
        }
    }
}

rustc_query_append! { [print_stats!][<'tcx>] }
