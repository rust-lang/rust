use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::ty::query::query_storage;
use rustc_middle::ty::TyCtxt;
use rustc_query_system::query::{QueryCache, QueryCacheStore};

use std::any::type_name;
use std::mem;

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
    key_size: usize,
    key_type: &'static str,
    value_size: usize,
    value_type: &'static str,
    entry_count: usize,
    local_def_id_keys: Option<usize>,
}

fn stats<C>(name: &'static str, map: &QueryCacheStore<C>) -> QueryStats
where
    C: QueryCache,
{
    let mut stats = QueryStats {
        name,
        key_size: mem::size_of::<C::Key>(),
        key_type: type_name::<C::Key>(),
        value_size: mem::size_of::<C::Value>(),
        value_type: type_name::<C::Value>(),
        entry_count: 0,
        local_def_id_keys: None,
    };
    map.iter_results(&mut |key, _, _| {
        stats.entry_count += 1;
        key.key_stats(&mut stats)
    });
    stats
}

pub fn print_stats(tcx: TyCtxt<'_>) {
    let queries = query_stats(tcx);

    let mut query_key_sizes = queries.clone();
    query_key_sizes.sort_by_key(|q| q.key_size);
    eprintln!("\nLarge query keys:");
    for q in query_key_sizes.iter().rev().filter(|q| q.key_size > 8) {
        eprintln!("   {} - {} x {} - {}", q.name, q.key_size, q.entry_count, q.key_type);
    }

    let mut query_value_sizes = queries.clone();
    query_value_sizes.sort_by_key(|q| q.value_size);
    eprintln!("\nLarge query values:");
    for q in query_value_sizes.iter().rev().filter(|q| q.value_size > 8) {
        eprintln!("   {} - {} x {} - {}", q.name, q.value_size, q.entry_count, q.value_type);
    }

    let mut query_value_count = queries.clone();
    query_value_count.sort_by_key(|q| q.entry_count);
    eprintln!("\nQuery value count:");
    for q in query_value_count.iter().rev() {
        eprintln!("   {} - {}", q.name, q.entry_count);
    }

    let mut def_id_density: Vec<_> =
        queries.iter().filter(|q| q.local_def_id_keys.is_some()).collect();
    def_id_density.sort_by_key(|q| q.local_def_id_keys.unwrap());
    eprintln!("\nLocal DefId density:");
    let total = tcx.resolutions(()).definitions.def_index_count() as f64;
    for q in def_id_density.iter().rev() {
        let local = q.local_def_id_keys.unwrap();
        eprintln!("   {} - {} = ({}%)", q.name, local, (local as f64 * 100.0) / total);
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
                    query_storage::$name<'_>,
                >(
                    stringify!($name),
                    &tcx.query_caches.$name,
                ));
            )*

            queries
        }
    }
}

rustc_query_append! { [print_stats!][<'tcx>] }
