use crate::ty::TyCtxt;
use rustc_crate::cstore::{LibSource, LinkagePreference};
use rustc_span::def_id::{CrateNum, LOCAL_CRATE};

pub mod exported_symbols;
pub mod lang_items;
pub mod lib_features {
    use rustc_data_structures::fx::{FxHashMap, FxHashSet};
    use rustc_span::symbol::Symbol;

    #[derive(HashStable)]
    pub struct LibFeatures {
        // A map from feature to stabilisation version.
        pub stable: FxHashMap<Symbol, Symbol>,
        pub unstable: FxHashSet<Symbol>,
    }

    impl LibFeatures {
        pub fn to_vec(&self) -> Vec<(Symbol, Option<Symbol>)> {
            let mut all_features: Vec<_> = self
                .stable
                .iter()
                .map(|(f, s)| (*f, Some(*s)))
                .chain(self.unstable.iter().map(|f| (*f, None)))
                .collect();
            all_features.sort_unstable_by_key(|f| f.0.as_str());
            all_features
        }
    }
}
pub mod limits;
pub mod privacy;
pub mod region;
pub mod resolve_lifetime;
pub mod stability;

// This method is used when generating the command line to pass through to
// system linker. The linker expects undefined symbols on the left of the
// command line to be defined in libraries on the right, not the other way
// around. For more info, see some comments in the add_used_library function
// below.
//
// In order to get this left-to-right dependency ordering, we perform a
// topological sort of all crates putting the leaves at the right-most
// positions.
pub fn used_crates(tcx: TyCtxt<'_>, prefer: LinkagePreference) -> Vec<(CrateNum, LibSource)> {
    let mut libs = tcx
        .crates()
        .iter()
        .cloned()
        .filter_map(|cnum| {
            if tcx.dep_kind(cnum).macros_only() {
                return None;
            }
            let source = tcx.used_crate_source(cnum);
            let path = match prefer {
                LinkagePreference::RequireDynamic => source.dylib.clone().map(|p| p.0),
                LinkagePreference::RequireStatic => source.rlib.clone().map(|p| p.0),
            };
            let path = match path {
                Some(p) => LibSource::Some(p),
                None => {
                    if source.rmeta.is_some() {
                        LibSource::MetadataOnly
                    } else {
                        LibSource::None
                    }
                }
            };
            Some((cnum, path))
        })
        .collect::<Vec<_>>();
    let mut ordering = tcx.postorder_cnums(LOCAL_CRATE).to_owned();
    ordering.reverse();
    libs.sort_by_cached_key(|&(a, _)| ordering.iter().position(|x| *x == a));
    libs
}
