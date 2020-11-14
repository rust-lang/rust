use crate::ty::TyCtxt;
use rustc_crate::cstore::{LibSource, LinkagePreference};
use rustc_span::def_id::{CrateNum, LOCAL_CRATE};

pub mod exported_symbols;
pub mod lang_items;
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
