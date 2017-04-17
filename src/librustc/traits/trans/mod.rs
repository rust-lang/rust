use dep_graph::{DepGraph, DepNode, DepTrackingMap, DepTrackingMapConfig};
use hir::def_id::DefId;
use std::cell::RefCell;
use std::marker::PhantomData;
use traits::Vtable;
use ty::{self, Ty};

/// Specializes caches used in trans -- in particular, they assume all
/// types are fully monomorphized and that free regions can be erased.
pub struct TransTraitCaches<'tcx> {
    pub trait_cache: RefCell<DepTrackingMap<TraitSelectionCache<'tcx>>>,
    pub project_cache: RefCell<DepTrackingMap<ProjectionCache<'tcx>>>,
}

impl<'tcx> TransTraitCaches<'tcx> {
    pub fn new(graph: DepGraph) -> Self {
        TransTraitCaches {
            trait_cache: RefCell::new(DepTrackingMap::new(graph.clone())),
            project_cache: RefCell::new(DepTrackingMap::new(graph)),
        }
    }
}

// Implement DepTrackingMapConfig for `trait_cache`
pub struct TraitSelectionCache<'tcx> {
    data: PhantomData<&'tcx ()>
}

impl<'tcx> DepTrackingMapConfig for TraitSelectionCache<'tcx> {
    type Key = ty::PolyTraitRef<'tcx>;
    type Value = Vtable<'tcx, ()>;
    fn to_dep_node(key: &ty::PolyTraitRef<'tcx>) -> DepNode<DefId> {
        key.to_poly_trait_predicate().dep_node()
    }
}

// # Global Cache

pub struct ProjectionCache<'gcx> {
    data: PhantomData<&'gcx ()>
}

impl<'gcx> DepTrackingMapConfig for ProjectionCache<'gcx> {
    type Key = Ty<'gcx>;
    type Value = Ty<'gcx>;
    fn to_dep_node(key: &Self::Key) -> DepNode<DefId> {
        // Ideally, we'd just put `key` into the dep-node, but we
        // can't put full types in there. So just collect up all the
        // def-ids of structs/enums as well as any traits that we
        // project out of. It doesn't matter so much what we do here,
        // except that if we are too coarse, we'll create overly
        // coarse edges between impls and the trans. For example, if
        // we just used the def-id of things we are projecting out of,
        // then the key for `<Foo as SomeTrait>::T` and `<Bar as
        // SomeTrait>::T` would both share a dep-node
        // (`TraitSelect(SomeTrait)`), and hence the impls for both
        // `Foo` and `Bar` would be considered inputs. So a change to
        // `Bar` would affect things that just normalized `Foo`.
        // Anyway, this heuristic is not ideal, but better than
        // nothing.
        let def_ids: Vec<DefId> =
            key.walk()
               .filter_map(|t| match t.sty {
                   ty::TyAdt(adt_def, _) => Some(adt_def.did),
                   ty::TyProjection(ref proj) => Some(proj.trait_ref.def_id),
                   _ => None,
               })
               .collect();

        DepNode::ProjectionCache { def_ids: def_ids }
    }
}

