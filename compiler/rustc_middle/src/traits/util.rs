use rustc_data_structures::stable_set::FxHashSet;

use crate::ty::{PolyTraitRef, TyCtxt};

/// Given a PolyTraitRef, get the PolyTraitRefs of the trait's (transitive) supertraits.
///
/// A simplfied version of the same function at `rustc_infer::traits::util::supertraits`.
pub fn supertraits<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: PolyTraitRef<'tcx>,
) -> impl Iterator<Item = PolyTraitRef<'tcx>> {
    Elaborator { tcx, visited: FxHashSet::from_iter([trait_ref]), stack: vec![trait_ref] }
}

struct Elaborator<'tcx> {
    tcx: TyCtxt<'tcx>,
    visited: FxHashSet<PolyTraitRef<'tcx>>,
    stack: Vec<PolyTraitRef<'tcx>>,
}

impl<'tcx> Elaborator<'tcx> {
    fn elaborate(&mut self, trait_ref: PolyTraitRef<'tcx>) {
        let supertrait_refs = self
            .tcx
            .super_predicates_of(trait_ref.def_id())
            .predicates
            .into_iter()
            .flat_map(|(pred, _)| {
                pred.subst_supertrait(self.tcx, &trait_ref).to_opt_poly_trait_pred()
            })
            .map(|t| t.map_bound(|pred| pred.trait_ref))
            .filter(|supertrait_ref| self.visited.insert(*supertrait_ref));

        self.stack.extend(supertrait_refs);
    }
}

impl<'tcx> Iterator for Elaborator<'tcx> {
    type Item = PolyTraitRef<'tcx>;

    fn next(&mut self) -> Option<PolyTraitRef<'tcx>> {
        if let Some(trait_ref) = self.stack.pop() {
            self.elaborate(trait_ref);
            Some(trait_ref)
        } else {
            None
        }
    }
}
