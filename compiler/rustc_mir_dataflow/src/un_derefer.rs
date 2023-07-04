use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::*;

use std::iter::once;

/// Used for reverting changes made by `DerefSeparator`
#[derive(Default, Debug)]
pub struct UnDerefer<'tcx> {
    deref_chains: FxHashMap<Local, Vec<Place<'tcx>>>,
}

impl<'tcx> UnDerefer<'tcx> {
    pub fn insert(&mut self, local: Local, reffed: Place<'tcx>) {
        let mut chain = self.deref_chains.remove(&reffed.local).unwrap_or_default();
        chain.push(reffed);
        self.deref_chains.insert(local, chain);
    }

    /// Returns the chain of places behind `DerefTemp` locals
    pub fn deref_chain(&self, local: Local) -> &[Place<'tcx>] {
        self.deref_chains.get(&local).map(Vec::as_slice).unwrap_or_default()
    }

    pub fn iter_projections(
        &self,
        place: PlaceRef<'tcx>,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> + DoubleEndedIterator + '_ {
        let deref_chain = self.deref_chain(place.local);

        deref_chain
            .iter()
            .map(Place::as_ref)
            .chain(once(place))
            .flat_map(|place| place.iter_projections())
    }
}
