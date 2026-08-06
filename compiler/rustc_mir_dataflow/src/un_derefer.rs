use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::*;

/// Used for reverting changes made by `DerefSeparator`
#[derive(Default, Debug)]
pub(crate) struct UnDerefer<'tcx> {
    deref_chains: FxHashMap<Local, Vec<PlaceRef<'tcx>>>,
}

impl<'tcx> UnDerefer<'tcx> {
    #[inline]
    pub(crate) fn insert(&mut self, local: Local, reffed: PlaceRef<'tcx>) {
        let mut chain = self.deref_chains.remove(&reffed.local).unwrap_or_default();
        chain.push(reffed);
        self.deref_chains.insert(local, chain);
    }

    /// Returns the chain of places behind `DerefTemp` locals
    #[inline]
    pub(crate) fn deref_chain(&self, local: Local) -> &[PlaceRef<'tcx>] {
        self.deref_chains.get(&local).map(Vec::as_slice).unwrap_or_default()
    }

    /// Iterates over the projections of a place and its deref chain.
    ///
    /// See [`PlaceRef::iter_projections`]
    #[inline]
    pub(crate) fn iter_projections(
        &self,
        place: PlaceRef<'tcx>,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> {
        self.co_iter_projections(place).into_iter(self)
    }

    /// Like [`UnDerefer::iter_projections`], but doesn't capture the self reference in the returned type.
    /// Instead, getting the next element requires passing a reference to this `UnDerefer` for each iteration.
    #[inline]
    pub(crate) fn co_iter_projections(&self, place: PlaceRef<'tcx>) -> ProjectionCoroutine<'tcx> {
        ProjectionCoroutine::new(self.deref_chain(place.local), place)
    }
}

pub(crate) enum ProjectionCoroutine<'tcx> {
    InChain { current: PlaceRef<'tcx>, proj_idx: usize, last: PlaceRef<'tcx>, chain_idx: usize },
    Last { last: PlaceRef<'tcx>, proj_idx: usize },
    Finished,
}

impl<'tcx> ProjectionCoroutine<'tcx> {
    fn new(deref_chain: &[PlaceRef<'tcx>], place: PlaceRef<'tcx>) -> Self {
        if let &[first, ..] = deref_chain {
            Self::InChain { current: first, proj_idx: 0, last: place, chain_idx: 0 }
        } else {
            if place.as_local().is_none() {
                Self::Last { last: place, proj_idx: 0 }
            } else {
                Self::Finished
            }
        }
    }

    fn advance_chain(&mut self, un_derefer: &UnDerefer<'tcx>) {
        *self = match self {
            &mut Self::InChain { last, chain_idx, .. } => {
                let chain = un_derefer.deref_chain(last.local);

                if let Some(&next) = chain.get(chain_idx + 1) {
                    Self::InChain { current: next, proj_idx: 0, last, chain_idx: chain_idx + 1 }
                } else {
                    Self::Last { last, proj_idx: 0 }
                }
            }
            &mut Self::Last { .. } => Self::Finished,
            &mut Self::Finished => unreachable!(),
        }
    }

    /// Returns the next `PlaceRef` and `PlaceElem` pair,
    /// or `None` if the entire place has been iterated through.
    ///
    /// `un_derefer` must be the same instance that produced `self`.
    pub(crate) fn next(
        &mut self,
        un_derefer: &UnDerefer<'tcx>,
    ) -> Option<(PlaceRef<'tcx>, PlaceElem<'tcx>)> {
        let (place, proj_idx) = match self {
            &mut Self::InChain { current, ref mut proj_idx, .. } => (current, proj_idx),
            &mut Self::Last { last, ref mut proj_idx } => (last, proj_idx),
            &mut Self::Finished => return None,
        };

        // the projection should never be empty except for a bare local which is handled in new
        let partial_place =
            PlaceRef { local: place.local, projection: &place.projection[..*proj_idx] };
        let elem = place.projection[*proj_idx];

        if *proj_idx == place.projection.len() - 1 {
            self.advance_chain(un_derefer);
        } else {
            *proj_idx += 1;
        }

        Some((partial_place, elem))
    }

    /// Returns a normal iterator over the remaining elements.
    ///
    /// `un_derefer` must be the same instance that produced `self`.
    pub(crate) fn into_iter(
        mut self,
        un_derefer: &UnDerefer<'tcx>,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> {
        std::iter::from_fn(move || self.next(un_derefer))
    }
}
