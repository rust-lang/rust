use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::*;

/// Used for reverting changes made by `DerefSeparator`
#[derive(Default, Debug)]
pub struct UnDerefer<'tcx> {
    deref_chains: FxHashMap<Local, Vec<PlaceRef<'tcx>>>,
}

impl<'tcx> UnDerefer<'tcx> {
    #[inline]
    pub fn insert(&mut self, local: Local, reffed: PlaceRef<'tcx>) {
        let mut chain = self.deref_chains.remove(&reffed.local).unwrap_or_default();
        chain.push(reffed);
        self.deref_chains.insert(local, chain);
    }

    /// Returns the chain of places behind `DerefTemp` locals
    #[inline]
    pub fn deref_chain(&self, local: Local) -> &[PlaceRef<'tcx>] {
        self.deref_chains.get(&local).map(Vec::as_slice).unwrap_or_default()
    }

    /// Iterates over the projections of a place and its deref chain.
    ///
    /// See [`PlaceRef::iter_projections`]
    #[inline]
    pub fn iter_projections(
        &self,
        place: PlaceRef<'tcx>,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> + '_ {
        ProjectionIter::new(self.deref_chain(place.local), place)
    }
}

/// The iterator returned by [`UnDerefer::iter_projections`].
struct ProjectionIter<'a, 'tcx> {
    places: SlicePlusOne<'a, PlaceRef<'tcx>>,
    proj_idx: usize,
}

impl<'a, 'tcx> ProjectionIter<'a, 'tcx> {
    #[inline]
    fn new(deref_chain: &'a [PlaceRef<'tcx>], place: PlaceRef<'tcx>) -> Self {
        // just return an empty iterator for a bare local
        let last = if place.as_local().is_none() {
            Some(place)
        } else {
            debug_assert!(deref_chain.is_empty());
            None
        };

        ProjectionIter { places: SlicePlusOne { slice: deref_chain, last }, proj_idx: 0 }
    }
}

impl<'tcx> Iterator for ProjectionIter<'_, 'tcx> {
    type Item = (PlaceRef<'tcx>, PlaceElem<'tcx>);

    #[inline]
    fn next(&mut self) -> Option<(PlaceRef<'tcx>, PlaceElem<'tcx>)> {
        let place = self.places.read()?;

        // the projection should never be empty except for a bare local which is handled in new
        let partial_place =
            PlaceRef { local: place.local, projection: &place.projection[..self.proj_idx] };
        let elem = place.projection[self.proj_idx];

        if self.proj_idx == place.projection.len() - 1 {
            self.proj_idx = 0;
            self.places.advance();
        } else {
            self.proj_idx += 1;
        }

        Some((partial_place, elem))
    }
}

struct SlicePlusOne<'a, T> {
    slice: &'a [T],
    last: Option<T>,
}

impl<T: Copy> SlicePlusOne<'_, T> {
    #[inline]
    fn read(&self) -> Option<T> {
        self.slice.first().copied().or(self.last)
    }

    #[inline]
    fn advance(&mut self) {
        match self.slice {
            [_, ref remainder @ ..] => {
                self.slice = remainder;
            }
            [] => self.last = None,
        }
    }
}
