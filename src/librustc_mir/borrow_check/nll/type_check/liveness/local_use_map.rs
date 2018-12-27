use borrow_check::nll::region_infer::values::{PointIndex, RegionValueElements};
use borrow_check::nll::type_check::liveness::liveness_map::{LiveVar, NllLivenessMap};
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::mir::{Local, Location, Mir};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::vec_linked_list as vll;
use util::liveness::{categorize, DefUse, LiveVariableMap};

/// A map that cross references each local with the locations where it
/// is defined (assigned), used, or dropped. Used during liveness
/// computation.
crate struct LocalUseMap<'me> {
    liveness_map: &'me NllLivenessMap,

    /// Head of a linked list of **definitions** of each variable --
    /// definition in this context means assignment, e.g., `x` is
    /// defined in `x = y` but not `y`; that first def is the head of
    /// a linked list that lets you enumerate all places the variable
    /// is assigned.
    first_def_at: IndexVec<LiveVar, Option<AppearanceIndex>>,

    /// Head of a linked list of **uses** of each variable -- use in
    /// this context means that the existing value of the variable is
    /// read or modified. e.g., `y` is used in `x = y` but not `x`.
    /// Note that `DROP(x)` terminators are excluded from this list.
    first_use_at: IndexVec<LiveVar, Option<AppearanceIndex>>,

    /// Head of a linked list of **drops** of each variable -- these
    /// are a special category of uses corresponding to the drop that
    /// we add for each local variable.
    first_drop_at: IndexVec<LiveVar, Option<AppearanceIndex>>,

    appearances: IndexVec<AppearanceIndex, Appearance>,
}

struct Appearance {
    point_index: PointIndex,
    next: Option<AppearanceIndex>,
}

newtype_index! {
    pub struct AppearanceIndex { .. }
}

impl vll::LinkElem for Appearance {
    type LinkIndex = AppearanceIndex;

    fn next(elem: &Self) -> Option<AppearanceIndex> {
        elem.next
    }
}

impl LocalUseMap<'me> {
    crate fn build(
        liveness_map: &'me NllLivenessMap,
        elements: &RegionValueElements,
        mir: &Mir<'_>,
    ) -> Self {
        let nones = IndexVec::from_elem_n(None, liveness_map.num_variables());
        let mut local_use_map = LocalUseMap {
            liveness_map,
            first_def_at: nones.clone(),
            first_use_at: nones.clone(),
            first_drop_at: nones,
            appearances: IndexVec::new(),
        };

        LocalUseMapBuild {
            local_use_map: &mut local_use_map,
            elements,
        }.visit_mir(mir);

        local_use_map
    }

    crate fn defs(&self, local: LiveVar) -> impl Iterator<Item = PointIndex> + '_ {
        vll::iter(self.first_def_at[local], &self.appearances)
            .map(move |aa| self.appearances[aa].point_index)
    }

    crate fn uses(&self, local: LiveVar) -> impl Iterator<Item = PointIndex> + '_ {
        vll::iter(self.first_use_at[local], &self.appearances)
            .map(move |aa| self.appearances[aa].point_index)
    }

    crate fn drops(&self, local: LiveVar) -> impl Iterator<Item = PointIndex> + '_ {
        vll::iter(self.first_drop_at[local], &self.appearances)
            .map(move |aa| self.appearances[aa].point_index)
    }
}

struct LocalUseMapBuild<'me, 'map: 'me> {
    local_use_map: &'me mut LocalUseMap<'map>,
    elements: &'me RegionValueElements,
}

impl LocalUseMapBuild<'_, '_> {
    fn insert_def(&mut self, local: LiveVar, location: Location) {
        Self::insert(
            self.elements,
            &mut self.local_use_map.first_def_at[local],
            &mut self.local_use_map.appearances,
            location,
        );
    }

    fn insert_use(&mut self, local: LiveVar, location: Location) {
        Self::insert(
            self.elements,
            &mut self.local_use_map.first_use_at[local],
            &mut self.local_use_map.appearances,
            location,
        );
    }

    fn insert_drop(&mut self, local: LiveVar, location: Location) {
        Self::insert(
            self.elements,
            &mut self.local_use_map.first_drop_at[local],
            &mut self.local_use_map.appearances,
            location,
        );
    }

    fn insert(
        elements: &RegionValueElements,
        first_appearance: &mut Option<AppearanceIndex>,
        appearances: &mut IndexVec<AppearanceIndex, Appearance>,
        location: Location,
    ) {
        let point_index = elements.point_from_location(location);
        let appearance_index = appearances.push(Appearance {
            point_index,
            next: *first_appearance,
        });
        *first_appearance = Some(appearance_index);
    }
}

impl Visitor<'tcx> for LocalUseMapBuild<'_, '_> {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext<'tcx>, location: Location) {
        if let Some(local_with_region) = self.local_use_map.liveness_map.from_local(local) {
            match categorize(context) {
                Some(DefUse::Def) => self.insert_def(local_with_region, location),
                Some(DefUse::Use) => self.insert_use(local_with_region, location),
                Some(DefUse::Drop) => self.insert_drop(local_with_region, location),
                _ => (),
            }
        }
    }
}
