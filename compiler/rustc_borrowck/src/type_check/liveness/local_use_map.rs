use rustc_index::IndexVec;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{Body, Local, Location};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};

use crate::def_use::{self, DefUse};

/// A map that cross references each local with the locations where it
/// is defined (assigned), used, or dropped. Used during liveness
/// computation.
///
/// We keep track only of `Local`s we'll do the liveness analysis later,
/// this means that our internal `IndexVec`s will only be sparsely populated.
/// In the time-memory trade-off between keeping compact vectors with new
/// indexes (and needing to continuously map the `Local` index to its compact
/// counterpart) and having `IndexVec`s that we only use a fraction of, time
/// (and code simplicity) was favored. The rationale is that we only keep
/// a small number of `IndexVec`s throughout the entire analysis while, in
/// contrast, we're accessing each `Local` *many* times.
pub(crate) struct LocalUseMap {
    /// Head of a linked list of **definitions** of each variable --
    /// definition in this context means assignment, e.g., `x` is
    /// defined in `x = y` but not `y`; that first def is the head of
    /// a linked list that lets you enumerate all places the variable
    /// is assigned.
    first_def_at: IndexVec<Local, Option<AppearanceIndex>>,

    /// Head of a linked list of **uses** of each variable -- use in
    /// this context means that the existing value of the variable is
    /// read or modified. e.g., `y` is used in `x = y` but not `x`.
    /// Note that `DROP(x)` terminators are excluded from this list.
    first_use_at: IndexVec<Local, Option<AppearanceIndex>>,

    /// Head of a linked list of **drops** of each variable -- these
    /// are a special category of uses corresponding to the drop that
    /// we add for each local variable.
    first_drop_at: IndexVec<Local, Option<AppearanceIndex>>,

    appearances: Appearances,
}

// The `Appearance::next` field effectively embeds a linked list within `Appearances`.
type Appearances = IndexVec<AppearanceIndex, Appearance>;

struct Appearance {
    point_index: PointIndex,
    next: Option<AppearanceIndex>,
}

rustc_index::newtype_index! {
    pub struct AppearanceIndex {}
}

fn appearances_iter(
    first: Option<AppearanceIndex>,
    appearances: &Appearances,
) -> impl Iterator<Item = AppearanceIndex> {
    AppearancesIter { appearances, current: first }
}

// Iterates over `Appearances` by following `next` fields.
struct AppearancesIter<'a> {
    appearances: &'a Appearances,
    current: Option<AppearanceIndex>,
}

impl<'a> Iterator for AppearancesIter<'a> {
    type Item = AppearanceIndex;

    fn next(&mut self) -> Option<AppearanceIndex> {
        if let Some(c) = self.current {
            self.current = self.appearances[c].next;
            Some(c)
        } else {
            None
        }
    }
}

//-----------------------------------------------------------------------------

impl LocalUseMap {
    pub(crate) fn build(
        live_locals: &[Local],
        location_map: &DenseLocationMap,
        body: &Body<'_>,
    ) -> Self {
        let nones = IndexVec::from_elem(None, &body.local_decls);
        let mut local_use_map = LocalUseMap {
            first_def_at: nones.clone(),
            first_use_at: nones.clone(),
            first_drop_at: nones,
            appearances: IndexVec::new(),
        };

        if live_locals.is_empty() {
            return local_use_map;
        }

        let mut locals_with_use_data: IndexVec<Local, bool> =
            IndexVec::from_elem(false, &body.local_decls);
        live_locals.iter().for_each(|&local| locals_with_use_data[local] = true);

        LocalUseMapBuild { local_use_map: &mut local_use_map, location_map, locals_with_use_data }
            .visit_body(body);

        local_use_map
    }

    pub(crate) fn defs(&self, local: Local) -> impl Iterator<Item = PointIndex> {
        appearances_iter(self.first_def_at[local], &self.appearances)
            .map(move |aa| self.appearances[aa].point_index)
    }

    pub(crate) fn uses(&self, local: Local) -> impl Iterator<Item = PointIndex> {
        appearances_iter(self.first_use_at[local], &self.appearances)
            .map(move |aa| self.appearances[aa].point_index)
    }

    pub(crate) fn drops(&self, local: Local) -> impl Iterator<Item = PointIndex> {
        appearances_iter(self.first_drop_at[local], &self.appearances)
            .map(move |aa| self.appearances[aa].point_index)
    }
}

struct LocalUseMapBuild<'me> {
    local_use_map: &'me mut LocalUseMap,
    location_map: &'me DenseLocationMap,

    // Vector used in `visit_local` to signal which `Local`s do we need
    // def/use/drop information on, constructed from `live_locals` (that
    // contains the variables we'll do the liveness analysis for).
    // This vector serves optimization purposes only: we could have
    // obtained the same information from `live_locals` but we want to
    // avoid repeatedly calling `Vec::contains()` (see `LocalUseMap` for
    // the rationale on the time-memory trade-off we're favoring here).
    locals_with_use_data: IndexVec<Local, bool>,
}

impl Visitor<'_> for LocalUseMapBuild<'_> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        if self.locals_with_use_data[local]
            && let Some(def_use) = def_use::categorize(context)
        {
            let first_appearance = match def_use {
                DefUse::Def => &mut self.local_use_map.first_def_at[local],
                DefUse::Use => &mut self.local_use_map.first_use_at[local],
                DefUse::Drop => &mut self.local_use_map.first_drop_at[local],
            };
            let point_index = self.location_map.point_from_location(location);
            let appearance_index = self
                .local_use_map
                .appearances
                .push(Appearance { point_index, next: *first_appearance });
            *first_appearance = Some(appearance_index);
        }
    }
}
