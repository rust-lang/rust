use rustc::mir::{BasicBlock, Location, Body};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

/// Maps between a MIR Location, which identifies a particular
/// statement within a basic block, to a "rich location", which
/// identifies at a finer granularity. In particular, we distinguish
/// the *start* of a statement and the *mid-point*. The mid-point is
/// the point *just* before the statement takes effect; in particular,
/// for an assignment `A = B`, it is the point where B is about to be
/// written into A. This mid-point is a kind of hack to work around
/// our inability to track the position information at sufficient
/// granularity through outlives relations; however, the rich location
/// table serves another purpose: it compresses locations from
/// multiple words into a single u32.
crate struct LocationTable {
    num_points: usize,
    statements_before_block: IndexVec<BasicBlock, usize>,
}

newtype_index! {
    pub struct LocationIndex {
        DEBUG_FORMAT = "LocationIndex({})"
    }
}

#[derive(Copy, Clone, Debug)]
crate enum RichLocation {
    Start(Location),
    Mid(Location),
}

impl LocationTable {
    crate fn new(body: &Body<'_>) -> Self {
        let mut num_points = 0;
        let statements_before_block = body.basic_blocks()
            .iter()
            .map(|block_data| {
                let v = num_points;
                num_points += (block_data.statements.len() + 1) * 2;
                v
            })
            .collect();

        debug!(
            "LocationTable(statements_before_block={:#?})",
            statements_before_block
        );
        debug!("LocationTable: num_points={:#?}", num_points);

        Self {
            num_points,
            statements_before_block,
        }
    }

    crate fn all_points(&self) -> impl Iterator<Item = LocationIndex> {
        (0..self.num_points).map(LocationIndex::new)
    }

    crate fn start_index(&self, location: Location) -> LocationIndex {
        let Location {
            block,
            statement_index,
        } = location;
        let start_index = self.statements_before_block[block];
        LocationIndex::new(start_index + statement_index * 2)
    }

    crate fn mid_index(&self, location: Location) -> LocationIndex {
        let Location {
            block,
            statement_index,
        } = location;
        let start_index = self.statements_before_block[block];
        LocationIndex::new(start_index + statement_index * 2 + 1)
    }

    crate fn to_location(&self, index: LocationIndex) -> RichLocation {
        let point_index = index.index();

        // Find the basic block. We have a vector with the
        // starting index of the statement in each block. Imagine
        // we have statement #22, and we have a vector like:
        //
        // [0, 10, 20]
        //
        // In that case, this represents point_index 2 of
        // basic block BB2. We know this because BB0 accounts for
        // 0..10, BB1 accounts for 11..20, and BB2 accounts for
        // 20...
        //
        // To compute this, we could do a binary search, but
        // because I am lazy we instead iterate through to find
        // the last point where the "first index" (0, 10, or 20)
        // was less than the statement index (22). In our case, this will
        // be (BB2, 20).
        let (block, &first_index) = self.statements_before_block
            .iter_enumerated()
            .filter(|(_, first_index)| **first_index <= point_index)
            .last()
            .unwrap();

        let statement_index = (point_index - first_index) / 2;
        if index.is_start() {
            RichLocation::Start(Location { block, statement_index })
        } else {
            RichLocation::Mid(Location { block, statement_index })
        }
    }
}

impl LocationIndex {
    fn is_start(&self) -> bool {
        // even indices are start points; odd indices are mid points
        (self.index() % 2) == 0
    }
}
