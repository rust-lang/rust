// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::region_infer::values::PointIndex;
use borrow_check::nll::region_infer::values::RegionValueElements;
use rustc::mir::{BasicBlock, Location, Mir};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use std::rc::Rc;

/// A little data structure that makes it more efficient to find the
/// predecessors of each point.
crate struct PointIndexMap<'me, 'tcx> {
    elements: &'me Rc<RegionValueElements>,
    mir: &'me Mir<'tcx>,
    basic_block_heads: IndexVec<PointIndex, Option<BasicBlock>>,
}

impl PointIndexMap<'m, 'tcx> {
    crate fn new(elements: &'m Rc<RegionValueElements>, mir: &'m Mir<'tcx>) -> Self {
        let mut basic_block_heads = IndexVec::from_elem_n(None, elements.num_points());

        for (bb, first_point) in elements.head_indices() {
            basic_block_heads[first_point] = Some(bb);
        }

        PointIndexMap {
            elements,
            mir,
            basic_block_heads,
        }
    }

    crate fn num_points(&self) -> usize {
        self.elements.num_points()
    }

    crate fn location_of(&self, index: PointIndex) -> Location {
        let mut statement_index = 0;

        for &opt_bb in self.basic_block_heads.raw[..= index.index()].iter().rev() {
            if let Some(block) = opt_bb {
                return Location { block, statement_index };
            }

            statement_index += 1;
        }

        bug!("did not find basic block as expected for index = {:?}", index)
    }

    crate fn push_predecessors(&self, index: PointIndex, stack: &mut Vec<PointIndex>) {
        match self.basic_block_heads[index] {
            // If this is a basic block head, then the predecessors are
            // the the terminators of other basic blocks
            Some(bb_head) => {
                stack.extend(
                    self.mir
                        .predecessors_for(bb_head)
                        .iter()
                        .map(|&pred_bb| self.mir.terminator_loc(pred_bb))
                        .map(|pred_loc| self.elements.point_from_location(pred_loc)),
                );
            }

            // Otherwise, the pred is just the previous statement
            None => {
                stack.push(PointIndex::new(index.index() - 1));
            }
        }
    }
}
