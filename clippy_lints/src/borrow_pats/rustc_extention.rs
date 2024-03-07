use mid::mir::{Local, Place};
use rustc_hir as hir;
use rustc_middle::{self as mid, mir};
use std::collections::VecDeque;

/// Extending the [`mir::Body`] where needed.
///
/// This is such a bad name for a trait, sorry.
pub trait BodyMagic {
    fn are_bbs_exclusive(&self, a: mir::BasicBlock, b: mir::BasicBlock) -> bool;
}

impl<'tcx> BodyMagic for mir::Body<'tcx> {
    fn are_bbs_exclusive(&self, a: mir::BasicBlock, b: mir::BasicBlock) -> bool {
        #[expect(clippy::comparison_chain)]
        if a == b {
            return false;
        } else if a > b {
            return self.are_bbs_exclusive(b, a);
        }

        let mut visited = Vec::with_capacity(16);
        let mut queue = VecDeque::with_capacity(16);

        queue.push_back(a);
        while let Some(bbi) = queue.pop_front() {
            // Check we don't know the node yet
            if visited.contains(&bbi) {
                continue;
            }

            // Found our connection
            if bbi == b {
                return false;
            }

            self.basic_blocks[bbi]
                .terminator()
                .successors()
                .collect_into(&mut queue);
            visited.push(bbi);
        }

        true
    }
}

pub trait PlaceMagic {
    /// This returns true, if this is only a part of the local. A field or array
    /// element would be a part of a local.
    fn is_part(&self) -> bool;

    /// Returns true if this is only a local. Any projections, field accesses or
    /// other non local things will return false.
    fn just_local(&self) -> bool;
}

impl PlaceMagic for mir::Place<'_> {
    fn is_part(&self) -> bool {
        self.projection.iter().any(|x| {
            matches!(
                x,
                mir::PlaceElem::Field(_, _)
                    | mir::PlaceElem::Index(_)
                    | mir::PlaceElem::ConstantIndex { .. }
                    | mir::PlaceElem::Subslice { .. }
            )
        })
    }

    fn just_local(&self) -> bool {
        self.projection.is_empty()
    }
}

pub trait LocalMagic {
    fn as_place(self) -> Place<'static>;
}

impl LocalMagic for Local {
    fn as_place(self) -> Place<'static> {
        Place {
            local: self,
            projection: rustc_middle::ty::List::empty(),
        }
    }
}

pub fn print_body(body: &mir::Body<'_>) {
    for (idx, data) in body.basic_blocks.iter_enumerated() {
        println!("bb{}:", idx.index());
        for stmt in &data.statements {
            println!("    {stmt:#?}");
        }
        println!("    {:#?}", data.terminator().kind);

        println!();
    }

    //println!("{body:#?}");
}
