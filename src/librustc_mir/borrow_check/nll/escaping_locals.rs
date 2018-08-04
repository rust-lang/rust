// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Identify those variables whose entire value will eventually be
//! returned from the fn via the RETURN_PLACE. As an optimization, we
//! can skip computing liveness results for those variables. The idea
//! is that the return type of the fn only ever contains free
//! regions. Therefore, the types of those variables are going to
//! ultimately be contrained to outlive those free regions -- since
//! free regions are always live for the entire body, this implies
//! that the liveness results are not important for those regions.
//! This is most important in the "fns" that we create to represent static
//! values, since those are often really quite large, and all regions in them
//! will ultimately be constrained to be `'static`. Two examples:
//!
//! ```
//! fn foo() -> &'static [u32] { &[] }
//! static FOO: &[u32] = &[];
//! ```
//!
//! In both these cases, the return value will only have static lifetime.
//!
//! NB: The simple logic here relies on the fact that outlives
//! relations in our analysis don't have locations. Otherwise, we
//! would have to restrict ourselves to values that are
//! *unconditionally* returned (which would still cover the "big
//! static value" case).
//!
//! The way that this code works is to use union-find -- we iterate
//! over the MIR and union together two variables X and Y if all
//! regions in the value of Y are going to be stored into X -- that
//! is, if `typeof(X): 'a` requires that `typeof(Y): 'a`. This means
//! that e.g. we can union together `x` and `y` if we have something
//! like `x = (y, 22)`, but not something like `x = y.f` (since there
//! may be regions in the type of `y` that do not appear in the field
//! `f`).

use rustc::mir::visit::Visitor;
use rustc::mir::*;
use rustc::ty::TyCtxt;

use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::unify as ut;

crate struct EscapingLocals {
    unification_table: ut::UnificationTable<ut::InPlace<AssignedLocal>>,
}

impl EscapingLocals {
    crate fn compute(tcx: TyCtxt<'_, '_, 'tcx>, mir: &Mir<'tcx>) -> Self {
        let mut visitor = GatherAssignedLocalsVisitor::new(tcx, mir);
        visitor.visit_mir(mir);

        EscapingLocals {
            unification_table: visitor.unification_table,
        }
    }

    /// True if `local` is known to escape into static
    /// memory.
    crate fn escapes_into_return(&mut self, local: Local) -> bool {
        let return_place = AssignedLocal::from(RETURN_PLACE);
        let other_place = AssignedLocal::from(local);
        self.unification_table.unioned(return_place, other_place)
    }
}

/// The MIR visitor gathering the union-find of the locals used in
/// assignments.
struct GatherAssignedLocalsVisitor<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    unification_table: ut::UnificationTable<ut::InPlace<AssignedLocal>>,
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    mir: &'cx Mir<'tcx>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct AssignedLocal(u32);

impl ut::UnifyKey for AssignedLocal {
    type Value = ();

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(i: u32) -> AssignedLocal {
        AssignedLocal(i)
    }

    fn tag() -> &'static str {
        "AssignedLocal"
    }
}

impl From<Local> for AssignedLocal {
    fn from(item: Local) -> Self {
        // newtype_indexes use usize but are u32s.
        assert!(item.index() < ::std::u32::MAX as usize);
        AssignedLocal(item.index() as u32)
    }
}

impl GatherAssignedLocalsVisitor<'cx, 'gcx, 'tcx> {
    fn new(tcx: TyCtxt<'cx, 'gcx, 'tcx>, mir: &'cx Mir<'tcx>) -> Self {
        Self {
            unification_table: ut::UnificationTable::new(),
            tcx,
            mir,
        }
    }

    fn union_locals_if_needed(&mut self, lvalue: Option<Local>, rvalue: Option<Local>) {
        if let Some(lvalue) = lvalue {
            if let Some(rvalue) = rvalue {
                if lvalue != rvalue {
                    debug!("EscapingLocals: union {:?} and {:?}", lvalue, rvalue);
                    self.unification_table
                        .union(AssignedLocal::from(lvalue), AssignedLocal::from(rvalue));
                }
            }
        }
    }
}

// Returns the potential `Local` associated to this `Place` or `PlaceProjection`
fn find_local_in_place(place: &Place) -> Option<Local> {
    match place {
        Place::Local(local) => Some(*local),

        // If you do e.g. `x = a.f` then only *part* of the type of
        // `a` escapes into `x` (the part contained in `f`); if `a`'s
        // type has regions that don't appear in `f`, those might not
        // escape.
        Place::Projection(..) => None,

        Place::Static { .. } | Place::Promoted { .. } => None,
    }
}

// Returns the potential `Local` in this `Operand`.
fn find_local_in_operand(op: &Operand) -> Option<Local> {
    // Conservatively check a subset of `Operand`s we know our
    // benchmarks track, for example `html5ever`.
    match op {
        Operand::Copy(place) | Operand::Move(place) => find_local_in_place(place),
        Operand::Constant(_) => None,
    }
}

impl Visitor<'tcx> for GatherAssignedLocalsVisitor<'_, '_, 'tcx> {
    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        // We need as many union-find keys as there are locals
        for _ in 0..mir.local_decls.len() {
            self.unification_table.new_key(());
        }

        self.super_mir(mir);
    }

    fn visit_assign(
        &mut self,
        block: BasicBlock,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location,
    ) {
        let local = find_local_in_place(place);

        // Conservatively check a subset of `Rvalue`s we know our
        // benchmarks track, for example `html5ever`.
        match rvalue {
            Rvalue::Use(op) => self.union_locals_if_needed(local, find_local_in_operand(op)),
            Rvalue::Ref(_, _, place) => {
                // Special case: if you have `X = &*Y` (or `X = &**Y`
                // etc), then the outlives relationships will ensure
                // that all regions in `Y` are constrained by regions
                // in `X` -- this is because the lifetimes of the
                // references we deref through are required to outlive
                // the borrow lifetime (which appears in `X`).
                //
                // (We don't actually need to check the type of `Y`:
                // since `ProjectionElem::Deref` represents a built-in
                // deref and not an overloaded deref, if the thing we
                // deref through is not a reference, then it must be a
                // `Box` or `*const`, in which case it contains no
                // references.)
                let mut place_ref = place;
                while let Place::Projection(proj) = place_ref {
                    if let ProjectionElem::Deref = proj.elem {
                        place_ref = &proj.base;
                    } else {
                        break;
                    }
                }

                self.union_locals_if_needed(local, find_local_in_place(place_ref))
            }

            Rvalue::Cast(kind, op, _) => match kind {
                CastKind::Unsize => self.union_locals_if_needed(local, find_local_in_operand(op)),
                _ => (),
            },

            Rvalue::Aggregate(_, ops) => {
                for rvalue in ops.iter().map(find_local_in_operand) {
                    self.union_locals_if_needed(local, rvalue);
                }
            }

            _ => (),
        };

        self.super_assign(block, place, rvalue, location);
    }
}
