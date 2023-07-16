//! We denote as "SSA" the set of locals that verify the following properties:
//! 1/ They are only assigned-to once, either as a function parameter, or in an assign statement;
//! 2/ This single assignment dominates all uses;
//!
//! As a consequence of rule 2, we consider that borrowed locals are not SSA, even if they are
//! `Freeze`, as we do not track that the assignment dominates all uses of the borrow.

use either::Either;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::BitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::middle::resolve_bound_vars::Set1;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;

#[derive(Debug)]
pub struct SsaLocals {
    /// Assignments to each local. This defines whether the local is SSA.
    assignments: IndexVec<Local, Set1<LocationExtended>>,
    /// We visit the body in reverse postorder, to ensure each local is assigned before it is used.
    /// We remember the order in which we saw the assignments to compute the SSA values in a single
    /// pass.
    assignment_order: Vec<Local>,
    /// Copy equivalence classes between locals. See `copy_classes` for documentation.
    copy_classes: IndexVec<Local, Local>,
    /// Number of "direct" uses of each local, ie. uses that are not dereferences.
    /// We ignore non-uses (Storage statements, debuginfo).
    direct_uses: IndexVec<Local, u32>,
}

/// We often encounter MIR bodies with 1 or 2 basic blocks. In those cases, it's unnecessary to
/// actually compute dominators, we can just compare block indices because bb0 is always the first
/// block, and in any body all other blocks are always dominated by bb0.
struct SmallDominators<'a> {
    inner: Option<&'a Dominators<BasicBlock>>,
}

impl SmallDominators<'_> {
    fn dominates(&self, first: Location, second: Location) -> bool {
        if first.block == second.block {
            first.statement_index <= second.statement_index
        } else if let Some(inner) = &self.inner {
            inner.dominates(first.block, second.block)
        } else {
            first.block < second.block
        }
    }

    fn check_dominates(&mut self, set: &mut Set1<LocationExtended>, loc: Location) {
        let assign_dominates = match *set {
            Set1::Empty | Set1::Many => false,
            Set1::One(LocationExtended::Arg) => true,
            Set1::One(LocationExtended::Plain(assign)) => {
                self.dominates(assign.successor_within_block(), loc)
            }
        };
        // We are visiting a use that is not dominated by an assignment.
        // Either there is a cycle involved, or we are reading for uninitialized local.
        // Bail out.
        if !assign_dominates {
            *set = Set1::Many;
        }
    }
}

impl SsaLocals {
    pub fn new<'tcx>(body: &Body<'tcx>) -> SsaLocals {
        let assignment_order = Vec::with_capacity(body.local_decls.len());

        let assignments = IndexVec::from_elem(Set1::Empty, &body.local_decls);
        let dominators =
            if body.basic_blocks.len() > 2 { Some(body.basic_blocks.dominators()) } else { None };
        let dominators = SmallDominators { inner: dominators };

        let direct_uses = IndexVec::from_elem(0, &body.local_decls);
        let mut visitor = SsaVisitor { assignments, assignment_order, dominators, direct_uses };

        for local in body.args_iter() {
            visitor.assignments[local] = Set1::One(LocationExtended::Arg);
        }

        if body.basic_blocks.len() > 2 {
            for (bb, data) in traversal::reverse_postorder(body) {
                visitor.visit_basic_block_data(bb, data);
            }
        } else {
            for (bb, data) in body.basic_blocks.iter_enumerated() {
                visitor.visit_basic_block_data(bb, data);
            }
        }

        for var_debug_info in &body.var_debug_info {
            visitor.visit_var_debug_info(var_debug_info);
        }

        debug!(?visitor.assignments);
        debug!(?visitor.direct_uses);

        visitor
            .assignment_order
            .retain(|&local| matches!(visitor.assignments[local], Set1::One(_)));
        debug!(?visitor.assignment_order);

        let mut ssa = SsaLocals {
            assignments: visitor.assignments,
            assignment_order: visitor.assignment_order,
            direct_uses: visitor.direct_uses,
            // This is filled by `compute_copy_classes`.
            copy_classes: IndexVec::default(),
        };
        compute_copy_classes(&mut ssa, body);
        ssa
    }

    pub fn num_locals(&self) -> usize {
        self.assignments.len()
    }

    pub fn locals(&self) -> impl Iterator<Item = Local> {
        self.assignments.indices()
    }

    pub fn is_ssa(&self, local: Local) -> bool {
        matches!(self.assignments[local], Set1::One(_))
    }

    /// Return the number of uses if a local that are not "Deref".
    pub fn num_direct_uses(&self, local: Local) -> u32 {
        self.direct_uses[local]
    }

    pub fn assignments<'a, 'tcx>(
        &'a self,
        body: &'a Body<'tcx>,
    ) -> impl Iterator<Item = (Local, &'a Rvalue<'tcx>, Location)> + 'a {
        self.assignment_order.iter().filter_map(|&local| {
            if let Set1::One(LocationExtended::Plain(loc)) = self.assignments[local] {
                // `loc` must point to a direct assignment to `local`.
                let Either::Left(stmt) = body.stmt_at(loc) else { bug!() };
                let Some((target, rvalue)) = stmt.kind.as_assign() else { bug!() };
                assert_eq!(target.as_local(), Some(local));
                Some((local, rvalue, loc))
            } else {
                None
            }
        })
    }

    /// Compute the equivalence classes for locals, based on copy statements.
    ///
    /// The returned vector maps each local to the one it copies. In the following case:
    ///   _a = &mut _0
    ///   _b = move? _a
    ///   _c = move? _a
    ///   _d = move? _c
    /// We return the mapping
    ///   _a => _a // not a copy so, represented by itself
    ///   _b => _a
    ///   _c => _a
    ///   _d => _a // transitively through _c
    ///
    /// Exception: we do not see through the return place, as it cannot be substituted.
    pub fn copy_classes(&self) -> &IndexSlice<Local, Local> {
        &self.copy_classes
    }

    /// Make a property uniform on a copy equivalence class by removing elements.
    pub fn meet_copy_equivalence(&self, property: &mut BitSet<Local>) {
        // Consolidate to have a local iff all its copies are.
        //
        // `copy_classes` defines equivalence classes between locals. The `local`s that recursively
        // move/copy the same local all have the same `head`.
        for (local, &head) in self.copy_classes.iter_enumerated() {
            // If any copy does not have `property`, then the head is not.
            if !property.contains(local) {
                property.remove(head);
            }
        }
        for (local, &head) in self.copy_classes.iter_enumerated() {
            // If any copy does not have `property`, then the head doesn't either,
            // then no copy has `property`.
            if !property.contains(head) {
                property.remove(local);
            }
        }

        // Verify that we correctly computed equivalence classes.
        #[cfg(debug_assertions)]
        for (local, &head) in self.copy_classes.iter_enumerated() {
            assert_eq!(property.contains(local), property.contains(head));
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LocationExtended {
    Plain(Location),
    Arg,
}

struct SsaVisitor<'a> {
    dominators: SmallDominators<'a>,
    assignments: IndexVec<Local, Set1<LocationExtended>>,
    assignment_order: Vec<Local>,
    direct_uses: IndexVec<Local, u32>,
}

impl<'tcx> Visitor<'tcx> for SsaVisitor<'_> {
    fn visit_local(&mut self, local: Local, ctxt: PlaceContext, loc: Location) {
        match ctxt {
            PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => bug!(),
            // Anything can happen with raw pointers, so remove them.
            // We do not verify that all uses of the borrow dominate the assignment to `local`,
            // so we have to remove them too.
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::ShallowBorrow
                | NonMutatingUseContext::AddressOf,
            )
            | PlaceContext::MutatingUse(_) => {
                self.assignments[local] = Set1::Many;
            }
            PlaceContext::NonMutatingUse(_) => {
                self.dominators.check_dominates(&mut self.assignments[local], loc);
                self.direct_uses[local] += 1;
            }
            PlaceContext::NonUse(_) => {}
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, ctxt: PlaceContext, loc: Location) {
        if place.projection.first() == Some(&PlaceElem::Deref) {
            // Do not do anything for storage statements and debuginfo.
            if ctxt.is_use() {
                // Only change the context if it is a real use, not a "use" in debuginfo.
                let new_ctxt = PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy);

                self.visit_projection(place.as_ref(), new_ctxt, loc);
                self.dominators.check_dominates(&mut self.assignments[place.local], loc);
            }
            return;
        } else {
            self.visit_projection(place.as_ref(), ctxt, loc);
            self.visit_local(place.local, ctxt, loc);
        }
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, loc: Location) {
        if let Some(local) = place.as_local() {
            self.assignments[local].insert(LocationExtended::Plain(loc));
            if let Set1::One(_) = self.assignments[local] {
                // Only record if SSA-like, to avoid growing the vector needlessly.
                self.assignment_order.push(local);
            }
        } else {
            self.visit_place(place, PlaceContext::MutatingUse(MutatingUseContext::Store), loc);
        }
        self.visit_rvalue(rvalue, loc);
    }
}

#[instrument(level = "trace", skip(ssa, body))]
fn compute_copy_classes(ssa: &mut SsaLocals, body: &Body<'_>) {
    let mut direct_uses = std::mem::take(&mut ssa.direct_uses);
    let mut copies = IndexVec::from_fn_n(|l| l, body.local_decls.len());

    for (local, rvalue, _) in ssa.assignments(body) {
        let (Rvalue::Use(Operand::Copy(place) | Operand::Move(place))
        | Rvalue::CopyForDeref(place)) = rvalue
        else {
            continue;
        };

        let Some(rhs) = place.as_local() else { continue };
        let local_ty = body.local_decls()[local].ty;
        let rhs_ty = body.local_decls()[rhs].ty;
        if local_ty != rhs_ty {
            // FIXME(#112651): This can be removed afterwards.
            trace!("skipped `{local:?} = {rhs:?}` due to subtyping: {local_ty} != {rhs_ty}");
            continue;
        }

        if !ssa.is_ssa(rhs) {
            continue;
        }

        // We visit in `assignment_order`, ie. reverse post-order, so `rhs` has been
        // visited before `local`, and we just have to copy the representing local.
        let head = copies[rhs];

        if local == RETURN_PLACE {
            // `_0` is special, we cannot rename it. Instead, rename the class of `rhs` to
            // `RETURN_PLACE`. This is only possible if the class head is a temporary, not an
            // argument.
            if body.local_kind(head) != LocalKind::Temp {
                continue;
            }
            for h in copies.iter_mut() {
                if *h == head {
                    *h = RETURN_PLACE;
                }
            }
        } else {
            copies[local] = head;
        }
        direct_uses[rhs] -= 1;
    }

    debug!(?copies);
    debug!(?direct_uses);

    // Invariant: `copies` must point to the head of an equivalence class.
    #[cfg(debug_assertions)]
    for &head in copies.iter() {
        assert_eq!(copies[head], head);
    }
    debug_assert_eq!(copies[RETURN_PLACE], RETURN_PLACE);

    ssa.direct_uses = direct_uses;
    ssa.copy_classes = copies;
}

#[derive(Debug)]
pub(crate) struct StorageLiveLocals {
    /// Set of "StorageLive" statements for each local.
    storage_live: IndexVec<Local, Set1<LocationExtended>>,
}

impl StorageLiveLocals {
    pub(crate) fn new(
        body: &Body<'_>,
        always_storage_live_locals: &BitSet<Local>,
    ) -> StorageLiveLocals {
        let mut storage_live = IndexVec::from_elem(Set1::Empty, &body.local_decls);
        for local in always_storage_live_locals.iter() {
            storage_live[local] = Set1::One(LocationExtended::Arg);
        }
        for (block, bbdata) in body.basic_blocks.iter_enumerated() {
            for (statement_index, statement) in bbdata.statements.iter().enumerate() {
                if let StatementKind::StorageLive(local) = statement.kind {
                    storage_live[local]
                        .insert(LocationExtended::Plain(Location { block, statement_index }));
                }
            }
        }
        debug!(?storage_live);
        StorageLiveLocals { storage_live }
    }

    #[inline]
    pub(crate) fn has_single_storage(&self, local: Local) -> bool {
        matches!(self.storage_live[local], Set1::One(_))
    }
}
