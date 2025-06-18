//! We denote as "SSA" the set of locals that verify the following properties:
//! 1/ They are only assigned-to once, either as a function parameter, or in an assign statement;
//! 2/ This single assignment dominates all uses;
//!
//! As we do not track indirect assignments, a local that has its address taken (via a borrow or raw
//! borrow operator) is considered non-SSA. However, it is UB to modify through an immutable borrow
//! of a `Freeze` local. Those can still be considered to be SSA.

use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::bug;
use rustc_middle::middle::resolve_bound_vars::Set1;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use tracing::{debug, instrument, trace};

pub(super) struct SsaLocals {
    /// Assignments to each local. This defines whether the local is SSA.
    assignments: IndexVec<Local, Set1<DefLocation>>,
    /// We visit the body in reverse postorder, to ensure each local is assigned before it is used.
    /// We remember the order in which we saw the assignments to compute the SSA values in a single
    /// pass.
    assignment_order: Vec<Local>,
    /// Copy equivalence classes between locals. See `copy_classes` for documentation.
    copy_classes: IndexVec<Local, Local>,
    /// Number of "direct" uses of each local, ie. uses that are not dereferences.
    /// We ignore non-uses (Storage statements, debuginfo).
    direct_uses: IndexVec<Local, u32>,
    /// Set of SSA locals that are immutably borrowed.
    borrowed_locals: DenseBitSet<Local>,
}

impl SsaLocals {
    pub(super) fn new<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> SsaLocals {
        let assignment_order = Vec::with_capacity(body.local_decls.len());

        let assignments = IndexVec::from_elem(Set1::Empty, &body.local_decls);
        let dominators = body.basic_blocks.dominators();

        let direct_uses = IndexVec::from_elem(0, &body.local_decls);
        let borrowed_locals = DenseBitSet::new_empty(body.local_decls.len());
        let mut visitor = SsaVisitor {
            body,
            assignments,
            assignment_order,
            dominators,
            direct_uses,
            borrowed_locals,
        };

        for local in body.args_iter() {
            visitor.assignments[local] = Set1::One(DefLocation::Argument);
            visitor.assignment_order.push(local);
        }

        // For SSA assignments, a RPO visit will see the assignment before it sees any use.
        // We only visit reachable nodes: computing `dominates` on an unreachable node ICEs.
        for (bb, data) in traversal::reverse_postorder(body) {
            visitor.visit_basic_block_data(bb, data);
        }

        for var_debug_info in &body.var_debug_info {
            visitor.visit_var_debug_info(var_debug_info);
        }

        // The immutability of shared borrows only works on `Freeze` locals. If the visitor found
        // borrows, we need to check the types. For raw pointers and mutable borrows, the locals
        // have already been marked as non-SSA.
        debug!(?visitor.borrowed_locals);
        for local in visitor.borrowed_locals.iter() {
            if !body.local_decls[local].ty.is_freeze(tcx, typing_env) {
                visitor.assignments[local] = Set1::Many;
            }
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
            borrowed_locals: visitor.borrowed_locals,
            // This is filled by `compute_copy_classes`.
            copy_classes: IndexVec::default(),
        };
        compute_copy_classes(&mut ssa, body);
        ssa
    }

    pub(super) fn num_locals(&self) -> usize {
        self.assignments.len()
    }

    pub(super) fn locals(&self) -> impl Iterator<Item = Local> {
        self.assignments.indices()
    }

    pub(super) fn is_ssa(&self, local: Local) -> bool {
        matches!(self.assignments[local], Set1::One(_))
    }

    /// Return the number of uses if a local that are not "Deref".
    pub(super) fn num_direct_uses(&self, local: Local) -> u32 {
        self.direct_uses[local]
    }

    #[inline]
    pub(super) fn assignment_dominates(
        &self,
        dominators: &Dominators<BasicBlock>,
        local: Local,
        location: Location,
    ) -> bool {
        match self.assignments[local] {
            Set1::One(def) => def.dominates(location, dominators),
            _ => false,
        }
    }

    pub(super) fn assignments<'a, 'tcx>(
        &'a self,
        body: &'a Body<'tcx>,
    ) -> impl Iterator<Item = (Local, &'a Rvalue<'tcx>, Location)> {
        self.assignment_order.iter().filter_map(|&local| {
            if let Set1::One(DefLocation::Assignment(loc)) = self.assignments[local] {
                let stmt = body.stmt_at(loc).left()?;
                // `loc` must point to a direct assignment to `local`.
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
    /// Exception: we do not see through the return place, as it cannot be instantiated.
    pub(super) fn copy_classes(&self) -> &IndexSlice<Local, Local> {
        &self.copy_classes
    }

    /// Set of SSA locals that are immutably borrowed.
    pub(super) fn borrowed_locals(&self) -> &DenseBitSet<Local> {
        &self.borrowed_locals
    }

    /// Make a property uniform on a copy equivalence class by removing elements.
    pub(super) fn meet_copy_equivalence(&self, property: &mut DenseBitSet<Local>) {
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

struct SsaVisitor<'a, 'tcx> {
    body: &'a Body<'tcx>,
    dominators: &'a Dominators<BasicBlock>,
    assignments: IndexVec<Local, Set1<DefLocation>>,
    assignment_order: Vec<Local>,
    direct_uses: IndexVec<Local, u32>,
    // Track locals that are immutably borrowed, so we can check their type is `Freeze` later.
    borrowed_locals: DenseBitSet<Local>,
}

impl SsaVisitor<'_, '_> {
    fn check_dominates(&mut self, local: Local, loc: Location) {
        let set = &mut self.assignments[local];
        let assign_dominates = match *set {
            Set1::Empty | Set1::Many => false,
            Set1::One(def) => def.dominates(loc, self.dominators),
        };
        // We are visiting a use that is not dominated by an assignment.
        // Either there is a cycle involved, or we are reading for uninitialized local.
        // Bail out.
        if !assign_dominates {
            *set = Set1::Many;
        }
    }
}

impl<'tcx> Visitor<'tcx> for SsaVisitor<'_, 'tcx> {
    fn visit_local(&mut self, local: Local, ctxt: PlaceContext, loc: Location) {
        match ctxt {
            PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => bug!(),
            // Anything can happen with raw pointers, so remove them.
            PlaceContext::NonMutatingUse(NonMutatingUseContext::RawBorrow)
            | PlaceContext::MutatingUse(_) => {
                self.assignments[local] = Set1::Many;
            }
            // Immutable borrows are ok, but we need to delay a check that the type is `Freeze`.
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::SharedBorrow | NonMutatingUseContext::FakeBorrow,
            ) => {
                self.borrowed_locals.insert(local);
                self.check_dominates(local, loc);
                self.direct_uses[local] += 1;
            }
            PlaceContext::NonMutatingUse(_) => {
                self.check_dominates(local, loc);
                self.direct_uses[local] += 1;
            }
            PlaceContext::NonUse(_) => {}
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, ctxt: PlaceContext, loc: Location) {
        let location = match ctxt {
            PlaceContext::MutatingUse(MutatingUseContext::Store) => {
                Some(DefLocation::Assignment(loc))
            }
            PlaceContext::MutatingUse(MutatingUseContext::Call) => {
                let call = loc.block;
                let TerminatorKind::Call { target, .. } =
                    self.body.basic_blocks[call].terminator().kind
                else {
                    bug!()
                };
                Some(DefLocation::CallReturn { call, target })
            }
            _ => None,
        };
        if let Some(location) = location
            && let Some(local) = place.as_local()
        {
            self.assignments[local].insert(location);
            if let Set1::One(_) = self.assignments[local] {
                // Only record if SSA-like, to avoid growing the vector needlessly.
                self.assignment_order.push(local);
            }
        } else if place.projection.first() == Some(&PlaceElem::Deref) {
            // Do not do anything for debuginfo.
            if ctxt.is_use() {
                // Only change the context if it is a real use, not a "use" in debuginfo.
                let new_ctxt = PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy);

                self.visit_projection(place.as_ref(), new_ctxt, loc);
                self.check_dominates(place.local, loc);
            }
        } else {
            self.visit_projection(place.as_ref(), ctxt, loc);
            self.visit_local(place.local, ctxt, loc);
        }
    }
}

#[instrument(level = "trace", skip(ssa, body))]
fn compute_copy_classes(ssa: &mut SsaLocals, body: &Body<'_>) {
    let mut direct_uses = std::mem::take(&mut ssa.direct_uses);
    let mut copies = IndexVec::from_fn_n(|l| l, body.local_decls.len());
    // We must not unify two locals that are borrowed. But this is fine if one is borrowed and
    // the other is not. This bitset is keyed by *class head* and contains whether any member of
    // the class is borrowed.
    let mut borrowed_classes = ssa.borrowed_locals().clone();

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

        // Do not unify two borrowed locals.
        if borrowed_classes.contains(local) && borrowed_classes.contains(head) {
            continue;
        }

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
            if borrowed_classes.contains(head) {
                borrowed_classes.insert(RETURN_PLACE);
            }
        } else {
            copies[local] = head;
            if borrowed_classes.contains(local) {
                borrowed_classes.insert(head);
            }
        }
        direct_uses[rhs] -= 1;
    }

    debug!(?copies);
    debug!(?direct_uses);
    debug!(?borrowed_classes);

    // Invariant: `copies` must point to the head of an equivalence class.
    #[cfg(debug_assertions)]
    for &head in copies.iter() {
        assert_eq!(copies[head], head);
    }
    debug_assert_eq!(copies[RETURN_PLACE], RETURN_PLACE);

    // Invariant: `borrowed_classes` must be true if any member of the class is borrowed.
    #[cfg(debug_assertions)]
    for &head in copies.iter() {
        let any_borrowed = ssa.borrowed_locals.iter().any(|l| copies[l] == head);
        assert_eq!(borrowed_classes.contains(head), any_borrowed);
    }

    ssa.direct_uses = direct_uses;
    ssa.copy_classes = copies;
}

#[derive(Debug)]
pub(crate) struct StorageLiveLocals {
    /// Set of "StorageLive" statements for each local.
    storage_live: IndexVec<Local, Set1<DefLocation>>,
}

impl StorageLiveLocals {
    pub(crate) fn new(
        body: &Body<'_>,
        always_storage_live_locals: &DenseBitSet<Local>,
    ) -> StorageLiveLocals {
        let mut storage_live = IndexVec::from_elem(Set1::Empty, &body.local_decls);
        for local in always_storage_live_locals.iter() {
            storage_live[local] = Set1::One(DefLocation::Argument);
        }
        for (block, bbdata) in body.basic_blocks.iter_enumerated() {
            for (statement_index, statement) in bbdata.statements.iter().enumerate() {
                if let StatementKind::StorageLive(local) = statement.kind {
                    storage_live[local]
                        .insert(DefLocation::Assignment(Location { block, statement_index }));
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
