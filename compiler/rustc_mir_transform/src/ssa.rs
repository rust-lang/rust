use either::Either;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::middle::resolve_lifetime::Set1;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, TyCtxt};

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
}

impl SsaLocals {
    pub fn new<'tcx>(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        body: &Body<'tcx>,
        borrowed_locals: &BitSet<Local>,
    ) -> SsaLocals {
        let assignment_order = Vec::new();

        let assignments = IndexVec::from_elem(Set1::Empty, &body.local_decls);
        let dominators = body.basic_blocks.dominators();
        let mut visitor = SsaVisitor { assignments, assignment_order, dominators };

        for (local, decl) in body.local_decls.iter_enumerated() {
            if matches!(body.local_kind(local), LocalKind::Arg) {
                visitor.assignments[local] = Set1::One(LocationExtended::Arg);
            }
            if borrowed_locals.contains(local) && !decl.ty.is_freeze(tcx, param_env) {
                visitor.assignments[local] = Set1::Many;
            }
        }

        for (bb, data) in traversal::reverse_postorder(body) {
            visitor.visit_basic_block_data(bb, data);
        }

        for var_debug_info in &body.var_debug_info {
            visitor.visit_var_debug_info(var_debug_info);
        }

        debug!(?visitor.assignments);

        visitor
            .assignment_order
            .retain(|&local| matches!(visitor.assignments[local], Set1::One(_)));
        debug!(?visitor.assignment_order);

        let copy_classes = compute_copy_classes(&visitor, body);

        SsaLocals {
            assignments: visitor.assignments,
            assignment_order: visitor.assignment_order,
            copy_classes,
        }
    }

    pub fn is_ssa(&self, local: Local) -> bool {
        matches!(self.assignments[local], Set1::One(_))
    }

    pub fn assignments<'a, 'tcx>(
        &'a self,
        body: &'a Body<'tcx>,
    ) -> impl Iterator<Item = (Local, &'a Rvalue<'tcx>)> + 'a {
        self.assignment_order.iter().filter_map(|&local| {
            if let Set1::One(LocationExtended::Plain(loc)) = self.assignments[local] {
                // `loc` must point to a direct assignment to `local`.
                let Either::Left(stmt) = body.stmt_at(loc) else { bug!() };
                let Some((target, rvalue)) = stmt.kind.as_assign() else { bug!() };
                assert_eq!(target.as_local(), Some(local));
                Some((local, rvalue))
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
    pub fn copy_classes(&self) -> &IndexVec<Local, Local> {
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

struct SsaVisitor {
    dominators: Dominators<BasicBlock>,
    assignments: IndexVec<Local, Set1<LocationExtended>>,
    assignment_order: Vec<Local>,
}

impl<'tcx> Visitor<'tcx> for SsaVisitor {
    fn visit_local(&mut self, local: Local, ctxt: PlaceContext, loc: Location) {
        match ctxt {
            PlaceContext::MutatingUse(MutatingUseContext::Store) => {
                self.assignments[local].insert(LocationExtended::Plain(loc));
                self.assignment_order.push(local);
            }
            // Anything can happen with raw pointers, so remove them.
            PlaceContext::NonMutatingUse(NonMutatingUseContext::AddressOf)
            | PlaceContext::MutatingUse(_) => self.assignments[local] = Set1::Many,
            // Immutable borrows are taken into account in `SsaLocals::new` by
            // removing non-freeze locals.
            PlaceContext::NonMutatingUse(_) => {
                let set = &mut self.assignments[local];
                let assign_dominates = match *set {
                    Set1::Empty | Set1::Many => false,
                    Set1::One(LocationExtended::Arg) => true,
                    Set1::One(LocationExtended::Plain(assign)) => {
                        assign.successor_within_block().dominates(loc, &self.dominators)
                    }
                };
                // We are visiting a use that is not dominated by an assignment.
                // Either there is a cycle involved, or we are reading for uninitialized local.
                // Bail out.
                if !assign_dominates {
                    *set = Set1::Many;
                }
            }
            PlaceContext::NonUse(_) => {}
        }
    }
}

#[instrument(level = "trace", skip(ssa, body))]
fn compute_copy_classes(ssa: &SsaVisitor, body: &Body<'_>) -> IndexVec<Local, Local> {
    let mut copies = IndexVec::from_fn_n(|l| l, body.local_decls.len());

    for &local in &ssa.assignment_order {
        debug!(?local);

        if local == RETURN_PLACE {
            // `_0` is special, we cannot rename it.
            continue;
        }

        // This is not SSA: mark that we don't know the value.
        debug!(assignments = ?ssa.assignments[local]);
        let Set1::One(LocationExtended::Plain(loc)) = ssa.assignments[local] else { continue };

        // `loc` must point to a direct assignment to `local`.
        let Either::Left(stmt) = body.stmt_at(loc) else { bug!() };
        let Some((_target, rvalue)) = stmt.kind.as_assign() else { bug!() };
        assert_eq!(_target.as_local(), Some(local));

        let (Rvalue::Use(Operand::Copy(place) | Operand::Move(place)) | Rvalue::CopyForDeref(place))
            = rvalue
        else { continue };

        let Some(rhs) = place.as_local() else { continue };
        let Set1::One(_) = ssa.assignments[rhs] else { continue };

        // We visit in `assignment_order`, ie. reverse post-order, so `rhs` has been
        // visited before `local`, and we just have to copy the representing local.
        copies[local] = copies[rhs];
    }

    debug!(?copies);

    // Invariant: `copies` must point to the head of an equivalence class.
    #[cfg(debug_assertions)]
    for &head in copies.iter() {
        assert_eq!(copies[head], head);
    }

    copies
}
