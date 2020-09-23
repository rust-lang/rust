//! Performs various peephole optimizations.

use crate::transform::{MirPass, MirSource};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::Mutability;
use rustc_index::vec::Idx;
use rustc_middle::mir::UnOp;
use rustc_middle::mir::{
    visit::PlaceContext,
    visit::{MutVisitor, Visitor},
    Statement,
};
use rustc_middle::mir::{
    BinOp, Body, BorrowKind, Constant, Local, Location, Operand, Place, PlaceRef, ProjectionElem,
    Rvalue,
};
use rustc_middle::ty::{self, TyCtxt};
use smallvec::SmallVec;
use std::mem;

pub struct InstCombine;

impl<'tcx> MirPass<'tcx> for InstCombine {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        // First, find optimization opportunities. This is done in a pre-pass to keep the MIR
        // read-only so that we can do global analyses on the MIR in the process (e.g.
        // `Place::ty()`).
        let optimizations = {
            let mut optimization_finder = OptimizationFinder::new(body, tcx);
            optimization_finder.visit_body(body);
            optimization_finder.optimizations
        };

        // Since eq_not has elements removed in the visitor, we clone it here,
        // such that we can still do the post visitor cleanup.
        let clone_eq_not = optimizations.eq_not.clone();
        // Then carry out those optimizations.
        MutVisitor::visit_body(&mut InstCombineVisitor { optimizations, tcx }, body);
        eq_not_post_visitor_mutations(body, clone_eq_not);
    }
}

fn eq_not_post_visitor_mutations<'tcx>(
    body: &mut Body<'tcx>,
    eq_not_opts: FxHashMap<Location, EqNotOptInfo<'tcx>>,
) {
    for (location, eq_not_opt_info) in eq_not_opts.iter() {
        let statements = &mut body.basic_blocks_mut()[location.block].statements;
        // We have to make sure that Ne is before any StorageDead as the operand being killed is used in the Ne
        if let Some(storage_dead_idx_to_swap_with) = eq_not_opt_info.storage_dead_to_swap_with_ne {
            statements.swap(location.statement_index, storage_dead_idx_to_swap_with);
        }
        if let Some(eq_stmt_idx) = eq_not_opt_info.can_remove_eq {
            statements[eq_stmt_idx].make_nop();
        }
    }
}

pub struct InstCombineVisitor<'tcx> {
    optimizations: OptimizationList<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for InstCombineVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if self.optimizations.and_stars.remove(&location) {
            debug!("replacing `&*`: {:?}", rvalue);
            let new_place = match rvalue {
                Rvalue::Ref(_, _, place) => {
                    if let &[ref proj_l @ .., proj_r] = place.projection.as_ref() {
                        place.projection = self.tcx().intern_place_elems(&[proj_r]);

                        Place {
                            // Replace with dummy
                            local: mem::replace(&mut place.local, Local::new(0)),
                            projection: self.tcx().intern_place_elems(proj_l),
                        }
                    } else {
                        unreachable!();
                    }
                }
                _ => bug!("Detected `&*` but didn't find `&*`!"),
            };
            *rvalue = Rvalue::Use(Operand::Copy(new_place))
        }

        if let Some(constant) = self.optimizations.arrays_lengths.remove(&location) {
            debug!("replacing `Len([_; N])`: {:?}", rvalue);
            *rvalue = Rvalue::Use(Operand::Constant(box constant));
        }

        if let Some(operand) = self.optimizations.unneeded_equality_comparison.remove(&location) {
            debug!("replacing {:?} with {:?}", rvalue, operand);
            *rvalue = Rvalue::Use(operand);
        }

        if let Some(place) = self.optimizations.unneeded_deref.remove(&location) {
            debug!("unneeded_deref: replacing {:?} with {:?}", rvalue, place);
            *rvalue = Rvalue::Use(Operand::Copy(place));
        }

        if let Some(eq_not_opt_info) = self.optimizations.eq_not.remove(&location) {
            *rvalue = Rvalue::BinaryOp(BinOp::Ne, eq_not_opt_info.op1, eq_not_opt_info.op2);
            debug!("replacing Eq and Not with {:?}", rvalue);
        }

        self.super_rvalue(rvalue, location)
    }
}

struct MutatingUseVisitor {
    has_mutating_use: bool,
    local_to_look_for: Local,
}

impl MutatingUseVisitor {
    fn has_mutating_use_in_stmt(local: Local, stmt: &Statement<'tcx>, location: Location) -> bool {
        let mut _self = Self { has_mutating_use: false, local_to_look_for: local };
        _self.visit_statement(stmt, location);
        _self.has_mutating_use
    }
}

impl<'tcx> Visitor<'tcx> for MutatingUseVisitor {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, _: Location) {
        if *local == self.local_to_look_for {
            self.has_mutating_use |= context.is_mutating_use();
        }
    }
}

/// Finds optimization opportunities on the MIR.
struct OptimizationFinder<'b, 'tcx> {
    body: &'b Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    optimizations: OptimizationList<'tcx>,
}

impl OptimizationFinder<'b, 'tcx> {
    fn new(body: &'b Body<'tcx>, tcx: TyCtxt<'tcx>) -> OptimizationFinder<'b, 'tcx> {
        OptimizationFinder { body, tcx, optimizations: OptimizationList::default() }
    }

    fn find_deref_of_address(&mut self, rvalue: &Rvalue<'tcx>, location: Location) -> Option<()> {
        // Look for the sequence
        //
        // _2 = &_1;
        // ...
        // _5 = (*_2);
        //
        // which we can replace the last statement with `_5 = _1;` to avoid the load of `_2`.
        if let Rvalue::Use(op) = rvalue {
            let local_being_derefed = match op.place()?.as_ref() {
                PlaceRef { local, projection: [ProjectionElem::Deref] } => Some(local),
                _ => None,
            }?;

            let stmt_index = location.statement_index;
            // Look behind for statement that assigns the local from a address of operator.
            // 6 is chosen as a heuristic determined by seeing the number of times
            // the optimization kicked in compiling rust std.
            let lower_index = stmt_index.saturating_sub(6);
            let statements_to_look_in = self.body.basic_blocks()[location.block].statements
                [lower_index..stmt_index]
                .iter()
                .rev();
            for stmt in statements_to_look_in {
                match &stmt.kind {
                    // Exhaustive match on statements to detect conditions that warrant we bail out of the optimization.
                    rustc_middle::mir::StatementKind::Assign(box (l, r))
                        if l.local == local_being_derefed =>
                    {
                        match r {
                            // Looking for immutable reference e.g _local_being_deref = &_1;
                            Rvalue::Ref(
                                _,
                                // Only apply the optimization if it is an immutable borrow.
                                BorrowKind::Shared,
                                place_taken_address_of,
                            ) => {
                                self.optimizations
                                    .unneeded_deref
                                    .insert(location, *place_taken_address_of);
                                return Some(());
                            }

                            // We found an assignment of `local_being_deref` that is not an immutable ref, e.g the following sequence
                            // _2 = &_1;
                            // _3 = &5
                            // _2 = _3;  <-- this means it is no longer valid to replace the last statement with `_5 = _1;`
                            // _5 = (*_2);
                            _ => return None,
                        }
                    }

                    // Inline asm can do anything, so bail out of the optimization.
                    rustc_middle::mir::StatementKind::LlvmInlineAsm(_) => return None,

                    // Check that `local_being_deref` is not being used in a mutating way which can cause misoptimization.
                    rustc_middle::mir::StatementKind::Assign(box (_, _))
                    | rustc_middle::mir::StatementKind::Coverage(_)
                    | rustc_middle::mir::StatementKind::Nop
                    | rustc_middle::mir::StatementKind::FakeRead(_, _)
                    | rustc_middle::mir::StatementKind::StorageLive(_)
                    | rustc_middle::mir::StatementKind::StorageDead(_)
                    | rustc_middle::mir::StatementKind::Retag(_, _)
                    | rustc_middle::mir::StatementKind::AscribeUserType(_, _)
                    | rustc_middle::mir::StatementKind::SetDiscriminant { .. } => {
                        if MutatingUseVisitor::has_mutating_use_in_stmt(
                            local_being_derefed,
                            stmt,
                            location,
                        ) {
                            return None;
                        }
                    }
                }
            }
        }
        Some(())
    }

    fn find_unneeded_equality_comparison(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        // find Ne(_place, false) or Ne(false, _place)
        // or   Eq(_place, true) or Eq(true, _place)
        if let Rvalue::BinaryOp(op, l, r) = rvalue {
            let const_to_find = if *op == BinOp::Ne {
                false
            } else if *op == BinOp::Eq {
                true
            } else {
                return;
            };
            // (const, _place)
            if let Some(o) = self.find_operand_in_equality_comparison_pattern(l, r, const_to_find) {
                self.optimizations.unneeded_equality_comparison.insert(location, o.clone());
            }
            // (_place, const)
            else if let Some(o) =
                self.find_operand_in_equality_comparison_pattern(r, l, const_to_find)
            {
                self.optimizations.unneeded_equality_comparison.insert(location, o.clone());
            }
        }
    }

    fn find_eq_not(&mut self, rvalue: &Rvalue<'tcx>, location: Location) -> Option<()> {
        // Optimize the sequence
        // _4 = Eq(move _5, const 2_u8);
        // StorageDead(_5);
        // _3 = Not(move _4);
        //
        // into _3 = Ne(move _5, const 2_u8)
        if let Rvalue::UnaryOp(UnOp::Not, op) = rvalue {
            let place = op.place()?;
            // See if we can find a Eq that assigns `place`.
            // We limit the search to 3 statements lookback.
            // Usually the first 2 statements are `StorageDead`s for operands for Eq.
            // We record what is marked dead so that we can reorder StorageDead so it comes after Ne

            // We will maximum see 2 StorageDeads
            let mut seen_storage_deads: SmallVec<[_; 2]> = SmallVec::new();
            let lower_index = location.statement_index.saturating_sub(3);
            for (stmt_idx, stmt) in self.body.basic_blocks()[location.block].statements
                [lower_index..location.statement_index]
                .iter()
                .enumerate()
                .rev()
            {
                match &stmt.kind {
                    rustc_middle::mir::StatementKind::Assign(box (l, r)) => {
                        if *l == place {
                            match r {
                                // FIXME(simonvandel): extend for Ne-Not pair
                                Rvalue::BinaryOp(BinOp::Eq, op1, op2) => {
                                    // We need to make sure that the StorageDeads we saw are for
                                    // either `op1`or `op2` of Eq. Else we bail the optimization.
                                    for (dead_local, _) in seen_storage_deads.iter() {
                                        let dead_local_matches = [op1, op2].iter().any(|x| {
                                            Some(*dead_local) == x.place().map(|x| x.local)
                                        });
                                        if !dead_local_matches {
                                            return None;
                                        }
                                    }

                                    // Recall that we are optimizing a sequence that looks like
                                    // this:
                                    // _4 = Eq(move _5, move _6);
                                    // StorageDead(_5);
                                    // StorageDead(_6);
                                    // _3 = Not(move _4);
                                    //
                                    // If we do a naive replace of Not -> Ne, we up with this:
                                    // StorageDead(_5);
                                    // StorageDead(_6);
                                    // _3 = Ne(move _5, move _6);
                                    //
                                    // Notice that `_5` and `_6` are marked dead before being used.
                                    // To combat this we want to swap Ne with the StorageDead
                                    // closest to Eq, i.e `StorageDead(_5)` in this example.
                                    let storage_dead_to_swap =
                                        seen_storage_deads.last().map(|(_, idx)| *idx);

                                    // If the operand of Not is moved into it,
                                    // and that same operand is the lhs of the Eq assignment,
                                    // then we can safely remove the Eq
                                    let can_remove_eq = if op.is_move() {
                                        Some(stmt_idx + lower_index)
                                    } else {
                                        None
                                    };

                                    self.optimizations.eq_not.insert(
                                        location,
                                        EqNotOptInfo {
                                            op1: op1.clone(),
                                            op2: op2.clone(),
                                            storage_dead_to_swap_with_ne: storage_dead_to_swap,
                                            can_remove_eq,
                                        },
                                    );
                                    return Some(());
                                }
                                _ => {}
                            }
                        }
                    }
                    rustc_middle::mir::StatementKind::StorageDead(dead) => {
                        seen_storage_deads.push((*dead, stmt_idx + lower_index));
                    }
                    // If we see a pattern other than (0..=2) StorageDeads and then an Eq assignment, we conservatively bail
                    _ => return None,
                }
            }
        }
        Some(())
    }

    fn find_operand_in_equality_comparison_pattern(
        &self,
        l: &Operand<'tcx>,
        r: &'a Operand<'tcx>,
        const_to_find: bool,
    ) -> Option<&'a Operand<'tcx>> {
        let const_ = l.constant()?;
        if const_.literal.ty == self.tcx.types.bool
            && const_.literal.val.try_to_bool() == Some(const_to_find)
        {
            if r.place().is_some() {
                return Some(r);
            }
        }

        None
    }
}

impl Visitor<'tcx> for OptimizationFinder<'b, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Ref(_, _, place) = rvalue {
            if let PlaceRef { local, projection: &[ref proj_base @ .., ProjectionElem::Deref] } =
                place.as_ref()
            {
                // The dereferenced place must have type `&_`.
                let ty = Place::ty_from(local, proj_base, self.body, self.tcx).ty;
                if let ty::Ref(_, _, Mutability::Not) = ty.kind() {
                    self.optimizations.and_stars.insert(location);
                }
            }
        }

        if let Rvalue::Len(ref place) = *rvalue {
            let place_ty = place.ty(&self.body.local_decls, self.tcx).ty;
            if let ty::Array(_, len) = place_ty.kind() {
                let span = self.body.source_info(location).span;
                let constant = Constant { span, literal: len, user_ty: None };
                self.optimizations.arrays_lengths.insert(location, constant);
            }
        }

        let _ = self.find_deref_of_address(rvalue, location);

        let _ = self.find_eq_not(rvalue, location);

        self.find_unneeded_equality_comparison(rvalue, location);

        self.super_rvalue(rvalue, location)
    }
}

#[derive(Clone)]
struct EqNotOptInfo<'tcx> {
    op1: Operand<'tcx>,
    op2: Operand<'tcx>,
    storage_dead_to_swap_with_ne: Option<usize>,
    can_remove_eq: Option<usize>,
}

#[derive(Default)]
struct OptimizationList<'tcx> {
    and_stars: FxHashSet<Location>,
    arrays_lengths: FxHashMap<Location, Constant<'tcx>>,
    unneeded_equality_comparison: FxHashMap<Location, Operand<'tcx>>,
    unneeded_deref: FxHashMap<Location, Place<'tcx>>,
    eq_not: FxHashMap<Location, EqNotOptInfo<'tcx>>,
}
