//! Performs various peephole optimizations.

use crate::transform::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::Mutability;
use rustc_index::vec::Idx;
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
use std::mem;

pub struct InstCombine;

impl<'tcx> MirPass<'tcx> for InstCombine {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // First, find optimization opportunities. This is done in a pre-pass to keep the MIR
        // read-only so that we can do global analyses on the MIR in the process (e.g.
        // `Place::ty()`).
        let optimizations = {
            let mut optimization_finder = OptimizationFinder::new(body, tcx);
            optimization_finder.visit_body(body);
            optimization_finder.optimizations
        };

        // Then carry out those optimizations.
        MutVisitor::visit_body(&mut InstCombineVisitor { optimizations, tcx }, body);
    }
}

pub struct InstCombineVisitor<'tcx> {
    optimizations: OptimizationList<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> InstCombineVisitor<'tcx> {
    fn should_combine(&self, rvalue: &Rvalue<'tcx>, location: Location) -> bool {
        self.tcx.consider_optimizing(|| {
            format!("InstCombine - Rvalue: {:?} Location: {:?}", rvalue, location)
        })
    }
}

impl<'tcx> MutVisitor<'tcx> for InstCombineVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if self.optimizations.and_stars.remove(&location) && self.should_combine(rvalue, location) {
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
            if self.should_combine(rvalue, location) {
                debug!("replacing `Len([_; N])`: {:?}", rvalue);
                *rvalue = Rvalue::Use(Operand::Constant(box constant));
            }
        }

        if let Some(operand) = self.optimizations.unneeded_equality_comparison.remove(&location) {
            if self.should_combine(rvalue, location) {
                debug!("replacing {:?} with {:?}", rvalue, operand);
                *rvalue = Rvalue::Use(operand);
            }
        }

        if let Some(place) = self.optimizations.unneeded_deref.remove(&location) {
            if self.should_combine(rvalue, location) {
                debug!("unneeded_deref: replacing {:?} with {:?}", rvalue, place);
                *rvalue = Rvalue::Use(Operand::Copy(place));
            }
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
        // FIXME(#78192): This optimization can result in unsoundness.
        if !self.tcx.sess.opts.debugging_opts.unsound_mir_opts {
            return None;
        }

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

            let mut dead_locals_seen = vec![];

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
                                // Make sure that the place has not been marked dead
                                if dead_locals_seen.contains(&place_taken_address_of.local) {
                                    return None;
                                }

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

                    // Remember `StorageDead`s, as the local being marked dead could be the
                    // place RHS we are looking for, in which case we need to abort to avoid UB
                    // using an uninitialized place
                    rustc_middle::mir::StatementKind::StorageDead(dead) => {
                        dead_locals_seen.push(*dead)
                    }

                    // Check that `local_being_deref` is not being used in a mutating way which can cause misoptimization.
                    rustc_middle::mir::StatementKind::Assign(box (_, _))
                    | rustc_middle::mir::StatementKind::Coverage(_)
                    | rustc_middle::mir::StatementKind::Nop
                    | rustc_middle::mir::StatementKind::FakeRead(_, _)
                    | rustc_middle::mir::StatementKind::StorageLive(_)
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

        self.find_unneeded_equality_comparison(rvalue, location);

        self.super_rvalue(rvalue, location)
    }
}

#[derive(Default)]
struct OptimizationList<'tcx> {
    and_stars: FxHashSet<Location>,
    arrays_lengths: FxHashMap<Location, Constant<'tcx>>,
    unneeded_equality_comparison: FxHashMap<Location, Operand<'tcx>>,
    unneeded_deref: FxHashMap<Location, Place<'tcx>>,
}
