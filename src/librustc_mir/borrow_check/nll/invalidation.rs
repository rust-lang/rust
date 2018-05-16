// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::{BorrowSet, BorrowData};
use borrow_check::location::LocationTable;
use borrow_check::{JustWrite, WriteAndRead};
use borrow_check::{ShallowOrDeep, Deep, Shallow};
use borrow_check::{ReadOrWrite, Activation, Read, Reservation, Write};
use borrow_check::{Context, ContextKind};
use borrow_check::{LocalMutationIsAllowed, MutateMode};
use borrow_check::ArtificialField;
use borrow_check::{ReadKind, WriteKind, Overlap};
use borrow_check::nll::facts::AllFacts;
use dataflow::move_paths::indexes::BorrowIndex;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::mir::visit::Visitor;
use rustc::mir::{BasicBlock, Location, Mir, Place, Rvalue, Projection};
use rustc::mir::{Local, ProjectionElem};
use rustc::mir::{Statement, StatementKind};
use rustc::mir::{Terminator, TerminatorKind};
use rustc::mir::{Field, Operand, BorrowKind};
use rustc::ty::{self, ParamEnv};
use rustc_data_structures::indexed_vec::Idx;
use std::iter;

pub(super) fn generate_invalidates<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    all_facts: &mut Option<AllFacts>,
    location_table: &LocationTable,
    mir: &Mir<'tcx>,
    mir_def_id: DefId,
    borrow_set: &BorrowSet<'tcx>,
) {
    if !all_facts.is_some() {
        // Nothing to do if we don't have any facts
        return;
    }

    let param_env = infcx.tcx.param_env(mir_def_id);

    let mut all_facts_taken = all_facts.take().unwrap();
    {
        let mut ig = InvalidationGenerator {
            all_facts: &mut all_facts_taken,
            borrow_set,
            infcx,
            location_table,
            mir,
            param_env,
        };
        ig.visit_mir(mir);
    }
    *all_facts = Some(all_facts_taken);
}

/// 'cg = the duration of the constraint generation process itself.
struct InvalidationGenerator<'cg, 'cx: 'cg, 'tcx: 'cx, 'gcx: 'tcx> {
    infcx: &'cg InferCtxt<'cx, 'gcx, 'tcx>,
    all_facts: &'cg mut AllFacts,
    location_table: &'cg LocationTable,
    mir: &'cg Mir<'tcx>,
    borrow_set: &'cg BorrowSet<'tcx>,
    param_env: ParamEnv<'gcx>,
}

/// Visits the whole MIR and generates invalidates() facts
/// Most of the code implementing this was stolen from borrow_check/mod.rs
impl<'cg, 'cx, 'tcx, 'gcx> Visitor<'tcx> for InvalidationGenerator<'cg, 'cx, 'tcx, 'gcx> {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &Statement<'tcx>,
                       location: Location) {
        match statement.kind {
            StatementKind::Assign(ref lhs, ref rhs) => {
                self.consume_rvalue(
                    ContextKind::AssignRhs.new(location),
                    rhs,
                );

                self.mutate_place(
                    ContextKind::AssignLhs.new(location),
                    lhs,
                    Shallow(None),
                    JustWrite
                );
            }
            StatementKind::SetDiscriminant {
                ref place,
                variant_index: _,
            } => {
                self.mutate_place(
                    ContextKind::SetDiscrim.new(location),
                    place,
                    Shallow(Some(ArtificialField::Discriminant)),
                    JustWrite,
                );
            }
            StatementKind::InlineAsm {
                ref asm,
                ref outputs,
                ref inputs,
            } => {
                let context = ContextKind::InlineAsm.new(location);
                for (o, output) in asm.outputs.iter().zip(outputs) {
                    if o.is_indirect {
                        // FIXME(eddyb) indirect inline asm outputs should
                        // be encoeded through MIR place derefs instead.
                        self.access_place(
                            context,
                            output,
                            (Deep, Read(ReadKind::Copy)),
                            LocalMutationIsAllowed::No,
                        );
                    } else {
                        self.mutate_place(
                            context,
                            output,
                            if o.is_rw { Deep } else { Shallow(None) },
                            if o.is_rw { WriteAndRead } else { JustWrite },
                        );
                    }
                }
                for input in inputs {
                    self.consume_operand(context, input);
                }
            }
            // EndRegion matters to older NLL/MIR AST borrowck, not to alias NLL
            StatementKind::EndRegion(..) |
            StatementKind::Nop |
            StatementKind::UserAssertTy(..) |
            StatementKind::Validate(..) |
            StatementKind::StorageLive(..) => {
                // `Nop`, `UserAssertTy`, `Validate`, and `StorageLive` are irrelevant
                // to borrow check.
            }
            StatementKind::StorageDead(local) => {
                self.access_place(
                    ContextKind::StorageDead.new(location),
                    &Place::Local(local),
                    (Shallow(None), Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                );
            }
        }

        self.super_statement(block, statement, location);
    }

    fn visit_terminator(
        &mut self,
        block: BasicBlock,
        terminator: &Terminator<'tcx>,
        location: Location
    ) {
        match terminator.kind {
            TerminatorKind::SwitchInt {
                ref discr,
                switch_ty: _,
                values: _,
                targets: _,
            } => {
                self.consume_operand(ContextKind::SwitchInt.new(location), discr);
            }
            TerminatorKind::Drop {
                location: ref drop_place,
                target: _,
                unwind: _,
            } => {
                let tcx = self.infcx.tcx;
                let gcx = tcx.global_tcx();
                let drop_place_ty = drop_place.ty(self.mir, tcx);
                let drop_place_ty = tcx.erase_regions(&drop_place_ty).to_ty(tcx);
                let drop_place_ty = gcx.lift(&drop_place_ty).unwrap();
                self.visit_terminator_drop(location, terminator, drop_place, drop_place_ty);
            }
            TerminatorKind::DropAndReplace {
                location: ref drop_place,
                value: ref new_value,
                target: _,
                unwind: _,
            } => {
                self.mutate_place(
                    ContextKind::DropAndReplace.new(location),
                    drop_place,
                    Deep,
                    JustWrite,
                );
                self.consume_operand(
                    ContextKind::DropAndReplace.new(location),
                    new_value,
                );
            }
            TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup: _,
            } => {
                self.consume_operand(ContextKind::CallOperator.new(location), func);
                for arg in args {
                    self.consume_operand(ContextKind::CallOperand.new(location), arg);
                }
                if let Some((ref dest, _ /*bb*/)) = *destination {
                    self.mutate_place(
                        ContextKind::CallDest.new(location),
                        dest,
                        Deep,
                        JustWrite,
                    );
                }
            }
            TerminatorKind::Assert {
                ref cond,
                expected: _,
                ref msg,
                target: _,
                cleanup: _,
            } => {
                self.consume_operand(ContextKind::Assert.new(location), cond);
                use rustc::mir::interpret::EvalErrorKind::BoundsCheck;
                if let BoundsCheck { ref len, ref index } = *msg {
                    self.consume_operand(ContextKind::Assert.new(location), len);
                    self.consume_operand(ContextKind::Assert.new(location), index);
                }
            }
            TerminatorKind::Yield {
                ref value,
                resume: _,
                drop: _,
            } => {
                self.consume_operand(ContextKind::Yield.new(location), value);

                // ** FIXME(bob_twinkles) figure out what the equivalent of this is
                // if self.movable_generator {
                //     // Look for any active borrows to locals
                //     let borrow_set = self.borrow_set.clone();
                //     flow_state.with_outgoing_borrows(|borrows| {
                //         for i in borrows {
                //             let borrow = &borrow_set[i];
                //             self.check_for_local_borrow(borrow, span);
                //         }
                //     });
                // }
            }
            TerminatorKind::Resume | TerminatorKind::Return | TerminatorKind::GeneratorDrop => {
                // ** FIXME(bob_twinkles) figure out what the equivalent of this is
                // // Returning from the function implicitly kills storage for all locals and
                // // statics.
                // // Often, the storage will already have been killed by an explicit
                // // StorageDead, but we don't always emit those (notably on unwind paths),
                // // so this "extra check" serves as a kind of backup.
                // let borrow_set = self.borrow_set.clone();
                // flow_state.with_outgoing_borrows(|borrows| {
                //     for i in borrows {
                //         let borrow = &borrow_set[i];
                //         let context = ContextKind::StorageDead.new(loc);
                //         self.check_for_invalidation_at_exit(context, borrow, span);
                //     }
                // });
            }
            TerminatorKind::Goto { target: _ }
            | TerminatorKind::Abort
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdges {
                real_target: _,
                imaginary_targets: _,
            }
            | TerminatorKind::FalseUnwind {
                real_target: _,
                unwind: _,
            } => {
                // no data used, thus irrelevant to borrowck
            }
        }

        self.super_terminator(block, terminator, location);
    }
}

impl<'cg, 'cx, 'tcx, 'gcx> InvalidationGenerator<'cg, 'cx, 'tcx, 'gcx> {
    /// Simulates dropping of a variable
    fn visit_terminator_drop(
        &mut self,
        loc: Location,
        term: &Terminator<'tcx>,
        drop_place: &Place<'tcx>,
        erased_drop_place_ty: ty::Ty<'gcx>,
    ) {
        let gcx = self.infcx.tcx.global_tcx();
        let drop_field = |
            ig: &mut InvalidationGenerator<'cg, 'cx, 'gcx, 'tcx>,
            (index, field): (usize, ty::Ty<'gcx>),
        | {
            let field_ty = gcx.normalize_erasing_regions(ig.param_env, field);
            let place = drop_place.clone().field(Field::new(index), field_ty);

            ig.visit_terminator_drop(loc, term, &place, field_ty);
        };

        match erased_drop_place_ty.sty {
            // When a struct is being dropped, we need to check
            // whether it has a destructor, if it does, then we can
            // call it, if it does not then we need to check the
            // individual fields instead. This way if `foo` has a
            // destructor but `bar` does not, we will only check for
            // borrows of `x.foo` and not `x.bar`. See #47703.
            ty::TyAdt(def, substs) if def.is_struct() && !def.has_dtor(self.infcx.tcx) => {
                def.all_fields()
                    .map(|field| field.ty(gcx, substs))
                    .enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            // Same as above, but for tuples.
            ty::TyTuple(tys) => {
                tys.iter().cloned().enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            // Closures and generators also have disjoint fields, but they are only
            // directly accessed in the body of the closure/generator.
            ty::TyGenerator(def, substs, ..)
                if *drop_place == Place::Local(Local::new(1)) && !self.mir.upvar_decls.is_empty()
            => {
                substs.upvar_tys(def, self.infcx.tcx).enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            ty::TyClosure(def, substs)
                if *drop_place == Place::Local(Local::new(1)) && !self.mir.upvar_decls.is_empty()
                => {
                    substs.upvar_tys(def, self.infcx.tcx).enumerate()
                        .for_each(|field| drop_field(self, field));
                }
            _ => {
                // We have now refined the type of the value being
                // dropped (potentially) to just the type of a
                // subfield; so check whether that field's type still
                // "needs drop". If so, we assume that the destructor
                // may access any data it likes (i.e., a Deep Write).
                if erased_drop_place_ty.needs_drop(gcx, self.param_env) {
                    self.access_place(
                        ContextKind::Drop.new(loc),
                        drop_place,
                        (Deep, Write(WriteKind::StorageDeadOrDrop)),
                        LocalMutationIsAllowed::Yes,
                    );
                }
            }
        }
    }

    /// Simulates mutation of a place
    fn mutate_place(
        &mut self,
        context: Context,
        place: &Place<'tcx>,
        kind: ShallowOrDeep,
        _mode: MutateMode,
    ) {
        self.access_place(
            context,
            place,
            (kind, Write(WriteKind::Mutate)),
            LocalMutationIsAllowed::ExceptUpvars,
        );
    }

    /// Simulates consumption of an operand
    fn consume_operand(
        &mut self,
        context: Context,
        operand: &Operand<'tcx>,
    ) {
        match *operand {
            Operand::Copy(ref place) => {
                self.access_place(
                    context,
                    place,
                    (Deep, Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                );
            }
            Operand::Move(ref place) => {
                self.access_place(
                    context,
                    place,
                    (Deep, Write(WriteKind::Move)),
                    LocalMutationIsAllowed::Yes,
                );
            }
            Operand::Constant(_) => {}
        }
    }

    // Simulates consumption of an rvalue
    fn consume_rvalue(
        &mut self,
        context: Context,
        rvalue: &Rvalue<'tcx>,
    ) {
        match *rvalue {
            Rvalue::Ref(_ /*rgn*/, bk, ref place) => {
                let access_kind = match bk {
                    BorrowKind::Shared => (Deep, Read(ReadKind::Borrow(bk))),
                    BorrowKind::Unique | BorrowKind::Mut { .. } => {
                        let wk = WriteKind::MutableBorrow(bk);
                        if self.allow_two_phase_borrow(bk) {
                            (Deep, Reservation(wk))
                        } else {
                            (Deep, Write(wk))
                        }
                    }
                };

                self.access_place(
                    context,
                    place,
                    access_kind,
                    LocalMutationIsAllowed::No,
                );
            }

            Rvalue::Use(ref operand)
            | Rvalue::Repeat(ref operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, ref operand)
            | Rvalue::Cast(_ /*cast_kind*/, ref operand, _ /*ty*/) => {
                self.consume_operand(context, operand)
            }

            Rvalue::Len(ref place) | Rvalue::Discriminant(ref place) => {
                let af = match *rvalue {
                    Rvalue::Len(..) => ArtificialField::ArrayLength,
                    Rvalue::Discriminant(..) => ArtificialField::Discriminant,
                    _ => unreachable!(),
                };
                self.access_place(
                    context,
                    place,
                    (Shallow(Some(af)), Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                );
            }

            Rvalue::BinaryOp(_bin_op, ref operand1, ref operand2)
            | Rvalue::CheckedBinaryOp(_bin_op, ref operand1, ref operand2) => {
                self.consume_operand(context, operand1);
                self.consume_operand(context, operand2);
            }

            Rvalue::NullaryOp(_op, _ty) => {
            }

            Rvalue::Aggregate(_, ref operands) => {
                for operand in operands {
                    self.consume_operand(context, operand);
                }
            }
        }
    }

    /// Simulates an access to a place
    fn access_place(
        &mut self,
        context: Context,
        place: &Place<'tcx>,
        kind: (ShallowOrDeep, ReadOrWrite),
        _is_local_mutation_allowed: LocalMutationIsAllowed,
    ) {
        let (sd, rw) = kind;
        // note: not doing check_access_permissions checks because they don't generate invalidates
        self.check_access_for_conflict(context, place, sd, rw);
    }

    fn check_access_for_conflict(
        &mut self,
        context: Context,
        place: &Place<'tcx>,
        sd: ShallowOrDeep,
        rw: ReadOrWrite,
    ) {
        debug!(
            "invalidation::check_access_for_conflict(context={:?}, place={:?}, sd={:?}, \
             rw={:?})",
            context,
            place,
            sd,
            rw,
        );
        self.each_borrow_involving_path(
            context,
            (sd, place),
            |this, borrow_index, borrow| match (rw, borrow.kind) {
                // Obviously an activation is compatible with its own
                // reservation (or even prior activating uses of same
                // borrow); so don't check if they interfere.
                //
                // NOTE: *reservations* do conflict with themselves;
                // thus aren't injecting unsoundenss w/ this check.)
                (Activation(_, activating), _) if activating == borrow_index => {
                    // Activating a borrow doesn't generate any invalidations, since we
                    // have already taken the reservation
                }

                (Read(_), BorrowKind::Shared) | (Reservation(..), BorrowKind::Shared) => {
                    // Reads/reservations don't invalidate shared borrows
                }

                (Read(_), BorrowKind::Unique) | (Read(_), BorrowKind::Mut { .. }) => {
                    // Reading from mere reservations of mutable-borrows is OK.
                    if !this.is_active(borrow, context.loc) {
                        // If the borrow isn't active yet, reads don't invalidate it
                        assert!(this.allow_two_phase_borrow(borrow.kind));
                        return;
                    }

                    // Unique and mutable borrows are invalidated by reads from any
                    // involved path
                    this.generate_invalidates(borrow_index, context.loc);
                }

                (Reservation(_), BorrowKind::Unique)
                | (Reservation(_), BorrowKind::Mut { .. })
                | (Activation(_, _), _)
                | (Write(_), _) => {
                    // unique or mutable borrows are invalidated by writes.
                    // Reservations count as writes since we need to check
                    // that activating the borrow will be OK
                    // TOOD(bob_twinkles) is this actually the right thing to do?
                    this.generate_invalidates(borrow_index, context.loc);
                }
            },
        );
    }

    /// Returns true if the borrow represented by `kind` is
    /// allowed to be split into separate Reservation and
    /// Activation phases.
    fn allow_two_phase_borrow(&self, kind: BorrowKind) -> bool {
        self.infcx.tcx.two_phase_borrows()
            && (kind.allows_two_phase_borrow()
                || self.infcx.tcx.sess.opts.debugging_opts.two_phase_beyond_autoref)
    }

    /// Generate a new invalidates(L, B) fact
    fn generate_invalidates(&mut self, b: BorrowIndex, l: Location) {
        let lidx = self.location_table.mid_index(l);
        self.all_facts.invalidates.push((lidx, b));
    }

    /// This function iterates over all borrows that intersect with an
    /// an access to a place, invoking the `op` callback for each one.
    ///
    /// "Current borrow" here means a borrow that reaches the point in
    /// the control-flow where the access occurs.
    ///
    /// The borrow's phase is represented by the IsActive parameter
    /// passed to the callback.
    fn each_borrow_involving_path<F>(
        &mut self,
        _context: Context,
        access_place: (ShallowOrDeep, &Place<'tcx>),
        mut op: F,
    ) where
        F: FnMut(&mut Self, BorrowIndex, &BorrowData<'tcx>),
    {
        let (access, place) = access_place;

        // FIXME: analogous code in check_loans first maps `place` to
        // its base_path.

        // check for loan restricting path P being used. Accounts for
        // borrows of P, P.a.b, etc.
        let borrow_set = self.borrow_set.clone();
        for i in borrow_set.borrows.indices() {
            let borrowed = &borrow_set[i];

            if self.places_conflict(&borrowed.borrowed_place, place, access) {
                debug!(
                    "each_borrow_involving_path: {:?} @ {:?} vs. {:?}/{:?}",
                    i, borrowed, place, access
                );
                op(self, i, borrowed);
            }
        }
    }

    /// Determine if a given two-phase borrow is active using dominator information
    fn is_active(&self, borrow: &BorrowData, location: Location) -> bool {
        // If it's not two-phase, the borrow is definitely active
        if !self.allow_two_phase_borrow(borrow.kind) {
            return true;
        }
        if borrow.activation_location.is_none() {
            return false;
        }
        let activation_location = borrow.activation_location.unwrap();
        if activation_location.block == location.block {
            activation_location.statement_index >= location.statement_index
        } else {
            self.mir.dominators().is_dominated_by(location.block, activation_location.block)
        }
    }

    /// Returns whether an access of kind `access` to `access_place` conflicts with
    /// a borrow/full access to `borrow_place` (for deep accesses to mutable
    /// locations, this function is symmetric between `borrow_place` & `access_place`).
    fn places_conflict(
        &mut self,
        borrow_place: &Place<'tcx>,
        access_place: &Place<'tcx>,
        access: ShallowOrDeep,
    ) -> bool {
        debug!(
            "places_conflict({:?},{:?},{:?})",
            borrow_place, access_place, access
        );

        // Return all the prefixes of `place` in reverse order, including
        // downcasts.
        fn place_elements<'a, 'tcx>(place: &'a Place<'tcx>) -> Vec<&'a Place<'tcx>> {
            let mut result = vec![];
            let mut place = place;
            loop {
                result.push(place);
                match place {
                    Place::Projection(interior) => {
                        place = &interior.base;
                    }
                    Place::Local(_) | Place::Static(_) => {
                        result.reverse();
                        return result;
                    }
                }
            }
        }

        let borrow_components = place_elements(borrow_place);
        let access_components = place_elements(access_place);
        debug!(
            "places_conflict: components {:?} / {:?}",
            borrow_components, access_components
        );

        let borrow_components = borrow_components
            .into_iter()
            .map(Some)
            .chain(iter::repeat(None));
        let access_components = access_components
            .into_iter()
            .map(Some)
            .chain(iter::repeat(None));
        // The borrowck rules for proving disjointness are applied from the "root" of the
        // borrow forwards, iterating over "similar" projections in lockstep until
        // we can prove overlap one way or another. Essentially, we treat `Overlap` as
        // a monoid and report a conflict if the product ends up not being `Disjoint`.
        //
        // At each step, if we didn't run out of borrow or place, we know that our elements
        // have the same type, and that they only overlap if they are the identical.
        //
        // For example, if we are comparing these:
        // BORROW:  (*x1[2].y).z.a
        // ACCESS:  (*x1[i].y).w.b
        //
        // Then our steps are:
        //       x1         |   x1          -- places are the same
        //       x1[2]      |   x1[i]       -- equal or disjoint (disjoint if indexes differ)
        //       x1[2].y    |   x1[i].y     -- equal or disjoint
        //      *x1[2].y    |  *x1[i].y     -- equal or disjoint
        //     (*x1[2].y).z | (*x1[i].y).w  -- we are disjoint and don't need to check more!
        //
        // Because `zip` does potentially bad things to the iterator inside, this loop
        // also handles the case where the access might be a *prefix* of the borrow, e.g.
        //
        // BORROW:  (*x1[2].y).z.a
        // ACCESS:  x1[i].y
        //
        // Then our steps are:
        //       x1         |   x1          -- places are the same
        //       x1[2]      |   x1[i]       -- equal or disjoint (disjoint if indexes differ)
        //       x1[2].y    |   x1[i].y     -- equal or disjoint
        //
        // -- here we run out of access - the borrow can access a part of it. If this
        // is a full deep access, then we *know* the borrow conflicts with it. However,
        // if the access is shallow, then we can proceed:
        //
        //       x1[2].y    | (*x1[i].y)    -- a deref! the access can't get past this, so we
        //                                     are disjoint
        //
        // Our invariant is, that at each step of the iteration:
        //  - If we didn't run out of access to match, our borrow and access are comparable
        //    and either equal or disjoint.
        //  - If we did run out of accesss, the borrow can access a part of it.
        for (borrow_c, access_c) in borrow_components.zip(access_components) {
            // loop invariant: borrow_c is always either equal to access_c or disjoint from it.
            debug!("places_conflict: {:?} vs. {:?}", borrow_c, access_c);
            match (borrow_c, access_c) {
                (None, _) => {
                    // If we didn't run out of access, the borrow can access all of our
                    // place (e.g. a borrow of `a.b` with an access to `a.b.c`),
                    // so we have a conflict.
                    //
                    // If we did, then we still know that the borrow can access a *part*
                    // of our place that our access cares about (a borrow of `a.b.c`
                    // with an access to `a.b`), so we still have a conflict.
                    //
                    // FIXME: Differs from AST-borrowck; includes drive-by fix
                    // to #38899. Will probably need back-compat mode flag.
                    debug!("places_conflict: full borrow, CONFLICT");
                    return true;
                }
                (Some(borrow_c), None) => {
                    // We know that the borrow can access a part of our place. This
                    // is a conflict if that is a part our access cares about.

                    let (base, elem) = match borrow_c {
                        Place::Projection(box Projection { base, elem }) => (base, elem),
                        _ => bug!("place has no base?"),
                    };
                    let base_ty = base.ty(self.mir, self.infcx.tcx).to_ty(self.infcx.tcx);

                    match (elem, &base_ty.sty, access) {
                        (_, _, Shallow(Some(ArtificialField::Discriminant)))
                        | (_, _, Shallow(Some(ArtificialField::ArrayLength))) => {
                            // The discriminant and array length are like
                            // additional fields on the type; they do not
                            // overlap any existing data there. Furthermore,
                            // they cannot actually be a prefix of any
                            // borrowed place (at least in MIR as it is
                            // currently.)
                            //
                            // e.g. a (mutable) borrow of `a[5]` while we read the
                            // array length of `a`.
                            debug!("places_conflict: implicit field");
                            return false;
                        }

                        (ProjectionElem::Deref, _, Shallow(None)) => {
                            // e.g. a borrow of `*x.y` while we shallowly access `x.y` or some
                            // prefix thereof - the shallow access can't touch anything behind
                            // the pointer.
                            debug!("places_conflict: shallow access behind ptr");
                            return false;
                        }
                        (
                            ProjectionElem::Deref,
                            ty::TyRef( _, _, hir::MutImmutable),
                            _,
                        ) => {
                            // the borrow goes through a dereference of a shared reference.
                            //
                            // I'm not sure why we are tracking these borrows - shared
                            // references can *always* be aliased, which means the
                            // permission check already account for this borrow.
                            debug!("places_conflict: behind a shared ref");
                            return false;
                        }

                        (ProjectionElem::Deref, _, Deep)
                        | (ProjectionElem::Field { .. }, _, _)
                        | (ProjectionElem::Index { .. }, _, _)
                        | (ProjectionElem::ConstantIndex { .. }, _, _)
                        | (ProjectionElem::Subslice { .. }, _, _)
                        | (ProjectionElem::Downcast { .. }, _, _) => {
                            // Recursive case. This can still be disjoint on a
                            // further iteration if this a shallow access and
                            // there's a deref later on, e.g. a borrow
                            // of `*x.y` while accessing `x`.
                        }
                    }
                }
                (Some(borrow_c), Some(access_c)) => {
                    match self.place_element_conflict(&borrow_c, access_c) {
                        Overlap::Arbitrary => {
                            // We have encountered different fields of potentially
                            // the same union - the borrow now partially overlaps.
                            //
                            // There is no *easy* way of comparing the fields
                            // further on, because they might have different types
                            // (e.g. borrows of `u.a.0` and `u.b.y` where `.0` and
                            // `.y` come from different structs).
                            //
                            // We could try to do some things here - e.g. count
                            // dereferences - but that's probably not a good
                            // idea, at least for now, so just give up and
                            // report a conflict. This is unsafe code anyway so
                            // the user could always use raw pointers.
                            debug!("places_conflict: arbitrary -> conflict");
                            return true;
                        }
                        Overlap::EqualOrDisjoint => {
                            // This is the recursive case - proceed to the next element.
                        }
                        Overlap::Disjoint => {
                            // We have proven the borrow disjoint - further
                            // projections will remain disjoint.
                            debug!("places_conflict: disjoint");
                            return false;
                        }
                    }
                }
            }
        }
        unreachable!("iter::repeat returned None")
    }

    // Given that the bases of `elem1` and `elem2` are always either equal
    // or disjoint (and have the same type!), return the overlap situation
    // between `elem1` and `elem2`.
    fn place_element_conflict(&self, elem1: &Place<'tcx>, elem2: &Place<'tcx>) -> Overlap {
        match (elem1, elem2) {
            (Place::Local(l1), Place::Local(l2)) => {
                if l1 == l2 {
                    // the same local - base case, equal
                    debug!("place_element_conflict: DISJOINT-OR-EQ-LOCAL");
                    Overlap::EqualOrDisjoint
                } else {
                    // different locals - base case, disjoint
                    debug!("place_element_conflict: DISJOINT-LOCAL");
                    Overlap::Disjoint
                }
            }
            (Place::Static(static1), Place::Static(static2)) => {
                if static1.def_id != static2.def_id {
                    debug!("place_element_conflict: DISJOINT-STATIC");
                    Overlap::Disjoint
                } else if self.infcx.tcx.is_static(static1.def_id) ==
                          Some(hir::Mutability::MutMutable) {
                    // We ignore mutable statics - they can only be unsafe code.
                    debug!("place_element_conflict: IGNORE-STATIC-MUT");
                    Overlap::Disjoint
                } else {
                    debug!("place_element_conflict: DISJOINT-OR-EQ-STATIC");
                    Overlap::EqualOrDisjoint
                }
            }
            (Place::Local(_), Place::Static(_)) | (Place::Static(_), Place::Local(_)) => {
                debug!("place_element_conflict: DISJOINT-STATIC-LOCAL");
                Overlap::Disjoint
            }
            (Place::Projection(pi1), Place::Projection(pi2)) => {
                match (&pi1.elem, &pi2.elem) {
                    (ProjectionElem::Deref, ProjectionElem::Deref) => {
                        // derefs (e.g. `*x` vs. `*x`) - recur.
                        debug!("place_element_conflict: DISJOINT-OR-EQ-DEREF");
                        Overlap::EqualOrDisjoint
                    }
                    (ProjectionElem::Field(f1, _), ProjectionElem::Field(f2, _)) => {
                        if f1 == f2 {
                            // same field (e.g. `a.y` vs. `a.y`) - recur.
                            debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");
                            Overlap::EqualOrDisjoint
                        } else {
                            let ty = pi1.base.ty(self.mir, self.infcx.tcx).to_ty(self.infcx.tcx);
                            match ty.sty {
                                ty::TyAdt(def, _) if def.is_union() => {
                                    // Different fields of a union, we are basically stuck.
                                    debug!("place_element_conflict: STUCK-UNION");
                                    Overlap::Arbitrary
                                }
                                _ => {
                                    // Different fields of a struct (`a.x` vs. `a.y`). Disjoint!
                                    debug!("place_element_conflict: DISJOINT-FIELD");
                                    Overlap::Disjoint
                                }
                            }
                        }
                    }
                    (ProjectionElem::Downcast(_, v1), ProjectionElem::Downcast(_, v2)) => {
                        // different variants are treated as having disjoint fields,
                        // even if they occupy the same "space", because it's
                        // impossible for 2 variants of the same enum to exist
                        // (and therefore, to be borrowed) at the same time.
                        //
                        // Note that this is different from unions - we *do* allow
                        // this code to compile:
                        //
                        // ```
                        // fn foo(x: &mut Result<i32, i32>) {
                        //     let mut v = None;
                        //     if let Ok(ref mut a) = *x {
                        //         v = Some(a);
                        //     }
                        //     // here, you would *think* that the
                        //     // *entirety* of `x` would be borrowed,
                        //     // but in fact only the `Ok` variant is,
                        //     // so the `Err` variant is *entirely free*:
                        //     if let Err(ref mut a) = *x {
                        //         v = Some(a);
                        //     }
                        //     drop(v);
                        // }
                        // ```
                        if v1 == v2 {
                            debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");
                            Overlap::EqualOrDisjoint
                        } else {
                            debug!("place_element_conflict: DISJOINT-FIELD");
                            Overlap::Disjoint
                        }
                    }
                    (ProjectionElem::Index(..), ProjectionElem::Index(..))
                    | (ProjectionElem::Index(..), ProjectionElem::ConstantIndex { .. })
                    | (ProjectionElem::Index(..), ProjectionElem::Subslice { .. })
                    | (ProjectionElem::ConstantIndex { .. }, ProjectionElem::Index(..))
                    | (
                        ProjectionElem::ConstantIndex { .. },
                        ProjectionElem::ConstantIndex { .. },
                    )
                    | (ProjectionElem::ConstantIndex { .. }, ProjectionElem::Subslice { .. })
                    | (ProjectionElem::Subslice { .. }, ProjectionElem::Index(..))
                    | (ProjectionElem::Subslice { .. }, ProjectionElem::ConstantIndex { .. })
                    | (ProjectionElem::Subslice { .. }, ProjectionElem::Subslice { .. }) => {
                        // Array indexes (`a[0]` vs. `a[i]`). These can either be disjoint
                        // (if the indexes differ) or equal (if they are the same), so this
                        // is the recursive case that gives "equal *or* disjoint" its meaning.
                        //
                        // Note that by construction, MIR at borrowck can't subdivide
                        // `Subslice` accesses (e.g. `a[2..3][i]` will never be present) - they
                        // are only present in slice patterns, and we "merge together" nested
                        // slice patterns. That means we don't have to think about these. It's
                        // probably a good idea to assert this somewhere, but I'm too lazy.
                        //
                        // FIXME(#8636) we might want to return Disjoint if
                        // both projections are constant and disjoint.
                        debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY");
                        Overlap::EqualOrDisjoint
                    }

                    (ProjectionElem::Deref, _)
                    | (ProjectionElem::Field(..), _)
                    | (ProjectionElem::Index(..), _)
                    | (ProjectionElem::ConstantIndex { .. }, _)
                    | (ProjectionElem::Subslice { .. }, _)
                    | (ProjectionElem::Downcast(..), _) => bug!(
                        "mismatched projections in place_element_conflict: {:?} and {:?}",
                        elem1,
                        elem2
                    ),
                }
            }
            (Place::Projection(_), _) | (_, Place::Projection(_)) => bug!(
                "unexpected elements in place_element_conflict: {:?} and {:?}",
                elem1,
                elem2
            ),
        }
    }
}
