// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::BorrowSet;
use borrow_check::location::LocationTable;
use borrow_check::{JustWrite, WriteAndRead};
use borrow_check::{AccessDepth, Deep, Shallow};
use borrow_check::{ReadOrWrite, Activation, Read, Reservation, Write};
use borrow_check::{Context, ContextKind};
use borrow_check::{LocalMutationIsAllowed, MutateMode};
use borrow_check::ArtificialField;
use borrow_check::{ReadKind, WriteKind};
use borrow_check::nll::facts::AllFacts;
use borrow_check::path_utils::*;
use dataflow::move_paths::indexes::BorrowIndex;
use rustc::ty::TyCtxt;
use rustc::mir::visit::Visitor;
use rustc::mir::{BasicBlock, Location, Mir, Place, Rvalue};
use rustc::mir::{Statement, StatementKind};
use rustc::mir::{Terminator, TerminatorKind};
use rustc::mir::{Operand, BorrowKind};
use rustc_data_structures::graph::dominators::Dominators;

pub(super) fn generate_invalidates<'cx, 'gcx, 'tcx>(
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    all_facts: &mut Option<AllFacts>,
    location_table: &LocationTable,
    mir: &Mir<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) {
    if all_facts.is_none() {
        // Nothing to do if we don't have any facts
        return;
    }

    if let Some(all_facts) = all_facts {
        let dominators = mir.dominators();
        let mut ig = InvalidationGenerator {
            all_facts,
            borrow_set,
            tcx,
            location_table,
            mir,
            dominators,
        };
        ig.visit_mir(mir);
    }
}

struct InvalidationGenerator<'cx, 'tcx: 'cx, 'gcx: 'tcx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    all_facts: &'cx mut AllFacts,
    location_table: &'cx LocationTable,
    mir: &'cx Mir<'tcx>,
    dominators: Dominators<BasicBlock>,
    borrow_set: &'cx BorrowSet<'tcx>,
}

/// Visits the whole MIR and generates invalidates() facts
/// Most of the code implementing this was stolen from borrow_check/mod.rs
impl<'cx, 'tcx, 'gcx> Visitor<'tcx> for InvalidationGenerator<'cx, 'tcx, 'gcx> {
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
            StatementKind::FakeRead(_, ref place) => {
                self.access_place(
                    ContextKind::FakeRead.new(location),
                    place,
                    (Deep, Read(ReadKind::Borrow(BorrowKind::Shared))),
                    LocalMutationIsAllowed::No,
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
                for (o, output) in asm.outputs.iter().zip(outputs.iter()) {
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
                for (_, input) in inputs.iter() {
                    self.consume_operand(context, input);
                }
            }
            StatementKind::Nop |
            StatementKind::AscribeUserType(..) |
            StatementKind::Retag { .. } |
            StatementKind::EscapeToRaw { .. } |
            StatementKind::StorageLive(..) => {
                // `Nop`, `AscribeUserType`, `Retag`, and `StorageLive` are irrelevant
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
                self.access_place(
                    ContextKind::Drop.new(location),
                    drop_place,
                    (AccessDepth::Drop, Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                );
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
                from_hir_call: _,
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
                resume,
                drop: _,
            } => {
                self.consume_operand(ContextKind::Yield.new(location), value);

                // Invalidate all borrows of local places
                let borrow_set = self.borrow_set.clone();
                let resume = self.location_table.start_index(resume.start_location());
                for i in borrow_set.borrows.indices() {
                    if borrow_of_local_data(&borrow_set.borrows[i].borrowed_place) {
                        self.all_facts.invalidates.push((resume, i));
                    }
                }
            }
            TerminatorKind::Resume | TerminatorKind::Return | TerminatorKind::GeneratorDrop => {
                // Invalidate all borrows of local places
                let borrow_set = self.borrow_set.clone();
                let start = self.location_table.start_index(location);
                for i in borrow_set.borrows.indices() {
                    if borrow_of_local_data(&borrow_set.borrows[i].borrowed_place) {
                        self.all_facts.invalidates.push((start, i));
                    }
                }
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

impl<'cg, 'cx, 'tcx, 'gcx> InvalidationGenerator<'cx, 'tcx, 'gcx> {
    /// Simulates mutation of a place
    fn mutate_place(
        &mut self,
        context: Context,
        place: &Place<'tcx>,
        kind: AccessDepth,
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
                    BorrowKind::Shallow => {
                        (Shallow(Some(ArtificialField::ShallowBorrow)), Read(ReadKind::Borrow(bk)))
                    },
                    BorrowKind::Shared => (Deep, Read(ReadKind::Borrow(bk))),
                    BorrowKind::Unique | BorrowKind::Mut { .. } => {
                        let wk = WriteKind::MutableBorrow(bk);
                        if allow_two_phase_borrow(&self.tcx, bk) {
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
        kind: (AccessDepth, ReadOrWrite),
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
        sd: AccessDepth,
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
        let tcx = self.tcx;
        let mir = self.mir;
        let borrow_set = self.borrow_set.clone();
        let indices = self.borrow_set.borrows.indices();
        each_borrow_involving_path(
            self,
            tcx,
            mir,
            context,
            (sd, place),
            &borrow_set.clone(),
            indices,
            |this, borrow_index, borrow| {
                match (rw, borrow.kind) {
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

                    (Read(_), BorrowKind::Shallow) | (Reservation(..), BorrowKind::Shallow)
                    | (Read(_), BorrowKind::Shared) | (Reservation(..), BorrowKind::Shared) => {
                        // Reads/reservations don't invalidate shared or shallow borrows
                    }

                    (Read(_), BorrowKind::Unique) | (Read(_), BorrowKind::Mut { .. }) => {
                        // Reading from mere reservations of mutable-borrows is OK.
                        if !is_active(&this.dominators, borrow, context.loc) {
                            // If the borrow isn't active yet, reads don't invalidate it
                            assert!(allow_two_phase_borrow(&this.tcx, borrow.kind));
                            return Control::Continue;
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
                            // FIXME(bob_twinkles) is this actually the right thing to do?
                            this.generate_invalidates(borrow_index, context.loc);
                        }
                }
                Control::Continue
            },
        );
    }


    /// Generate a new invalidates(L, B) fact
    fn generate_invalidates(&mut self, b: BorrowIndex, l: Location) {
        let lidx = self.location_table.start_index(l);
        self.all_facts.invalidates.push((lidx, b));
    }
}

