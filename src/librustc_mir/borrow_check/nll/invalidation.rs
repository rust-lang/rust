use crate::borrow_check::borrow_set::BorrowSet;
use crate::borrow_check::location::LocationTable;
use crate::borrow_check::{JustWrite, WriteAndRead};
use crate::borrow_check::{AccessDepth, Deep, Shallow};
use crate::borrow_check::{ReadOrWrite, Activation, Read, Reservation, Write};
use crate::borrow_check::{LocalMutationIsAllowed, MutateMode};
use crate::borrow_check::ArtificialField;
use crate::borrow_check::{ReadKind, WriteKind};
use crate::borrow_check::nll::facts::AllFacts;
use crate::borrow_check::path_utils::*;
use crate::dataflow::indexes::BorrowIndex;
use rustc::ty::TyCtxt;
use rustc::mir::visit::Visitor;
use rustc::mir::{BasicBlock, Location, Body, Place, Rvalue};
use rustc::mir::{Statement, StatementKind};
use rustc::mir::TerminatorKind;
use rustc::mir::{Operand, BorrowKind};
use rustc_data_structures::graph::dominators::Dominators;

pub(super) fn generate_invalidates<'tcx>(
    tcx: TyCtxt<'tcx>,
    all_facts: &mut Option<AllFacts>,
    location_table: &LocationTable,
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) {
    if all_facts.is_none() {
        // Nothing to do if we don't have any facts
        return;
    }

    if let Some(all_facts) = all_facts {
        let dominators = body.dominators();
        let mut ig = InvalidationGenerator {
            all_facts,
            borrow_set,
            tcx,
            location_table,
            body,
            dominators,
        };
        ig.visit_body(body);
    }
}

struct InvalidationGenerator<'cx, 'tcx> {
    tcx: TyCtxt<'tcx>,
    all_facts: &'cx mut AllFacts,
    location_table: &'cx LocationTable,
    body: &'cx Body<'tcx>,
    dominators: Dominators<BasicBlock>,
    borrow_set: &'cx BorrowSet<'tcx>,
}

/// Visits the whole MIR and generates `invalidates()` facts.
/// Most of the code implementing this was stolen from `borrow_check/mod.rs`.
impl<'cx, 'tcx> Visitor<'tcx> for InvalidationGenerator<'cx, 'tcx> {
    fn visit_statement(
        &mut self,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        self.check_activations(location);

        match statement.kind {
            StatementKind::Assign(ref lhs, ref rhs) => {
                self.consume_rvalue(
                    location,
                    rhs,
                );

                self.mutate_place(
                    location,
                    lhs,
                    Shallow(None),
                    JustWrite
                );
            }
            StatementKind::FakeRead(_, _) => {
                // Only relavent for initialized/liveness/safety checks.
            }
            StatementKind::SetDiscriminant {
                ref place,
                variant_index: _,
            } => {
                self.mutate_place(
                    location,
                    place,
                    Shallow(None),
                    JustWrite,
                );
            }
            StatementKind::InlineAsm(ref asm) => {
                for (o, output) in asm.asm.outputs.iter().zip(asm.outputs.iter()) {
                    if o.is_indirect {
                        // FIXME(eddyb) indirect inline asm outputs should
                        // be encoded through MIR place derefs instead.
                        self.access_place(
                            location,
                            output,
                            (Deep, Read(ReadKind::Copy)),
                            LocalMutationIsAllowed::No,
                        );
                    } else {
                        self.mutate_place(
                            location,
                            output,
                            if o.is_rw { Deep } else { Shallow(None) },
                            if o.is_rw { WriteAndRead } else { JustWrite },
                        );
                    }
                }
                for (_, input) in asm.inputs.iter() {
                    self.consume_operand(location, input);
                }
            }
            StatementKind::Nop |
            StatementKind::AscribeUserType(..) |
            StatementKind::Retag { .. } |
            StatementKind::StorageLive(..) => {
                // `Nop`, `AscribeUserType`, `Retag`, and `StorageLive` are irrelevant
                // to borrow check.
            }
            StatementKind::StorageDead(local) => {
                self.access_place(
                    location,
                    &Place::from(local),
                    (Shallow(None), Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                );
            }
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator_kind(
        &mut self,
        kind: &TerminatorKind<'tcx>,
        location: Location
    ) {
        self.check_activations(location);

        match kind {
            TerminatorKind::SwitchInt {
                ref discr,
                switch_ty: _,
                values: _,
                targets: _,
            } => {
                self.consume_operand(location, discr);
            }
            TerminatorKind::Drop {
                location: ref drop_place,
                target: _,
                unwind: _,
            } => {
                self.access_place(
                    location,
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
                    location,
                    drop_place,
                    Deep,
                    JustWrite,
                );
                self.consume_operand(
                    location,
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
                self.consume_operand(location, func);
                for arg in args {
                    self.consume_operand(location, arg);
                }
                if let Some((ref dest, _ /*bb*/)) = *destination {
                    self.mutate_place(
                        location,
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
                self.consume_operand(location, cond);
                use rustc::mir::interpret::InterpError::BoundsCheck;
                if let BoundsCheck { ref len, ref index } = *msg {
                    self.consume_operand(location, len);
                    self.consume_operand(location, index);
                }
            }
            TerminatorKind::Yield {
                ref value,
                resume,
                drop: _,
            } => {
                self.consume_operand(location, value);

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
                imaginary_target: _,
            }
            | TerminatorKind::FalseUnwind {
                real_target: _,
                unwind: _,
            } => {
                // no data used, thus irrelevant to borrowck
            }
        }

        self.super_terminator_kind(kind, location);
    }
}

impl<'cx, 'tcx> InvalidationGenerator<'cx, 'tcx> {
    /// Simulates mutation of a place.
    fn mutate_place(
        &mut self,
        location: Location,
        place: &Place<'tcx>,
        kind: AccessDepth,
        _mode: MutateMode,
    ) {
        self.access_place(
            location,
            place,
            (kind, Write(WriteKind::Mutate)),
            LocalMutationIsAllowed::ExceptUpvars,
        );
    }

    /// Simulates consumption of an operand.
    fn consume_operand(
        &mut self,
        location: Location,
        operand: &Operand<'tcx>,
    ) {
        match *operand {
            Operand::Copy(ref place) => {
                self.access_place(
                    location,
                    place,
                    (Deep, Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                );
            }
            Operand::Move(ref place) => {
                self.access_place(
                    location,
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
        location: Location,
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
                        if allow_two_phase_borrow(bk) {
                            (Deep, Reservation(wk))
                        } else {
                            (Deep, Write(wk))
                        }
                    }
                };

                self.access_place(
                    location,
                    place,
                    access_kind,
                    LocalMutationIsAllowed::No,
                );
            }

            Rvalue::Use(ref operand)
            | Rvalue::Repeat(ref operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, ref operand)
            | Rvalue::Cast(_ /*cast_kind*/, ref operand, _ /*ty*/) => {
                self.consume_operand(location, operand)
            }

            Rvalue::Len(ref place) | Rvalue::Discriminant(ref place) => {
                let af = match *rvalue {
                    Rvalue::Len(..) => Some(ArtificialField::ArrayLength),
                    Rvalue::Discriminant(..) => None,
                    _ => unreachable!(),
                };
                self.access_place(
                    location,
                    place,
                    (Shallow(af), Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                );
            }

            Rvalue::BinaryOp(_bin_op, ref operand1, ref operand2)
            | Rvalue::CheckedBinaryOp(_bin_op, ref operand1, ref operand2) => {
                self.consume_operand(location, operand1);
                self.consume_operand(location, operand2);
            }

            Rvalue::NullaryOp(_op, _ty) => {
            }

            Rvalue::Aggregate(_, ref operands) => {
                for operand in operands {
                    self.consume_operand(location, operand);
                }
            }
        }
    }

    /// Simulates an access to a place.
    fn access_place(
        &mut self,
        location: Location,
        place: &Place<'tcx>,
        kind: (AccessDepth, ReadOrWrite),
        _is_local_mutation_allowed: LocalMutationIsAllowed,
    ) {
        let (sd, rw) = kind;
        // note: not doing check_access_permissions checks because they don't generate invalidates
        self.check_access_for_conflict(location, place, sd, rw);
    }

    fn check_access_for_conflict(
        &mut self,
        location: Location,
        place: &Place<'tcx>,
        sd: AccessDepth,
        rw: ReadOrWrite,
    ) {
        debug!(
            "invalidation::check_access_for_conflict(location={:?}, place={:?}, sd={:?}, \
             rw={:?})",
            location,
            place,
            sd,
            rw,
        );
        let tcx = self.tcx;
        let body = self.body;
        let borrow_set = self.borrow_set.clone();
        let indices = self.borrow_set.borrows.indices();
        each_borrow_involving_path(
            self,
            tcx,
            body,
            location,
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

                    (Read(_), BorrowKind::Shallow)
                    | (Read(_), BorrowKind::Shared)
                    | (Read(ReadKind::Borrow(BorrowKind::Shallow)), BorrowKind::Unique)
                    | (Read(ReadKind::Borrow(BorrowKind::Shallow)), BorrowKind::Mut { .. }) => {
                        // Reads don't invalidate shared or shallow borrows
                    }

                    (Read(_), BorrowKind::Unique) | (Read(_), BorrowKind::Mut { .. }) => {
                        // Reading from mere reservations of mutable-borrows is OK.
                        if !is_active(&this.dominators, borrow, location) {
                            // If the borrow isn't active yet, reads don't invalidate it
                            assert!(allow_two_phase_borrow(borrow.kind));
                            return Control::Continue;
                        }

                        // Unique and mutable borrows are invalidated by reads from any
                        // involved path
                        this.generate_invalidates(borrow_index, location);
                    }

                    (Reservation(_), _)
                    | (Activation(_, _), _)
                    | (Write(_), _) => {
                        // unique or mutable borrows are invalidated by writes.
                        // Reservations count as writes since we need to check
                        // that activating the borrow will be OK
                        // FIXME(bob_twinkles) is this actually the right thing to do?
                        this.generate_invalidates(borrow_index, location);
                    }
                }
                Control::Continue
            },
        );
    }


    /// Generates a new `invalidates(L, B)` fact.
    fn generate_invalidates(&mut self, b: BorrowIndex, l: Location) {
        let lidx = self.location_table.start_index(l);
        self.all_facts.invalidates.push((lidx, b));
    }

    fn check_activations(
        &mut self,
        location: Location,
    ) {
        // Two-phase borrow support: For each activation that is newly
        // generated at this statement, check if it interferes with
        // another borrow.
        for &borrow_index in self.borrow_set.activations_at_location(location) {
            let borrow = &self.borrow_set[borrow_index];

            // only mutable borrows should be 2-phase
            assert!(match borrow.kind {
                BorrowKind::Shared | BorrowKind::Shallow => false,
                BorrowKind::Unique | BorrowKind::Mut { .. } => true,
            });

            self.access_place(
                location,
                &borrow.borrowed_place,
                (
                    Deep,
                    Activation(WriteKind::MutableBorrow(borrow.kind), borrow_index),
                ),
                LocalMutationIsAllowed::No,
            );

            // We do not need to call `check_if_path_or_subpath_is_moved`
            // again, as we already called it when we made the
            // initial reservation.
        }
    }
}
