use rustc_data_structures::graph::dominators::Dominators;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{BasicBlock, Body, Location, Place, Rvalue};
use rustc_middle::mir::{BorrowKind, Mutability, Operand};
use rustc_middle::mir::{InlineAsmOperand, Terminator, TerminatorKind};
use rustc_middle::mir::{Statement, StatementKind};
use rustc_middle::ty::TyCtxt;
use std::iter;

use crate::{
    borrow_set::BorrowSet, facts::AllFacts, location::LocationTable, path_utils::*, AccessDepth,
    Activation, ArtificialField, BorrowIndex, Deep, JustWrite, LocalMutationIsAllowed, MutateMode,
    Read, ReadKind, ReadOrWrite, Reservation, Shallow, Write, WriteAndRead, WriteKind,
};

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
        let _prof_timer = tcx.prof.generic_activity("polonius_fact_generation");
        let dominators = body.dominators();
        let mut ig = InvalidationGenerator {
            all_facts,
            borrow_set,
            tcx,
            location_table,
            body: &body,
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
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        self.check_activations(location);

        match &statement.kind {
            StatementKind::Assign(box (lhs, rhs)) => {
                self.consume_rvalue(location, rhs);

                self.mutate_place(location, *lhs, Shallow(None), JustWrite);
            }
            StatementKind::FakeRead(box (_, _)) => {
                // Only relevant for initialized/liveness/safety checks.
            }
            StatementKind::SetDiscriminant { place, variant_index: _ } => {
                self.mutate_place(location, **place, Shallow(None), JustWrite);
            }
            StatementKind::LlvmInlineAsm(asm) => {
                for (o, output) in iter::zip(&asm.asm.outputs, &*asm.outputs) {
                    if o.is_indirect {
                        // FIXME(eddyb) indirect inline asm outputs should
                        // be encoded through MIR place derefs instead.
                        self.access_place(
                            location,
                            *output,
                            (Deep, Read(ReadKind::Copy)),
                            LocalMutationIsAllowed::No,
                        );
                    } else {
                        self.mutate_place(
                            location,
                            *output,
                            if o.is_rw { Deep } else { Shallow(None) },
                            if o.is_rw { WriteAndRead } else { JustWrite },
                        );
                    }
                }
                for (_, input) in asm.inputs.iter() {
                    self.consume_operand(location, input);
                }
            }
            StatementKind::CopyNonOverlapping(box rustc_middle::mir::CopyNonOverlapping {
                ref src,
                ref dst,
                ref count,
            }) => {
                self.consume_operand(location, src);
                self.consume_operand(location, dst);
                self.consume_operand(location, count);
            }
            StatementKind::Nop
            | StatementKind::Coverage(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Retag { .. }
            | StatementKind::StorageLive(..) => {
                // `Nop`, `AscribeUserType`, `Retag`, and `StorageLive` are irrelevant
                // to borrow check.
            }
            StatementKind::StorageDead(local) => {
                self.access_place(
                    location,
                    Place::from(*local),
                    (Shallow(None), Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                );
            }
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.check_activations(location);

        match &terminator.kind {
            TerminatorKind::SwitchInt { ref discr, switch_ty: _, targets: _ } => {
                self.consume_operand(location, discr);
            }
            TerminatorKind::Drop { place: drop_place, target: _, unwind: _ } => {
                self.access_place(
                    location,
                    *drop_place,
                    (AccessDepth::Drop, Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                );
            }
            TerminatorKind::DropAndReplace {
                place: drop_place,
                value: ref new_value,
                target: _,
                unwind: _,
            } => {
                self.mutate_place(location, *drop_place, Deep, JustWrite);
                self.consume_operand(location, new_value);
            }
            TerminatorKind::Call {
                ref func,
                ref args,
                destination,
                cleanup: _,
                from_hir_call: _,
                fn_span: _,
            } => {
                self.consume_operand(location, func);
                for arg in args {
                    self.consume_operand(location, arg);
                }
                if let Some((dest, _ /*bb*/)) = destination {
                    self.mutate_place(location, *dest, Deep, JustWrite);
                }
            }
            TerminatorKind::Assert { ref cond, expected: _, ref msg, target: _, cleanup: _ } => {
                self.consume_operand(location, cond);
                use rustc_middle::mir::AssertKind;
                if let AssertKind::BoundsCheck { ref len, ref index } = *msg {
                    self.consume_operand(location, len);
                    self.consume_operand(location, index);
                }
            }
            TerminatorKind::Yield { ref value, resume, resume_arg, drop: _ } => {
                self.consume_operand(location, value);

                // Invalidate all borrows of local places
                let borrow_set = self.borrow_set;
                let resume = self.location_table.start_index(resume.start_location());
                for (i, data) in borrow_set.iter_enumerated() {
                    if borrow_of_local_data(data.borrowed_place) {
                        self.all_facts.loan_invalidated_at.push((resume, i));
                    }
                }

                self.mutate_place(location, *resume_arg, Deep, JustWrite);
            }
            TerminatorKind::Resume | TerminatorKind::Return | TerminatorKind::GeneratorDrop => {
                // Invalidate all borrows of local places
                let borrow_set = self.borrow_set;
                let start = self.location_table.start_index(location);
                for (i, data) in borrow_set.iter_enumerated() {
                    if borrow_of_local_data(data.borrowed_place) {
                        self.all_facts.loan_invalidated_at.push((start, i));
                    }
                }
            }
            TerminatorKind::InlineAsm {
                template: _,
                ref operands,
                options: _,
                line_spans: _,
                destination: _,
            } => {
                for op in operands {
                    match *op {
                        InlineAsmOperand::In { reg: _, ref value } => {
                            self.consume_operand(location, value);
                        }
                        InlineAsmOperand::Out { reg: _, late: _, place, .. } => {
                            if let Some(place) = place {
                                self.mutate_place(location, place, Shallow(None), JustWrite);
                            }
                        }
                        InlineAsmOperand::InOut { reg: _, late: _, ref in_value, out_place } => {
                            self.consume_operand(location, in_value);
                            if let Some(out_place) = out_place {
                                self.mutate_place(location, out_place, Shallow(None), JustWrite);
                            }
                        }
                        InlineAsmOperand::Const { value: _ }
                        | InlineAsmOperand::SymFn { value: _ }
                        | InlineAsmOperand::SymStatic { def_id: _ } => {}
                    }
                }
            }
            TerminatorKind::Goto { target: _ }
            | TerminatorKind::Abort
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdge { real_target: _, imaginary_target: _ }
            | TerminatorKind::FalseUnwind { real_target: _, unwind: _ } => {
                // no data used, thus irrelevant to borrowck
            }
        }

        self.super_terminator(terminator, location);
    }
}

impl<'cx, 'tcx> InvalidationGenerator<'cx, 'tcx> {
    /// Simulates mutation of a place.
    fn mutate_place(
        &mut self,
        location: Location,
        place: Place<'tcx>,
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
    fn consume_operand(&mut self, location: Location, operand: &Operand<'tcx>) {
        match *operand {
            Operand::Copy(place) => {
                self.access_place(
                    location,
                    place,
                    (Deep, Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                );
            }
            Operand::Move(place) => {
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
    fn consume_rvalue(&mut self, location: Location, rvalue: &Rvalue<'tcx>) {
        match *rvalue {
            Rvalue::Ref(_ /*rgn*/, bk, place) => {
                let access_kind = match bk {
                    BorrowKind::Shallow => {
                        (Shallow(Some(ArtificialField::ShallowBorrow)), Read(ReadKind::Borrow(bk)))
                    }
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

                self.access_place(location, place, access_kind, LocalMutationIsAllowed::No);
            }

            Rvalue::AddressOf(mutability, place) => {
                let access_kind = match mutability {
                    Mutability::Mut => (
                        Deep,
                        Write(WriteKind::MutableBorrow(BorrowKind::Mut {
                            allow_two_phase_borrow: false,
                        })),
                    ),
                    Mutability::Not => (Deep, Read(ReadKind::Borrow(BorrowKind::Shared))),
                };

                self.access_place(location, place, access_kind, LocalMutationIsAllowed::No);
            }

            Rvalue::ThreadLocalRef(_) => {}

            Rvalue::Use(ref operand)
            | Rvalue::Repeat(ref operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, ref operand)
            | Rvalue::Cast(_ /*cast_kind*/, ref operand, _ /*ty*/)
            | Rvalue::ShallowInitBox(ref operand, _ /*ty*/) => {
                self.consume_operand(location, operand)
            }

            Rvalue::Len(place) | Rvalue::Discriminant(place) => {
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

            Rvalue::BinaryOp(_bin_op, box (ref operand1, ref operand2))
            | Rvalue::CheckedBinaryOp(_bin_op, box (ref operand1, ref operand2)) => {
                self.consume_operand(location, operand1);
                self.consume_operand(location, operand2);
            }

            Rvalue::NullaryOp(_op, _ty) => {}

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
        place: Place<'tcx>,
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
        place: Place<'tcx>,
        sd: AccessDepth,
        rw: ReadOrWrite,
    ) {
        debug!(
            "invalidation::check_access_for_conflict(location={:?}, place={:?}, sd={:?}, \
             rw={:?})",
            location, place, sd, rw,
        );
        let tcx = self.tcx;
        let body = self.body;
        let borrow_set = self.borrow_set;
        let indices = self.borrow_set.indices();
        each_borrow_involving_path(
            self,
            tcx,
            body,
            location,
            (sd, place),
            borrow_set,
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

                    (Read(_), BorrowKind::Shallow | BorrowKind::Shared)
                    | (
                        Read(ReadKind::Borrow(BorrowKind::Shallow)),
                        BorrowKind::Unique | BorrowKind::Mut { .. },
                    ) => {
                        // Reads don't invalidate shared or shallow borrows
                    }

                    (Read(_), BorrowKind::Unique | BorrowKind::Mut { .. }) => {
                        // Reading from mere reservations of mutable-borrows is OK.
                        if !is_active(&this.dominators, borrow, location) {
                            // If the borrow isn't active yet, reads don't invalidate it
                            assert!(allow_two_phase_borrow(borrow.kind));
                            return Control::Continue;
                        }

                        // Unique and mutable borrows are invalidated by reads from any
                        // involved path
                        this.emit_loan_invalidated_at(borrow_index, location);
                    }

                    (Reservation(_) | Activation(_, _) | Write(_), _) => {
                        // unique or mutable borrows are invalidated by writes.
                        // Reservations count as writes since we need to check
                        // that activating the borrow will be OK
                        // FIXME(bob_twinkles) is this actually the right thing to do?
                        this.emit_loan_invalidated_at(borrow_index, location);
                    }
                }
                Control::Continue
            },
        );
    }

    /// Generates a new `loan_invalidated_at(L, B)` fact.
    fn emit_loan_invalidated_at(&mut self, b: BorrowIndex, l: Location) {
        let lidx = self.location_table.start_index(l);
        self.all_facts.loan_invalidated_at.push((lidx, b));
    }

    fn check_activations(&mut self, location: Location) {
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
                borrow.borrowed_place,
                (Deep, Activation(WriteKind::MutableBorrow(borrow.kind), borrow_index)),
                LocalMutationIsAllowed::No,
            );

            // We do not need to call `check_if_path_or_subpath_is_moved`
            // again, as we already called it when we made the
            // initial reservation.
        }
    }
}
