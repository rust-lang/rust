// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The visitors in this module collect sizes and counts of the most important
// pieces of MIR. The resulting numbers are good approximations but not
// completely accurate (some things might be counted twice, others missed).

use rustc::mir::{AggregateKind, AssertMessage, BasicBlock, BasicBlockData};
use rustc::mir::{Constant, Location, Local, LocalDecl};
use rustc::mir::{Place, PlaceElem, PlaceProjection};
use rustc::mir::{Mir, Operand, ProjectionElem};
use rustc::mir::{Rvalue, SourceInfo, Statement, StatementKind};
use rustc::mir::{Terminator, TerminatorKind, SourceScope, SourceScopeData};
use rustc::mir::interpret::EvalErrorKind;
use rustc::mir::visit as mir_visit;
use rustc::ty::{self, ClosureSubsts, TyCtxt};
use rustc::util::nodemap::{FxHashMap};

struct NodeData {
    count: usize,
    size: usize,
}

struct StatCollector<'a, 'tcx: 'a> {
    _tcx: TyCtxt<'a, 'tcx, 'tcx>,
    data: FxHashMap<&'static str, NodeData>,
}

impl<'a, 'tcx> StatCollector<'a, 'tcx> {

    fn record_with_size(&mut self, label: &'static str, node_size: usize) {
        let entry = self.data.entry(label).or_insert(NodeData {
            count: 0,
            size: 0,
        });

        entry.count += 1;
        entry.size = node_size;
    }

    fn record<T>(&mut self, label: &'static str, node: &T) {
        self.record_with_size(label, ::std::mem::size_of_val(node));
    }
}

impl<'a, 'tcx> mir_visit::Visitor<'tcx> for StatCollector<'a, 'tcx> {
    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        self.record("Mir", mir);

        // since the `super_mir` method does not traverse the MIR of
        // promoted rvalues, (but we still want to gather statistics
        // on the structures represented there) we manually traverse
        // the promoted rvalues here.
        for promoted_mir in &mir.promoted {
            self.visit_mir(promoted_mir);
        }

        self.super_mir(mir);
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.record("BasicBlockData", data);
        self.super_basic_block_data(block, data);
    }

    fn visit_source_scope_data(&mut self, scope_data: &SourceScopeData) {
        self.record("SourceScopeData", scope_data);
        self.super_source_scope_data(scope_data);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &Statement<'tcx>,
                       location: Location) {
        self.record("Statement", statement);
        self.record(match statement.kind {
            StatementKind::Assign(..) => "StatementKind::Assign",
            StatementKind::FakeRead(..) => "StatementKind::FakeRead",
            StatementKind::Retag { .. } => "StatementKind::Retag",
            StatementKind::EscapeToRaw { .. } => "StatementKind::EscapeToRaw",
            StatementKind::SetDiscriminant { .. } => "StatementKind::SetDiscriminant",
            StatementKind::StorageLive(..) => "StatementKind::StorageLive",
            StatementKind::StorageDead(..) => "StatementKind::StorageDead",
            StatementKind::InlineAsm { .. } => "StatementKind::InlineAsm",
            StatementKind::AscribeUserType(..) => "StatementKind::AscribeUserType",
            StatementKind::Nop => "StatementKind::Nop",
        }, &statement.kind);
        self.super_statement(block, statement, location);
    }

    fn visit_terminator(&mut self,
                        block: BasicBlock,
                        terminator: &Terminator<'tcx>,
                        location: Location) {
        self.record("Terminator", terminator);
        self.super_terminator(block, terminator, location);
    }

    fn visit_terminator_kind(&mut self,
                             block: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        self.record("TerminatorKind", kind);
        self.record(match *kind {
            TerminatorKind::Goto { .. } => "TerminatorKind::Goto",
            TerminatorKind::SwitchInt { .. } => "TerminatorKind::SwitchInt",
            TerminatorKind::Resume => "TerminatorKind::Resume",
            TerminatorKind::Abort => "TerminatorKind::Abort",
            TerminatorKind::Return => "TerminatorKind::Return",
            TerminatorKind::Unreachable => "TerminatorKind::Unreachable",
            TerminatorKind::Drop { .. } => "TerminatorKind::Drop",
            TerminatorKind::DropAndReplace { .. } => "TerminatorKind::DropAndReplace",
            TerminatorKind::Call { .. } => "TerminatorKind::Call",
            TerminatorKind::Assert { .. } => "TerminatorKind::Assert",
            TerminatorKind::GeneratorDrop => "TerminatorKind::GeneratorDrop",
            TerminatorKind::Yield { .. } => "TerminatorKind::Yield",
            TerminatorKind::FalseEdges { .. } => "TerminatorKind::FalseEdges",
            TerminatorKind::FalseUnwind { .. } => "TerminatorKind::FalseUnwind",
        }, kind);
        self.super_terminator_kind(block, kind, location);
    }

    fn visit_assert_message(&mut self, msg: &AssertMessage<'tcx>, location: Location) {
        self.record("AssertMessage", msg);
        self.record(match *msg {
            EvalErrorKind::BoundsCheck { .. } => "AssertMessage::BoundsCheck",
            EvalErrorKind::Overflow(..) => "AssertMessage::Overflow",
            EvalErrorKind::OverflowNeg => "AssertMessage::OverflowNeg",
            EvalErrorKind::DivisionByZero => "AssertMessage::DivisionByZero",
            EvalErrorKind::RemainderByZero => "AssertMessage::RemainderByZero",
            EvalErrorKind::GeneratorResumedAfterReturn => {
                "AssertMessage::GeneratorResumedAfterReturn"
            }
            EvalErrorKind::GeneratorResumedAfterPanic => {
                "AssertMessage::GeneratorResumedAfterPanic"
            }
            _ => bug!(),
        }, msg);
        self.super_assert_message(msg, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.record("Rvalue", rvalue);
        let rvalue_kind = match *rvalue {
            Rvalue::Use(..) => "Rvalue::Use",
            Rvalue::Repeat(..) => "Rvalue::Repeat",
            Rvalue::Ref(..) => "Rvalue::Ref",
            Rvalue::Len(..) => "Rvalue::Len",
            Rvalue::Cast(..) => "Rvalue::Cast",
            Rvalue::BinaryOp(..) => "Rvalue::BinaryOp",
            Rvalue::CheckedBinaryOp(..) => "Rvalue::CheckedBinaryOp",
            Rvalue::UnaryOp(..) => "Rvalue::UnaryOp",
            Rvalue::Discriminant(..) => "Rvalue::Discriminant",
            Rvalue::NullaryOp(..) => "Rvalue::NullaryOp",
            Rvalue::Aggregate(ref kind, ref _operands) => {
                // AggregateKind is not distinguished by visit API, so
                // record it. (`super_rvalue` handles `_operands`.)
                self.record(match **kind {
                    AggregateKind::Array(_) => "AggregateKind::Array",
                    AggregateKind::Tuple => "AggregateKind::Tuple",
                    AggregateKind::Adt(..) => "AggregateKind::Adt",
                    AggregateKind::Closure(..) => "AggregateKind::Closure",
                    AggregateKind::Generator(..) => "AggregateKind::Generator",
                }, kind);

                "Rvalue::Aggregate"
            }
        };
        self.record(rvalue_kind, rvalue);
        self.super_rvalue(rvalue, location);
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        self.record("Operand", operand);
        self.record(match *operand {
            Operand::Copy(..) => "Operand::Copy",
            Operand::Move(..) => "Operand::Move",
            Operand::Constant(..) => "Operand::Constant",
        }, operand);
        self.super_operand(operand, location);
    }

    fn visit_place(&mut self,
                    place: &Place<'tcx>,
                    context: mir_visit::PlaceContext<'tcx>,
                    location: Location) {
        self.record("Place", place);
        self.record(match *place {
            Place::Local(..) => "Place::Local",
            Place::Static(..) => "Place::Static",
            Place::Promoted(..) => "Place::Promoted",
            Place::Projection(..) => "Place::Projection",
        }, place);
        self.super_place(place, context, location);
    }

    fn visit_projection(&mut self,
                        place: &PlaceProjection<'tcx>,
                        context: mir_visit::PlaceContext<'tcx>,
                        location: Location) {
        self.record("PlaceProjection", place);
        self.super_projection(place, context, location);
    }

    fn visit_projection_elem(&mut self,
                             place: &PlaceElem<'tcx>,
                             location: Location) {
        self.record("PlaceElem", place);
        self.record(match *place {
            ProjectionElem::Deref => "PlaceElem::Deref",
            ProjectionElem::Subslice { .. } => "PlaceElem::Subslice",
            ProjectionElem::Field(..) => "PlaceElem::Field",
            ProjectionElem::Index(..) => "PlaceElem::Index",
            ProjectionElem::ConstantIndex { .. } => "PlaceElem::ConstantIndex",
            ProjectionElem::Downcast(..) => "PlaceElem::Downcast",
        }, place);
        self.super_projection_elem(place, location);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        self.record("Constant", constant);
        self.super_constant(constant, location);
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        self.record("SourceInfo", source_info);
        self.super_source_info(source_info);
    }

    fn visit_closure_substs(&mut self, substs: &ClosureSubsts<'tcx>, _: Location) {
        self.record("ClosureSubsts", substs);
        self.super_closure_substs(substs);
    }

    fn visit_const(&mut self, constant: &&'tcx ty::Const<'tcx>, _: Location) {
        self.record("Const", constant);
        self.super_const(constant);
    }

    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        self.record("LocalDecl", local_decl);
        self.super_local_decl(local, local_decl);
    }

    fn visit_source_scope(&mut self, scope: &SourceScope) {
        self.record("VisiblityScope", scope);
        self.super_source_scope(scope);
    }
}
