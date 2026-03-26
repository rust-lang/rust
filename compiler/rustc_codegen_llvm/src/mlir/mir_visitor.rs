/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! MIR visitor for MLIR codegen backend.
//!
//! This module provides utilities to traverse and log MIR structures
//! for debugging and development of the MLIR codegen backend.

use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Local, LocalDecl, Operand, Place, ProjectionElem, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{Instance, Ty, TyCtxt};
use tracing::info;

/// Visitor for MIR structures that logs all encountered elements.
pub struct MirVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    indent: usize,
}

impl<'tcx> MirVisitor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, indent: 0 }
    }

    fn indent_str(&self) -> String {
        "  ".repeat(self.indent)
    }

    fn log(&self, msg: &str) {
        info!("{}{}", self.indent_str(), msg);
    }

    fn with_indent<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.indent += 1;
        let result = f(self);
        self.indent -= 1;
        result
    }

    /// Visit an entire function instance.
    pub fn visit_instance(&mut self, instance: Instance<'tcx>) {
        eprintln!("[DEBUG] MirVisitor::visit_instance called for instance: {:?}", instance);
        self.log(&format!("=== Instance: {:?} ===", instance));
        self.log(&format!("DefId: {:?}", instance.def_id()));
        self.log(&format!("Args: {:?}", instance.args));

        eprintln!("[DEBUG] Getting MIR for instance");
        let mir = self.tcx.instance_mir(instance.def);
        eprintln!("[DEBUG] Got MIR, calling visit_body");
        self.visit_body(mir, instance);
        eprintln!("[DEBUG] Completed visit_body");
    }

    /// Visit a MIR body (function body).
    pub fn visit_body(&mut self, body: &Body<'tcx>, instance: Instance<'tcx>) {
        eprintln!("[DEBUG] MirVisitor::visit_body called");
        self.log(&format!("--- MIR Body for {:?} ---", instance.def_id()));

        self.with_indent(|this| {
            // Log return type
            this.log(&format!("Return type: {:?}", body.return_ty()));

            // Log all local declarations
            this.log("Local declarations:");
            this.with_indent(|this| {
                for (local, decl) in body.local_decls.iter_enumerated() {
                    this.visit_local_decl(local, decl);
                }
            });

            // Log argument count
            this.log(&format!("Argument count: {}", body.arg_count));

            // Log spread_arg if present
            if let Some(spread_arg) = body.spread_arg {
                this.log(&format!("Spread arg: {:?}", spread_arg));
            }

            // Log var debug info
            this.log(&format!("Var debug info count: {}", body.var_debug_info.len()));

            // Log all basic blocks
            this.log(&format!("Basic blocks ({})", body.basic_blocks.len()));
            this.with_indent(|this| {
                for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
                    this.visit_basic_block(bb, bb_data);
                }
            });

            // Log source scopes
            this.log(&format!("Source scopes count: {}", body.source_scopes.len()));
        });
    }

    /// Visit a local declaration.
    fn visit_local_decl(&mut self, local: Local, decl: &LocalDecl<'tcx>) {
        self.log(&format!("{:?}: {:?} (mutability: {:?})", local, decl.ty, decl.mutability));
        self.with_indent(|this| {
            this.log(&format!("Local info: {:?}", decl.local_info));
        });
    }

    /// Visit a basic block.
    fn visit_basic_block(&mut self, bb: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.log(&format!("BasicBlock {:?} (is_cleanup: {})", bb, data.is_cleanup));

        self.with_indent(|this| {
            // Log all statements
            this.log(&format!("Statements ({}):", data.statements.len()));
            this.with_indent(|this| {
                for (idx, stmt) in data.statements.iter().enumerate() {
                    this.visit_statement(idx, stmt);
                }
            });

            // Log the terminator
            this.log("Terminator:");
            this.with_indent(|this| {
                this.visit_terminator(data.terminator());
            });
        });
    }

    /// Visit a statement.
    fn visit_statement(&mut self, idx: usize, stmt: &Statement<'tcx>) {
        self.log(&format!("[{}] {:?}", idx, stmt.kind));

        self.with_indent(|this| {
            this.log(&format!("Source info: {:?}", stmt.source_info));

            match &stmt.kind {
                StatementKind::Assign(assign) => {
                    let (place, rvalue) = assign.as_ref();
                    this.log("Assignment:");
                    this.with_indent(|this| {
                        this.visit_place("LHS", place);
                        this.visit_rvalue(rvalue);
                    });
                }
                StatementKind::SetDiscriminant { place, variant_index } => {
                    this.log(&format!("SetDiscriminant: variant {:?}", variant_index));
                    this.with_indent(|this| {
                        this.visit_place("Place", place);
                    });
                }
                StatementKind::StorageLive(local) => {
                    this.log(&format!("StorageLive: {:?}", local));
                }
                StatementKind::StorageDead(local) => {
                    this.log(&format!("StorageDead: {:?}", local));
                }
                StatementKind::Intrinsic(intrinsic) => {
                    this.log(&format!("Intrinsic: {:?}", intrinsic));
                }
                StatementKind::Coverage(coverage) => {
                    this.log(&format!("Coverage: {:?}", coverage));
                }
                StatementKind::Retag(retag_kind, place) => {
                    this.log(&format!("Retag: {:?}", retag_kind));
                    this.visit_place("Place", place);
                }
                StatementKind::FakeRead(fake_read) => {
                    let (cause, place) = fake_read.as_ref();
                    this.log(&format!("FakeRead: {:?}", cause));
                    this.visit_place("Place", place);
                }
                StatementKind::AscribeUserType(ascribe, variance) => {
                    let (place, user_ty) = ascribe.as_ref();
                    this.log(&format!("AscribeUserType: {:?} {:?}", user_ty, variance));
                    this.visit_place("Place", place);
                }
                StatementKind::PlaceMention(place) => {
                    this.visit_place("PlaceMention", place);
                }
                StatementKind::ConstEvalCounter => {
                    this.log("ConstEvalCounter");
                }
                StatementKind::Nop => {
                    this.log("Nop");
                }
                StatementKind::BackwardIncompatibleDropHint { place, reason } => {
                    this.log(&format!("BackwardIncompatibleDropHint: {:?}", reason));
                    this.visit_place("Place", place);
                }
            }
        });
    }

    /// Visit a place (lvalue).
    fn visit_place(&mut self, label: &str, place: &Place<'tcx>) {
        self.log(&format!("{}: {:?}", label, place));

        self.with_indent(|this| {
            this.log(&format!("Local: {:?}", place.local));

            if !place.projection.is_empty() {
                this.log("Projections:");
                this.with_indent(|this| {
                    for (idx, elem) in place.projection.iter().enumerate() {
                        this.visit_projection_elem(idx, elem);
                    }
                });
            }
        });
    }

    /// Visit a projection element.
    fn visit_projection_elem(&mut self, idx: usize, elem: ProjectionElem<Local, Ty<'tcx>>) {
        match elem {
            ProjectionElem::Deref => {
                self.log(&format!("[{}] Deref", idx));
            }
            ProjectionElem::Field(field, ty) => {
                self.log(&format!("[{}] Field({:?}, {:?})", idx, field, ty));
            }
            ProjectionElem::Index(local) => {
                self.log(&format!("[{}] Index({:?})", idx, local));
            }
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                self.log(&format!(
                    "[{}] ConstantIndex {{ offset: {}, min_length: {}, from_end: {} }}",
                    idx, offset, min_length, from_end
                ));
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                self.log(&format!(
                    "[{}] Subslice {{ from: {}, to: {}, from_end: {} }}",
                    idx, from, to, from_end
                ));
            }
            ProjectionElem::Downcast(name, variant_idx) => {
                self.log(&format!("[{}] Downcast({:?}, {:?})", idx, name, variant_idx));
            }
            ProjectionElem::OpaqueCast(ty) => {
                self.log(&format!("[{}] OpaqueCast({:?})", idx, ty));
            }
            ProjectionElem::UnwrapUnsafeBinder(ty) => {
                self.log(&format!("[{}] UnwrapUnsafeBinder({:?})", idx, ty));
            }
        }
    }

    /// Visit an rvalue (right-hand side of assignment).
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        self.log(&format!("RHS: {:?}", rvalue));

        self.with_indent(|this| match rvalue {
            Rvalue::Use(operand) => {
                this.log("Use:");
                this.visit_operand(operand);
            }
            Rvalue::Repeat(operand, count) => {
                this.log(&format!("Repeat (count: {:?}):", count));
                this.visit_operand(operand);
            }
            Rvalue::Ref(region, borrow_kind, place) => {
                this.log(&format!("Ref: region={:?}, kind={:?}", region, borrow_kind));
                this.visit_place("Place", place);
            }
            Rvalue::RawPtr(raw_ptr_kind, place) => {
                this.log(&format!("RawPtr: {:?}", raw_ptr_kind));
                this.visit_place("Place", place);
            }
            Rvalue::Cast(cast_kind, operand, ty) => {
                this.log(&format!("Cast: {:?} -> {:?}", cast_kind, ty));
                this.visit_operand(operand);
            }
            Rvalue::BinaryOp(bin_op, operands) => {
                let (lhs, rhs) = operands.as_ref();
                this.log(&format!("BinaryOp: {:?}", bin_op));
                this.with_indent(|this| {
                    this.log("LHS:");
                    this.visit_operand(lhs);
                    this.log("RHS:");
                    this.visit_operand(rhs);
                });
            }
            Rvalue::UnaryOp(un_op, operand) => {
                this.log(&format!("UnaryOp: {:?}", un_op));
                this.visit_operand(operand);
            }
            Rvalue::Discriminant(place) => {
                this.log("Discriminant:");
                this.visit_place("Place", place);
            }
            Rvalue::Aggregate(aggregate_kind, operands) => {
                this.log(&format!("Aggregate: {:?}", aggregate_kind));
                this.with_indent(|this| {
                    for (idx, op) in operands.iter().enumerate() {
                        this.log(&format!("Field {}:", idx));
                        this.visit_operand(op);
                    }
                });
            }
            Rvalue::ShallowInitBox(operand, ty) => {
                this.log(&format!("ShallowInitBox: type={:?}", ty));
                this.visit_operand(operand);
            }
            Rvalue::CopyForDeref(place) => {
                this.log("CopyForDeref:");
                this.visit_place("Place", place);
            }
            Rvalue::ThreadLocalRef(def_id) => {
                this.log(&format!("ThreadLocalRef: {:?}", def_id));
            }
            Rvalue::WrapUnsafeBinder(operand, ty) => {
                this.log(&format!("WrapUnsafeBinder: type={:?}", ty));
                this.visit_operand(operand);
            }
        });
    }

    /// Visit an operand.
    fn visit_operand(&mut self, operand: &Operand<'tcx>) {
        self.with_indent(|this| match operand {
            Operand::Copy(place) => {
                this.log("Copy:");
                this.visit_place("Place", place);
            }
            Operand::Move(place) => {
                this.log("Move:");
                this.visit_place("Place", place);
            }
            Operand::Constant(constant) => {
                this.log(&format!("Constant: {:?}", constant));
                this.with_indent(|this| {
                    this.log(&format!("Type: {:?}", constant.ty()));
                    this.log(&format!("Const: {:?}", constant.const_));
                });
            }
            Operand::RuntimeChecks(checks) => {
                this.log(&format!("RuntimeChecks: {:?}", checks));
            }
        });
    }

    /// Visit a terminator.
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>) {
        self.log(&format!("Kind: {:?}", terminator.kind));
        self.log(&format!("Source info: {:?}", terminator.source_info));

        self.with_indent(|this| match &terminator.kind {
            TerminatorKind::Goto { target } => {
                this.log(&format!("Goto: {:?}", target));
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                this.log("SwitchInt:");
                this.visit_operand(discr);
                this.log(&format!("Targets: {:?}", targets));
            }
            TerminatorKind::UnwindResume => {
                this.log("UnwindResume");
            }
            TerminatorKind::UnwindTerminate(reason) => {
                this.log(&format!("UnwindTerminate: {:?}", reason));
            }
            TerminatorKind::Return => {
                this.log("Return");
            }
            TerminatorKind::Unreachable => {
                this.log("Unreachable");
            }
            TerminatorKind::Drop { place, target, unwind, replace, drop, async_fut } => {
                this.log(&format!(
                    "Drop: target={:?}, unwind={:?}, replace={}, drop={:?}, async_fut={:?}",
                    target, unwind, replace, drop, async_fut
                ));
                this.visit_place("Place", place);
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source,
                fn_span,
            } => {
                this.log(&format!(
                    "Call: target={:?}, unwind={:?}, call_source={:?}, span={:?}",
                    target, unwind, call_source, fn_span
                ));
                this.log("Function:");
                this.visit_operand(func);
                this.log(&format!("Args ({}):", args.len()));
                this.with_indent(|this| {
                    for (idx, arg) in args.iter().enumerate() {
                        this.log(&format!("Arg {}:", idx));
                        this.visit_operand(&arg.node);
                    }
                });
                this.visit_place("Destination", destination);
            }
            TerminatorKind::TailCall { func, args, fn_span } => {
                this.log(&format!("TailCall: span={:?}", fn_span));
                this.log("Function:");
                this.visit_operand(func);
                this.log(&format!("Args ({}):", args.len()));
                this.with_indent(|this| {
                    for (idx, arg) in args.iter().enumerate() {
                        this.log(&format!("Arg {}:", idx));
                        this.visit_operand(&arg.node);
                    }
                });
            }
            TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                this.log(&format!(
                    "Assert: expected={}, target={:?}, unwind={:?}",
                    expected, target, unwind
                ));
                this.log(&format!("Message: {:?}", msg));
                this.log("Condition:");
                this.visit_operand(cond);
            }
            TerminatorKind::Yield { value, resume, resume_arg, drop } => {
                this.log(&format!("Yield: resume={:?}, drop={:?}", resume, drop));
                this.log("Value:");
                this.visit_operand(value);
                this.visit_place("ResumeArg", resume_arg);
            }
            TerminatorKind::CoroutineDrop => {
                this.log("CoroutineDrop");
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                this.log(&format!(
                    "FalseEdge: real={:?}, imaginary={:?}",
                    real_target, imaginary_target
                ));
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                this.log(&format!("FalseUnwind: real={:?}, unwind={:?}", real_target, unwind));
            }
            TerminatorKind::InlineAsm {
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind,
            } => {
                this.log(&format!(
                    "InlineAsm: macro={:?}, options={:?}, targets={:?}, unwind={:?}",
                    asm_macro, options, targets, unwind
                ));
                this.log(&format!("Template: {:?}", template));
                this.log(&format!("Operands ({}):", operands.len()));
                this.with_indent(|this| {
                    for (idx, op) in operands.iter().enumerate() {
                        this.log(&format!("[{}] {:?}", idx, op));
                    }
                });
            }
        });
    }
}

/// Summary of MIR for quick overview.
pub struct MirSummary<'tcx> {
    pub instance: Instance<'tcx>,
    pub local_count: usize,
    pub arg_count: usize,
    pub basic_block_count: usize,
    pub statement_count: usize,
    pub return_ty: Ty<'tcx>,
}

impl<'tcx> MirSummary<'tcx> {
    pub fn from_instance(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Self {
        let mir = tcx.instance_mir(instance.def);
        let statement_count: usize = mir.basic_blocks.iter().map(|bb| bb.statements.len()).sum();

        Self {
            instance,
            local_count: mir.local_decls.len(),
            arg_count: mir.arg_count,
            basic_block_count: mir.basic_blocks.len(),
            statement_count,
            return_ty: mir.return_ty(),
        }
    }

    pub fn log(&self) {
        info!("MIR Summary for {:?}:", self.instance.def_id());
        info!("  Locals: {}", self.local_count);
        info!("  Args: {}", self.arg_count);
        info!("  Basic blocks: {}", self.basic_block_count);
        info!("  Statements: {}", self.statement_count);
        info!("  Return type: {:?}", self.return_ty);
    }
}
