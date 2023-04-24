//! Module that implements what will become the rustc side of Stable MIR.
//!
//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use crate::stable_mir::{self, ty::TyKind, Context};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};
use tracing::debug;

impl<'tcx> Context for Tables<'tcx> {
    fn local_crate(&self) -> stable_mir::Crate {
        smir_crate(self.tcx, LOCAL_CRATE)
    }

    fn external_crates(&self) -> Vec<stable_mir::Crate> {
        self.tcx.crates(()).iter().map(|crate_num| smir_crate(self.tcx, *crate_num)).collect()
    }

    fn find_crate(&self, name: &str) -> Option<stable_mir::Crate> {
        [LOCAL_CRATE].iter().chain(self.tcx.crates(()).iter()).find_map(|crate_num| {
            let crate_name = self.tcx.crate_name(*crate_num).to_string();
            (name == crate_name).then(|| smir_crate(self.tcx, *crate_num))
        })
    }

    fn all_local_items(&mut self) -> stable_mir::CrateItems {
        self.tcx.mir_keys(()).iter().map(|item| self.crate_item(item.to_def_id())).collect()
    }
    fn entry_fn(&mut self) -> Option<stable_mir::CrateItem> {
        Some(self.crate_item(self.tcx.entry_fn(())?.0))
    }
    fn mir_body(&mut self, item: &stable_mir::CrateItem) -> stable_mir::mir::Body {
        let def_id = self.item_def_id(item);
        let mir = self.tcx.optimized_mir(def_id);
        stable_mir::mir::Body {
            blocks: mir
                .basic_blocks
                .iter()
                .map(|block| stable_mir::mir::BasicBlock {
                    terminator: rustc_terminator_to_terminator(block.terminator()),
                    statements: block.statements.iter().map(rustc_statement_to_statement).collect(),
                })
                .collect(),
            locals: mir.local_decls.iter().map(|decl| self.intern_ty(decl.ty)).collect(),
        }
    }

    fn rustc_tables(&mut self, f: &mut dyn FnMut(&mut Tables<'_>)) {
        f(self)
    }

    fn ty_kind(&mut self, ty: crate::stable_mir::ty::Ty) -> TyKind {
        self.rustc_ty_to_ty(self.types[ty.0])
    }
}

pub struct Tables<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub def_ids: Vec<DefId>,
    pub types: Vec<Ty<'tcx>>,
}

impl<'tcx> Tables<'tcx> {
    fn rustc_ty_to_ty(&mut self, ty: Ty<'tcx>) -> TyKind {
        match ty.kind() {
            ty::Bool => TyKind::Bool,
            ty::Char => todo!(),
            ty::Int(_) => todo!(),
            ty::Uint(_) => todo!(),
            ty::Float(_) => todo!(),
            ty::Adt(_, _) => todo!(),
            ty::Foreign(_) => todo!(),
            ty::Str => todo!(),
            ty::Array(_, _) => todo!(),
            ty::Slice(_) => todo!(),
            ty::RawPtr(_) => todo!(),
            ty::Ref(_, _, _) => todo!(),
            ty::FnDef(_, _) => todo!(),
            ty::FnPtr(_) => todo!(),
            ty::Placeholder(..) => todo!(),
            ty::Dynamic(_, _, _) => todo!(),
            ty::Closure(_, _) => todo!(),
            ty::Generator(_, _, _) => todo!(),
            ty::GeneratorWitness(_) => todo!(),
            ty::GeneratorWitnessMIR(_, _) => todo!(),
            ty::Never => todo!(),
            ty::Tuple(fields) => {
                TyKind::Tuple(fields.iter().map(|ty| self.intern_ty(ty)).collect())
            }
            ty::Alias(_, _) => todo!(),
            ty::Param(_) => todo!(),
            ty::Bound(_, _) => todo!(),
            ty::Infer(_) => todo!(),
            ty::Error(_) => todo!(),
        }
    }

    fn intern_ty(&mut self, ty: Ty<'tcx>) -> stable_mir::ty::Ty {
        if let Some(id) = self.types.iter().position(|&t| t == ty) {
            return stable_mir::ty::Ty(id);
        }
        let id = self.types.len();
        self.types.push(ty);
        stable_mir::ty::Ty(id)
    }
}

/// Build a stable mir crate from a given crate number.
fn smir_crate(tcx: TyCtxt<'_>, crate_num: CrateNum) -> stable_mir::Crate {
    let crate_name = tcx.crate_name(crate_num).to_string();
    let is_local = crate_num == LOCAL_CRATE;
    debug!(?crate_name, ?crate_num, "smir_crate");
    stable_mir::Crate { id: crate_num.into(), name: crate_name, is_local }
}

fn rustc_statement_to_statement(
    s: &rustc_middle::mir::Statement<'_>,
) -> stable_mir::mir::Statement {
    use rustc_middle::mir::StatementKind::*;
    match &s.kind {
        Assign(assign) => stable_mir::mir::Statement::Assign(
            rustc_place_to_place(&assign.0),
            rustc_rvalue_to_rvalue(&assign.1),
        ),
        FakeRead(_) => todo!(),
        SetDiscriminant { .. } => todo!(),
        Deinit(_) => todo!(),
        StorageLive(_) => todo!(),
        StorageDead(_) => todo!(),
        Retag(_, _) => todo!(),
        PlaceMention(_) => todo!(),
        AscribeUserType(_, _) => todo!(),
        Coverage(_) => todo!(),
        Intrinsic(_) => todo!(),
        ConstEvalCounter => todo!(),
        Nop => stable_mir::mir::Statement::Nop,
    }
}

fn rustc_rvalue_to_rvalue(rvalue: &rustc_middle::mir::Rvalue<'_>) -> stable_mir::mir::Rvalue {
    use rustc_middle::mir::Rvalue::*;
    match rvalue {
        Use(op) => stable_mir::mir::Rvalue::Use(rustc_op_to_op(op)),
        Repeat(_, _) => todo!(),
        Ref(_, _, _) => todo!(),
        ThreadLocalRef(_) => todo!(),
        AddressOf(_, _) => todo!(),
        Len(_) => todo!(),
        Cast(_, _, _) => todo!(),
        BinaryOp(_, _) => todo!(),
        CheckedBinaryOp(bin_op, ops) => stable_mir::mir::Rvalue::CheckedBinaryOp(
            rustc_bin_op_to_bin_op(bin_op),
            rustc_op_to_op(&ops.0),
            rustc_op_to_op(&ops.1),
        ),
        NullaryOp(_, _) => todo!(),
        UnaryOp(un_op, op) => {
            stable_mir::mir::Rvalue::UnaryOp(rustc_un_op_to_un_op(un_op), rustc_op_to_op(op))
        }
        Discriminant(_) => todo!(),
        Aggregate(_, _) => todo!(),
        ShallowInitBox(_, _) => todo!(),
        CopyForDeref(_) => todo!(),
    }
}

fn rustc_op_to_op(op: &rustc_middle::mir::Operand<'_>) -> stable_mir::mir::Operand {
    use rustc_middle::mir::Operand::*;
    match op {
        Copy(place) => stable_mir::mir::Operand::Copy(rustc_place_to_place(place)),
        Move(place) => stable_mir::mir::Operand::Move(rustc_place_to_place(place)),
        Constant(c) => stable_mir::mir::Operand::Constant(c.to_string()),
    }
}

fn rustc_place_to_place(place: &rustc_middle::mir::Place<'_>) -> stable_mir::mir::Place {
    stable_mir::mir::Place {
        local: place.local.as_usize(),
        projection: format!("{:?}", place.projection),
    }
}

fn rustc_unwind_to_unwind(
    unwind: &rustc_middle::mir::UnwindAction,
) -> stable_mir::mir::UnwindAction {
    use rustc_middle::mir::UnwindAction;
    match unwind {
        UnwindAction::Continue => stable_mir::mir::UnwindAction::Continue,
        UnwindAction::Unreachable => stable_mir::mir::UnwindAction::Unreachable,
        UnwindAction::Terminate => stable_mir::mir::UnwindAction::Terminate,
        UnwindAction::Cleanup(bb) => stable_mir::mir::UnwindAction::Cleanup(bb.as_usize()),
    }
}

fn rustc_assert_msg_to_msg<'tcx>(
    assert_message: &rustc_middle::mir::AssertMessage<'tcx>,
) -> stable_mir::mir::AssertMessage {
    use rustc_middle::mir::AssertKind;
    match assert_message {
        AssertKind::BoundsCheck { len, index } => stable_mir::mir::AssertMessage::BoundsCheck {
            len: rustc_op_to_op(len),
            index: rustc_op_to_op(index),
        },
        AssertKind::Overflow(bin_op, op1, op2) => stable_mir::mir::AssertMessage::Overflow(
            rustc_bin_op_to_bin_op(bin_op),
            rustc_op_to_op(op1),
            rustc_op_to_op(op2),
        ),
        AssertKind::OverflowNeg(op) => {
            stable_mir::mir::AssertMessage::OverflowNeg(rustc_op_to_op(op))
        }
        AssertKind::DivisionByZero(op) => {
            stable_mir::mir::AssertMessage::DivisionByZero(rustc_op_to_op(op))
        }
        AssertKind::RemainderByZero(op) => {
            stable_mir::mir::AssertMessage::RemainderByZero(rustc_op_to_op(op))
        }
        AssertKind::ResumedAfterReturn(generator) => {
            stable_mir::mir::AssertMessage::ResumedAfterReturn(rustc_generator_to_generator(
                generator,
            ))
        }
        AssertKind::ResumedAfterPanic(generator) => {
            stable_mir::mir::AssertMessage::ResumedAfterPanic(rustc_generator_to_generator(
                generator,
            ))
        }
        AssertKind::MisalignedPointerDereference { required, found } => {
            stable_mir::mir::AssertMessage::MisalignedPointerDereference {
                required: rustc_op_to_op(required),
                found: rustc_op_to_op(found),
            }
        }
    }
}

fn rustc_bin_op_to_bin_op(bin_op: &rustc_middle::mir::BinOp) -> stable_mir::mir::BinOp {
    use rustc_middle::mir::BinOp;
    match bin_op {
        BinOp::Add => stable_mir::mir::BinOp::Add,
        BinOp::Sub => stable_mir::mir::BinOp::Sub,
        BinOp::Mul => stable_mir::mir::BinOp::Mul,
        BinOp::Div => stable_mir::mir::BinOp::Div,
        BinOp::Rem => stable_mir::mir::BinOp::Rem,
        BinOp::BitXor => stable_mir::mir::BinOp::BitXor,
        BinOp::BitAnd => stable_mir::mir::BinOp::BitAnd,
        BinOp::BitOr => stable_mir::mir::BinOp::BitOr,
        BinOp::Shl => stable_mir::mir::BinOp::Shl,
        BinOp::Shr => stable_mir::mir::BinOp::Shr,
        BinOp::Eq => stable_mir::mir::BinOp::Eq,
        BinOp::Lt => stable_mir::mir::BinOp::Lt,
        BinOp::Le => stable_mir::mir::BinOp::Le,
        BinOp::Ne => stable_mir::mir::BinOp::Ne,
        BinOp::Ge => stable_mir::mir::BinOp::Ge,
        BinOp::Gt => stable_mir::mir::BinOp::Gt,
        BinOp::Offset => stable_mir::mir::BinOp::Offset,
    }
}

fn rustc_un_op_to_un_op(unary_op: &rustc_middle::mir::UnOp) -> stable_mir::mir::UnOp {
    use rustc_middle::mir::UnOp;
    match unary_op {
        UnOp::Not => stable_mir::mir::UnOp::Not,
        UnOp::Neg => stable_mir::mir::UnOp::Neg,
    }
}

fn rustc_generator_to_generator(
    generator: &rustc_hir::GeneratorKind,
) -> stable_mir::mir::GeneratorKind {
    use rustc_hir::{AsyncGeneratorKind, GeneratorKind};
    match generator {
        GeneratorKind::Async(async_gen) => {
            let async_gen = match async_gen {
                AsyncGeneratorKind::Block => stable_mir::mir::AsyncGeneratorKind::Block,
                AsyncGeneratorKind::Closure => stable_mir::mir::AsyncGeneratorKind::Closure,
                AsyncGeneratorKind::Fn => stable_mir::mir::AsyncGeneratorKind::Fn,
            };
            stable_mir::mir::GeneratorKind::Async(async_gen)
        }
        GeneratorKind::Gen => stable_mir::mir::GeneratorKind::Gen,
    }
}

fn rustc_terminator_to_terminator(
    terminator: &rustc_middle::mir::Terminator<'_>,
) -> stable_mir::mir::Terminator {
    use rustc_middle::mir::TerminatorKind::*;
    use stable_mir::mir::Terminator;
    match &terminator.kind {
        Goto { target } => Terminator::Goto { target: target.as_usize() },
        SwitchInt { discr, targets } => Terminator::SwitchInt {
            discr: rustc_op_to_op(discr),
            targets: targets
                .iter()
                .map(|(value, target)| stable_mir::mir::SwitchTarget {
                    value,
                    target: target.as_usize(),
                })
                .collect(),
            otherwise: targets.otherwise().as_usize(),
        },
        Resume => Terminator::Resume,
        Terminate => Terminator::Abort,
        Return => Terminator::Return,
        Unreachable => Terminator::Unreachable,
        Drop { place, target, unwind } => Terminator::Drop {
            place: rustc_place_to_place(place),
            target: target.as_usize(),
            unwind: rustc_unwind_to_unwind(unwind),
        },
        Call { func, args, destination, target, unwind, from_hir_call: _, fn_span: _ } => {
            Terminator::Call {
                func: rustc_op_to_op(func),
                args: args.iter().map(|arg| rustc_op_to_op(arg)).collect(),
                destination: rustc_place_to_place(destination),
                target: target.map(|t| t.as_usize()),
                unwind: rustc_unwind_to_unwind(unwind),
            }
        }
        Assert { cond, expected, msg, target, unwind } => Terminator::Assert {
            cond: rustc_op_to_op(cond),
            expected: *expected,
            msg: rustc_assert_msg_to_msg(msg),
            target: target.as_usize(),
            unwind: rustc_unwind_to_unwind(unwind),
        },
        Yield { .. } => todo!(),
        GeneratorDrop => Terminator::GeneratorDrop,
        FalseEdge { .. } => todo!(),
        FalseUnwind { .. } => todo!(),
        InlineAsm { .. } => todo!(),
    }
}
