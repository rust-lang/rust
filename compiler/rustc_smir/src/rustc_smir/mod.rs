//! Module that implements what will become the rustc side of Stable MIR.
//!
//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use crate::rustc_internal::{self, opaque};
use crate::stable_mir::ty::{AdtSubsts, FloatTy, GenericArgKind, IntTy, RigidTy, TyKind, UintTy};
use crate::stable_mir::{self, Context};
use rustc_middle::mir;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_target::abi::FieldIdx;
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
                    terminator: block.terminator().stable(),
                    statements: block.statements.iter().map(mir::Statement::stable).collect(),
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
            ty::Bool => TyKind::RigidTy(RigidTy::Bool),
            ty::Char => TyKind::RigidTy(RigidTy::Char),
            ty::Int(int_ty) => match int_ty {
                ty::IntTy::Isize => TyKind::RigidTy(RigidTy::Int(IntTy::Isize)),
                ty::IntTy::I8 => TyKind::RigidTy(RigidTy::Int(IntTy::I8)),
                ty::IntTy::I16 => TyKind::RigidTy(RigidTy::Int(IntTy::I16)),
                ty::IntTy::I32 => TyKind::RigidTy(RigidTy::Int(IntTy::I32)),
                ty::IntTy::I64 => TyKind::RigidTy(RigidTy::Int(IntTy::I64)),
                ty::IntTy::I128 => TyKind::RigidTy(RigidTy::Int(IntTy::I128)),
            },
            ty::Uint(uint_ty) => match uint_ty {
                ty::UintTy::Usize => TyKind::RigidTy(RigidTy::Uint(UintTy::Usize)),
                ty::UintTy::U8 => TyKind::RigidTy(RigidTy::Uint(UintTy::U8)),
                ty::UintTy::U16 => TyKind::RigidTy(RigidTy::Uint(UintTy::U16)),
                ty::UintTy::U32 => TyKind::RigidTy(RigidTy::Uint(UintTy::U32)),
                ty::UintTy::U64 => TyKind::RigidTy(RigidTy::Uint(UintTy::U64)),
                ty::UintTy::U128 => TyKind::RigidTy(RigidTy::Uint(UintTy::U128)),
            },
            ty::Float(float_ty) => match float_ty {
                ty::FloatTy::F32 => TyKind::RigidTy(RigidTy::Float(FloatTy::F32)),
                ty::FloatTy::F64 => TyKind::RigidTy(RigidTy::Float(FloatTy::F64)),
            },
            ty::Adt(adt_def, substs) => TyKind::RigidTy(RigidTy::Adt(
                rustc_internal::adt_def(adt_def.did()),
                AdtSubsts(
                    substs
                        .iter()
                        .map(|arg| match arg.unpack() {
                            ty::GenericArgKind::Lifetime(region) => {
                                GenericArgKind::Lifetime(opaque(&region))
                            }
                            ty::GenericArgKind::Type(ty) => {
                                GenericArgKind::Type(self.intern_ty(ty))
                            }
                            ty::GenericArgKind::Const(const_) => {
                                GenericArgKind::Const(opaque(&const_))
                            }
                        })
                        .collect(),
                ),
            )),
            ty::Foreign(_) => todo!(),
            ty::Str => TyKind::RigidTy(RigidTy::Str),
            ty::Array(ty, constant) => {
                TyKind::RigidTy(RigidTy::Array(self.intern_ty(*ty), opaque(constant)))
            }
            ty::Slice(ty) => TyKind::RigidTy(RigidTy::Slice(self.intern_ty(*ty))),
            ty::RawPtr(_) => todo!(),
            ty::Ref(_, _, _) => todo!(),
            ty::FnDef(_, _) => todo!(),
            ty::FnPtr(_) => todo!(),
            ty::Dynamic(_, _, _) => todo!(),
            ty::Closure(_, _) => todo!(),
            ty::Generator(_, _, _) => todo!(),
            ty::Never => todo!(),
            ty::Tuple(fields) => TyKind::RigidTy(RigidTy::Tuple(
                fields.iter().map(|ty| self.intern_ty(ty)).collect(),
            )),
            ty::Alias(_, _) => todo!(),
            ty::Param(_) => todo!(),
            ty::Bound(_, _) => todo!(),
            ty::Placeholder(..)
            | ty::GeneratorWitness(_)
            | ty::GeneratorWitnessMIR(_, _)
            | ty::Infer(_)
            | ty::Error(_) => {
                unreachable!();
            }
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

/// Trait used to convert between an internal MIR type to a Stable MIR type.
pub(crate) trait Stable {
    /// The stable representation of the type implementing Stable.
    type T;
    /// Converts an object to the equivalent Stable MIR representation.
    fn stable(&self) -> Self::T;
}

impl<'tcx> Stable for mir::Statement<'tcx> {
    type T = stable_mir::mir::Statement;
    fn stable(&self) -> Self::T {
        use rustc_middle::mir::StatementKind::*;
        match &self.kind {
            Assign(assign) => {
                stable_mir::mir::Statement::Assign(assign.0.stable(), assign.1.stable())
            }
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
}

impl<'tcx> Stable for mir::Rvalue<'tcx> {
    type T = stable_mir::mir::Rvalue;
    fn stable(&self) -> Self::T {
        use mir::Rvalue::*;
        match self {
            Use(op) => stable_mir::mir::Rvalue::Use(op.stable()),
            Repeat(_, _) => todo!(),
            Ref(region, kind, place) => {
                stable_mir::mir::Rvalue::Ref(opaque(region), kind.stable(), place.stable())
            }
            ThreadLocalRef(def_id) => {
                stable_mir::mir::Rvalue::ThreadLocalRef(rustc_internal::crate_item(*def_id))
            }
            AddressOf(mutability, place) => {
                stable_mir::mir::Rvalue::AddressOf(mutability.stable(), place.stable())
            }
            Len(place) => stable_mir::mir::Rvalue::Len(place.stable()),
            Cast(_, _, _) => todo!(),
            BinaryOp(bin_op, ops) => {
                stable_mir::mir::Rvalue::BinaryOp(bin_op.stable(), ops.0.stable(), ops.1.stable())
            }
            CheckedBinaryOp(bin_op, ops) => stable_mir::mir::Rvalue::CheckedBinaryOp(
                bin_op.stable(),
                ops.0.stable(),
                ops.1.stable(),
            ),
            NullaryOp(_, _) => todo!(),
            UnaryOp(un_op, op) => stable_mir::mir::Rvalue::UnaryOp(un_op.stable(), op.stable()),
            Discriminant(place) => stable_mir::mir::Rvalue::Discriminant(place.stable()),
            Aggregate(_, _) => todo!(),
            ShallowInitBox(_, _) => todo!(),
            CopyForDeref(place) => stable_mir::mir::Rvalue::CopyForDeref(place.stable()),
        }
    }
}

impl Stable for mir::Mutability {
    type T = stable_mir::mir::Mutability;
    fn stable(&self) -> Self::T {
        use mir::Mutability::*;
        match *self {
            Not => stable_mir::mir::Mutability::Not,
            Mut => stable_mir::mir::Mutability::Mut,
        }
    }
}

impl Stable for mir::BorrowKind {
    type T = stable_mir::mir::BorrowKind;
    fn stable(&self) -> Self::T {
        use mir::BorrowKind::*;
        match *self {
            Shared => stable_mir::mir::BorrowKind::Shared,
            Shallow => stable_mir::mir::BorrowKind::Shallow,
            Mut { kind } => stable_mir::mir::BorrowKind::Mut { kind: kind.stable() },
        }
    }
}

impl Stable for mir::MutBorrowKind {
    type T = stable_mir::mir::MutBorrowKind;
    fn stable(&self) -> Self::T {
        use mir::MutBorrowKind::*;
        match *self {
            Default => stable_mir::mir::MutBorrowKind::Default,
            TwoPhaseBorrow => stable_mir::mir::MutBorrowKind::TwoPhaseBorrow,
            ClosureCapture => stable_mir::mir::MutBorrowKind::ClosureCapture,
        }
    }
}

impl<'tcx> Stable for mir::NullOp<'tcx> {
    type T = stable_mir::mir::NullOp;
    fn stable(&self) -> Self::T {
        use mir::NullOp::*;
        match self {
            SizeOf => stable_mir::mir::NullOp::SizeOf,
            AlignOf => stable_mir::mir::NullOp::AlignOf,
            OffsetOf(indices) => {
                stable_mir::mir::NullOp::OffsetOf(indices.iter().map(|idx| idx.stable()).collect())
            }
        }
    }
}

impl Stable for mir::CastKind {
    type T = stable_mir::mir::CastKind;
    fn stable(&self) -> Self::T {
        use mir::CastKind::*;
        match self {
            PointerExposeAddress => stable_mir::mir::CastKind::PointerExposeAddress,
            PointerFromExposedAddress => stable_mir::mir::CastKind::PointerFromExposedAddress,
            PointerCoercion(c) => stable_mir::mir::CastKind::PointerCoercion(c.stable()),
            DynStar => stable_mir::mir::CastKind::DynStar,
            IntToInt => stable_mir::mir::CastKind::IntToInt,
            FloatToInt => stable_mir::mir::CastKind::FloatToInt,
            FloatToFloat => stable_mir::mir::CastKind::FloatToFloat,
            IntToFloat => stable_mir::mir::CastKind::IntToFloat,
            PtrToPtr => stable_mir::mir::CastKind::PtrToPtr,
            FnPtrToPtr => stable_mir::mir::CastKind::FnPtrToPtr,
            Transmute => stable_mir::mir::CastKind::Transmute,
        }
    }
}

impl Stable for ty::adjustment::PointerCoercion {
    type T = stable_mir::mir::PointerCoercion;
    fn stable(&self) -> Self::T {
        use ty::adjustment::PointerCoercion;
        match self {
            PointerCoercion::ReifyFnPointer => stable_mir::mir::PointerCoercion::ReifyFnPointer,
            PointerCoercion::UnsafeFnPointer => stable_mir::mir::PointerCoercion::UnsafeFnPointer,
            PointerCoercion::ClosureFnPointer(unsafety) => {
                stable_mir::mir::PointerCoercion::ClosureFnPointer(unsafety.stable())
            }
            PointerCoercion::MutToConstPointer => {
                stable_mir::mir::PointerCoercion::MutToConstPointer
            }
            PointerCoercion::ArrayToPointer => stable_mir::mir::PointerCoercion::ArrayToPointer,
            PointerCoercion::Unsize => stable_mir::mir::PointerCoercion::Unsize,
        }
    }
}

impl Stable for rustc_hir::Unsafety {
    type T = stable_mir::mir::Safety;
    fn stable(&self) -> Self::T {
        match self {
            rustc_hir::Unsafety::Unsafe => stable_mir::mir::Safety::Unsafe,
            rustc_hir::Unsafety::Normal => stable_mir::mir::Safety::Normal,
        }
    }
}

impl Stable for FieldIdx {
    type T = usize;
    fn stable(&self) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable for mir::Operand<'tcx> {
    type T = stable_mir::mir::Operand;
    fn stable(&self) -> Self::T {
        use mir::Operand::*;
        match self {
            Copy(place) => stable_mir::mir::Operand::Copy(place.stable()),
            Move(place) => stable_mir::mir::Operand::Move(place.stable()),
            Constant(c) => stable_mir::mir::Operand::Constant(c.to_string()),
        }
    }
}

impl<'tcx> Stable for mir::Place<'tcx> {
    type T = stable_mir::mir::Place;
    fn stable(&self) -> Self::T {
        stable_mir::mir::Place {
            local: self.local.as_usize(),
            projection: format!("{:?}", self.projection),
        }
    }
}

impl Stable for mir::UnwindAction {
    type T = stable_mir::mir::UnwindAction;
    fn stable(&self) -> Self::T {
        use rustc_middle::mir::UnwindAction;
        match self {
            UnwindAction::Continue => stable_mir::mir::UnwindAction::Continue,
            UnwindAction::Unreachable => stable_mir::mir::UnwindAction::Unreachable,
            UnwindAction::Terminate => stable_mir::mir::UnwindAction::Terminate,
            UnwindAction::Cleanup(bb) => stable_mir::mir::UnwindAction::Cleanup(bb.as_usize()),
        }
    }
}

impl<'tcx> Stable for mir::AssertMessage<'tcx> {
    type T = stable_mir::mir::AssertMessage;
    fn stable(&self) -> Self::T {
        use rustc_middle::mir::AssertKind;
        match self {
            AssertKind::BoundsCheck { len, index } => stable_mir::mir::AssertMessage::BoundsCheck {
                len: len.stable(),
                index: index.stable(),
            },
            AssertKind::Overflow(bin_op, op1, op2) => stable_mir::mir::AssertMessage::Overflow(
                bin_op.stable(),
                op1.stable(),
                op2.stable(),
            ),
            AssertKind::OverflowNeg(op) => stable_mir::mir::AssertMessage::OverflowNeg(op.stable()),
            AssertKind::DivisionByZero(op) => {
                stable_mir::mir::AssertMessage::DivisionByZero(op.stable())
            }
            AssertKind::RemainderByZero(op) => {
                stable_mir::mir::AssertMessage::RemainderByZero(op.stable())
            }
            AssertKind::ResumedAfterReturn(generator) => {
                stable_mir::mir::AssertMessage::ResumedAfterReturn(generator.stable())
            }
            AssertKind::ResumedAfterPanic(generator) => {
                stable_mir::mir::AssertMessage::ResumedAfterPanic(generator.stable())
            }
            AssertKind::MisalignedPointerDereference { required, found } => {
                stable_mir::mir::AssertMessage::MisalignedPointerDereference {
                    required: required.stable(),
                    found: found.stable(),
                }
            }
        }
    }
}

impl Stable for mir::BinOp {
    type T = stable_mir::mir::BinOp;
    fn stable(&self) -> Self::T {
        use mir::BinOp;
        match self {
            BinOp::Add => stable_mir::mir::BinOp::Add,
            BinOp::AddUnchecked => stable_mir::mir::BinOp::AddUnchecked,
            BinOp::Sub => stable_mir::mir::BinOp::Sub,
            BinOp::SubUnchecked => stable_mir::mir::BinOp::SubUnchecked,
            BinOp::Mul => stable_mir::mir::BinOp::Mul,
            BinOp::MulUnchecked => stable_mir::mir::BinOp::MulUnchecked,
            BinOp::Div => stable_mir::mir::BinOp::Div,
            BinOp::Rem => stable_mir::mir::BinOp::Rem,
            BinOp::BitXor => stable_mir::mir::BinOp::BitXor,
            BinOp::BitAnd => stable_mir::mir::BinOp::BitAnd,
            BinOp::BitOr => stable_mir::mir::BinOp::BitOr,
            BinOp::Shl => stable_mir::mir::BinOp::Shl,
            BinOp::ShlUnchecked => stable_mir::mir::BinOp::ShlUnchecked,
            BinOp::Shr => stable_mir::mir::BinOp::Shr,
            BinOp::ShrUnchecked => stable_mir::mir::BinOp::ShrUnchecked,
            BinOp::Eq => stable_mir::mir::BinOp::Eq,
            BinOp::Lt => stable_mir::mir::BinOp::Lt,
            BinOp::Le => stable_mir::mir::BinOp::Le,
            BinOp::Ne => stable_mir::mir::BinOp::Ne,
            BinOp::Ge => stable_mir::mir::BinOp::Ge,
            BinOp::Gt => stable_mir::mir::BinOp::Gt,
            BinOp::Offset => stable_mir::mir::BinOp::Offset,
        }
    }
}

impl Stable for mir::UnOp {
    type T = stable_mir::mir::UnOp;
    fn stable(&self) -> Self::T {
        use mir::UnOp;
        match self {
            UnOp::Not => stable_mir::mir::UnOp::Not,
            UnOp::Neg => stable_mir::mir::UnOp::Neg,
        }
    }
}

impl Stable for rustc_hir::GeneratorKind {
    type T = stable_mir::mir::GeneratorKind;
    fn stable(&self) -> Self::T {
        use rustc_hir::{AsyncGeneratorKind, GeneratorKind};
        match self {
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
}

impl<'tcx> Stable for mir::InlineAsmOperand<'tcx> {
    type T = stable_mir::mir::InlineAsmOperand;
    fn stable(&self) -> Self::T {
        use rustc_middle::mir::InlineAsmOperand;

        let (in_value, out_place) = match self {
            InlineAsmOperand::In { value, .. } => (Some(value.stable()), None),
            InlineAsmOperand::Out { place, .. } => (None, place.map(|place| place.stable())),
            InlineAsmOperand::InOut { in_value, out_place, .. } => {
                (Some(in_value.stable()), out_place.map(|place| place.stable()))
            }
            InlineAsmOperand::Const { .. }
            | InlineAsmOperand::SymFn { .. }
            | InlineAsmOperand::SymStatic { .. } => (None, None),
        };

        stable_mir::mir::InlineAsmOperand { in_value, out_place, raw_rpr: format!("{:?}", self) }
    }
}

impl<'tcx> Stable for mir::Terminator<'tcx> {
    type T = stable_mir::mir::Terminator;
    fn stable(&self) -> Self::T {
        use rustc_middle::mir::TerminatorKind::*;
        use stable_mir::mir::Terminator;
        match &self.kind {
            Goto { target } => Terminator::Goto { target: target.as_usize() },
            SwitchInt { discr, targets } => Terminator::SwitchInt {
                discr: discr.stable(),
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
            Drop { place, target, unwind, replace: _ } => Terminator::Drop {
                place: place.stable(),
                target: target.as_usize(),
                unwind: unwind.stable(),
            },
            Call { func, args, destination, target, unwind, call_source: _, fn_span: _ } => {
                Terminator::Call {
                    func: func.stable(),
                    args: args.iter().map(|arg| arg.stable()).collect(),
                    destination: destination.stable(),
                    target: target.map(|t| t.as_usize()),
                    unwind: unwind.stable(),
                }
            }
            Assert { cond, expected, msg, target, unwind } => Terminator::Assert {
                cond: cond.stable(),
                expected: *expected,
                msg: msg.stable(),
                target: target.as_usize(),
                unwind: unwind.stable(),
            },
            InlineAsm { template, operands, options, line_spans, destination, unwind } => {
                Terminator::InlineAsm {
                    template: format!("{:?}", template),
                    operands: operands.iter().map(|operand| operand.stable()).collect(),
                    options: format!("{:?}", options),
                    line_spans: format!("{:?}", line_spans),
                    destination: destination.map(|d| d.as_usize()),
                    unwind: unwind.stable(),
                }
            }
            Yield { .. } | GeneratorDrop | FalseEdge { .. } | FalseUnwind { .. } => unreachable!(),
        }
    }
}
