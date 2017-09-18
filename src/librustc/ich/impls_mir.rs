// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains `HashStable` implementations for various MIR data
//! types in no particular order.

use ich::StableHashingContext;
use mir;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};
use std::mem;

impl_stable_hash_for!(struct mir::GeneratorLayout<'tcx> { fields });
impl_stable_hash_for!(struct mir::SourceInfo { span, scope });
impl_stable_hash_for!(enum mir::Mutability { Mut, Not });
impl_stable_hash_for!(enum mir::BorrowKind { Shared, Unique, Mut });
impl_stable_hash_for!(enum mir::LocalKind { Var, Temp, Arg, ReturnPointer });
impl_stable_hash_for!(struct mir::LocalDecl<'tcx> {
    mutability,
    ty,
    name,
    source_info,
    internal,
    is_user_variable
});
impl_stable_hash_for!(struct mir::UpvarDecl { debug_name, by_ref });
impl_stable_hash_for!(struct mir::BasicBlockData<'tcx> { statements, terminator, is_cleanup });

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for mir::Terminator<'gcx> {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let mir::Terminator {
            ref kind,
            ref source_info,
        } = *self;

        let hash_spans_unconditionally = match *kind {
            mir::TerminatorKind::Assert { .. } => {
                // Assert terminators generate a panic message that contains the
                // source location, so we always have to feed its span into the
                // ICH.
                true
            }
            mir::TerminatorKind::Goto { .. } |
            mir::TerminatorKind::SwitchInt { .. } |
            mir::TerminatorKind::Resume |
            mir::TerminatorKind::Return |
            mir::TerminatorKind::GeneratorDrop |
            mir::TerminatorKind::Unreachable |
            mir::TerminatorKind::Drop { .. } |
            mir::TerminatorKind::DropAndReplace { .. } |
            mir::TerminatorKind::Yield { .. } |
            mir::TerminatorKind::Call { .. } => false,
        };

        if hash_spans_unconditionally {
            hcx.while_hashing_spans(true, |hcx| {
                source_info.hash_stable(hcx, hasher);
            })
        } else {
            source_info.hash_stable(hcx, hasher);
        }

        kind.hash_stable(hcx, hasher);
    }
}


impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Local {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::BasicBlock {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Field {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for mir::VisibilityScope {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Promoted {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for mir::TerminatorKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::TerminatorKind::Goto { ref target } => {
                target.hash_stable(hcx, hasher);
            }
            mir::TerminatorKind::SwitchInt { ref discr,
                                             switch_ty,
                                             ref values,
                                             ref targets } => {
                discr.hash_stable(hcx, hasher);
                switch_ty.hash_stable(hcx, hasher);
                values.hash_stable(hcx, hasher);
                targets.hash_stable(hcx, hasher);
            }
            mir::TerminatorKind::Resume |
            mir::TerminatorKind::Return |
            mir::TerminatorKind::GeneratorDrop |
            mir::TerminatorKind::Unreachable => {}
            mir::TerminatorKind::Drop { ref location, target, unwind } => {
                location.hash_stable(hcx, hasher);
                target.hash_stable(hcx, hasher);
                unwind.hash_stable(hcx, hasher);
            }
            mir::TerminatorKind::DropAndReplace { ref location,
                                                  ref value,
                                                  target,
                                                  unwind, } => {
                location.hash_stable(hcx, hasher);
                value.hash_stable(hcx, hasher);
                target.hash_stable(hcx, hasher);
                unwind.hash_stable(hcx, hasher);
            }
            mir::TerminatorKind::Yield { ref value,
                                        resume,
                                        drop } => {
                value.hash_stable(hcx, hasher);
                resume.hash_stable(hcx, hasher);
                drop.hash_stable(hcx, hasher);
            }
            mir::TerminatorKind::Call { ref func,
                                        ref args,
                                        ref destination,
                                        cleanup } => {
                func.hash_stable(hcx, hasher);
                args.hash_stable(hcx, hasher);
                destination.hash_stable(hcx, hasher);
                cleanup.hash_stable(hcx, hasher);
            }
            mir::TerminatorKind::Assert { ref cond,
                                          expected,
                                          ref msg,
                                          target,
                                          cleanup } => {
                cond.hash_stable(hcx, hasher);
                expected.hash_stable(hcx, hasher);
                msg.hash_stable(hcx, hasher);
                target.hash_stable(hcx, hasher);
                cleanup.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for mir::AssertMessage<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                len.hash_stable(hcx, hasher);
                index.hash_stable(hcx, hasher);
            }
            mir::AssertMessage::Math(ref const_math_err) => {
                const_math_err.hash_stable(hcx, hasher);
            }
            mir::AssertMessage::GeneratorResumedAfterReturn => (),
            mir::AssertMessage::GeneratorResumedAfterPanic => (),
        }
    }
}

impl_stable_hash_for!(struct mir::Statement<'tcx> { source_info, kind });

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for mir::StatementKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                lvalue.hash_stable(hcx, hasher);
                rvalue.hash_stable(hcx, hasher);
            }
            mir::StatementKind::SetDiscriminant { ref lvalue, variant_index } => {
                lvalue.hash_stable(hcx, hasher);
                variant_index.hash_stable(hcx, hasher);
            }
            mir::StatementKind::StorageLive(ref lvalue) |
            mir::StatementKind::StorageDead(ref lvalue) => {
                lvalue.hash_stable(hcx, hasher);
            }
            mir::StatementKind::EndRegion(ref region_scope) => {
                region_scope.hash_stable(hcx, hasher);
            }
            mir::StatementKind::Validate(ref op, ref lvalues) => {
                op.hash_stable(hcx, hasher);
                lvalues.hash_stable(hcx, hasher);
            }
            mir::StatementKind::Nop => {}
            mir::StatementKind::InlineAsm { ref asm, ref outputs, ref inputs } => {
                asm.hash_stable(hcx, hasher);
                outputs.hash_stable(hcx, hasher);
                inputs.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'gcx, T> HashStable<StableHashingContext<'gcx>>
    for mir::ValidationOperand<'gcx, T>
    where T: HashStable<StableHashingContext<'gcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>)
    {
        self.lval.hash_stable(hcx, hasher);
        self.ty.hash_stable(hcx, hasher);
        self.re.hash_stable(hcx, hasher);
        self.mutbl.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(enum mir::ValidationOp { Acquire, Release, Suspend(region_scope) });

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Lvalue<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::Lvalue::Local(ref local) => {
                local.hash_stable(hcx, hasher);
            }
            mir::Lvalue::Static(ref statik) => {
                statik.hash_stable(hcx, hasher);
            }
            mir::Lvalue::Projection(ref lvalue_projection) => {
                lvalue_projection.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'gcx, B, V, T> HashStable<StableHashingContext<'gcx>>
for mir::Projection<'gcx, B, V, T>
    where B: HashStable<StableHashingContext<'gcx>>,
          V: HashStable<StableHashingContext<'gcx>>,
          T: HashStable<StableHashingContext<'gcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let mir::Projection {
            ref base,
            ref elem,
        } = *self;

        base.hash_stable(hcx, hasher);
        elem.hash_stable(hcx, hasher);
    }
}

impl<'gcx, V, T> HashStable<StableHashingContext<'gcx>>
for mir::ProjectionElem<'gcx, V, T>
    where V: HashStable<StableHashingContext<'gcx>>,
          T: HashStable<StableHashingContext<'gcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::ProjectionElem::Deref => {}
            mir::ProjectionElem::Field(field, ref ty) => {
                field.hash_stable(hcx, hasher);
                ty.hash_stable(hcx, hasher);
            }
            mir::ProjectionElem::Index(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            mir::ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                offset.hash_stable(hcx, hasher);
                min_length.hash_stable(hcx, hasher);
                from_end.hash_stable(hcx, hasher);
            }
            mir::ProjectionElem::Subslice { from, to } => {
                from.hash_stable(hcx, hasher);
                to.hash_stable(hcx, hasher);
            }
            mir::ProjectionElem::Downcast(adt_def, variant) => {
                adt_def.hash_stable(hcx, hasher);
                variant.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct mir::VisibilityScopeData { span, parent_scope });

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Operand<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::Operand::Consume(ref lvalue) => {
                lvalue.hash_stable(hcx, hasher);
            }
            mir::Operand::Constant(ref constant) => {
                constant.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Rvalue<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::Rvalue::Use(ref operand) => {
                operand.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Repeat(ref operand, ref val) => {
                operand.hash_stable(hcx, hasher);
                val.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Ref(region, borrow_kind, ref lvalue) => {
                region.hash_stable(hcx, hasher);
                borrow_kind.hash_stable(hcx, hasher);
                lvalue.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Len(ref lvalue) => {
                lvalue.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Cast(cast_kind, ref operand, ty) => {
                cast_kind.hash_stable(hcx, hasher);
                operand.hash_stable(hcx, hasher);
                ty.hash_stable(hcx, hasher);
            }
            mir::Rvalue::BinaryOp(op, ref operand1, ref operand2) |
            mir::Rvalue::CheckedBinaryOp(op, ref operand1, ref operand2) => {
                op.hash_stable(hcx, hasher);
                operand1.hash_stable(hcx, hasher);
                operand2.hash_stable(hcx, hasher);
            }
            mir::Rvalue::UnaryOp(op, ref operand) => {
                op.hash_stable(hcx, hasher);
                operand.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Discriminant(ref lvalue) => {
                lvalue.hash_stable(hcx, hasher);
            }
            mir::Rvalue::NullaryOp(op, ty) => {
                op.hash_stable(hcx, hasher);
                ty.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                kind.hash_stable(hcx, hasher);
                operands.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum mir::CastKind {
    Misc,
    ReifyFnPointer,
    ClosureFnPointer,
    UnsafeFnPointer,
    Unsize
});

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for mir::AggregateKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::AggregateKind::Tuple => {}
            mir::AggregateKind::Array(t) => {
                t.hash_stable(hcx, hasher);
            }
            mir::AggregateKind::Adt(adt_def, idx, substs, active_field) => {
                adt_def.hash_stable(hcx, hasher);
                idx.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
                active_field.hash_stable(hcx, hasher);
            }
            mir::AggregateKind::Closure(def_id, ref substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            mir::AggregateKind::Generator(def_id, ref substs, ref interior) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
                interior.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum mir::BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Offset
});

impl_stable_hash_for!(enum mir::UnOp {
    Not,
    Neg
});

impl_stable_hash_for!(enum mir::NullOp {
    Box,
    SizeOf
});

impl_stable_hash_for!(struct mir::Constant<'tcx> { span, ty, literal });

impl<'gcx> HashStable<StableHashingContext<'gcx>> for mir::Literal<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::Literal::Value { ref value } => {
                value.hash_stable(hcx, hasher);
            }
            mir::Literal::Promoted { index } => {
                index.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct mir::Location { block, statement_index });
