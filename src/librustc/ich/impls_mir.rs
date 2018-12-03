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
impl_stable_hash_for!(enum mir::LocalKind { Var, Temp, Arg, ReturnPointer });
impl_stable_hash_for!(struct mir::LocalDecl<'tcx> {
    mutability,
    ty,
    user_ty,
    name,
    source_info,
    visibility_scope,
    internal,
    is_block_tail,
    is_user_variable
});
impl_stable_hash_for!(struct mir::UpvarDecl { debug_name, var_hir_id, by_ref, mutability });
impl_stable_hash_for!(struct mir::BasicBlockData<'tcx> { statements, terminator, is_cleanup });
impl_stable_hash_for!(struct mir::UnsafetyViolation { source_info, description, details, kind });
impl_stable_hash_for!(struct mir::UnsafetyCheckResult { violations, unsafe_blocks });

impl_stable_hash_for!(enum mir::BorrowKind {
    Shared,
    Shallow,
    Unique,
    Mut { allow_two_phase_borrow },
});

impl_stable_hash_for!(enum mir::UnsafetyViolationKind {
    General,
    MinConstFn,
    ExternStatic(lint_node_id),
    BorrowPacked(lint_node_id),
});

impl_stable_hash_for!(struct mir::Terminator<'tcx> {
    kind,
    source_info
});

impl_stable_hash_for!(
    impl<T> for enum mir::ClearCrossCrate<T> [ mir::ClearCrossCrate ] {
        Clear,
        Set(value),
    }
);

impl<'a> HashStable<StableHashingContext<'a>> for mir::Local {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for mir::BasicBlock {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for mir::Field {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>>
for mir::SourceScope {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for mir::Promoted {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for mir::TerminatorKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
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
            mir::TerminatorKind::Abort |
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
                                        cleanup,
                                        from_hir_call, } => {
                func.hash_stable(hcx, hasher);
                args.hash_stable(hcx, hasher);
                destination.hash_stable(hcx, hasher);
                cleanup.hash_stable(hcx, hasher);
                from_hir_call.hash_stable(hcx, hasher);
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
            mir::TerminatorKind::FalseEdges { ref real_target, ref imaginary_targets } => {
                real_target.hash_stable(hcx, hasher);
                for target in imaginary_targets {
                    target.hash_stable(hcx, hasher);
                }
            }
            mir::TerminatorKind::FalseUnwind { ref real_target, ref unwind } => {
                real_target.hash_stable(hcx, hasher);
                unwind.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct mir::Statement<'tcx> { source_info, kind });

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for mir::StatementKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::StatementKind::Assign(ref place, ref rvalue) => {
                place.hash_stable(hcx, hasher);
                rvalue.hash_stable(hcx, hasher);
            }
            mir::StatementKind::FakeRead(ref cause, ref place) => {
                cause.hash_stable(hcx, hasher);
                place.hash_stable(hcx, hasher);
            }
            mir::StatementKind::SetDiscriminant { ref place, variant_index } => {
                place.hash_stable(hcx, hasher);
                variant_index.hash_stable(hcx, hasher);
            }
            mir::StatementKind::StorageLive(ref place) |
            mir::StatementKind::StorageDead(ref place) => {
                place.hash_stable(hcx, hasher);
            }
            mir::StatementKind::EscapeToRaw(ref place) => {
                place.hash_stable(hcx, hasher);
            }
            mir::StatementKind::Retag { fn_entry, ref place } => {
                fn_entry.hash_stable(hcx, hasher);
                place.hash_stable(hcx, hasher);
            }
            mir::StatementKind::AscribeUserType(ref place, ref variance, ref c_ty) => {
                place.hash_stable(hcx, hasher);
                variance.hash_stable(hcx, hasher);
                c_ty.hash_stable(hcx, hasher);
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

impl_stable_hash_for!(enum mir::FakeReadCause { ForMatchGuard, ForMatchedPlace, ForLet });

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for mir::Place<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::Place::Local(ref local) => {
                local.hash_stable(hcx, hasher);
            }
            mir::Place::Static(ref statik) => {
                statik.hash_stable(hcx, hasher);
            }
            mir::Place::Promoted(ref promoted) => {
                promoted.hash_stable(hcx, hasher);
            }
            mir::Place::Projection(ref place_projection) => {
                place_projection.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx, B, V, T> HashStable<StableHashingContext<'a>>
for mir::Projection<'gcx, B, V, T>
    where B: HashStable<StableHashingContext<'a>>,
          V: HashStable<StableHashingContext<'a>>,
          T: HashStable<StableHashingContext<'a>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let mir::Projection {
            ref base,
            ref elem,
        } = *self;

        base.hash_stable(hcx, hasher);
        elem.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, V, T> HashStable<StableHashingContext<'a>>
for mir::ProjectionElem<'gcx, V, T>
    where V: HashStable<StableHashingContext<'a>>,
          T: HashStable<StableHashingContext<'a>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
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

impl_stable_hash_for!(struct mir::SourceScopeData { span, parent_scope });
impl_stable_hash_for!(struct mir::SourceScopeLocalData {
    lint_root, safety
});

impl<'a> HashStable<StableHashingContext<'a>> for mir::Safety {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::Safety::Safe |
            mir::Safety::BuiltinUnsafe |
            mir::Safety::FnUnsafe => {}
            mir::Safety::ExplicitUnsafe(node_id) => {
                node_id.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for mir::Operand<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            mir::Operand::Copy(ref place) => {
                place.hash_stable(hcx, hasher);
            }
            mir::Operand::Move(ref place) => {
                place.hash_stable(hcx, hasher);
            }
            mir::Operand::Constant(ref constant) => {
                constant.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for mir::Rvalue<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
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
            mir::Rvalue::Ref(region, borrow_kind, ref place) => {
                region.hash_stable(hcx, hasher);
                borrow_kind.hash_stable(hcx, hasher);
                place.hash_stable(hcx, hasher);
            }
            mir::Rvalue::Len(ref place) => {
                place.hash_stable(hcx, hasher);
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
            mir::Rvalue::Discriminant(ref place) => {
                place.hash_stable(hcx, hasher);
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

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for mir::AggregateKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::AggregateKind::Tuple => {}
            mir::AggregateKind::Array(t) => {
                t.hash_stable(hcx, hasher);
            }
            mir::AggregateKind::Adt(adt_def, idx, substs, user_substs, active_field) => {
                adt_def.hash_stable(hcx, hasher);
                idx.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
                user_substs.hash_stable(hcx, hasher);
                active_field.hash_stable(hcx, hasher);
            }
            mir::AggregateKind::Closure(def_id, ref substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            mir::AggregateKind::Generator(def_id, ref substs, movability) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
                movability.hash_stable(hcx, hasher);
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

impl_stable_hash_for!(struct mir::Constant<'tcx> { span, ty, user_ty, literal });

impl_stable_hash_for!(struct mir::Location { block, statement_index });

impl_stable_hash_for!(struct mir::BorrowCheckResult<'tcx> {
    closure_requirements,
    used_mut_upvars
});

impl_stable_hash_for!(struct mir::ClosureRegionRequirements<'tcx> {
    num_external_vids,
    outlives_requirements
});

impl_stable_hash_for!(struct mir::ClosureOutlivesRequirement<'tcx> {
    subject,
    outlived_free_region,
    blame_span,
    category
});

impl_stable_hash_for!(enum mir::ConstraintCategory {
    Return,
    UseAsConst,
    UseAsStatic,
    TypeAnnotation,
    Cast,
    ClosureBounds,
    CallArgument,
    CopyBound,
    SizedBound,
    Assignment,
    OpaqueType,
    Boring,
    BoringNoLocation,
    Internal,
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for mir::ClosureOutlivesSubject<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::ClosureOutlivesSubject::Ty(ref ty) => {
                ty.hash_stable(hcx, hasher);
            }
            mir::ClosureOutlivesSubject::Region(ref region) => {
                region.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct mir::interpret::GlobalId<'tcx> { instance, promoted });

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for mir::UserTypeAnnotation<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            mir::UserTypeAnnotation::Ty(ref ty) => {
                ty.hash_stable(hcx, hasher);
            }
            mir::UserTypeAnnotation::TypeOf(ref def_id, ref substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct mir::UserTypeProjection<'tcx> { base, projs });
impl_stable_hash_for!(struct mir::UserTypeProjections<'tcx> { contents });
