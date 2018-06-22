// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains `HashStable` implementations for various data types
//! from rustc::ty in no particular order.

use ich::{Fingerprint, StableHashingContext, NodeIdHashingMode};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use std::cell::RefCell;
use std::hash as std_hash;
use std::mem;
use middle::region;
use infer;
use traits;
use ty;
use mir;

impl<'a, 'gcx, T> HashStable<StableHashingContext<'a>>
for &'gcx ty::Slice<T>
    where T: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<(usize, usize), Fingerprint>> =
                RefCell::new(FxHashMap());
        }

        let hash = CACHE.with(|cache| {
            let key = (self.as_ptr() as usize, self.len());
            if let Some(&hash) = cache.borrow().get(&key) {
                return hash;
            }

            let mut hasher = StableHasher::new();
            (&self[..]).hash_stable(hcx, &mut hasher);

            let hash: Fingerprint = hasher.finish();
            cache.borrow_mut().insert(key, hash);
            hash
        });

        hash.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, T> ToStableHashKey<StableHashingContext<'a>> for &'gcx ty::Slice<T>
    where T: HashStable<StableHashingContext<'a>>
{
    type KeyType = Fingerprint;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &StableHashingContext<'a>) -> Fingerprint {
        let mut hasher = StableHasher::new();
        let mut hcx: StableHashingContext<'a> = hcx.clone();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish()
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for ty::subst::Kind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.unpack().hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::subst::UnpackedKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            ty::subst::UnpackedKind::Lifetime(lt) => lt.hash_stable(hcx, hasher),
            ty::subst::UnpackedKind::Type(ty) => ty.hash_stable(hcx, hasher),
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>>
for ty::RegionKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::ReErased |
            ty::ReStatic |
            ty::ReEmpty => {
                // No variant fields to hash for these ...
            }
            ty::ReCanonical(c) => {
                c.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrAnon(i)) => {
                db.hash_stable(hcx, hasher);
                i.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrNamed(def_id, name)) => {
                db.hash_stable(hcx, hasher);
                def_id.hash_stable(hcx, hasher);
                name.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrEnv) => {
                db.hash_stable(hcx, hasher);
            }
            ty::ReEarlyBound(ty::EarlyBoundRegion { def_id, index, name }) => {
                def_id.hash_stable(hcx, hasher);
                index.hash_stable(hcx, hasher);
                name.hash_stable(hcx, hasher);
            }
            ty::ReScope(scope) => {
                scope.hash_stable(hcx, hasher);
            }
            ty::ReFree(ref free_region) => {
                free_region.hash_stable(hcx, hasher);
            }
            ty::ReClosureBound(vid) => {
                vid.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(..) |
            ty::ReVar(..) |
            ty::ReSkolemized(..) => {
                bug!("StableHasher: unexpected region {:?}", *self)
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::RegionVid {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::CanonicalVar {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use rustc_data_structures::indexed_vec::Idx;
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::adjustment::AutoBorrow<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::adjustment::AutoBorrow::Ref(ref region, mutability) => {
                region.hash_stable(hcx, hasher);
                mutability.hash_stable(hcx, hasher);
            }
            ty::adjustment::AutoBorrow::RawPtr(mutability) => {
                mutability.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::adjustment::Adjust<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::adjustment::Adjust::NeverToAny |
            ty::adjustment::Adjust::ReifyFnPointer |
            ty::adjustment::Adjust::UnsafeFnPointer |
            ty::adjustment::Adjust::ClosureFnPointer |
            ty::adjustment::Adjust::MutToConstPointer |
            ty::adjustment::Adjust::Unsize => {}
            ty::adjustment::Adjust::Deref(ref overloaded) => {
                overloaded.hash_stable(hcx, hasher);
            }
            ty::adjustment::Adjust::Borrow(ref autoref) => {
                autoref.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ty::adjustment::Adjustment<'tcx> { kind, target });
impl_stable_hash_for!(struct ty::adjustment::OverloadedDeref<'tcx> { region, mutbl });
impl_stable_hash_for!(struct ty::UpvarBorrow<'tcx> { kind, region });
impl_stable_hash_for!(enum ty::adjustment::AllowTwoPhase {
    Yes,
    No
});

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::adjustment::AutoBorrowMutability {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::adjustment::AutoBorrowMutability::Mutable { ref allow_two_phase_borrow } => {
                allow_two_phase_borrow.hash_stable(hcx, hasher);
            }
            ty::adjustment::AutoBorrowMutability::Immutable => {}
        }
    }
}

impl_stable_hash_for!(struct ty::UpvarId { var_id, closure_expr_id });

impl_stable_hash_for!(enum ty::BorrowKind {
    ImmBorrow,
    UniqueImmBorrow,
    MutBorrow
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::UpvarCapture<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::UpvarCapture::ByValue => {}
            ty::UpvarCapture::ByRef(ref up_var_borrow) => {
                up_var_borrow.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ty::GenSig<'tcx> {
    yield_ty,
    return_ty
});

impl_stable_hash_for!(struct ty::FnSig<'tcx> {
    inputs_and_output,
    variadic,
    unsafety,
    abi
});

impl<'a, 'gcx, T> HashStable<StableHashingContext<'a>> for ty::Binder<T>
    where T: HashStable<StableHashingContext<'a>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.skip_binder().hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(enum ty::ClosureKind { Fn, FnMut, FnOnce });

impl_stable_hash_for!(enum ty::Visibility {
    Public,
    Restricted(def_id),
    Invisible
});

impl_stable_hash_for!(struct ty::TraitRef<'tcx> { def_id, substs });
impl_stable_hash_for!(struct ty::TraitPredicate<'tcx> { trait_ref });
impl_stable_hash_for!(struct ty::SubtypePredicate<'tcx> { a_is_expected, a, b });

impl<'a, 'gcx, A, B> HashStable<StableHashingContext<'a>>
for ty::OutlivesPredicate<A, B>
    where A: HashStable<StableHashingContext<'a>>,
          B: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::OutlivesPredicate(ref a, ref b) = *self;
        a.hash_stable(hcx, hasher);
        b.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::ProjectionPredicate<'tcx> { projection_ty, ty });
impl_stable_hash_for!(struct ty::ProjectionTy<'tcx> { substs, item_def_id });


impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for ty::Predicate<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::Predicate::Trait(ref pred) => {
                pred.hash_stable(hcx, hasher);
            }
            ty::Predicate::Subtype(ref pred) => {
                pred.hash_stable(hcx, hasher);
            }
            ty::Predicate::RegionOutlives(ref pred) => {
                pred.hash_stable(hcx, hasher);
            }
            ty::Predicate::TypeOutlives(ref pred) => {
                pred.hash_stable(hcx, hasher);
            }
            ty::Predicate::Projection(ref pred) => {
                pred.hash_stable(hcx, hasher);
            }
            ty::Predicate::WellFormed(ty) => {
                ty.hash_stable(hcx, hasher);
            }
            ty::Predicate::ObjectSafe(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            ty::Predicate::ClosureKind(def_id, closure_substs, closure_kind) => {
                def_id.hash_stable(hcx, hasher);
                closure_substs.hash_stable(hcx, hasher);
                closure_kind.hash_stable(hcx, hasher);
            }
            ty::Predicate::ConstEvaluatable(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::AdtFlags {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        std_hash::Hash::hash(self, hasher);
    }
}

impl_stable_hash_for!(struct ty::VariantDef {
    did,
    name,
    discr,
    fields,
    ctor_kind
});

impl_stable_hash_for!(enum ty::VariantDiscr {
    Explicit(def_id),
    Relative(distance)
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for ty::FieldDef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::FieldDef { did, ident, vis } = *self;

        did.hash_stable(hcx, hasher);
        ident.name.hash_stable(hcx, hasher);
        vis.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ::middle::const_val::ConstVal<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use middle::const_val::ConstVal::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Unevaluated(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            Value(ref value) => {
                value.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ::mir::interpret::ConstValue<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use mir::interpret::ConstValue::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Scalar(val) => {
                val.hash_stable(hcx, hasher);
            }
            ScalarPair(a, b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            ByRef(alloc, offset) => {
                alloc.hash_stable(hcx, hasher);
                offset.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum mir::interpret::Value {
    Scalar(v),
    ScalarPair(a, b),
    ByRef(ptr, align)
});

impl_stable_hash_for!(struct mir::interpret::Pointer {
    alloc_id,
    offset
});

impl<'a> HashStable<StableHashingContext<'a>> for mir::interpret::AllocId {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'a>,
        hasher: &mut StableHasher<W>,
    ) {
        ty::tls::with_opt(|tcx| {
            trace!("hashing {:?}", *self);
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            let alloc_kind = tcx.alloc_map.lock().get(*self).expect("no value for AllocId");
            alloc_kind.hash_stable(hcx, hasher);
        });
    }
}

impl<'a, 'gcx, M: HashStable<StableHashingContext<'a>>> HashStable<StableHashingContext<'a>>
for mir::interpret::AllocType<'gcx, M> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use mir::interpret::AllocType::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Function(instance) => instance.hash_stable(hcx, hasher),
            Static(def_id) => def_id.hash_stable(hcx, hasher),
            Memory(ref mem) => mem.hash_stable(hcx, hasher),
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for mir::interpret::Allocation {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'a>,
        hasher: &mut StableHasher<W>,
    ) {
        self.bytes.hash_stable(hcx, hasher);
        for reloc in self.relocations.iter() {
            reloc.hash_stable(hcx, hasher);
        }
        self.undef_mask.hash_stable(hcx, hasher);
        self.align.hash_stable(hcx, hasher);
        self.runtime_mutability.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(enum ::syntax::ast::Mutability {
    Immutable,
    Mutable
});


impl<'a> HashStable<StableHashingContext<'a>>
for ::mir::interpret::Scalar {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use mir::interpret::Scalar::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            Bits { bits, defined } => {
                bits.hash_stable(hcx, hasher);
                defined.hash_stable(hcx, hasher);
            },
            Ptr(ptr) => ptr.hash_stable(hcx, hasher),
        }
    }
}

impl_stable_hash_for!(struct ty::Const<'tcx> {
    ty,
    val
});

impl_stable_hash_for!(struct ::middle::const_val::ConstEvalErr<'tcx> {
    span,
    kind
});

impl_stable_hash_for!(struct ::middle::const_val::FrameInfo {
    span,
    lint_root,
    location
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ::middle::const_val::ErrKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use middle::const_val::ErrKind::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            TypeckError |
            CouldNotResolve |
            CheckMatchError => {
                // nothing to do
            }
            Miri(ref err, ref trace) => {
                err.hash_stable(hcx, hasher);
                trace.hash_stable(hcx, hasher);
            },
        }
    }
}

impl_stable_hash_for!(struct ty::ClosureSubsts<'tcx> { substs });
impl_stable_hash_for!(struct ty::GeneratorSubsts<'tcx> { substs });

impl_stable_hash_for!(struct ty::GenericPredicates<'tcx> {
    parent,
    predicates
});


impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ::mir::interpret::EvalError<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.kind.hash_stable(hcx, hasher)
    }
}

impl<'a, 'gcx, O: HashStable<StableHashingContext<'a>>> HashStable<StableHashingContext<'a>>
for ::mir::interpret::EvalErrorKind<'gcx, O> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use mir::interpret::EvalErrorKind::*;

        mem::discriminant(&self).hash_stable(hcx, hasher);

        match *self {
            DanglingPointerDeref |
            DoubleFree |
            InvalidMemoryAccess |
            InvalidFunctionPointer |
            InvalidBool |
            InvalidDiscriminant |
            InvalidNullPointerUsage |
            ReadPointerAsBytes |
            ReadBytesAsPointer |
            InvalidPointerMath |
            ReadUndefBytes |
            DeadLocal |
            StackFrameLimitReached |
            OutOfTls |
            TlsOutOfBounds |
            CalledClosureAsFunction |
            VtableForArgumentlessMethod |
            ModifiedConstantMemory |
            AssumptionNotHeld |
            InlineAsm |
            ReallocateNonBasePtr |
            DeallocateNonBasePtr |
            HeapAllocZeroBytes |
            Unreachable |
            Panic |
            ReadFromReturnPointer |
            UnimplementedTraitSelection |
            TypeckError |
            DerefFunctionPointer |
            ExecuteMemory |
            OverflowNeg |
            RemainderByZero |
            DivisionByZero |
            GeneratorResumedAfterReturn |
            GeneratorResumedAfterPanic => {}
            ReferencedConstant(ref err) => err.hash_stable(hcx, hasher),
            MachineError(ref err) => err.hash_stable(hcx, hasher),
            FunctionPointerTyMismatch(a, b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher)
            },
            NoMirFor(ref s) => s.hash_stable(hcx, hasher),
            UnterminatedCString(ptr) => ptr.hash_stable(hcx, hasher),
            PointerOutOfBounds {
                ptr,
                access,
                allocation_size,
            } => {
                ptr.hash_stable(hcx, hasher);
                access.hash_stable(hcx, hasher);
                allocation_size.hash_stable(hcx, hasher)
            },
            InvalidBoolOp(bop) => bop.hash_stable(hcx, hasher),
            Unimplemented(ref s) => s.hash_stable(hcx, hasher),
            BoundsCheck { ref len, ref index } => {
                len.hash_stable(hcx, hasher);
                index.hash_stable(hcx, hasher)
            },
            Intrinsic(ref s) => s.hash_stable(hcx, hasher),
            InvalidChar(c) => c.hash_stable(hcx, hasher),
            AbiViolation(ref s) => s.hash_stable(hcx, hasher),
            AlignmentCheckFailed {
                required,
                has,
            } => {
                required.hash_stable(hcx, hasher);
                has.hash_stable(hcx, hasher)
            },
            MemoryLockViolation {
                ptr,
                len,
                frame,
                access,
                ref lock,
            } =>  {
                ptr.hash_stable(hcx, hasher);
                len.hash_stable(hcx, hasher);
                frame.hash_stable(hcx, hasher);
                access.hash_stable(hcx, hasher);
                lock.hash_stable(hcx, hasher)
            },
            MemoryAcquireConflict {
                ptr,
                len,
                kind,
                ref lock,
            } =>  {
                ptr.hash_stable(hcx, hasher);
                len.hash_stable(hcx, hasher);
                kind.hash_stable(hcx, hasher);
                lock.hash_stable(hcx, hasher)
            },
            InvalidMemoryLockRelease {
                ptr,
                len,
                frame,
                ref lock,
            } =>  {
                ptr.hash_stable(hcx, hasher);
                len.hash_stable(hcx, hasher);
                frame.hash_stable(hcx, hasher);
                lock.hash_stable(hcx, hasher)
            },
            DeallocatedLockedMemory {
                ptr,
                ref lock,
            } => {
                ptr.hash_stable(hcx, hasher);
                lock.hash_stable(hcx, hasher)
            },
            ValidationFailure(ref s) => s.hash_stable(hcx, hasher),
            TypeNotPrimitive(ty) => ty.hash_stable(hcx, hasher),
            ReallocatedWrongMemoryKind(ref a, ref b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher)
            },
            DeallocatedWrongMemoryKind(ref a, ref b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher)
            },
            IncorrectAllocationInformation(a, b, c, d) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
                c.hash_stable(hcx, hasher);
                d.hash_stable(hcx, hasher)
            },
            Layout(lay) => lay.hash_stable(hcx, hasher),
            HeapAllocNonPowerOfTwoAlignment(n) => n.hash_stable(hcx, hasher),
            PathNotFound(ref v) => v.hash_stable(hcx, hasher),
            Overflow(op) => op.hash_stable(hcx, hasher),
        }
    }
}

impl_stable_hash_for!(enum mir::interpret::Lock {
    NoLock,
    WriteLock(dl),
    ReadLock(v)
});

impl_stable_hash_for!(struct mir::interpret::DynamicLifetime {
    frame,
    region
});

impl_stable_hash_for!(enum mir::interpret::AccessKind {
    Read,
    Write
});

impl_stable_hash_for!(enum ty::Variance {
    Covariant,
    Invariant,
    Contravariant,
    Bivariant
});

impl_stable_hash_for!(enum ty::adjustment::CustomCoerceUnsized {
    Struct(index)
});

impl<'a> HashStable<StableHashingContext<'a>> for ty::Generics {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::Generics {
            parent,
            ref parent_count,
            ref params,

            // Reverse map to each param's `index` field, from its `def_id`.
            param_def_id_to_index: _, // Don't hash this
            has_self,
            has_late_bound_regions,
        } = *self;

        parent.hash_stable(hcx, hasher);
        parent_count.hash_stable(hcx, hasher);
        params.hash_stable(hcx, hasher);
        has_self.hash_stable(hcx, hasher);
        has_late_bound_regions.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::GenericParamDef {
    name,
    def_id,
    index,
    pure_wrt_drop,
    kind
});

impl<'a> HashStable<StableHashingContext<'a>> for ty::GenericParamDefKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::GenericParamDefKind::Lifetime => {}
            ty::GenericParamDefKind::Type {
                has_default,
                ref object_lifetime_default,
                ref synthetic,
            } => {
                has_default.hash_stable(hcx, hasher);
                object_lifetime_default.hash_stable(hcx, hasher);
                synthetic.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a, 'gcx, T> HashStable<StableHashingContext<'a>>
for ::middle::resolve_lifetime::Set1<T>
    where T: HashStable<StableHashingContext<'a>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use middle::resolve_lifetime::Set1;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            Set1::Empty |
            Set1::Many => {
                // Nothing to do.
            }
            Set1::One(ref value) => {
                value.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum ::middle::resolve_lifetime::LifetimeDefOrigin {
    Explicit,
    InBand
});

impl_stable_hash_for!(enum ::middle::resolve_lifetime::Region {
    Static,
    EarlyBound(index, decl, is_in_band),
    LateBound(db_index, decl, is_in_band),
    LateBoundAnon(db_index, anon_index),
    Free(call_site_scope_data, decl)
});

impl_stable_hash_for!(enum ty::cast::CastKind {
    CoercionCast,
    PtrPtrCast,
    PtrAddrCast,
    AddrPtrCast,
    NumericCast,
    EnumCast,
    PrimIntCast,
    U8CharCast,
    ArrayPtrCast,
    FnPtrPtrCast,
    FnPtrAddrCast
});

impl_stable_hash_for!(tuple_struct ::middle::region::FirstStatementIndex { idx });
impl_stable_hash_for!(struct ::middle::region::Scope { id, code });

impl<'a> ToStableHashKey<StableHashingContext<'a>> for region::Scope {
    type KeyType = region::Scope;

    #[inline]
    fn to_stable_hash_key(&self, _: &StableHashingContext<'a>) -> region::Scope {
        *self
    }
}

impl_stable_hash_for!(struct ::middle::region::BlockRemainder {
    block,
    first_statement_index
});

impl_stable_hash_for!(struct ty::adjustment::CoerceUnsizedInfo {
    custom_kind
});

impl_stable_hash_for!(struct ty::FreeRegion {
    scope,
    bound_region
});

impl_stable_hash_for!(enum ty::BoundRegion {
    BrAnon(index),
    BrNamed(def_id, name),
    BrFresh(index),
    BrEnv
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::TypeVariants<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use ty::TypeVariants::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            TyBool  |
            TyChar  |
            TyStr   |
            TyError |
            TyNever => {
                // Nothing more to hash.
            }
            TyInt(int_ty) => {
                int_ty.hash_stable(hcx, hasher);
            }
            TyUint(uint_ty) => {
                uint_ty.hash_stable(hcx, hasher);
            }
            TyFloat(float_ty)  => {
                float_ty.hash_stable(hcx, hasher);
            }
            TyAdt(adt_def, substs) => {
                adt_def.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            TyArray(inner_ty, len) => {
                inner_ty.hash_stable(hcx, hasher);
                len.hash_stable(hcx, hasher);
            }
            TySlice(inner_ty) => {
                inner_ty.hash_stable(hcx, hasher);
            }
            TyRawPtr(pointee_ty) => {
                pointee_ty.hash_stable(hcx, hasher);
            }
            TyRef(region, pointee_ty, mutbl) => {
                region.hash_stable(hcx, hasher);
                pointee_ty.hash_stable(hcx, hasher);
                mutbl.hash_stable(hcx, hasher);
            }
            TyFnDef(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            TyFnPtr(ref sig) => {
                sig.hash_stable(hcx, hasher);
            }
            TyDynamic(ref existential_predicates, region) => {
                existential_predicates.hash_stable(hcx, hasher);
                region.hash_stable(hcx, hasher);
            }
            TyClosure(def_id, closure_substs) => {
                def_id.hash_stable(hcx, hasher);
                closure_substs.hash_stable(hcx, hasher);
            }
            TyGenerator(def_id, generator_substs, movability) => {
                def_id.hash_stable(hcx, hasher);
                generator_substs.hash_stable(hcx, hasher);
                movability.hash_stable(hcx, hasher);
            }
            TyGeneratorWitness(types) => {
                types.hash_stable(hcx, hasher)
            }
            TyTuple(inner_tys) => {
                inner_tys.hash_stable(hcx, hasher);
            }
            TyProjection(ref projection_ty) => {
                projection_ty.hash_stable(hcx, hasher);
            }
            TyAnon(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            TyParam(param_ty) => {
                param_ty.hash_stable(hcx, hasher);
            }
            TyForeign(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            TyInfer(infer_ty) => {
                infer_ty.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(enum ty::InferTy {
    TyVar(a),
    IntVar(a),
    FloatVar(a),
    FreshTy(a),
    FreshIntTy(a),
    FreshFloatTy(a),
    CanonicalTy(a),
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::TyVid
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          _hcx: &mut StableHashingContext<'a>,
                                          _hasher: &mut StableHasher<W>) {
        // TyVid values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TypeVariants::hash_stable() - can't hash a TyVid {:?}.", *self)
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::IntVid
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          _hcx: &mut StableHashingContext<'a>,
                                          _hasher: &mut StableHasher<W>) {
        // IntVid values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TypeVariants::hash_stable() - can't hash an IntVid {:?}.", *self)
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::FloatVid
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          _hcx: &mut StableHashingContext<'a>,
                                          _hasher: &mut StableHasher<W>) {
        // FloatVid values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TypeVariants::hash_stable() - can't hash a FloatVid {:?}.", *self)
    }
}

impl_stable_hash_for!(struct ty::ParamTy {
    idx,
    name
});

impl_stable_hash_for!(struct ty::TypeAndMut<'tcx> {
    ty,
    mutbl
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::ExistentialPredicate<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::ExistentialPredicate::Trait(ref trait_ref) => {
                trait_ref.hash_stable(hcx, hasher);
            }
            ty::ExistentialPredicate::Projection(ref projection) => {
                projection.hash_stable(hcx, hasher);
            }
            ty::ExistentialPredicate::AutoTrait(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ty::ExistentialTraitRef<'tcx> {
    def_id,
    substs
});

impl_stable_hash_for!(struct ty::ExistentialProjection<'tcx> {
    item_def_id,
    substs,
    ty
});

impl_stable_hash_for!(struct ty::Instance<'tcx> {
    def,
    substs
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for ty::InstanceDef<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            ty::InstanceDef::Item(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::Intrinsic(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::FnPtrShim(def_id, ty) => {
                def_id.hash_stable(hcx, hasher);
                ty.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::Virtual(def_id, n) => {
                def_id.hash_stable(hcx, hasher);
                n.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::ClosureOnceShim { call_once } => {
                call_once.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::DropGlue(def_id, ty) => {
                def_id.hash_stable(hcx, hasher);
                ty.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::CloneShim(def_id, ty) => {
                def_id.hash_stable(hcx, hasher);
                ty.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::TraitDef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::TraitDef {
            // We already have the def_path_hash below, no need to hash it twice
            def_id: _,
            unsafety,
            paren_sugar,
            has_auto_impl,
            def_path_hash,
        } = *self;

        unsafety.hash_stable(hcx, hasher);
        paren_sugar.hash_stable(hcx, hasher);
        has_auto_impl.hash_stable(hcx, hasher);
        def_path_hash.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::Destructor {
    did
});

impl<'a> HashStable<StableHashingContext<'a>> for ty::CrateVariancesMap {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::CrateVariancesMap {
            ref variances,
            // This is just an irrelevant helper value.
            empty_variance: _,
        } = *self;

        variances.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for ty::CratePredicatesMap<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::CratePredicatesMap {
            ref predicates,
            // This is just an irrelevant helper value.
            empty_predicate: _,
        } = *self;

        predicates.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::AssociatedItem {
    def_id,
    name,
    kind,
    vis,
    defaultness,
    container,
    method_has_self_argument
});

impl_stable_hash_for!(enum ty::AssociatedKind {
    Const,
    Method,
    Type
});

impl_stable_hash_for!(enum ty::AssociatedItemContainer {
    TraitContainer(def_id),
    ImplContainer(def_id)
});


impl<'a, 'gcx, T> HashStable<StableHashingContext<'a>>
for ty::steal::Steal<T>
    where T: HashStable<StableHashingContext<'a>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.borrow().hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::ParamEnv<'tcx> {
    caller_bounds,
    reveal
});

impl_stable_hash_for!(enum traits::Reveal {
    UserFacing,
    All
});

impl_stable_hash_for!(enum ::middle::privacy::AccessLevel {
    Reachable,
    Exported,
    Public
});

impl<'a> HashStable<StableHashingContext<'a>>
for ::middle::privacy::AccessLevels {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let ::middle::privacy::AccessLevels {
                ref map
            } = *self;

            map.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(struct ty::CrateInherentImpls {
    inherent_impls
});

impl_stable_hash_for!(enum ::session::CompileIncomplete {
    Stopped,
    Errored(error_reported)
});

impl_stable_hash_for!(struct ::util::common::ErrorReported {});

impl_stable_hash_for!(tuple_struct ::middle::reachable::ReachableSet {
    reachable_set
});

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::Vtable<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::Vtable::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match self {
            &VtableImpl(ref table_impl) => table_impl.hash_stable(hcx, hasher),
            &VtableAutoImpl(ref table_def_impl) => table_def_impl.hash_stable(hcx, hasher),
            &VtableParam(ref table_param) => table_param.hash_stable(hcx, hasher),
            &VtableObject(ref table_obj) => table_obj.hash_stable(hcx, hasher),
            &VtableBuiltin(ref table_builtin) => table_builtin.hash_stable(hcx, hasher),
            &VtableClosure(ref table_closure) => table_closure.hash_stable(hcx, hasher),
            &VtableFnPointer(ref table_fn_pointer) => table_fn_pointer.hash_stable(hcx, hasher),
            &VtableGenerator(ref table_generator) => table_generator.hash_stable(hcx, hasher),
        }
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableImplData<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableImplData {
            impl_def_id,
            substs,
            ref nested,
        } = *self;
        impl_def_id.hash_stable(hcx, hasher);
        substs.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableAutoImplData<N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableAutoImplData {
            trait_def_id,
            ref nested,
        } = *self;
        trait_def_id.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableObjectData<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableObjectData {
            upcast_trait_ref,
            vtable_base,
            ref nested,
        } = *self;
        upcast_trait_ref.hash_stable(hcx, hasher);
        vtable_base.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableBuiltinData<N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableBuiltinData {
            ref nested,
        } = *self;
        nested.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableClosureData<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableClosureData {
            closure_def_id,
            substs,
            ref nested,
        } = *self;
        closure_def_id.hash_stable(hcx, hasher);
        substs.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableFnPointerData<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableFnPointerData {
            fn_ty,
            ref nested,
        } = *self;
        fn_ty.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableGeneratorData<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableGeneratorData {
            generator_def_id,
            substs,
            ref nested,
        } = *self;
        generator_def_id.hash_stable(hcx, hasher);
        substs.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(
    impl<'tcx, V> for struct infer::canonical::Canonical<'tcx, V> {
        variables, value
    }
);

impl_stable_hash_for!(
    impl<'tcx> for struct infer::canonical::CanonicalVarValues<'tcx> {
        var_values
    }
);

impl_stable_hash_for!(struct infer::canonical::CanonicalVarInfo {
    kind
});

impl_stable_hash_for!(enum infer::canonical::CanonicalVarKind {
    Ty(k),
    Region
});

impl_stable_hash_for!(enum infer::canonical::CanonicalTyVarKind {
    General,
    Int,
    Float
});

impl_stable_hash_for!(
    impl<'tcx, R> for struct infer::canonical::QueryResult<'tcx, R> {
        var_values, region_constraints, certainty, value
    }
);

impl_stable_hash_for!(enum infer::canonical::Certainty {
    Proven, Ambiguous
});

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::WhereClause<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::WhereClause::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Implemented(trait_ref) => trait_ref.hash_stable(hcx, hasher),
            ProjectionEq(projection) => projection.hash_stable(hcx, hasher),
            TypeOutlives(ty_outlives) => ty_outlives.hash_stable(hcx, hasher),
            RegionOutlives(region_outlives) => region_outlives.hash_stable(hcx, hasher),
        }
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::WellFormed<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::WellFormed::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Trait(trait_ref) => trait_ref.hash_stable(hcx, hasher),
            Ty(ty) => ty.hash_stable(hcx, hasher),
        }
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::FromEnv<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::FromEnv::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Trait(trait_ref) => trait_ref.hash_stable(hcx, hasher),
            Ty(ty) => ty.hash_stable(hcx, hasher),
        }
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::DomainGoal<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::DomainGoal::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Holds(wc) => wc.hash_stable(hcx, hasher),
            WellFormed(wf) => wf.hash_stable(hcx, hasher),
            FromEnv(from_env) => from_env.hash_stable(hcx, hasher),
            Normalize(projection) => projection.hash_stable(hcx, hasher),
        }
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::Goal<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::Goal::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Implies(hypotheses, goal) => {
                hypotheses.hash_stable(hcx, hasher);
                goal.hash_stable(hcx, hasher);
            },
            And(goal1, goal2) => {
                goal1.hash_stable(hcx, hasher);
                goal2.hash_stable(hcx, hasher);
            }
            Not(goal) => goal.hash_stable(hcx, hasher),
            DomainGoal(domain_goal) => domain_goal.hash_stable(hcx, hasher),
            Quantified(quantifier, goal) => {
                quantifier.hash_stable(hcx, hasher);
                goal.hash_stable(hcx, hasher);
            },
            CannotProve => { },
        }
    }
}

impl_stable_hash_for!(
    impl<'tcx> for struct traits::ProgramClause<'tcx> {
        goal, hypotheses
    }
);

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::Clause<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use traits::Clause::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Implies(clause) => clause.hash_stable(hcx, hasher),
            ForAll(clause) => clause.hash_stable(hcx, hasher),
        }
    }
}

impl_stable_hash_for!(enum traits::QuantifierKind {
    Universal,
    Existential
});
