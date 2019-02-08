//! This module contains `HashStable` implementations for various data types
//! from rustc::ty in no particular order.

use crate::ich::{Fingerprint, StableHashingContext, NodeIdHashingMode};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use std::cell::RefCell;
use std::hash as std_hash;
use std::mem;
use crate::middle::region;
use crate::infer;
use crate::traits;
use crate::ty;
use crate::mir;

impl<'a, 'gcx, T> HashStable<StableHashingContext<'a>>
for &'gcx ty::List<T>
    where T: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<(usize, usize), Fingerprint>> =
                RefCell::new(Default::default());
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

impl<'a, 'gcx, T> ToStableHashKey<StableHashingContext<'a>> for &'gcx ty::List<T>
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
            ty::RePlaceholder(..) => {
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
        self.index().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::BoundVar {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
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

impl_stable_hash_for!(tuple_struct ty::util::NeedsDrop { value });

impl_stable_hash_for!(tuple_struct ty::AdtSizedConstraint<'tcx> { list });

impl_stable_hash_for!(struct ty::UpvarPath { hir_id });

impl_stable_hash_for!(struct ty::UpvarId { var_path, closure_expr_id });

impl_stable_hash_for!(enum ty::BorrowKind {
    ImmBorrow,
    UniqueImmBorrow,
    MutBorrow
});

impl_stable_hash_for!(impl<'gcx> for enum ty::UpvarCapture<'gcx> [ ty::UpvarCapture ] {
    ByValue,
    ByRef(up_var_borrow),
});

impl_stable_hash_for!(struct ty::GenSig<'tcx> {
    yield_ty,
    return_ty
});

impl_stable_hash_for!(struct ty::FnSig<'tcx> {
    inputs_and_output,
    c_variadic,
    unsafety,
    abi
});

impl_stable_hash_for!(struct ty::ResolvedOpaqueTy<'tcx> {
    concrete_type,
    substs
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
impl_stable_hash_for!(impl<A, B> for tuple_struct ty::OutlivesPredicate<A, B> { a, b });
impl_stable_hash_for!(struct ty::ProjectionPredicate<'tcx> { projection_ty, ty });
impl_stable_hash_for!(struct ty::ProjectionTy<'tcx> { substs, item_def_id });

impl_stable_hash_for!(
    impl<'tcx> for enum ty::Predicate<'tcx> [ ty::Predicate ] {
        Trait(pred),
        Subtype(pred),
        RegionOutlives(pred),
        TypeOutlives(pred),
        Projection(pred),
        WellFormed(ty),
        ObjectSafe(def_id),
        ClosureKind(def_id, closure_substs, closure_kind),
        ConstEvaluatable(def_id, substs),
    }
);

impl<'a> HashStable<StableHashingContext<'a>> for ty::AdtFlags {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        std_hash::Hash::hash(self, hasher);
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::VariantFlags {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        std_hash::Hash::hash(self, hasher);
    }
}

impl_stable_hash_for!(enum ty::VariantDiscr {
    Explicit(def_id),
    Relative(distance)
});

impl_stable_hash_for!(struct ty::FieldDef {
    did,
    ident -> (ident.name),
    vis,
});

impl_stable_hash_for!(
    impl<'tcx> for enum mir::interpret::ConstValue<'tcx> [ mir::interpret::ConstValue ] {
        Scalar(val),
        Slice(a, b),
        ByRef(ptr, alloc),
    }
);
impl_stable_hash_for!(struct crate::mir::interpret::RawConst<'tcx> {
    alloc_id,
    ty,
});

impl_stable_hash_for! {
    impl<Tag> for struct mir::interpret::Pointer<Tag> {
        alloc_id,
        offset,
        tag,
    }
}

impl_stable_hash_for!(
    impl<Tag> for enum mir::interpret::Scalar<Tag> [ mir::interpret::Scalar ] {
        Bits { bits, size },
        Ptr(ptr),
    }
);

impl_stable_hash_for!(
    impl<'tcx> for enum mir::interpret::AllocKind<'tcx> [ mir::interpret::AllocKind ] {
        Function(instance),
        Static(def_id),
        Memory(mem),
    }
);

// AllocIds get resolved to whatever they point to (to be stable)
impl<'a> HashStable<StableHashingContext<'a>> for mir::interpret::AllocId {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'a>,
        hasher: &mut StableHasher<W>,
    ) {
        ty::tls::with_opt(|tcx| {
            trace!("hashing {:?}", *self);
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            let alloc_kind = tcx.alloc_map.lock().get(*self);
            alloc_kind.hash_stable(hcx, hasher);
        });
    }
}

// Allocations treat their relocations specially
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
        self.mutability.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(enum ::syntax::ast::Mutability {
    Immutable,
    Mutable
});

impl_stable_hash_for!(struct ty::Const<'tcx> {
    ty,
    val
});

impl_stable_hash_for!(impl<'tcx> for enum ty::LazyConst<'tcx> [ty::LazyConst] {
    Unevaluated(did, substs),
    Evaluated(c)
});

impl_stable_hash_for!(enum mir::interpret::ErrorHandled {
    Reported,
    TooGeneric
});

impl_stable_hash_for!(struct mir::interpret::FrameInfo<'tcx> {
    call_site,
    lint_root,
    instance
});

impl_stable_hash_for!(struct ty::ClosureSubsts<'tcx> { substs });
impl_stable_hash_for!(struct ty::GeneratorSubsts<'tcx> { substs });

impl_stable_hash_for!(struct ty::GenericPredicates<'tcx> {
    parent,
    predicates
});

impl_stable_hash_for!(
    impl<'tcx, O> for enum mir::interpret::EvalErrorKind<'tcx, O>
        [ mir::interpret::EvalErrorKind ]
    {
        FunctionArgCountMismatch,
        DanglingPointerDeref,
        DoubleFree,
        InvalidMemoryAccess,
        InvalidFunctionPointer,
        InvalidBool,
        InvalidNullPointerUsage,
        ReadPointerAsBytes,
        ReadBytesAsPointer,
        ReadForeignStatic,
        InvalidPointerMath,
        DeadLocal,
        StackFrameLimitReached,
        OutOfTls,
        TlsOutOfBounds,
        CalledClosureAsFunction,
        VtableForArgumentlessMethod,
        ModifiedConstantMemory,
        ModifiedStatic,
        AssumptionNotHeld,
        InlineAsm,
        ReallocateNonBasePtr,
        DeallocateNonBasePtr,
        HeapAllocZeroBytes,
        Unreachable,
        ReadFromReturnPointer,
        UnimplementedTraitSelection,
        TypeckError,
        TooGeneric,
        DerefFunctionPointer,
        ExecuteMemory,
        OverflowNeg,
        RemainderByZero,
        DivisionByZero,
        GeneratorResumedAfterReturn,
        GeneratorResumedAfterPanic,
        ReferencedConstant,
        InfiniteLoop,
        ReadUndefBytes(offset),
        InvalidDiscriminant(val),
        Panic { msg, file, line, col },
        MachineError(err),
        FunctionAbiMismatch(a, b),
        FunctionArgMismatch(a, b),
        FunctionRetMismatch(a, b),
        NoMirFor(s),
        UnterminatedCString(ptr),
        PointerOutOfBounds { ptr, check, allocation_size },
        InvalidBoolOp(bop),
        Unimplemented(s),
        BoundsCheck { len, index },
        Intrinsic(s),
        InvalidChar(c),
        AbiViolation(s),
        AlignmentCheckFailed { required, has },
        ValidationFailure(s),
        TypeNotPrimitive(ty),
        ReallocatedWrongMemoryKind(a, b),
        DeallocatedWrongMemoryKind(a, b),
        IncorrectAllocationInformation(a, b, c, d),
        Layout(lay),
        HeapAllocNonPowerOfTwoAlignment(n),
        PathNotFound(v),
        Overflow(op),
    }
);

impl_stable_hash_for!(enum mir::interpret::InboundsCheck {
    Live,
    MaybeDead
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

impl_stable_hash_for!(struct ty::Generics {
    parent,
    parent_count,
    params,
    // Reverse map to each param's `index` field, from its `def_id`.
    param_def_id_to_index -> _, // Don't hash this
    has_self,
    has_late_bound_regions,
});

impl_stable_hash_for!(struct ty::GenericParamDef {
    name,
    def_id,
    index,
    pure_wrt_drop,
    kind
});

impl_stable_hash_for!(enum ty::GenericParamDefKind {
    Lifetime,
    Type { has_default, object_lifetime_default, synthetic },
});

impl_stable_hash_for!(
    impl<T> for enum crate::middle::resolve_lifetime::Set1<T>
        [ crate::middle::resolve_lifetime::Set1 ]
    {
        Empty,
        Many,
        One(value),
    }
);

impl_stable_hash_for!(enum crate::middle::resolve_lifetime::LifetimeDefOrigin {
    ExplicitOrElided,
    InBand,
    Error,
});

impl_stable_hash_for!(enum crate::middle::resolve_lifetime::Region {
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

impl_stable_hash_for!(struct crate::middle::region::Scope { id, data });

impl_stable_hash_for!(enum crate::middle::region::ScopeData {
    Node,
    CallSite,
    Arguments,
    Destruction,
    Remainder(first_statement_index)
});

impl<'a> ToStableHashKey<StableHashingContext<'a>> for region::Scope {
    type KeyType = region::Scope;

    #[inline]
    fn to_stable_hash_key(&self, _: &StableHashingContext<'a>) -> region::Scope {
        *self
    }
}

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
for ty::TyKind<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use crate::ty::TyKind::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            Bool  |
            Char  |
            Str   |
            Error |
            Never => {
                // Nothing more to hash.
            }
            Int(int_ty) => {
                int_ty.hash_stable(hcx, hasher);
            }
            Uint(uint_ty) => {
                uint_ty.hash_stable(hcx, hasher);
            }
            Float(float_ty)  => {
                float_ty.hash_stable(hcx, hasher);
            }
            Adt(adt_def, substs) => {
                adt_def.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            Array(inner_ty, len) => {
                inner_ty.hash_stable(hcx, hasher);
                len.hash_stable(hcx, hasher);
            }
            Slice(inner_ty) => {
                inner_ty.hash_stable(hcx, hasher);
            }
            RawPtr(pointee_ty) => {
                pointee_ty.hash_stable(hcx, hasher);
            }
            Ref(region, pointee_ty, mutbl) => {
                region.hash_stable(hcx, hasher);
                pointee_ty.hash_stable(hcx, hasher);
                mutbl.hash_stable(hcx, hasher);
            }
            FnDef(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            FnPtr(ref sig) => {
                sig.hash_stable(hcx, hasher);
            }
            Dynamic(ref existential_predicates, region) => {
                existential_predicates.hash_stable(hcx, hasher);
                region.hash_stable(hcx, hasher);
            }
            Closure(def_id, closure_substs) => {
                def_id.hash_stable(hcx, hasher);
                closure_substs.hash_stable(hcx, hasher);
            }
            Generator(def_id, generator_substs, movability) => {
                def_id.hash_stable(hcx, hasher);
                generator_substs.hash_stable(hcx, hasher);
                movability.hash_stable(hcx, hasher);
            }
            GeneratorWitness(types) => {
                types.hash_stable(hcx, hasher)
            }
            Tuple(inner_tys) => {
                inner_tys.hash_stable(hcx, hasher);
            }
            Projection(ref data) | UnnormalizedProjection(ref data) => {
                data.hash_stable(hcx, hasher);
            }
            Opaque(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            Param(param_ty) => {
                param_ty.hash_stable(hcx, hasher);
            }
            Bound(debruijn, bound_ty) => {
                debruijn.hash_stable(hcx, hasher);
                bound_ty.hash_stable(hcx, hasher);
            }
            ty::Placeholder(placeholder_ty) => {
                placeholder_ty.hash_stable(hcx, hasher);
            }
            Foreign(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            Infer(infer_ty) => {
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
});

impl<'a, 'gcx> HashStable<StableHashingContext<'a>>
for ty::TyVid
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          _hcx: &mut StableHashingContext<'a>,
                                          _hasher: &mut StableHasher<W>) {
        // TyVid values are confined to an inference context and hence
        // should not be hashed.
        bug!("ty::TyKind::hash_stable() - can't hash a TyVid {:?}.", *self)
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
        bug!("ty::TyKind::hash_stable() - can't hash an IntVid {:?}.", *self)
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
        bug!("ty::TyKind::hash_stable() - can't hash a FloatVid {:?}.", *self)
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
            ty::InstanceDef::VtableShim(def_id) => {
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

impl_stable_hash_for!(struct ty::TraitDef {
    // We already have the def_path_hash below, no need to hash it twice
    def_id -> _,
    unsafety,
    paren_sugar,
    has_auto_impl,
    is_marker,
    def_path_hash,
});

impl_stable_hash_for!(struct ty::Destructor {
    did
});

impl_stable_hash_for!(struct ty::CrateVariancesMap {
    variances,
    // This is just an irrelevant helper value.
    empty_variance -> _,
});

impl_stable_hash_for!(struct ty::CratePredicatesMap<'tcx> {
    predicates,
    // This is just an irrelevant helper value.
    empty_predicate -> _,
});

impl_stable_hash_for!(struct ty::AssociatedItem {
    def_id,
    ident -> (ident.name),
    kind,
    vis,
    defaultness,
    container,
    method_has_self_argument
});

impl_stable_hash_for!(enum ty::AssociatedKind {
    Const,
    Method,
    Existential,
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
    reveal,
    def_id
});

impl_stable_hash_for!(enum traits::Reveal {
    UserFacing,
    All
});

impl_stable_hash_for!(enum crate::middle::privacy::AccessLevel {
    ReachableFromImplTrait,
    Reachable,
    Exported,
    Public
});

impl<'a> HashStable<StableHashingContext<'a>>
for crate::middle::privacy::AccessLevels {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            let crate::middle::privacy::AccessLevels {
                ref map
            } = *self;

            map.hash_stable(hcx, hasher);
        });
    }
}

impl_stable_hash_for!(struct ty::CrateInherentImpls {
    inherent_impls
});

impl_stable_hash_for!(enum crate::session::CompileIncomplete {
    Stopped,
    Errored(error_reported)
});

impl_stable_hash_for!(struct crate::util::common::ErrorReported {});

impl_stable_hash_for!(tuple_struct crate::middle::reachable::ReachableSet {
    reachable_set
});

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::Vtable<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use crate::traits::Vtable::*;

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
            &VtableTraitAlias(ref table_alias) => table_alias.hash_stable(hcx, hasher),
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

impl<'a, 'gcx, N> HashStable<StableHashingContext<'a>>
for traits::VtableTraitAliasData<'gcx, N> where N: HashStable<StableHashingContext<'a>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let traits::VtableTraitAliasData {
            alias_def_id,
            substs,
            ref nested,
        } = *self;
        alias_def_id.hash_stable(hcx, hasher);
        substs.hash_stable(hcx, hasher);
        nested.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(
    impl<'tcx, V> for struct infer::canonical::Canonical<'tcx, V> {
        max_universe, variables, value
    }
);

impl_stable_hash_for!(
    struct infer::canonical::CanonicalVarValues<'tcx> {
        var_values
    }
);

impl_stable_hash_for!(struct infer::canonical::CanonicalVarInfo {
    kind
});

impl_stable_hash_for!(enum infer::canonical::CanonicalVarKind {
    Ty(k),
    PlaceholderTy(placeholder),
    Region(ui),
    PlaceholderRegion(placeholder),
});

impl_stable_hash_for!(enum infer::canonical::CanonicalTyVarKind {
    General(ui),
    Int,
    Float
});

impl_stable_hash_for!(
    impl<'tcx, R> for struct infer::canonical::QueryResponse<'tcx, R> {
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
        use crate::traits::WhereClause::*;

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
        use crate::traits::WellFormed::*;

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
        use crate::traits::FromEnv::*;

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
        use crate::traits::DomainGoal::*;

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
        use crate::traits::GoalKind::*;

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
            Subtype(a, b) => {
                a.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            CannotProve => { },
        }
    }
}

impl_stable_hash_for!(
    struct traits::ProgramClause<'tcx> {
        goal, hypotheses, category
    }
);

impl_stable_hash_for!(enum traits::ProgramClauseCategory {
    ImpliedBound,
    WellFormed,
    Other,
});

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for traits::Clause<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        use crate::traits::Clause::*;

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

impl_stable_hash_for!(struct ty::subst::UserSubsts<'tcx> { substs, user_self_ty });

impl_stable_hash_for!(struct ty::subst::UserSelfTy<'tcx> { impl_def_id, self_ty });

impl_stable_hash_for!(
    struct traits::Environment<'tcx> {
        clauses,
    }
);

impl_stable_hash_for!(
    impl<'tcx, G> for struct traits::InEnvironment<'tcx, G> {
        environment,
        goal,
    }
);

impl_stable_hash_for!(
    struct ty::CanonicalUserTypeAnnotation<'tcx> {
        user_ty, span, inferred_ty
    }
);

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for ty::UserType<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::UserType::Ty(ref ty) => {
                ty.hash_stable(hcx, hasher);
            }
            ty::UserType::TypeOf(ref def_id, ref substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ty::UserTypeAnnotationIndex {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        self.index().hash_stable(hcx, hasher);
    }
}
