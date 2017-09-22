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

use ich::{StableHashingContext, NodeIdHashingMode};
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use std::hash as std_hash;
use std::mem;
use middle::region;
use traits;
use ty;

impl<'gcx, T> HashStable<StableHashingContext<'gcx>>
for &'gcx ty::Slice<T>
    where T: HashStable<StableHashingContext<'gcx>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        (&self[..]).hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::subst::Kind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        self.as_type().hash_stable(hcx, hasher);
        self.as_region().hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::RegionKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::ReErased |
            ty::ReStatic |
            ty::ReEmpty => {
                // No variant fields to hash for these ...
            }
            ty::ReLateBound(db, ty::BrAnon(i)) => {
                db.depth.hash_stable(hcx, hasher);
                i.hash_stable(hcx, hasher);
            }
            ty::ReLateBound(db, ty::BrNamed(def_id, name)) => {
                db.depth.hash_stable(hcx, hasher);
                def_id.hash_stable(hcx, hasher);
                name.hash_stable(hcx, hasher);
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
            ty::ReLateBound(..) |
            ty::ReVar(..) |
            ty::ReSkolemized(..) => {
                bug!("TypeIdHasher: unexpected region {:?}", *self)
            }
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::adjustment::AutoBorrow<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::adjustment::Adjust<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl_stable_hash_for!(struct ty::UpvarId { var_id, closure_expr_id });

impl_stable_hash_for!(enum ty::BorrowKind {
    ImmBorrow,
    UniqueImmBorrow,
    MutBorrow
});

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::UpvarCapture<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl<'gcx, T> HashStable<StableHashingContext<'gcx>> for ty::Binder<T>
    where T: HashStable<StableHashingContext<'gcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::Binder(ref inner) = *self;
        inner.hash_stable(hcx, hasher);
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
impl_stable_hash_for!(tuple_struct ty::EquatePredicate<'tcx> { t1, t2 });
impl_stable_hash_for!(struct ty::SubtypePredicate<'tcx> { a_is_expected, a, b });

impl<'gcx, A, B> HashStable<StableHashingContext<'gcx>>
for ty::OutlivesPredicate<A, B>
    where A: HashStable<StableHashingContext<'gcx>>,
          B: HashStable<StableHashingContext<'gcx>>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::OutlivesPredicate(ref a, ref b) = *self;
        a.hash_stable(hcx, hasher);
        b.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::ProjectionPredicate<'tcx> { projection_ty, ty });
impl_stable_hash_for!(struct ty::ProjectionTy<'tcx> { substs, item_def_id });


impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::Predicate<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ty::Predicate::Trait(ref pred) => {
                pred.hash_stable(hcx, hasher);
            }
            ty::Predicate::Equate(ref pred) => {
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
            ty::Predicate::ClosureKind(def_id, closure_kind) => {
                def_id.hash_stable(hcx, hasher);
                closure_kind.hash_stable(hcx, hasher);
            }
            ty::Predicate::ConstEvaluatable(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::AdtFlags {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'gcx>,
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

impl_stable_hash_for!(struct ty::FieldDef {
    did,
    name,
    vis
});

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ::middle::const_val::ConstVal<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use middle::const_val::ConstVal::*;
        use middle::const_val::ConstAggregate::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Integral(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            Float(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            Str(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ByteStr(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            Bool(value) => {
                value.hash_stable(hcx, hasher);
            }
            Char(value) => {
                value.hash_stable(hcx, hasher);
            }
            Variant(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            Function(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                    substs.hash_stable(hcx, hasher);
                });
            }
            Aggregate(Struct(ref name_values)) => {
                let mut values = name_values.to_vec();
                values.sort_unstable_by_key(|&(ref name, _)| name.clone());
                values.hash_stable(hcx, hasher);
            }
            Aggregate(Tuple(ref value)) => {
                value.hash_stable(hcx, hasher);
            }
            Aggregate(Array(ref value)) => {
                value.hash_stable(hcx, hasher);
            }
            Aggregate(Repeat(ref value, times)) => {
                value.hash_stable(hcx, hasher);
                times.hash_stable(hcx, hasher);
            }
            Unevaluated(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ::middle::const_val::ByteArray<'tcx> {
    data
});

impl_stable_hash_for!(struct ty::Const<'tcx> {
    ty,
    val
});

impl_stable_hash_for!(struct ::middle::const_val::ConstEvalErr<'tcx> {
    span,
    kind
});

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ::middle::const_val::ErrKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        use middle::const_val::ErrKind::*;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            CannotCast |
            MissingStructField |
            NonConstPath |
            ExpectedConstTuple |
            ExpectedConstStruct |
            IndexedNonVec |
            IndexNotUsize |
            MiscBinaryOp |
            MiscCatchAll |
            IndexOpFeatureGated |
            TypeckError => {
                // nothing to do
            }
            UnimplementedConstVal(s) => {
                s.hash_stable(hcx, hasher);
            }
            IndexOutOfBounds { len, index } => {
                len.hash_stable(hcx, hasher);
                index.hash_stable(hcx, hasher);
            }
            Math(ref const_math_err) => {
                const_math_err.hash_stable(hcx, hasher);
            }
            LayoutError(ref layout_error) => {
                layout_error.hash_stable(hcx, hasher);
            }
            ErroneousReferencedConstant(ref const_val) => {
                const_val.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ty::ClosureSubsts<'tcx> { substs });

impl_stable_hash_for!(struct ty::GeneratorInterior<'tcx> { witness });

impl_stable_hash_for!(struct ty::GenericPredicates<'tcx> {
    parent,
    predicates
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

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::Generics {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::Generics {
            parent,
            parent_regions,
            parent_types,
            ref regions,
            ref types,

            // Reverse map to each `TypeParameterDef`'s `index` field, from
            // `def_id.index` (`def_id.krate` is the same as the item's).
            type_param_to_index: _, // Don't hash this
            has_self,
            has_late_bound_regions,
        } = *self;

        parent.hash_stable(hcx, hasher);
        parent_regions.hash_stable(hcx, hasher);
        parent_types.hash_stable(hcx, hasher);
        regions.hash_stable(hcx, hasher);
        types.hash_stable(hcx, hasher);
        has_self.hash_stable(hcx, hasher);
        has_late_bound_regions.hash_stable(hcx, hasher);
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::RegionParameterDef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::RegionParameterDef {
            name,
            def_id,
            index,
            pure_wrt_drop
        } = *self;

        name.hash_stable(hcx, hasher);
        def_id.hash_stable(hcx, hasher);
        index.hash_stable(hcx, hasher);
        pure_wrt_drop.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::TypeParameterDef {
    name,
    def_id,
    index,
    has_default,
    object_lifetime_default,
    pure_wrt_drop
});

impl<'gcx, T> HashStable<StableHashingContext<'gcx>>
for ::middle::resolve_lifetime::Set1<T>
    where T: HashStable<StableHashingContext<'gcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl_stable_hash_for!(enum ::middle::resolve_lifetime::Region {
    Static,
    EarlyBound(index, decl),
    LateBound(db_index, decl),
    LateBoundAnon(db_index, anon_index),
    Free(call_site_scope_data, decl)
});

impl_stable_hash_for!(struct ty::DebruijnIndex {
    depth
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

impl_stable_hash_for!(enum ::middle::region::Scope {
    Node(local_id),
    Destruction(local_id),
    CallSite(local_id),
    Arguments(local_id),
    Remainder(block_remainder)
});

impl<'gcx> ToStableHashKey<StableHashingContext<'gcx>> for region::Scope {
    type KeyType = region::Scope;

    #[inline]
    fn to_stable_hash_key(&self, _: &StableHashingContext<'gcx>) -> region::Scope {
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

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::TypeVariants<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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
            TyRef(region, pointee_ty) => {
                region.hash_stable(hcx, hasher);
                pointee_ty.hash_stable(hcx, hasher);
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
            TyGenerator(def_id, closure_substs, interior)
             => {
                def_id.hash_stable(hcx, hasher);
                closure_substs.hash_stable(hcx, hasher);
                interior.hash_stable(hcx, hasher);
            }
            TyTuple(inner_tys, from_diverging_type_var) => {
                inner_tys.hash_stable(hcx, hasher);
                from_diverging_type_var.hash_stable(hcx, hasher);
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
            TyInfer(..) => {
                bug!("ty::TypeVariants::hash_stable() - Unexpected variant {:?}.", *self)
            }
        }
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

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ty::ExistentialPredicate<'gcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::InstanceDef<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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
            ty::InstanceDef::DropGlue(def_id, t) => {
                def_id.hash_stable(hcx, hasher);
                t.hash_stable(hcx, hasher);
            }
            ty::InstanceDef::CloneShim(def_id, t) => {
                def_id.hash_stable(hcx, hasher);
                t.hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::TraitDef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::TraitDef {
            // We already have the def_path_hash below, no need to hash it twice
            def_id: _,
            unsafety,
            paren_sugar,
            has_default_impl,
            def_path_hash,
        } = *self;

        unsafety.hash_stable(hcx, hasher);
        paren_sugar.hash_stable(hcx, hasher);
        has_default_impl.hash_stable(hcx, hasher);
        def_path_hash.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::Destructor {
    did
});

impl_stable_hash_for!(struct ty::DtorckConstraint<'tcx> {
    outlives,
    dtorck_types
});


impl<'gcx> HashStable<StableHashingContext<'gcx>> for ty::CrateVariancesMap {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::CrateVariancesMap {
            ref dependencies,
            ref variances,
            // This is just an irrelevant helper value.
            empty_variance: _,
        } = *self;

        dependencies.hash_stable(hcx, hasher);
        variances.hash_stable(hcx, hasher);
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


impl<'gcx, T> HashStable<StableHashingContext<'gcx>>
for ty::steal::Steal<T>
    where T: HashStable<StableHashingContext<'gcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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

impl<'gcx> HashStable<StableHashingContext<'gcx>>
for ::middle::privacy::AccessLevels {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
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
