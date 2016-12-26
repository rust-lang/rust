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

use ich::{self, StableHashingContext, NodeIdHashingMode};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};
use std::hash as std_hash;
use std::mem;
use syntax_pos::symbol::InternedString;
use ty;

impl<'a, 'gcx, 'tcx, T> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for &'tcx ty::Slice<T>
    where T: HashStable<StableHashingContext<'a, 'gcx, 'tcx>> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        (&self[..]).hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::subst::Kind<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        self.as_type().hash_stable(hcx, hasher);
        self.as_region().hash_stable(hcx, hasher);
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::RegionKind {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
            ty::ReEarlyBound(ty::EarlyBoundRegion { def_id, index, name }) => {
                def_id.hash_stable(hcx, hasher);
                index.hash_stable(hcx, hasher);
                name.hash_stable(hcx, hasher);
            }
            ty::ReScope(code_extent) => {
                code_extent.hash_stable(hcx, hasher);
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::adjustment::AutoBorrow<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::adjustment::Adjust<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
impl_stable_hash_for!(struct ty::UpvarId { var_id, closure_expr_id });
impl_stable_hash_for!(struct ty::UpvarBorrow<'tcx> { kind, region });

impl_stable_hash_for!(enum ty::BorrowKind {
    ImmBorrow,
    UniqueImmBorrow,
    MutBorrow
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::UpvarCapture<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
    impl_arg_ty,
    suspend_ty,
    return_ty
});

impl_stable_hash_for!(struct ty::FnSig<'tcx> {
    inputs_and_output,
    variadic,
    unsafety,
    abi
});

impl<'a, 'gcx, 'tcx, T> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ty::Binder<T>
    where T: HashStable<StableHashingContext<'a, 'gcx, 'tcx>> + ty::fold::TypeFoldable<'tcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        hcx.tcx().anonymize_late_bound_regions(self).0.hash_stable(hcx, hasher);
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

impl<'a, 'gcx, 'tcx, A, B> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::OutlivesPredicate<A, B>
    where A: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>,
          B: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::OutlivesPredicate(ref a, ref b) = *self;
        a.hash_stable(hcx, hasher);
        b.hash_stable(hcx, hasher);
    }
}

impl_stable_hash_for!(struct ty::ProjectionPredicate<'tcx> { projection_ty, ty });
impl_stable_hash_for!(struct ty::ProjectionTy<'tcx> { substs, item_def_id });


impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ty::Predicate<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
        }
    }
}

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ty::AdtFlags {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ::middle::const_val::ConstVal<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        use middle::const_val::ConstVal;

        mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            ConstVal::Float(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Integral(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Str(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::ByteStr(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Bool(value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Char(value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Variant(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            ConstVal::Function(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            ConstVal::Struct(ref name_value_map) => {
                let mut values: Vec<(InternedString, &ConstVal)> =
                    name_value_map.iter()
                                  .map(|(name, val)| (name.as_str(), val))
                                  .collect();

                values.sort_unstable_by_key(|&(ref name, _)| name.clone());
                values.hash_stable(hcx, hasher);
            }
            ConstVal::Tuple(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Array(ref value) => {
                value.hash_stable(hcx, hasher);
            }
            ConstVal::Repeat(ref value, times) => {
                value.hash_stable(hcx, hasher);
                times.hash_stable(hcx, hasher);
            }
        }
    }
}

impl_stable_hash_for!(struct ty::ClosureSubsts<'tcx> { substs });

impl_stable_hash_for!(tuple_struct ty::GeneratorInterior<'tcx> { ty });

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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ty::Generics {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::RegionParameterDef {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::RegionParameterDef {
            name,
            def_id,
            index,
            issue_32330: _,
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


impl<'a, 'gcx, 'tcx, T> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ::middle::resolve_lifetime::Set1<T>
    where T: HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ::middle::region::CodeExtent
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        use middle::region::CodeExtent;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            CodeExtent::Misc(node_id) |
            CodeExtent::DestructionScope(node_id) => {
                node_id.hash_stable(hcx, hasher);
            }
            CodeExtent::CallSiteScope(body_id) |
            CodeExtent::ParameterScope(body_id) => {
                body_id.hash_stable(hcx, hasher);
            }
            CodeExtent::Remainder(block_remainder) => {
                block_remainder.hash_stable(hcx, hasher);
            }
        }
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::TypeVariants<'tcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        use ty::TypeVariants::*;

        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            TyBool  |
            TyChar  |
            TyStr   |
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

            TyError     |
            TyInfer(..) => {
                bug!("ty::TypeVariants::hash_stable() - Unexpected variant.")
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

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::ExistentialPredicate<'tcx>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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


impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>>
for ty::TypeckTables<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
                                          hasher: &mut StableHasher<W>) {
        let ty::TypeckTables {
            ref type_dependent_defs,
            ref node_types,
            ref node_substs,
            ref adjustments,
            ref upvar_capture_map,
            ref closure_tys,
            ref closure_kinds,
            ref generator_interiors,
            ref generator_sigs,
            ref liberated_fn_sigs,
            ref fru_field_types,

            ref cast_kinds,

            // FIXME(#41184): This is still ignored at the moment.
            lints: _,
            ref used_trait_imports,
            tainted_by_errors,
            ref free_region_map,
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            ich::hash_stable_nodemap(hcx, hasher, type_dependent_defs);
            ich::hash_stable_nodemap(hcx, hasher, node_types);
            ich::hash_stable_nodemap(hcx, hasher, node_substs);
            ich::hash_stable_nodemap(hcx, hasher, adjustments);
            ich::hash_stable_hashmap(hcx, hasher, upvar_capture_map, |hcx, up_var_id| {
                let ty::UpvarId {
                    var_id,
                    closure_expr_id
                } = *up_var_id;

                let var_def_id = hcx.tcx().hir.local_def_id(var_id);
                let closure_def_id = hcx.tcx().hir.local_def_id(closure_expr_id);
                (hcx.def_path_hash(var_def_id), hcx.def_path_hash(closure_def_id))
            });

            ich::hash_stable_nodemap(hcx, hasher, closure_tys);
            ich::hash_stable_nodemap(hcx, hasher, closure_kinds);
            ich::hash_stable_nodemap(hcx, hasher, generator_interiors);
            ich::hash_stable_nodemap(hcx, hasher, generator_sigs);
            ich::hash_stable_nodemap(hcx, hasher, liberated_fn_sigs);
            ich::hash_stable_nodemap(hcx, hasher, fru_field_types);
            ich::hash_stable_nodemap(hcx, hasher, cast_kinds);

            ich::hash_stable_hashset(hcx, hasher, used_trait_imports, |hcx, def_id| {
                hcx.def_path_hash(*def_id)
            });

            tainted_by_errors.hash_stable(hcx, hasher);
            free_region_map.hash_stable(hcx, hasher);
        })
    }
}

impl_stable_hash_for!(enum ty::fast_reject::SimplifiedType {
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(int_ty),
    UintSimplifiedType(int_ty),
    FloatSimplifiedType(float_ty),
    AdtSimplifiedType(def_id),
    StrSimplifiedType,
    ArraySimplifiedType,
    PtrSimplifiedType,
    NeverSimplifiedType,
    TupleSimplifiedType(size),
    TraitSimplifiedType(def_id),
    ClosureSimplifiedType(def_id),
    AnonSimplifiedType(def_id),
    FunctionSimplifiedType(params),
    ParameterSimplifiedType
});

impl_stable_hash_for!(struct ty::Instance<'tcx> {
    def,
    substs
});

impl<'a, 'gcx, 'tcx> HashStable<StableHashingContext<'a, 'gcx, 'tcx>> for ty::InstanceDef<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a, 'gcx, 'tcx>,
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
        }
    }
}

