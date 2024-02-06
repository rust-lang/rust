//! Conversion of internal Rust compiler `ty` items to stable ones.

use rustc_middle::ty::Ty;
use rustc_middle::{mir, ty};
use stable_mir::ty::{
    AdtKind, Const, ConstantKind, FloatTy, GenericArgs, GenericParamDef, IntTy, Region, RigidTy,
    TyKind, UintTy,
};

use crate::rustc_smir::{alloc, Stable, Tables};

impl<'tcx> Stable<'tcx> for ty::AliasKind {
    type T = stable_mir::ty::AliasKind;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::Projection => stable_mir::ty::AliasKind::Projection,
            ty::Inherent => stable_mir::ty::AliasKind::Inherent,
            ty::Opaque => stable_mir::ty::AliasKind::Opaque,
            ty::Weak => stable_mir::ty::AliasKind::Weak,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasTy<'tcx> {
    type T = stable_mir::ty::AliasTy;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::AliasTy { args, def_id, .. } = self;
        stable_mir::ty::AliasTy { def_id: tables.alias_def(*def_id), args: args.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::DynKind {
    type T = stable_mir::ty::DynKind;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::Dyn => stable_mir::ty::DynKind::Dyn,
            ty::DynStar => stable_mir::ty::DynKind::DynStar,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialPredicate<'tcx> {
    type T = stable_mir::ty::ExistentialPredicate;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::ExistentialPredicate::*;
        match self {
            ty::ExistentialPredicate::Trait(existential_trait_ref) => {
                Trait(existential_trait_ref.stable(tables))
            }
            ty::ExistentialPredicate::Projection(existential_projection) => {
                Projection(existential_projection.stable(tables))
            }
            ty::ExistentialPredicate::AutoTrait(def_id) => AutoTrait(tables.trait_def(*def_id)),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialTraitRef<'tcx> {
    type T = stable_mir::ty::ExistentialTraitRef;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::ExistentialTraitRef { def_id, args } = self;
        stable_mir::ty::ExistentialTraitRef {
            def_id: tables.trait_def(*def_id),
            generic_args: args.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TermKind<'tcx> {
    type T = stable_mir::ty::TermKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::TermKind;
        match self {
            ty::TermKind::Ty(ty) => TermKind::Type(ty.stable(tables)),
            ty::TermKind::Const(cnst) => {
                let cnst = cnst.stable(tables);
                TermKind::Const(cnst)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialProjection<'tcx> {
    type T = stable_mir::ty::ExistentialProjection;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::ExistentialProjection { def_id, args, term } = self;
        stable_mir::ty::ExistentialProjection {
            def_id: tables.trait_def(*def_id),
            generic_args: args.stable(tables),
            term: term.unpack().stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::adjustment::PointerCoercion {
    type T = stable_mir::mir::PointerCoercion;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::ty::adjustment::PointerCoercion;
        match self {
            PointerCoercion::ReifyFnPointer => stable_mir::mir::PointerCoercion::ReifyFnPointer,
            PointerCoercion::UnsafeFnPointer => stable_mir::mir::PointerCoercion::UnsafeFnPointer,
            PointerCoercion::ClosureFnPointer(unsafety) => {
                stable_mir::mir::PointerCoercion::ClosureFnPointer(unsafety.stable(tables))
            }
            PointerCoercion::MutToConstPointer => {
                stable_mir::mir::PointerCoercion::MutToConstPointer
            }
            PointerCoercion::ArrayToPointer => stable_mir::mir::PointerCoercion::ArrayToPointer,
            PointerCoercion::Unsize => stable_mir::mir::PointerCoercion::Unsize,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::UserTypeAnnotationIndex {
    type T = usize;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for ty::AdtKind {
    type T = AdtKind;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        match self {
            ty::AdtKind::Struct => AdtKind::Struct,
            ty::AdtKind::Union => AdtKind::Union,
            ty::AdtKind::Enum => AdtKind::Enum,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::FieldDef {
    type T = stable_mir::ty::FieldDef;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        stable_mir::ty::FieldDef {
            def: tables.create_def_id(self.did),
            name: self.name.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::GenericArgs<'tcx> {
    type T = stable_mir::ty::GenericArgs;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        GenericArgs(self.iter().map(|arg| arg.unpack().stable(tables)).collect())
    }
}

impl<'tcx> Stable<'tcx> for ty::GenericArgKind<'tcx> {
    type T = stable_mir::ty::GenericArgKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::GenericArgKind;
        match self {
            ty::GenericArgKind::Lifetime(region) => GenericArgKind::Lifetime(region.stable(tables)),
            ty::GenericArgKind::Type(ty) => GenericArgKind::Type(ty.stable(tables)),
            ty::GenericArgKind::Const(cnst) => GenericArgKind::Const(cnst.stable(tables)),
        }
    }
}

impl<'tcx, S, V> Stable<'tcx> for ty::Binder<'tcx, S>
where
    S: Stable<'tcx, T = V>,
{
    type T = stable_mir::ty::Binder<V>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::Binder;

        Binder {
            value: self.as_ref().skip_binder().stable(tables),
            bound_vars: self
                .bound_vars()
                .iter()
                .map(|bound_var| bound_var.stable(tables))
                .collect(),
        }
    }
}

impl<'tcx, S, V> Stable<'tcx> for ty::EarlyBinder<S>
where
    S: Stable<'tcx, T = V>,
{
    type T = stable_mir::ty::EarlyBinder<V>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::EarlyBinder;

        EarlyBinder { value: self.as_ref().skip_binder().stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::FnSig<'tcx> {
    type T = stable_mir::ty::FnSig;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_target::spec::abi;
        use stable_mir::ty::{Abi, FnSig};

        FnSig {
            inputs_and_output: self.inputs_and_output.iter().map(|ty| ty.stable(tables)).collect(),
            c_variadic: self.c_variadic,
            unsafety: self.unsafety.stable(tables),
            abi: match self.abi {
                abi::Abi::Rust => Abi::Rust,
                abi::Abi::C { unwind } => Abi::C { unwind },
                abi::Abi::Cdecl { unwind } => Abi::Cdecl { unwind },
                abi::Abi::Stdcall { unwind } => Abi::Stdcall { unwind },
                abi::Abi::Fastcall { unwind } => Abi::Fastcall { unwind },
                abi::Abi::Vectorcall { unwind } => Abi::Vectorcall { unwind },
                abi::Abi::Thiscall { unwind } => Abi::Thiscall { unwind },
                abi::Abi::Aapcs { unwind } => Abi::Aapcs { unwind },
                abi::Abi::Win64 { unwind } => Abi::Win64 { unwind },
                abi::Abi::SysV64 { unwind } => Abi::SysV64 { unwind },
                abi::Abi::PtxKernel => Abi::PtxKernel,
                abi::Abi::Msp430Interrupt => Abi::Msp430Interrupt,
                abi::Abi::X86Interrupt => Abi::X86Interrupt,
                abi::Abi::EfiApi => Abi::EfiApi,
                abi::Abi::AvrInterrupt => Abi::AvrInterrupt,
                abi::Abi::AvrNonBlockingInterrupt => Abi::AvrNonBlockingInterrupt,
                abi::Abi::CCmseNonSecureCall => Abi::CCmseNonSecureCall,
                abi::Abi::Wasm => Abi::Wasm,
                abi::Abi::System { unwind } => Abi::System { unwind },
                abi::Abi::RustIntrinsic => Abi::RustIntrinsic,
                abi::Abi::RustCall => Abi::RustCall,
                abi::Abi::PlatformIntrinsic => Abi::PlatformIntrinsic,
                abi::Abi::Unadjusted => Abi::Unadjusted,
                abi::Abi::RustCold => Abi::RustCold,
                abi::Abi::RiscvInterruptM => Abi::RiscvInterruptM,
                abi::Abi::RiscvInterruptS => Abi::RiscvInterruptS,
            },
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTyKind {
    type T = stable_mir::ty::BoundTyKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::BoundTyKind;

        match self {
            ty::BoundTyKind::Anon => BoundTyKind::Anon,
            ty::BoundTyKind::Param(def_id, symbol) => {
                BoundTyKind::Param(tables.param_def(*def_id), symbol.to_string())
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundRegionKind {
    type T = stable_mir::ty::BoundRegionKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::BoundRegionKind;

        match self {
            ty::BoundRegionKind::BrAnon => BoundRegionKind::BrAnon,
            ty::BoundRegionKind::BrNamed(def_id, symbol) => {
                BoundRegionKind::BrNamed(tables.br_named_def(*def_id), symbol.to_string())
            }
            ty::BoundRegionKind::BrEnv => BoundRegionKind::BrEnv,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundVariableKind {
    type T = stable_mir::ty::BoundVariableKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::BoundVariableKind;

        match self {
            ty::BoundVariableKind::Ty(bound_ty_kind) => {
                BoundVariableKind::Ty(bound_ty_kind.stable(tables))
            }
            ty::BoundVariableKind::Region(bound_region_kind) => {
                BoundVariableKind::Region(bound_region_kind.stable(tables))
            }
            ty::BoundVariableKind::Const => BoundVariableKind::Const,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::IntTy {
    type T = IntTy;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::IntTy::Isize => IntTy::Isize,
            ty::IntTy::I8 => IntTy::I8,
            ty::IntTy::I16 => IntTy::I16,
            ty::IntTy::I32 => IntTy::I32,
            ty::IntTy::I64 => IntTy::I64,
            ty::IntTy::I128 => IntTy::I128,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::UintTy {
    type T = UintTy;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::UintTy::Usize => UintTy::Usize,
            ty::UintTy::U8 => UintTy::U8,
            ty::UintTy::U16 => UintTy::U16,
            ty::UintTy::U32 => UintTy::U32,
            ty::UintTy::U64 => UintTy::U64,
            ty::UintTy::U128 => UintTy::U128,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::FloatTy {
    type T = FloatTy;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::FloatTy::F32 => FloatTy::F32,
            ty::FloatTy::F64 => FloatTy::F64,
        }
    }
}

impl<'tcx> Stable<'tcx> for Ty<'tcx> {
    type T = stable_mir::ty::Ty;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        tables.intern_ty(tables.tcx.lift(*self).unwrap())
    }
}

impl<'tcx> Stable<'tcx> for ty::TyKind<'tcx> {
    type T = stable_mir::ty::TyKind;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            ty::Bool => TyKind::RigidTy(RigidTy::Bool),
            ty::Char => TyKind::RigidTy(RigidTy::Char),
            ty::Int(int_ty) => TyKind::RigidTy(RigidTy::Int(int_ty.stable(tables))),
            ty::Uint(uint_ty) => TyKind::RigidTy(RigidTy::Uint(uint_ty.stable(tables))),
            ty::Float(float_ty) => TyKind::RigidTy(RigidTy::Float(float_ty.stable(tables))),
            ty::Adt(adt_def, generic_args) => TyKind::RigidTy(RigidTy::Adt(
                tables.adt_def(adt_def.did()),
                generic_args.stable(tables),
            )),
            ty::Foreign(def_id) => TyKind::RigidTy(RigidTy::Foreign(tables.foreign_def(*def_id))),
            ty::Str => TyKind::RigidTy(RigidTy::Str),
            ty::Array(ty, constant) => {
                TyKind::RigidTy(RigidTy::Array(ty.stable(tables), constant.stable(tables)))
            }
            ty::Slice(ty) => TyKind::RigidTy(RigidTy::Slice(ty.stable(tables))),
            ty::RawPtr(ty::TypeAndMut { ty, mutbl }) => {
                TyKind::RigidTy(RigidTy::RawPtr(ty.stable(tables), mutbl.stable(tables)))
            }
            ty::Ref(region, ty, mutbl) => TyKind::RigidTy(RigidTy::Ref(
                region.stable(tables),
                ty.stable(tables),
                mutbl.stable(tables),
            )),
            ty::FnDef(def_id, generic_args) => {
                TyKind::RigidTy(RigidTy::FnDef(tables.fn_def(*def_id), generic_args.stable(tables)))
            }
            ty::FnPtr(poly_fn_sig) => TyKind::RigidTy(RigidTy::FnPtr(poly_fn_sig.stable(tables))),
            ty::Dynamic(existential_predicates, region, dyn_kind) => {
                TyKind::RigidTy(RigidTy::Dynamic(
                    existential_predicates
                        .iter()
                        .map(|existential_predicate| existential_predicate.stable(tables))
                        .collect(),
                    region.stable(tables),
                    dyn_kind.stable(tables),
                ))
            }
            ty::Closure(def_id, generic_args) => TyKind::RigidTy(RigidTy::Closure(
                tables.closure_def(*def_id),
                generic_args.stable(tables),
            )),
            ty::CoroutineClosure(..) => todo!("FIXME(async_closures): Lower these to SMIR"),
            ty::Coroutine(def_id, generic_args) => TyKind::RigidTy(RigidTy::Coroutine(
                tables.coroutine_def(*def_id),
                generic_args.stable(tables),
                tables.tcx.coroutine_movability(*def_id).stable(tables),
            )),
            ty::Never => TyKind::RigidTy(RigidTy::Never),
            ty::Tuple(fields) => {
                TyKind::RigidTy(RigidTy::Tuple(fields.iter().map(|ty| ty.stable(tables)).collect()))
            }
            ty::Alias(alias_kind, alias_ty) => {
                TyKind::Alias(alias_kind.stable(tables), alias_ty.stable(tables))
            }
            ty::Param(param_ty) => TyKind::Param(param_ty.stable(tables)),
            ty::Bound(debruijn_idx, bound_ty) => {
                TyKind::Bound(debruijn_idx.as_usize(), bound_ty.stable(tables))
            }
            ty::CoroutineWitness(def_id, args) => TyKind::RigidTy(RigidTy::CoroutineWitness(
                tables.coroutine_witness_def(*def_id),
                args.stable(tables),
            )),
            ty::Placeholder(..) | ty::Infer(_) | ty::Error(_) => {
                unreachable!();
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Const<'tcx> {
    type T = stable_mir::ty::Const;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let kind = match self.kind() {
            ty::Value(val) => {
                let val = match val {
                    ty::ValTree::Leaf(scalar) => ty::ValTree::Leaf(scalar),
                    ty::ValTree::Branch(branch) => {
                        ty::ValTree::Branch(tables.tcx.lift(branch).unwrap())
                    }
                };
                let ty = tables.tcx.lift(self.ty()).unwrap();
                let const_val = tables.tcx.valtree_to_const_val((ty, val));
                if matches!(const_val, mir::ConstValue::ZeroSized) {
                    ConstantKind::ZeroSized
                } else {
                    stable_mir::ty::ConstantKind::Allocated(alloc::new_allocation(
                        ty, const_val, tables,
                    ))
                }
            }
            ty::ParamCt(param) => stable_mir::ty::ConstantKind::Param(param.stable(tables)),
            ty::ErrorCt(_) => unreachable!(),
            ty::InferCt(_) => unreachable!(),
            ty::BoundCt(_, _) => unimplemented!(),
            ty::PlaceholderCt(_) => unimplemented!(),
            ty::Unevaluated(uv) => {
                stable_mir::ty::ConstantKind::Unevaluated(stable_mir::ty::UnevaluatedConst {
                    def: tables.const_def(uv.def),
                    args: uv.args.stable(tables),
                    promoted: None,
                })
            }
            ty::ExprCt(_) => unimplemented!(),
        };
        let ty = self.ty().stable(tables);
        let id = tables.intern_const(mir::Const::Ty(tables.tcx.lift(*self).unwrap()));
        Const::new(kind, ty, id)
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamConst {
    type T = stable_mir::ty::ParamConst;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::ParamConst;
        ParamConst { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamTy {
    type T = stable_mir::ty::ParamTy;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::ParamTy;
        ParamTy { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTy {
    type T = stable_mir::ty::BoundTy;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::BoundTy;
        BoundTy { var: self.var.as_usize(), kind: self.kind.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::trait_def::TraitSpecializationKind {
    type T = stable_mir::ty::TraitSpecializationKind;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::TraitSpecializationKind;

        match self {
            ty::trait_def::TraitSpecializationKind::None => TraitSpecializationKind::None,
            ty::trait_def::TraitSpecializationKind::Marker => TraitSpecializationKind::Marker,
            ty::trait_def::TraitSpecializationKind::AlwaysApplicable => {
                TraitSpecializationKind::AlwaysApplicable
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitDef {
    type T = stable_mir::ty::TraitDecl;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::opaque;
        use stable_mir::ty::TraitDecl;

        TraitDecl {
            def_id: tables.trait_def(self.def_id),
            unsafety: self.unsafety.stable(tables),
            paren_sugar: self.paren_sugar,
            has_auto_impl: self.has_auto_impl,
            is_marker: self.is_marker,
            is_coinductive: self.is_coinductive,
            skip_array_during_method_dispatch: self.skip_array_during_method_dispatch,
            specialization_kind: self.specialization_kind.stable(tables),
            must_implement_one_of: self
                .must_implement_one_of
                .as_ref()
                .map(|idents| idents.iter().map(|ident| opaque(ident)).collect()),
            implement_via_object: self.implement_via_object,
            deny_explicit_impl: self.deny_explicit_impl,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitRef<'tcx> {
    type T = stable_mir::ty::TraitRef;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::TraitRef;

        TraitRef::try_new(tables.trait_def(self.def_id), self.args.stable(tables)).unwrap()
    }
}

impl<'tcx> Stable<'tcx> for ty::Generics {
    type T = stable_mir::ty::Generics;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::Generics;

        let params: Vec<_> = self.params.iter().map(|param| param.stable(tables)).collect();
        let param_def_id_to_index =
            params.iter().map(|param| (param.def_id, param.index)).collect();

        Generics {
            parent: self.parent.map(|did| tables.generic_def(did)),
            parent_count: self.parent_count,
            params,
            param_def_id_to_index,
            has_self: self.has_self,
            has_late_bound_regions: self
                .has_late_bound_regions
                .as_ref()
                .map(|late_bound_regions| late_bound_regions.stable(tables)),
            host_effect_index: self.host_effect_index,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::GenericParamDefKind {
    type T = stable_mir::ty::GenericParamDefKind;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::GenericParamDefKind;
        match self {
            ty::GenericParamDefKind::Lifetime => GenericParamDefKind::Lifetime,
            ty::GenericParamDefKind::Type { has_default, synthetic } => {
                GenericParamDefKind::Type { has_default: *has_default, synthetic: *synthetic }
            }
            ty::GenericParamDefKind::Const { has_default, is_host_effect: _ } => {
                GenericParamDefKind::Const { has_default: *has_default }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::GenericParamDef {
    type T = stable_mir::ty::GenericParamDef;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        GenericParamDef {
            name: self.name.to_string(),
            def_id: tables.generic_def(self.def_id),
            index: self.index,
            pure_wrt_drop: self.pure_wrt_drop,
            kind: self.kind.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::PredicateKind<'tcx> {
    type T = stable_mir::ty::PredicateKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::ty::PredicateKind;
        match self {
            PredicateKind::Clause(clause_kind) => {
                stable_mir::ty::PredicateKind::Clause(clause_kind.stable(tables))
            }
            PredicateKind::ObjectSafe(did) => {
                stable_mir::ty::PredicateKind::ObjectSafe(tables.trait_def(*did))
            }
            PredicateKind::Subtype(subtype_predicate) => {
                stable_mir::ty::PredicateKind::SubType(subtype_predicate.stable(tables))
            }
            PredicateKind::Coerce(coerce_predicate) => {
                stable_mir::ty::PredicateKind::Coerce(coerce_predicate.stable(tables))
            }
            PredicateKind::ConstEquate(a, b) => {
                stable_mir::ty::PredicateKind::ConstEquate(a.stable(tables), b.stable(tables))
            }
            PredicateKind::Ambiguous => stable_mir::ty::PredicateKind::Ambiguous,
            PredicateKind::NormalizesTo(_pred) => unimplemented!(),
            PredicateKind::AliasRelate(a, b, alias_relation_direction) => {
                stable_mir::ty::PredicateKind::AliasRelate(
                    a.unpack().stable(tables),
                    b.unpack().stable(tables),
                    alias_relation_direction.stable(tables),
                )
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ClauseKind<'tcx> {
    type T = stable_mir::ty::ClauseKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::ty::ClauseKind;
        match *self {
            ClauseKind::Trait(trait_object) => {
                stable_mir::ty::ClauseKind::Trait(trait_object.stable(tables))
            }
            ClauseKind::RegionOutlives(region_outlives) => {
                stable_mir::ty::ClauseKind::RegionOutlives(region_outlives.stable(tables))
            }
            ClauseKind::TypeOutlives(type_outlives) => {
                let ty::OutlivesPredicate::<_, _>(a, b) = type_outlives;
                stable_mir::ty::ClauseKind::TypeOutlives(stable_mir::ty::OutlivesPredicate(
                    a.stable(tables),
                    b.stable(tables),
                ))
            }
            ClauseKind::Projection(projection_predicate) => {
                stable_mir::ty::ClauseKind::Projection(projection_predicate.stable(tables))
            }
            ClauseKind::ConstArgHasType(const_, ty) => stable_mir::ty::ClauseKind::ConstArgHasType(
                const_.stable(tables),
                ty.stable(tables),
            ),
            ClauseKind::WellFormed(generic_arg) => {
                stable_mir::ty::ClauseKind::WellFormed(generic_arg.unpack().stable(tables))
            }
            ClauseKind::ConstEvaluatable(const_) => {
                stable_mir::ty::ClauseKind::ConstEvaluatable(const_.stable(tables))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ClosureKind {
    type T = stable_mir::ty::ClosureKind;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use rustc_middle::ty::ClosureKind::*;
        match self {
            Fn => stable_mir::ty::ClosureKind::Fn,
            FnMut => stable_mir::ty::ClosureKind::FnMut,
            FnOnce => stable_mir::ty::ClosureKind::FnOnce,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::SubtypePredicate<'tcx> {
    type T = stable_mir::ty::SubtypePredicate;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::SubtypePredicate { a, b, a_is_expected: _ } = self;
        stable_mir::ty::SubtypePredicate { a: a.stable(tables), b: b.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::CoercePredicate<'tcx> {
    type T = stable_mir::ty::CoercePredicate;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::CoercePredicate { a, b } = self;
        stable_mir::ty::CoercePredicate { a: a.stable(tables), b: b.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasRelationDirection {
    type T = stable_mir::ty::AliasRelationDirection;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use rustc_middle::ty::AliasRelationDirection::*;
        match self {
            Equate => stable_mir::ty::AliasRelationDirection::Equate,
            Subtype => stable_mir::ty::AliasRelationDirection::Subtype,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitPredicate<'tcx> {
    type T = stable_mir::ty::TraitPredicate;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::TraitPredicate { trait_ref, polarity } = self;
        stable_mir::ty::TraitPredicate {
            trait_ref: trait_ref.stable(tables),
            polarity: polarity.stable(tables),
        }
    }
}

impl<'tcx, A, B, U, V> Stable<'tcx> for ty::OutlivesPredicate<A, B>
where
    A: Stable<'tcx, T = U>,
    B: Stable<'tcx, T = V>,
{
    type T = stable_mir::ty::OutlivesPredicate<U, V>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::OutlivesPredicate(a, b) = self;
        stable_mir::ty::OutlivesPredicate(a.stable(tables), b.stable(tables))
    }
}

impl<'tcx> Stable<'tcx> for ty::ProjectionPredicate<'tcx> {
    type T = stable_mir::ty::ProjectionPredicate;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let ty::ProjectionPredicate { projection_ty, term } = self;
        stable_mir::ty::ProjectionPredicate {
            projection_ty: projection_ty.stable(tables),
            term: term.unpack().stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ImplPolarity {
    type T = stable_mir::ty::ImplPolarity;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use rustc_middle::ty::ImplPolarity::*;
        match self {
            Positive => stable_mir::ty::ImplPolarity::Positive,
            Negative => stable_mir::ty::ImplPolarity::Negative,
            Reservation => stable_mir::ty::ImplPolarity::Reservation,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Region<'tcx> {
    type T = stable_mir::ty::Region;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        Region { kind: self.kind().stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::RegionKind<'tcx> {
    type T = stable_mir::ty::RegionKind;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::ty::{BoundRegion, EarlyParamRegion, RegionKind};
        match self {
            ty::ReEarlyParam(early_reg) => RegionKind::ReEarlyParam(EarlyParamRegion {
                def_id: tables.region_def(early_reg.def_id),
                index: early_reg.index,
                name: early_reg.name.to_string(),
            }),
            ty::ReBound(db_index, bound_reg) => RegionKind::ReBound(
                db_index.as_u32(),
                BoundRegion { var: bound_reg.var.as_u32(), kind: bound_reg.kind.stable(tables) },
            ),
            ty::ReStatic => RegionKind::ReStatic,
            ty::RePlaceholder(place_holder) => {
                RegionKind::RePlaceholder(stable_mir::ty::Placeholder {
                    universe: place_holder.universe.as_u32(),
                    bound: BoundRegion {
                        var: place_holder.bound.var.as_u32(),
                        kind: place_holder.bound.kind.stable(tables),
                    },
                })
            }
            ty::ReErased => RegionKind::ReErased,
            _ => unreachable!("{self:?}"),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Instance<'tcx> {
    type T = stable_mir::mir::mono::Instance;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let def = tables.instance_def(tables.tcx.lift(*self).unwrap());
        let kind = match self.def {
            ty::InstanceDef::Item(..) => stable_mir::mir::mono::InstanceKind::Item,
            ty::InstanceDef::Intrinsic(..) => stable_mir::mir::mono::InstanceKind::Intrinsic,
            ty::InstanceDef::Virtual(_def_id, idx) => {
                stable_mir::mir::mono::InstanceKind::Virtual { idx }
            }
            ty::InstanceDef::VTableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::FnPtrAddrShim(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceDef::CoroutineKindShim { .. }
            | ty::InstanceDef::ThreadLocalShim(..)
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..)
            | ty::InstanceDef::FnPtrShim(..) => stable_mir::mir::mono::InstanceKind::Shim,
        };
        stable_mir::mir::mono::Instance { def, kind }
    }
}

impl<'tcx> Stable<'tcx> for ty::Variance {
    type T = stable_mir::mir::Variance;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::Variance::Bivariant => stable_mir::mir::Variance::Bivariant,
            ty::Variance::Contravariant => stable_mir::mir::Variance::Contravariant,
            ty::Variance::Covariant => stable_mir::mir::Variance::Covariant,
            ty::Variance::Invariant => stable_mir::mir::Variance::Invariant,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Movability {
    type T = stable_mir::ty::Movability;

    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        match self {
            ty::Movability::Static => stable_mir::ty::Movability::Static,
            ty::Movability::Movable => stable_mir::ty::Movability::Movable,
        }
    }
}
