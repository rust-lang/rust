//! Module that implements what will become the rustc side of Stable MIR.

//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};
use stable_mir::abi::Layout;
use stable_mir::mir::mono::InstanceDef;
use stable_mir::ty::{ConstId, Span};
use stable_mir::{CtorKind, ItemKind};
use std::ops::RangeInclusive;
use tracing::debug;

use crate::rustc_internal::IndexMap;

mod alloc;
mod builder;
pub(crate) mod context;
mod convert;

pub struct Tables<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) def_ids: IndexMap<DefId, stable_mir::DefId>,
    pub(crate) alloc_ids: IndexMap<AllocId, stable_mir::mir::alloc::AllocId>,
    pub(crate) spans: IndexMap<rustc_span::Span, Span>,
    pub(crate) types: IndexMap<Ty<'tcx>, stable_mir::ty::Ty>,
    pub(crate) instances: IndexMap<ty::Instance<'tcx>, InstanceDef>,
    pub(crate) constants: IndexMap<mir::Const<'tcx>, ConstId>,
    pub(crate) layouts: IndexMap<rustc_target::abi::Layout<'tcx>, Layout>,
}

impl<'tcx> Tables<'tcx> {
    pub(crate) fn intern_ty(&mut self, ty: Ty<'tcx>) -> stable_mir::ty::Ty {
        self.types.create_or_fetch(ty)
    }

    pub(crate) fn intern_const(&mut self, constant: mir::Const<'tcx>) -> ConstId {
        self.constants.create_or_fetch(constant)
    }

    pub(crate) fn has_body(&self, instance: Instance<'tcx>) -> bool {
        let def_id = instance.def_id();
        self.tcx.is_mir_available(def_id)
            || !matches!(
                instance.def,
                ty::InstanceDef::Virtual(..)
                    | ty::InstanceDef::Intrinsic(..)
                    | ty::InstanceDef::Item(..)
            )
    }
}

/// Build a stable mir crate from a given crate number.
pub(crate) fn smir_crate(tcx: TyCtxt<'_>, crate_num: CrateNum) -> stable_mir::Crate {
    let crate_name = tcx.crate_name(crate_num).to_string();
    let is_local = crate_num == LOCAL_CRATE;
    debug!(?crate_name, ?crate_num, "smir_crate");
    stable_mir::Crate { id: crate_num.into(), name: crate_name, is_local }
}

pub(crate) fn new_item_kind(kind: DefKind) -> ItemKind {
    match kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::Impl { .. }
        | DefKind::GlobalAsm => {
            unreachable!("Not a valid item kind: {kind:?}");
        }
        DefKind::Closure | DefKind::AssocFn | DefKind::Fn => ItemKind::Fn,
        DefKind::Const | DefKind::InlineConst | DefKind::AssocConst | DefKind::AnonConst => {
            ItemKind::Const
        }
        DefKind::Static { .. } => ItemKind::Static,
        DefKind::Ctor(_, rustc_hir::def::CtorKind::Const) => ItemKind::Ctor(CtorKind::Const),
        DefKind::Ctor(_, rustc_hir::def::CtorKind::Fn) => ItemKind::Ctor(CtorKind::Fn),
    }
}

/// Trait used to convert between an internal MIR type to a Stable MIR type.
pub trait Stable<'cx> {
    /// The stable representation of the type implementing Stable.
    type T;
    /// Converts an object to the equivalent Stable MIR representation.
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T;
}

impl<'tcx, T> Stable<'tcx> for &T
where
    T: Stable<'tcx>,
{
    type T = T::T;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        (*self).stable(tables)
    }
}

impl<'tcx, T> Stable<'tcx> for Option<T>
where
    T: Stable<'tcx>,
{
    type T = Option<T::T>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        self.as_ref().map(|value| value.stable(tables))
    }
}

impl<'tcx> Stable<'tcx> for ty::FnSig<'tcx> {
    type T = stable_mir::ty::FnSig;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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
                abi::Abi::AmdGpuKernel => Abi::AmdGpuKernel,
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

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        match self {
            ty::FloatTy::F32 => FloatTy::F32,
            ty::FloatTy::F64 => FloatTy::F64,
        }
    }
}

impl<'tcx> Stable<'tcx> for hir::Movability {
    type T = Movability;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        match self {
            hir::Movability::Static => Movability::Static,
            hir::Movability::Movable => Movability::Movable,
        }
    }
}

impl<'tcx> Stable<'tcx> for Ty<'tcx> {
    type T = stable_mir::ty::Ty;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.intern_ty(*self)
    }
}

impl<'tcx> Stable<'tcx> for ty::TyKind<'tcx> {
    type T = stable_mir::ty::TyKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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
            ty::Coroutine(def_id, generic_args, movability) => TyKind::RigidTy(RigidTy::Coroutine(
                tables.coroutine_def(*def_id),
                generic_args.stable(tables),
                movability.stable(tables),
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
            ty::Placeholder(..) | ty::CoroutineWitness(..) | ty::Infer(_) | ty::Error(_) => {
                unreachable!();
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Const<'tcx> {
    type T = stable_mir::ty::Const;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let kind = match self.kind() {
            ty::Value(val) => {
                let const_val = tables.tcx.valtree_to_const_val((self.ty(), val));
                if matches!(const_val, mir::ConstValue::ZeroSized) {
                    ConstantKind::ZeroSized
                } else {
                    stable_mir::ty::ConstantKind::Allocated(alloc::new_allocation(
                        self.ty(),
                        const_val,
                        tables,
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
        let id = tables.intern_const(mir::Const::Ty(*self));
        Const::new(kind, ty, id)
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamConst {
    type T = stable_mir::ty::ParamConst;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::ParamConst;
        ParamConst { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamTy {
    type T = stable_mir::ty::ParamTy;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::ParamTy;
        ParamTy { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTy {
    type T = stable_mir::ty::BoundTy;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::BoundTy;
        BoundTy { var: self.var.as_usize(), kind: self.kind.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::Allocation {
    type T = stable_mir::ty::Allocation;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        alloc::allocation_filter(
            self,
            alloc_range(rustc_target::abi::Size::ZERO, self.size()),
            tables,
        )
    }
}

impl<'tcx> Stable<'tcx> for ty::trait_def::TraitSpecializationKind {
    type T = stable_mir::ty::TraitSpecializationKind;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
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
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::TraitDecl;

        TraitDecl {
            def_id: tables.trait_def(self.def_id),
            unsafety: self.unsafety.stable(tables),
            paren_sugar: self.paren_sugar,
            has_auto_impl: self.has_auto_impl,
            is_marker: self.is_marker,
            is_coinductive: self.is_coinductive,
            skip_array_during_method_dispatch: self.skip_array_during_method_dispatch,
            skip_boxed_slice_during_method_dispatch: self.skip_boxed_slice_during_method_dispatch,
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

impl<'tcx> Stable<'tcx> for rustc_middle::mir::Const<'tcx> {
    type T = stable_mir::ty::Const;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match *self {
            mir::Const::Ty(c) => c.stable(tables),
            mir::Const::Unevaluated(unev_const, ty) => {
                let kind =
                    stable_mir::ty::ConstantKind::Unevaluated(stable_mir::ty::UnevaluatedConst {
                        def: tables.const_def(unev_const.def),
                        args: unev_const.args.stable(tables),
                        promoted: unev_const.promoted.map(|u| u.as_u32()),
                    });
                let ty = ty.stable(tables);
                let id = tables.intern_const(*self);
                Const::new(kind, ty, id)
            }
            mir::Const::Val(val, ty) if matches!(val, mir::ConstValue::ZeroSized) => {
                let ty = ty.stable(tables);
                let id = tables.intern_const(*self);
                Const::new(ConstantKind::ZeroSized, ty, id)
            }
            mir::Const::Val(val, ty) => {
                let kind = ConstantKind::Allocated(alloc::new_allocation(ty, val, tables));
                let ty = ty.stable(tables);
                let id = tables.intern_const(*self);
                Const::new(kind, ty, id)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitRef<'tcx> {
    type T = stable_mir::ty::TraitRef;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::TraitRef;

        TraitRef { def_id: tables.trait_def(self.def_id), args: self.args.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::Generics {
    type T = stable_mir::ty::Generics;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use ty::PredicateKind;
        match self {
            PredicateKind::Clause(clause_kind) => {
                stable_mir::ty::PredicateKind::Clause(clause_kind.stable(tables))
            }
            PredicateKind::ObjectSafe(did) => {
                stable_mir::ty::PredicateKind::ObjectSafe(tables.trait_def(*did))
            }
            PredicateKind::ClosureKind(did, generic_args, closure_kind) => {
                stable_mir::ty::PredicateKind::ClosureKind(
                    tables.closure_def(*did),
                    generic_args.stable(tables),
                    closure_kind.stable(tables),
                )
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

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use ty::ClauseKind;
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::ClosureKind::*;
        match self {
            Fn => stable_mir::ty::ClosureKind::Fn,
            FnMut => stable_mir::ty::ClosureKind::FnMut,
            FnOnce => stable_mir::ty::ClosureKind::FnOnce,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::SubtypePredicate<'tcx> {
    type T = stable_mir::ty::SubtypePredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::SubtypePredicate { a, b, a_is_expected: _ } = self;
        stable_mir::ty::SubtypePredicate { a: a.stable(tables), b: b.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::CoercePredicate<'tcx> {
    type T = stable_mir::ty::CoercePredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::CoercePredicate { a, b } = self;
        stable_mir::ty::CoercePredicate { a: a.stable(tables), b: b.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasRelationDirection {
    type T = stable_mir::ty::AliasRelationDirection;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::AliasRelationDirection::*;
        match self {
            Equate => stable_mir::ty::AliasRelationDirection::Equate,
            Subtype => stable_mir::ty::AliasRelationDirection::Subtype,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitPredicate<'tcx> {
    type T = stable_mir::ty::TraitPredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::TraitPredicate { trait_ref, polarity } = self;
        stable_mir::ty::TraitPredicate {
            trait_ref: trait_ref.stable(tables),
            polarity: polarity.stable(tables),
        }
    }
}

impl<'tcx, A, B, U, V> Stable<'tcx> for ty::OutlivesPredicate<A, B>
where
    T: Stable<'tcx>,
    E: Stable<'tcx>,
{
    type T = Result<T::T, E::T>;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            Ok(val) => Ok(val.stable(tables)),
            Err(error) => Err(error.stable(tables)),
        }
    }
}

impl<'tcx, T> Stable<'tcx> for &[T]
where
    T: Stable<'tcx>,
{
    type T = Vec<T::T>;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        self.iter().map(|e| e.stable(tables)).collect()
    }
}

impl<'tcx, T, U> Stable<'tcx> for (T, U)
where
    T: Stable<'tcx>,
    U: Stable<'tcx>,
{
    type T = (T::T, U::T);
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        (self.0.stable(tables), self.1.stable(tables))
    }
}

impl<'tcx, T> Stable<'tcx> for RangeInclusive<T>
where
    T: Stable<'tcx>,
{
    type T = RangeInclusive<T::T>;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        RangeInclusive::new(self.start().stable(tables), self.end().stable(tables))
    }
}
