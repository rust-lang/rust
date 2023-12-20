//! Module containing the translation from stable mir constructs to the rustc counterpart.
//!
//! This module will only include a few constructs to allow users to invoke internal rustc APIs
//! due to incomplete stable coverage.

// Prefer importing stable_mir over internal rustc constructs to make this file more readable.
use crate::rustc_smir::Tables;
use rustc_middle::ty::{self as rustc_ty, Ty as InternalTy};
use rustc_span::Symbol;
use stable_mir::abi::Layout;
use stable_mir::mir::alloc::AllocId;
use stable_mir::mir::mono::{Instance, MonoItem, StaticDef};
use stable_mir::mir::{Mutability, Safety};
use stable_mir::ty::{
    Abi, AdtDef, Binder, BoundRegionKind, BoundTyKind, BoundVariableKind, ClosureKind, Const,
    DynKind, ExistentialPredicate, ExistentialProjection, ExistentialTraitRef, FloatTy, FnSig,
    GenericArgKind, GenericArgs, IndexedVal, IntTy, Movability, Region, RigidTy, Span, TermKind,
    TraitRef, Ty, UintTy, VariantDef, VariantIdx,
};
use stable_mir::{CrateItem, DefId};

use super::RustcInternal;

impl<'tcx> RustcInternal<'tcx> for CrateItem {
    type T = rustc_span::def_id::DefId;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.0.internal(tables)
    }
}

impl<'tcx> RustcInternal<'tcx> for DefId {
    type T = rustc_span::def_id::DefId;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.def_ids[*self]
    }
}

impl<'tcx> RustcInternal<'tcx> for GenericArgs {
    type T = rustc_ty::GenericArgsRef<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.tcx.mk_args_from_iter(self.0.iter().map(|arg| arg.internal(tables)))
    }
}

impl<'tcx> RustcInternal<'tcx> for GenericArgKind {
    type T = rustc_ty::GenericArg<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            GenericArgKind::Lifetime(reg) => reg.internal(tables).into(),
            GenericArgKind::Type(ty) => ty.internal(tables).into(),
            GenericArgKind::Const(cnst) => ty_const(cnst, tables).into(),
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Region {
    type T = rustc_ty::Region<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        // Cannot recover region. Use erased instead.
        tables.tcx.lifetimes.re_erased
    }
}

impl<'tcx> RustcInternal<'tcx> for Ty {
    type T = InternalTy<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.types[*self]
    }
}

impl<'tcx> RustcInternal<'tcx> for RigidTy {
    type T = rustc_ty::TyKind<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            RigidTy::Bool => rustc_ty::TyKind::Bool,
            RigidTy::Char => rustc_ty::TyKind::Char,
            RigidTy::Int(int_ty) => rustc_ty::TyKind::Int(int_ty.internal(tables)),
            RigidTy::Uint(uint_ty) => rustc_ty::TyKind::Uint(uint_ty.internal(tables)),
            RigidTy::Float(float_ty) => rustc_ty::TyKind::Float(float_ty.internal(tables)),
            RigidTy::Never => rustc_ty::TyKind::Never,
            RigidTy::Array(ty, cnst) => {
                rustc_ty::TyKind::Array(ty.internal(tables), ty_const(cnst, tables))
            }
            RigidTy::Adt(def, args) => {
                rustc_ty::TyKind::Adt(def.internal(tables), args.internal(tables))
            }
            RigidTy::Str => rustc_ty::TyKind::Str,
            RigidTy::Slice(ty) => rustc_ty::TyKind::Slice(ty.internal(tables)),
            RigidTy::RawPtr(ty, mutability) => rustc_ty::TyKind::RawPtr(rustc_ty::TypeAndMut {
                ty: ty.internal(tables),
                mutbl: mutability.internal(tables),
            }),
            RigidTy::Ref(region, ty, mutability) => rustc_ty::TyKind::Ref(
                region.internal(tables),
                ty.internal(tables),
                mutability.internal(tables),
            ),
            RigidTy::Foreign(def) => rustc_ty::TyKind::Foreign(def.0.internal(tables)),
            RigidTy::FnDef(def, args) => {
                rustc_ty::TyKind::FnDef(def.0.internal(tables), args.internal(tables))
            }
            RigidTy::FnPtr(sig) => rustc_ty::TyKind::FnPtr(sig.internal(tables)),
            RigidTy::Closure(def, args) => {
                rustc_ty::TyKind::Closure(def.0.internal(tables), args.internal(tables))
            }
            RigidTy::Coroutine(def, args, mov) => rustc_ty::TyKind::Coroutine(
                def.0.internal(tables),
                args.internal(tables),
                mov.internal(tables),
            ),
            RigidTy::CoroutineWitness(def, args) => {
                rustc_ty::TyKind::CoroutineWitness(def.0.internal(tables), args.internal(tables))
            }
            RigidTy::Dynamic(predicate, region, dyn_kind) => rustc_ty::TyKind::Dynamic(
                tables.tcx.mk_poly_existential_predicates(&predicate.internal(tables)),
                region.internal(tables),
                dyn_kind.internal(tables),
            ),
            RigidTy::Tuple(tys) => {
                rustc_ty::TyKind::Tuple(tables.tcx.mk_type_list(&tys.internal(tables)))
            }
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for IntTy {
    type T = rustc_ty::IntTy;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            IntTy::Isize => rustc_ty::IntTy::Isize,
            IntTy::I8 => rustc_ty::IntTy::I8,
            IntTy::I16 => rustc_ty::IntTy::I16,
            IntTy::I32 => rustc_ty::IntTy::I32,
            IntTy::I64 => rustc_ty::IntTy::I64,
            IntTy::I128 => rustc_ty::IntTy::I128,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for UintTy {
    type T = rustc_ty::UintTy;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            UintTy::Usize => rustc_ty::UintTy::Usize,
            UintTy::U8 => rustc_ty::UintTy::U8,
            UintTy::U16 => rustc_ty::UintTy::U16,
            UintTy::U32 => rustc_ty::UintTy::U32,
            UintTy::U64 => rustc_ty::UintTy::U64,
            UintTy::U128 => rustc_ty::UintTy::U128,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for FloatTy {
    type T = rustc_ty::FloatTy;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            FloatTy::F32 => rustc_ty::FloatTy::F32,
            FloatTy::F64 => rustc_ty::FloatTy::F64,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Mutability {
    type T = rustc_ty::Mutability;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            Mutability::Not => rustc_ty::Mutability::Not,
            Mutability::Mut => rustc_ty::Mutability::Mut,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Movability {
    type T = rustc_ty::Movability;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            Movability::Static => rustc_ty::Movability::Static,
            Movability::Movable => rustc_ty::Movability::Movable,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for FnSig {
    type T = rustc_ty::FnSig<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::FnSig {
            inputs_and_output: tables.tcx.mk_type_list(&self.inputs_and_output.internal(tables)),
            c_variadic: self.c_variadic,
            unsafety: self.unsafety.internal(tables),
            abi: self.abi.internal(tables),
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for VariantIdx {
    type T = rustc_target::abi::VariantIdx;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        rustc_target::abi::VariantIdx::from(self.to_index())
    }
}

impl<'tcx> RustcInternal<'tcx> for VariantDef {
    type T = &'tcx rustc_ty::VariantDef;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.adt_def.internal(tables).variant(self.idx.internal(tables))
    }
}

fn ty_const<'tcx>(constant: &Const, tables: &mut Tables<'tcx>) -> rustc_ty::Const<'tcx> {
    match constant.internal(tables) {
        rustc_middle::mir::Const::Ty(c) => c,
        cnst => {
            panic!("Trying to convert constant `{constant:?}` to type constant, but found {cnst:?}")
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Const {
    type T = rustc_middle::mir::Const<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.constants[self.id]
    }
}

impl<'tcx> RustcInternal<'tcx> for MonoItem {
    type T = rustc_middle::mir::mono::MonoItem<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::mono as rustc_mono;
        match self {
            MonoItem::Fn(instance) => rustc_mono::MonoItem::Fn(instance.internal(tables)),
            MonoItem::Static(def) => rustc_mono::MonoItem::Static(def.internal(tables)),
            MonoItem::GlobalAsm(_) => {
                unimplemented!()
            }
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Instance {
    type T = rustc_ty::Instance<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.instances[self.def]
    }
}

impl<'tcx> RustcInternal<'tcx> for StaticDef {
    type T = rustc_span::def_id::DefId;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.0.internal(tables)
    }
}

#[allow(rustc::usage_of_qualified_ty)]
impl<'tcx, T> RustcInternal<'tcx> for Binder<T>
where
    T: RustcInternal<'tcx>,
    T::T: rustc_ty::TypeVisitable<rustc_ty::TyCtxt<'tcx>>,
{
    type T = rustc_ty::Binder<'tcx, T::T>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::Binder::bind_with_vars(
            self.value.internal(tables),
            tables.tcx.mk_bound_variable_kinds_from_iter(
                self.bound_vars.iter().map(|bound| bound.internal(tables)),
            ),
        )
    }
}

impl<'tcx> RustcInternal<'tcx> for BoundVariableKind {
    type T = rustc_ty::BoundVariableKind;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            BoundVariableKind::Ty(kind) => rustc_ty::BoundVariableKind::Ty(match kind {
                BoundTyKind::Anon => rustc_ty::BoundTyKind::Anon,
                BoundTyKind::Param(def, symbol) => {
                    rustc_ty::BoundTyKind::Param(def.0.internal(tables), Symbol::intern(symbol))
                }
            }),
            BoundVariableKind::Region(kind) => rustc_ty::BoundVariableKind::Region(match kind {
                BoundRegionKind::BrAnon => rustc_ty::BoundRegionKind::BrAnon,
                BoundRegionKind::BrNamed(def, symbol) => rustc_ty::BoundRegionKind::BrNamed(
                    def.0.internal(tables),
                    Symbol::intern(symbol),
                ),
                BoundRegionKind::BrEnv => rustc_ty::BoundRegionKind::BrEnv,
            }),
            BoundVariableKind::Const => rustc_ty::BoundVariableKind::Const,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for DynKind {
    type T = rustc_ty::DynKind;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            DynKind::Dyn => rustc_ty::DynKind::Dyn,
            DynKind::DynStar => rustc_ty::DynKind::DynStar,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for ExistentialPredicate {
    type T = rustc_ty::ExistentialPredicate<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            ExistentialPredicate::Trait(trait_ref) => {
                rustc_ty::ExistentialPredicate::Trait(trait_ref.internal(tables))
            }
            ExistentialPredicate::Projection(proj) => {
                rustc_ty::ExistentialPredicate::Projection(proj.internal(tables))
            }
            ExistentialPredicate::AutoTrait(trait_def) => {
                rustc_ty::ExistentialPredicate::AutoTrait(trait_def.0.internal(tables))
            }
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for ExistentialProjection {
    type T = rustc_ty::ExistentialProjection<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::ExistentialProjection {
            def_id: self.def_id.0.internal(tables),
            args: self.generic_args.internal(tables),
            term: self.term.internal(tables),
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for TermKind {
    type T = rustc_ty::Term<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            TermKind::Type(ty) => ty.internal(tables).into(),
            TermKind::Const(const_) => ty_const(const_, tables).into(),
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for ExistentialTraitRef {
    type T = rustc_ty::ExistentialTraitRef<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::ExistentialTraitRef {
            def_id: self.def_id.0.internal(tables),
            args: self.generic_args.internal(tables),
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for TraitRef {
    type T = rustc_ty::TraitRef<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        rustc_ty::TraitRef::new(
            tables.tcx,
            self.def_id.0.internal(tables),
            self.args().internal(tables),
        )
    }
}

impl<'tcx> RustcInternal<'tcx> for AllocId {
    type T = rustc_middle::mir::interpret::AllocId;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.alloc_ids[*self]
    }
}

impl<'tcx> RustcInternal<'tcx> for ClosureKind {
    type T = rustc_ty::ClosureKind;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            ClosureKind::Fn => rustc_ty::ClosureKind::Fn,
            ClosureKind::FnMut => rustc_ty::ClosureKind::FnMut,
            ClosureKind::FnOnce => rustc_ty::ClosureKind::FnOnce,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for AdtDef {
    type T = rustc_ty::AdtDef<'tcx>;
    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.tcx.adt_def(self.0.internal(&mut *tables))
    }
}

impl<'tcx> RustcInternal<'tcx> for Abi {
    type T = rustc_target::spec::abi::Abi;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match *self {
            Abi::Rust => rustc_target::spec::abi::Abi::Rust,
            Abi::C { unwind } => rustc_target::spec::abi::Abi::C { unwind },
            Abi::Cdecl { unwind } => rustc_target::spec::abi::Abi::Cdecl { unwind },
            Abi::Stdcall { unwind } => rustc_target::spec::abi::Abi::Stdcall { unwind },
            Abi::Fastcall { unwind } => rustc_target::spec::abi::Abi::Fastcall { unwind },
            Abi::Vectorcall { unwind } => rustc_target::spec::abi::Abi::Vectorcall { unwind },
            Abi::Thiscall { unwind } => rustc_target::spec::abi::Abi::Thiscall { unwind },
            Abi::Aapcs { unwind } => rustc_target::spec::abi::Abi::Aapcs { unwind },
            Abi::Win64 { unwind } => rustc_target::spec::abi::Abi::Win64 { unwind },
            Abi::SysV64 { unwind } => rustc_target::spec::abi::Abi::SysV64 { unwind },
            Abi::PtxKernel => rustc_target::spec::abi::Abi::PtxKernel,
            Abi::Msp430Interrupt => rustc_target::spec::abi::Abi::Msp430Interrupt,
            Abi::X86Interrupt => rustc_target::spec::abi::Abi::X86Interrupt,
            Abi::AmdGpuKernel => rustc_target::spec::abi::Abi::AmdGpuKernel,
            Abi::EfiApi => rustc_target::spec::abi::Abi::EfiApi,
            Abi::AvrInterrupt => rustc_target::spec::abi::Abi::AvrInterrupt,
            Abi::AvrNonBlockingInterrupt => rustc_target::spec::abi::Abi::AvrNonBlockingInterrupt,
            Abi::CCmseNonSecureCall => rustc_target::spec::abi::Abi::CCmseNonSecureCall,
            Abi::Wasm => rustc_target::spec::abi::Abi::Wasm,
            Abi::System { unwind } => rustc_target::spec::abi::Abi::System { unwind },
            Abi::RustIntrinsic => rustc_target::spec::abi::Abi::RustIntrinsic,
            Abi::RustCall => rustc_target::spec::abi::Abi::RustCall,
            Abi::PlatformIntrinsic => rustc_target::spec::abi::Abi::PlatformIntrinsic,
            Abi::Unadjusted => rustc_target::spec::abi::Abi::Unadjusted,
            Abi::RustCold => rustc_target::spec::abi::Abi::RustCold,
            Abi::RiscvInterruptM => rustc_target::spec::abi::Abi::RiscvInterruptM,
            Abi::RiscvInterruptS => rustc_target::spec::abi::Abi::RiscvInterruptS,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Safety {
    type T = rustc_hir::Unsafety;

    fn internal(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            Safety::Unsafe => rustc_hir::Unsafety::Unsafe,
            Safety::Normal => rustc_hir::Unsafety::Normal,
        }
    }
}

impl<'tcx> RustcInternal<'tcx> for Span {
    type T = rustc_span::Span;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables[*self]
    }
}

impl<'tcx> RustcInternal<'tcx> for Layout {
    type T = rustc_target::abi::Layout<'tcx>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.layouts[*self]
    }
}

impl<'tcx, T> RustcInternal<'tcx> for &T
where
    T: RustcInternal<'tcx>,
{
    type T = T::T;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        (*self).internal(tables)
    }
}

impl<'tcx, T> RustcInternal<'tcx> for Option<T>
where
    T: RustcInternal<'tcx>,
{
    type T = Option<T::T>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.as_ref().map(|inner| inner.internal(tables))
    }
}

impl<'tcx, T> RustcInternal<'tcx> for Vec<T>
where
    T: RustcInternal<'tcx>,
{
    type T = Vec<T::T>;

    fn internal(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.iter().map(|e| e.internal(tables)).collect()
    }
}
