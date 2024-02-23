//! Module containing the translation from stable mir constructs to the rustc counterpart.
//!
//! This module will only include a few constructs to allow users to invoke internal rustc APIs
//! due to incomplete stable coverage.

// Prefer importing stable_mir over internal rustc constructs to make this file more readable.
use crate::rustc_smir::Tables;
use rustc_middle::ty::{self as rustc_ty, Ty as InternalTy, TyCtxt};
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
use stable_mir::{CrateItem, CrateNum, DefId};

use super::RustcInternal;

impl RustcInternal for CrateItem {
    type T<'tcx> = rustc_span::def_id::DefId;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.0.internal(tables, tcx)
    }
}

impl RustcInternal for CrateNum {
    type T<'tcx> = rustc_span::def_id::CrateNum;
    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_span::def_id::CrateNum::from_usize(*self)
    }
}

impl RustcInternal for DefId {
    type T<'tcx> = rustc_span::def_id::DefId;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.def_ids[*self]).unwrap()
    }
}

impl RustcInternal for GenericArgs {
    type T<'tcx> = rustc_ty::GenericArgsRef<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.mk_args_from_iter(self.0.iter().map(|arg| arg.internal(tables, tcx)))
    }
}

impl RustcInternal for GenericArgKind {
    type T<'tcx> = rustc_ty::GenericArg<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        let arg: rustc_ty::GenericArg<'tcx> = match self {
            GenericArgKind::Lifetime(reg) => reg.internal(tables, tcx).into(),
            GenericArgKind::Type(ty) => ty.internal(tables, tcx).into(),
            GenericArgKind::Const(cnst) => ty_const(cnst, tables, tcx).into(),
        };
        tcx.lift(arg).unwrap()
    }
}

impl RustcInternal for Region {
    type T<'tcx> = rustc_ty::Region<'tcx>;
    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        // Cannot recover region. Use erased for now.
        tcx.lifetimes.re_erased
    }
}

impl RustcInternal for Ty {
    type T<'tcx> = InternalTy<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.types[*self]).unwrap()
    }
}

impl RustcInternal for RigidTy {
    type T<'tcx> = rustc_ty::TyKind<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            RigidTy::Bool => rustc_ty::TyKind::Bool,
            RigidTy::Char => rustc_ty::TyKind::Char,
            RigidTy::Int(int_ty) => rustc_ty::TyKind::Int(int_ty.internal(tables, tcx)),
            RigidTy::Uint(uint_ty) => rustc_ty::TyKind::Uint(uint_ty.internal(tables, tcx)),
            RigidTy::Float(float_ty) => rustc_ty::TyKind::Float(float_ty.internal(tables, tcx)),
            RigidTy::Never => rustc_ty::TyKind::Never,
            RigidTy::Array(ty, cnst) => {
                rustc_ty::TyKind::Array(ty.internal(tables, tcx), ty_const(cnst, tables, tcx))
            }
            RigidTy::Adt(def, args) => {
                rustc_ty::TyKind::Adt(def.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::Str => rustc_ty::TyKind::Str,
            RigidTy::Slice(ty) => rustc_ty::TyKind::Slice(ty.internal(tables, tcx)),
            RigidTy::RawPtr(ty, mutability) => rustc_ty::TyKind::RawPtr(rustc_ty::TypeAndMut {
                ty: ty.internal(tables, tcx),
                mutbl: mutability.internal(tables, tcx),
            }),
            RigidTy::Ref(region, ty, mutability) => rustc_ty::TyKind::Ref(
                region.internal(tables, tcx),
                ty.internal(tables, tcx),
                mutability.internal(tables, tcx),
            ),
            RigidTy::Foreign(def) => rustc_ty::TyKind::Foreign(def.0.internal(tables, tcx)),
            RigidTy::FnDef(def, args) => {
                rustc_ty::TyKind::FnDef(def.0.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::FnPtr(sig) => rustc_ty::TyKind::FnPtr(sig.internal(tables, tcx)),
            RigidTy::Closure(def, args) => {
                rustc_ty::TyKind::Closure(def.0.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::Coroutine(def, args, _mov) => {
                rustc_ty::TyKind::Coroutine(def.0.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::CoroutineWitness(def, args) => rustc_ty::TyKind::CoroutineWitness(
                def.0.internal(tables, tcx),
                args.internal(tables, tcx),
            ),
            RigidTy::Dynamic(predicate, region, dyn_kind) => rustc_ty::TyKind::Dynamic(
                tcx.mk_poly_existential_predicates(&predicate.internal(tables, tcx)),
                region.internal(tables, tcx),
                dyn_kind.internal(tables, tcx),
            ),
            RigidTy::Tuple(tys) => {
                rustc_ty::TyKind::Tuple(tcx.mk_type_list(&tys.internal(tables, tcx)))
            }
        }
    }
}

impl RustcInternal for IntTy {
    type T<'tcx> = rustc_ty::IntTy;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
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

impl RustcInternal for UintTy {
    type T<'tcx> = rustc_ty::UintTy;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
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

impl RustcInternal for FloatTy {
    type T<'tcx> = rustc_ty::FloatTy;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            FloatTy::F32 => rustc_ty::FloatTy::F32,
            FloatTy::F64 => rustc_ty::FloatTy::F64,
        }
    }
}

impl RustcInternal for Mutability {
    type T<'tcx> = rustc_ty::Mutability;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Mutability::Not => rustc_ty::Mutability::Not,
            Mutability::Mut => rustc_ty::Mutability::Mut,
        }
    }
}

impl RustcInternal for Movability {
    type T<'tcx> = rustc_ty::Movability;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Movability::Static => rustc_ty::Movability::Static,
            Movability::Movable => rustc_ty::Movability::Movable,
        }
    }
}

impl RustcInternal for FnSig {
    type T<'tcx> = rustc_ty::FnSig<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(rustc_ty::FnSig {
            inputs_and_output: tcx.mk_type_list(&self.inputs_and_output.internal(tables, tcx)),
            c_variadic: self.c_variadic,
            unsafety: self.unsafety.internal(tables, tcx),
            abi: self.abi.internal(tables, tcx),
        })
        .unwrap()
    }
}

impl RustcInternal for VariantIdx {
    type T<'tcx> = rustc_target::abi::VariantIdx;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_target::abi::VariantIdx::from(self.to_index())
    }
}

impl RustcInternal for VariantDef {
    type T<'tcx> = &'tcx rustc_ty::VariantDef;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.adt_def.internal(tables, tcx).variant(self.idx.internal(tables, tcx))
    }
}

fn ty_const<'tcx>(
    constant: &Const,
    tables: &mut Tables<'_>,
    tcx: TyCtxt<'tcx>,
) -> rustc_ty::Const<'tcx> {
    match constant.internal(tables, tcx) {
        rustc_middle::mir::Const::Ty(c) => c,
        cnst => {
            panic!("Trying to convert constant `{constant:?}` to type constant, but found {cnst:?}")
        }
    }
}

impl RustcInternal for Const {
    type T<'tcx> = rustc_middle::mir::Const<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        let constant = tables.constants[self.id];
        match constant {
            rustc_middle::mir::Const::Ty(ty) => rustc_middle::mir::Const::Ty(tcx.lift(ty).unwrap()),
            rustc_middle::mir::Const::Unevaluated(uneval, ty) => {
                rustc_middle::mir::Const::Unevaluated(
                    tcx.lift(uneval).unwrap(),
                    tcx.lift(ty).unwrap(),
                )
            }
            rustc_middle::mir::Const::Val(const_val, ty) => {
                rustc_middle::mir::Const::Val(tcx.lift(const_val).unwrap(), tcx.lift(ty).unwrap())
            }
        }
    }
}

impl RustcInternal for MonoItem {
    type T<'tcx> = rustc_middle::mir::mono::MonoItem<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        use rustc_middle::mir::mono as rustc_mono;
        match self {
            MonoItem::Fn(instance) => rustc_mono::MonoItem::Fn(instance.internal(tables, tcx)),
            MonoItem::Static(def) => rustc_mono::MonoItem::Static(def.internal(tables, tcx)),
            MonoItem::GlobalAsm(_) => {
                unimplemented!()
            }
        }
    }
}

impl RustcInternal for Instance {
    type T<'tcx> = rustc_ty::Instance<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.instances[self.def]).unwrap()
    }
}

impl RustcInternal for StaticDef {
    type T<'tcx> = rustc_span::def_id::DefId;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.0.internal(tables, tcx)
    }
}

#[allow(rustc::usage_of_qualified_ty)]
impl<T> RustcInternal for Binder<T>
where
    T: RustcInternal,
    for<'tcx> T::T<'tcx>: rustc_ty::TypeVisitable<rustc_ty::TyCtxt<'tcx>>,
{
    type T<'tcx> = rustc_ty::Binder<'tcx, T::T<'tcx>>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_ty::Binder::bind_with_vars(
            self.value.internal(tables, tcx),
            tcx.mk_bound_variable_kinds_from_iter(
                self.bound_vars.iter().map(|bound| bound.internal(tables, tcx)),
            ),
        )
    }
}

impl RustcInternal for BoundVariableKind {
    type T<'tcx> = rustc_ty::BoundVariableKind;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            BoundVariableKind::Ty(kind) => rustc_ty::BoundVariableKind::Ty(match kind {
                BoundTyKind::Anon => rustc_ty::BoundTyKind::Anon,
                BoundTyKind::Param(def, symbol) => rustc_ty::BoundTyKind::Param(
                    def.0.internal(tables, tcx),
                    Symbol::intern(symbol),
                ),
            }),
            BoundVariableKind::Region(kind) => rustc_ty::BoundVariableKind::Region(match kind {
                BoundRegionKind::BrAnon => rustc_ty::BoundRegionKind::BrAnon,
                BoundRegionKind::BrNamed(def, symbol) => rustc_ty::BoundRegionKind::BrNamed(
                    def.0.internal(tables, tcx),
                    Symbol::intern(symbol),
                ),
                BoundRegionKind::BrEnv => rustc_ty::BoundRegionKind::BrEnv,
            }),
            BoundVariableKind::Const => rustc_ty::BoundVariableKind::Const,
        }
    }
}

impl RustcInternal for DynKind {
    type T<'tcx> = rustc_ty::DynKind;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            DynKind::Dyn => rustc_ty::DynKind::Dyn,
            DynKind::DynStar => rustc_ty::DynKind::DynStar,
        }
    }
}

impl RustcInternal for ExistentialPredicate {
    type T<'tcx> = rustc_ty::ExistentialPredicate<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            ExistentialPredicate::Trait(trait_ref) => {
                rustc_ty::ExistentialPredicate::Trait(trait_ref.internal(tables, tcx))
            }
            ExistentialPredicate::Projection(proj) => {
                rustc_ty::ExistentialPredicate::Projection(proj.internal(tables, tcx))
            }
            ExistentialPredicate::AutoTrait(trait_def) => {
                rustc_ty::ExistentialPredicate::AutoTrait(trait_def.0.internal(tables, tcx))
            }
        }
    }
}

impl RustcInternal for ExistentialProjection {
    type T<'tcx> = rustc_ty::ExistentialProjection<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_ty::ExistentialProjection {
            def_id: self.def_id.0.internal(tables, tcx),
            args: self.generic_args.internal(tables, tcx),
            term: self.term.internal(tables, tcx),
        }
    }
}

impl RustcInternal for TermKind {
    type T<'tcx> = rustc_ty::Term<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            TermKind::Type(ty) => ty.internal(tables, tcx).into(),
            TermKind::Const(const_) => ty_const(const_, tables, tcx).into(),
        }
    }
}

impl RustcInternal for ExistentialTraitRef {
    type T<'tcx> = rustc_ty::ExistentialTraitRef<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_ty::ExistentialTraitRef {
            def_id: self.def_id.0.internal(tables, tcx),
            args: self.generic_args.internal(tables, tcx),
        }
    }
}

impl RustcInternal for TraitRef {
    type T<'tcx> = rustc_ty::TraitRef<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_ty::TraitRef::new(
            tcx,
            self.def_id.0.internal(tables, tcx),
            self.args().internal(tables, tcx),
        )
    }
}

impl RustcInternal for AllocId {
    type T<'tcx> = rustc_middle::mir::interpret::AllocId;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.alloc_ids[*self]).unwrap()
    }
}

impl RustcInternal for ClosureKind {
    type T<'tcx> = rustc_ty::ClosureKind;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            ClosureKind::Fn => rustc_ty::ClosureKind::Fn,
            ClosureKind::FnMut => rustc_ty::ClosureKind::FnMut,
            ClosureKind::FnOnce => rustc_ty::ClosureKind::FnOnce,
        }
    }
}

impl RustcInternal for AdtDef {
    type T<'tcx> = rustc_ty::AdtDef<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.adt_def(self.0.internal(tables, tcx))
    }
}

impl RustcInternal for Abi {
    type T<'tcx> = rustc_target::spec::abi::Abi;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
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
            Abi::EfiApi => rustc_target::spec::abi::Abi::EfiApi,
            Abi::AvrInterrupt => rustc_target::spec::abi::Abi::AvrInterrupt,
            Abi::AvrNonBlockingInterrupt => rustc_target::spec::abi::Abi::AvrNonBlockingInterrupt,
            Abi::CCmseNonSecureCall => rustc_target::spec::abi::Abi::CCmseNonSecureCall,
            Abi::Wasm => rustc_target::spec::abi::Abi::Wasm,
            Abi::System { unwind } => rustc_target::spec::abi::Abi::System { unwind },
            Abi::RustIntrinsic => rustc_target::spec::abi::Abi::RustIntrinsic,
            Abi::RustCall => rustc_target::spec::abi::Abi::RustCall,
            Abi::Unadjusted => rustc_target::spec::abi::Abi::Unadjusted,
            Abi::RustCold => rustc_target::spec::abi::Abi::RustCold,
            Abi::RiscvInterruptM => rustc_target::spec::abi::Abi::RiscvInterruptM,
            Abi::RiscvInterruptS => rustc_target::spec::abi::Abi::RiscvInterruptS,
        }
    }
}

impl RustcInternal for Safety {
    type T<'tcx> = rustc_hir::Unsafety;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Safety::Unsafe => rustc_hir::Unsafety::Unsafe,
            Safety::Normal => rustc_hir::Unsafety::Normal,
        }
    }
}

impl RustcInternal for Span {
    type T<'tcx> = rustc_span::Span;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tables[*self]
    }
}

impl RustcInternal for Layout {
    type T<'tcx> = rustc_target::abi::Layout<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.layouts[*self]).unwrap()
    }
}

impl<T> RustcInternal for &T
where
    T: RustcInternal,
{
    type T<'tcx> = T::T<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        (*self).internal(tables, tcx)
    }
}

impl<T> RustcInternal for Option<T>
where
    T: RustcInternal,
{
    type T<'tcx> = Option<T::T<'tcx>>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.as_ref().map(|inner| inner.internal(tables, tcx))
    }
}

impl<T> RustcInternal for Vec<T>
where
    T: RustcInternal,
{
    type T<'tcx> = Vec<T::T<'tcx>>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.iter().map(|e| e.internal(tables, tcx)).collect()
    }
}
