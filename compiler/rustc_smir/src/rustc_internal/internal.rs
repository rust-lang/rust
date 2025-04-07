//! Module containing the translation from stable mir constructs to the rustc counterpart.
//!
//! This module will only include a few constructs to allow users to invoke internal rustc APIs
//! due to incomplete stable coverage.

// Prefer importing stable_mir over internal rustc constructs to make this file more readable.

use rustc_middle::ty::{self as rustc_ty, Const as InternalConst, Ty as InternalTy, TyCtxt};
use rustc_span::Symbol;
use stable_mir::abi::Layout;
use stable_mir::mir::alloc::AllocId;
use stable_mir::mir::mono::{Instance, MonoItem, StaticDef};
use stable_mir::mir::{BinOp, Mutability, Place, ProjectionElem, RawPtrKind, Safety, UnOp};
use stable_mir::ty::{
    Abi, AdtDef, Binder, BoundRegionKind, BoundTyKind, BoundVariableKind, ClosureKind, DynKind,
    ExistentialPredicate, ExistentialProjection, ExistentialTraitRef, FloatTy, FnSig,
    GenericArgKind, GenericArgs, IndexedVal, IntTy, MirConst, Movability, Pattern, Region, RigidTy,
    Span, TermKind, TraitRef, Ty, TyConst, UintTy, VariantDef, VariantIdx,
};
use stable_mir::{CrateItem, CrateNum, DefId};

use super::RustcInternal;
use crate::rustc_smir::Tables;
use crate::stable_mir;

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
            GenericArgKind::Const(cnst) => cnst.internal(tables, tcx).into(),
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

impl RustcInternal for TyConst {
    type T<'tcx> = InternalConst<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.ty_consts[self.id]).unwrap()
    }
}

impl RustcInternal for Pattern {
    type T<'tcx> = rustc_ty::Pattern<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.mk_pat(match self {
            Pattern::Range { start, end, include_end: _ } => rustc_ty::PatternKind::Range {
                start: start.as_ref().unwrap().internal(tables, tcx),
                end: end.as_ref().unwrap().internal(tables, tcx),
            },
        })
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
                rustc_ty::TyKind::Array(ty.internal(tables, tcx), cnst.internal(tables, tcx))
            }
            RigidTy::Pat(ty, pat) => {
                rustc_ty::TyKind::Pat(ty.internal(tables, tcx), pat.internal(tables, tcx))
            }
            RigidTy::Adt(def, args) => {
                rustc_ty::TyKind::Adt(def.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::Str => rustc_ty::TyKind::Str,
            RigidTy::Slice(ty) => rustc_ty::TyKind::Slice(ty.internal(tables, tcx)),
            RigidTy::RawPtr(ty, mutability) => {
                rustc_ty::TyKind::RawPtr(ty.internal(tables, tcx), mutability.internal(tables, tcx))
            }
            RigidTy::Ref(region, ty, mutability) => rustc_ty::TyKind::Ref(
                region.internal(tables, tcx),
                ty.internal(tables, tcx),
                mutability.internal(tables, tcx),
            ),
            RigidTy::Foreign(def) => rustc_ty::TyKind::Foreign(def.0.internal(tables, tcx)),
            RigidTy::FnDef(def, args) => {
                rustc_ty::TyKind::FnDef(def.0.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::FnPtr(sig) => {
                let (sig_tys, hdr) = sig.internal(tables, tcx).split();
                rustc_ty::TyKind::FnPtr(sig_tys, hdr)
            }
            RigidTy::Closure(def, args) => {
                rustc_ty::TyKind::Closure(def.0.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::Coroutine(def, args, _mov) => {
                rustc_ty::TyKind::Coroutine(def.0.internal(tables, tcx), args.internal(tables, tcx))
            }
            RigidTy::CoroutineClosure(def, args) => rustc_ty::TyKind::CoroutineClosure(
                def.0.internal(tables, tcx),
                args.internal(tables, tcx),
            ),
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
            FloatTy::F16 => rustc_ty::FloatTy::F16,
            FloatTy::F32 => rustc_ty::FloatTy::F32,
            FloatTy::F64 => rustc_ty::FloatTy::F64,
            FloatTy::F128 => rustc_ty::FloatTy::F128,
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

impl RustcInternal for RawPtrKind {
    type T<'tcx> = rustc_middle::mir::RawPtrKind;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            RawPtrKind::Mut => rustc_middle::mir::RawPtrKind::Mut,
            RawPtrKind::Const => rustc_middle::mir::RawPtrKind::Const,
            RawPtrKind::FakeForPtrMetadata => rustc_middle::mir::RawPtrKind::FakeForPtrMetadata,
        }
    }
}

impl RustcInternal for FnSig {
    type T<'tcx> = rustc_ty::FnSig<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(rustc_ty::FnSig {
            inputs_and_output: tcx.mk_type_list(&self.inputs_and_output.internal(tables, tcx)),
            c_variadic: self.c_variadic,
            safety: self.safety.internal(tables, tcx),
            abi: self.abi.internal(tables, tcx),
        })
        .unwrap()
    }
}

impl RustcInternal for VariantIdx {
    type T<'tcx> = rustc_abi::VariantIdx;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_abi::VariantIdx::from(self.to_index())
    }
}

impl RustcInternal for VariantDef {
    type T<'tcx> = &'tcx rustc_ty::VariantDef;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.adt_def.internal(tables, tcx).variant(self.idx.internal(tables, tcx))
    }
}

impl RustcInternal for MirConst {
    type T<'tcx> = rustc_middle::mir::Const<'tcx>;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        let constant = tables.mir_consts[self.id];
        match constant {
            rustc_middle::mir::Const::Ty(ty, ct) => {
                rustc_middle::mir::Const::Ty(tcx.lift(ty).unwrap(), tcx.lift(ct).unwrap())
            }
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
                BoundRegionKind::BrAnon => rustc_ty::BoundRegionKind::Anon,
                BoundRegionKind::BrNamed(def, symbol) => rustc_ty::BoundRegionKind::Named(
                    def.0.internal(tables, tcx),
                    Symbol::intern(symbol),
                ),
                BoundRegionKind::BrEnv => rustc_ty::BoundRegionKind::ClosureEnv,
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
        rustc_ty::ExistentialProjection::new_from_args(
            tcx,
            self.def_id.0.internal(tables, tcx),
            self.generic_args.internal(tables, tcx),
            self.term.internal(tables, tcx),
        )
    }
}

impl RustcInternal for TermKind {
    type T<'tcx> = rustc_ty::Term<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            TermKind::Type(ty) => ty.internal(tables, tcx).into(),
            TermKind::Const(cnst) => cnst.internal(tables, tcx).into(),
        }
    }
}

impl RustcInternal for ExistentialTraitRef {
    type T<'tcx> = rustc_ty::ExistentialTraitRef<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_ty::ExistentialTraitRef::new_from_args(
            tcx,
            self.def_id.0.internal(tables, tcx),
            self.generic_args.internal(tables, tcx),
        )
    }
}

impl RustcInternal for TraitRef {
    type T<'tcx> = rustc_ty::TraitRef<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_ty::TraitRef::new_from_args(
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
    type T<'tcx> = rustc_abi::ExternAbi;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match *self {
            Abi::Rust => rustc_abi::ExternAbi::Rust,
            Abi::C { unwind } => rustc_abi::ExternAbi::C { unwind },
            Abi::Cdecl { unwind } => rustc_abi::ExternAbi::Cdecl { unwind },
            Abi::Stdcall { unwind } => rustc_abi::ExternAbi::Stdcall { unwind },
            Abi::Fastcall { unwind } => rustc_abi::ExternAbi::Fastcall { unwind },
            Abi::Vectorcall { unwind } => rustc_abi::ExternAbi::Vectorcall { unwind },
            Abi::Thiscall { unwind } => rustc_abi::ExternAbi::Thiscall { unwind },
            Abi::Aapcs { unwind } => rustc_abi::ExternAbi::Aapcs { unwind },
            Abi::Win64 { unwind } => rustc_abi::ExternAbi::Win64 { unwind },
            Abi::SysV64 { unwind } => rustc_abi::ExternAbi::SysV64 { unwind },
            Abi::PtxKernel => rustc_abi::ExternAbi::PtxKernel,
            Abi::Msp430Interrupt => rustc_abi::ExternAbi::Msp430Interrupt,
            Abi::X86Interrupt => rustc_abi::ExternAbi::X86Interrupt,
            Abi::GpuKernel => rustc_abi::ExternAbi::GpuKernel,
            Abi::EfiApi => rustc_abi::ExternAbi::EfiApi,
            Abi::AvrInterrupt => rustc_abi::ExternAbi::AvrInterrupt,
            Abi::AvrNonBlockingInterrupt => rustc_abi::ExternAbi::AvrNonBlockingInterrupt,
            Abi::CCmseNonSecureCall => rustc_abi::ExternAbi::CCmseNonSecureCall,
            Abi::CCmseNonSecureEntry => rustc_abi::ExternAbi::CCmseNonSecureEntry,
            Abi::System { unwind } => rustc_abi::ExternAbi::System { unwind },
            Abi::RustCall => rustc_abi::ExternAbi::RustCall,
            Abi::Unadjusted => rustc_abi::ExternAbi::Unadjusted,
            Abi::RustCold => rustc_abi::ExternAbi::RustCold,
            Abi::RiscvInterruptM => rustc_abi::ExternAbi::RiscvInterruptM,
            Abi::RiscvInterruptS => rustc_abi::ExternAbi::RiscvInterruptS,
        }
    }
}

impl RustcInternal for Safety {
    type T<'tcx> = rustc_hir::Safety;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Safety::Unsafe => rustc_hir::Safety::Unsafe,
            Safety::Safe => rustc_hir::Safety::Safe,
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
    type T<'tcx> = rustc_abi::Layout<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.layouts[*self]).unwrap()
    }
}

impl RustcInternal for Place {
    type T<'tcx> = rustc_middle::mir::Place<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::Place {
            local: rustc_middle::mir::Local::from_usize(self.local),
            projection: tcx.mk_place_elems(&self.projection.internal(tables, tcx)),
        }
    }
}

impl RustcInternal for ProjectionElem {
    type T<'tcx> = rustc_middle::mir::PlaceElem<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            ProjectionElem::Deref => rustc_middle::mir::PlaceElem::Deref,
            ProjectionElem::Field(idx, ty) => {
                rustc_middle::mir::PlaceElem::Field((*idx).into(), ty.internal(tables, tcx))
            }
            ProjectionElem::Index(idx) => rustc_middle::mir::PlaceElem::Index((*idx).into()),
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                rustc_middle::mir::PlaceElem::ConstantIndex {
                    offset: *offset,
                    min_length: *min_length,
                    from_end: *from_end,
                }
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                rustc_middle::mir::PlaceElem::Subslice { from: *from, to: *to, from_end: *from_end }
            }
            ProjectionElem::Downcast(idx) => {
                rustc_middle::mir::PlaceElem::Downcast(None, idx.internal(tables, tcx))
            }
            ProjectionElem::OpaqueCast(ty) => {
                rustc_middle::mir::PlaceElem::OpaqueCast(ty.internal(tables, tcx))
            }
            ProjectionElem::Subtype(ty) => {
                rustc_middle::mir::PlaceElem::Subtype(ty.internal(tables, tcx))
            }
        }
    }
}

impl RustcInternal for BinOp {
    type T<'tcx> = rustc_middle::mir::BinOp;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            BinOp::Add => rustc_middle::mir::BinOp::Add,
            BinOp::AddUnchecked => rustc_middle::mir::BinOp::AddUnchecked,
            BinOp::Sub => rustc_middle::mir::BinOp::Sub,
            BinOp::SubUnchecked => rustc_middle::mir::BinOp::SubUnchecked,
            BinOp::Mul => rustc_middle::mir::BinOp::Mul,
            BinOp::MulUnchecked => rustc_middle::mir::BinOp::MulUnchecked,
            BinOp::Div => rustc_middle::mir::BinOp::Div,
            BinOp::Rem => rustc_middle::mir::BinOp::Rem,
            BinOp::BitXor => rustc_middle::mir::BinOp::BitXor,
            BinOp::BitAnd => rustc_middle::mir::BinOp::BitAnd,
            BinOp::BitOr => rustc_middle::mir::BinOp::BitOr,
            BinOp::Shl => rustc_middle::mir::BinOp::Shl,
            BinOp::ShlUnchecked => rustc_middle::mir::BinOp::ShlUnchecked,
            BinOp::Shr => rustc_middle::mir::BinOp::Shr,
            BinOp::ShrUnchecked => rustc_middle::mir::BinOp::ShrUnchecked,
            BinOp::Eq => rustc_middle::mir::BinOp::Eq,
            BinOp::Lt => rustc_middle::mir::BinOp::Lt,
            BinOp::Le => rustc_middle::mir::BinOp::Le,
            BinOp::Ne => rustc_middle::mir::BinOp::Ne,
            BinOp::Ge => rustc_middle::mir::BinOp::Ge,
            BinOp::Gt => rustc_middle::mir::BinOp::Gt,
            BinOp::Cmp => rustc_middle::mir::BinOp::Cmp,
            BinOp::Offset => rustc_middle::mir::BinOp::Offset,
        }
    }
}

impl RustcInternal for UnOp {
    type T<'tcx> = rustc_middle::mir::UnOp;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            UnOp::Not => rustc_middle::mir::UnOp::Not,
            UnOp::Neg => rustc_middle::mir::UnOp::Neg,
            UnOp::PtrMetadata => rustc_middle::mir::UnOp::PtrMetadata,
        }
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
