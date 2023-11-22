//! Module containing the translation from stable mir constructs to the rustc counterpart.
//!
//! This module will only include a few constructs to allow users to invoke internal rustc APIs
//! due to incomplete stable coverage.

// Prefer importing stable_mir over internal rustc constructs to make this file more readable.
use crate::rustc_smir::Tables;
use rustc_middle::ty::{self as rustc_ty, Ty as InternalTy};
use rustc_span::Symbol;
use stable_mir::mir::alloc::AllocId;
use stable_mir::mir::mono::{Instance, MonoItem, StaticDef};
use stable_mir::ty::{
    AdtDef, Binder, BoundRegionKind, BoundTyKind, BoundVariableKind, ClosureKind, Const,
    ExistentialTraitRef, FloatTy, GenericArgKind, GenericArgs, IntTy, Region, RigidTy, TraitRef,
    Ty, UintTy,
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
            RigidTy::RawPtr(..)
            | RigidTy::Ref(..)
            | RigidTy::Foreign(_)
            | RigidTy::FnDef(_, _)
            | RigidTy::FnPtr(_)
            | RigidTy::Closure(..)
            | RigidTy::Coroutine(..)
            | RigidTy::CoroutineWitness(..)
            | RigidTy::Dynamic(..)
            | RigidTy::Tuple(..) => {
                todo!()
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
