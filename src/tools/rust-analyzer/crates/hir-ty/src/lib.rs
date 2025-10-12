//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

// FIXME: We used to import `rustc_*` deps from `rustc_private` with `feature = "in-rust-tree" but
// temporarily switched to crates.io versions due to hardships that working on them from rustc
// demands corresponding changes on rust-analyzer at the same time.
// For details, see the zulip discussion below:
// https://rust-lang.zulipchat.com/#narrow/channel/185405-t-compiler.2Frust-analyzer/topic/relying.20on.20in-tree.20.60rustc_type_ir.60.2F.60rustc_next_trait_solver.60/with/541055689

extern crate ra_ap_rustc_index as rustc_index;

extern crate ra_ap_rustc_abi as rustc_abi;

extern crate ra_ap_rustc_pattern_analysis as rustc_pattern_analysis;

extern crate ra_ap_rustc_ast_ir as rustc_ast_ir;

extern crate ra_ap_rustc_type_ir as rustc_type_ir;

extern crate ra_ap_rustc_next_trait_solver as rustc_next_trait_solver;

extern crate self as hir_ty;

mod builder;
mod chalk_db;
mod chalk_ext;
mod infer;
mod inhabitedness;
mod interner;
mod lower;
mod lower_nextsolver;
mod mapping;
pub mod next_solver;
mod target_feature;
mod tls;
mod utils;

pub mod autoderef;
pub mod consteval;
mod consteval_chalk;
pub mod db;
pub mod diagnostics;
pub mod display;
pub mod drop;
pub mod dyn_compatibility;
pub mod generics;
pub mod lang_items;
pub mod layout;
pub mod method_resolution;
pub mod mir;
pub mod primitive;
pub mod traits;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod tests;
mod variance;

use std::hash::Hash;

use chalk_ir::{
    VariableKinds,
    fold::{Shift, TypeFoldable},
    interner::HasInterner,
};
use hir_def::{CallableDefId, GeneralConstId, TypeOrConstParamId, hir::ExprId, type_ref::Rawness};
use hir_expand::name::Name;
use indexmap::{IndexMap, map::Entry};
use intern::{Symbol, sym};
use la_arena::{Arena, Idx};
use mir::{MirEvalError, VTableMap};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use rustc_type_ir::{
    TypeSuperVisitable, TypeVisitableExt, UpcastFrom,
    inherent::{IntoKind, SliceLike, Ty as _},
};
use syntax::ast::{ConstArg, make};
use traits::FnTrait;
use triomphe::Arc;

#[cfg(not(debug_assertions))]
use crate::next_solver::ErrorGuaranteed;
use crate::{
    db::HirDatabase,
    display::{DisplayTarget, HirDisplay},
    generics::Generics,
    infer::unify::InferenceTable,
    next_solver::{
        DbInterner,
        mapping::{ChalkToNextSolver, NextSolverToChalk, convert_ty_for_result},
    },
};

pub use autoderef::autoderef;
pub use builder::{ParamKind, TyBuilder};
pub use chalk_ext::*;
pub use infer::{
    Adjust, Adjustment, AutoBorrow, BindingMode, InferenceDiagnostic, InferenceResult,
    InferenceTyDiagnosticSource, OverloadedDeref, PointerCast,
    cast::CastError,
    closure::analysis::{CaptureKind, CapturedItem},
    could_coerce, could_unify, could_unify_deeply,
};
pub use interner::Interner;
pub use lower::{ImplTraitLoweringMode, ParamLoweringMode, TyDefId, ValueTyDefId, diagnostics::*};
pub use lower_nextsolver::{
    LifetimeElisionKind, TyLoweringContext, associated_type_shorthand_candidates,
};
pub use mapping::{
    ToChalk, from_assoc_type_id, from_chalk_trait_id, from_foreign_def_id, from_placeholder_idx,
    lt_from_placeholder_idx, lt_to_placeholder_idx, to_assoc_type_id, to_chalk_trait_id,
    to_foreign_def_id, to_placeholder_idx, to_placeholder_idx_no_index,
};
pub use method_resolution::check_orphan_rules;
pub use next_solver::interner::{attach_db, attach_db_allow_change, with_attached_db};
pub use target_feature::TargetFeatures;
pub use traits::TraitEnvironment;
pub use utils::{
    TargetFeatureIsSafeInTarget, Unsafety, all_super_traits, direct_super_traits,
    is_fn_unsafe_to_call, target_feature_is_safe_in_target,
};
pub use variance::Variance;

use chalk_ir::{AdtId, BoundVar, DebruijnIndex, Safety, Scalar};

pub(crate) type ForeignDefId = chalk_ir::ForeignDefId<Interner>;
pub(crate) type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub(crate) type FnDefId = chalk_ir::FnDefId<Interner>;
pub(crate) type ClosureId = chalk_ir::ClosureId<Interner>;
pub(crate) type OpaqueTyId = chalk_ir::OpaqueTyId<Interner>;
pub(crate) type PlaceholderIndex = chalk_ir::PlaceholderIndex;

pub(crate) type CanonicalVarKinds = chalk_ir::CanonicalVarKinds<Interner>;

pub(crate) type VariableKind = chalk_ir::VariableKind<Interner>;
/// Represents generic parameters and an item bound by them. When the item has parent, the binders
/// also contain the generic parameters for its parent. See chalk's documentation for details.
///
/// One thing to keep in mind when working with `Binders` (and `Substitution`s, which represent
/// generic arguments) in rust-analyzer is that the ordering within *is* significant - the generic
/// parameters/arguments for an item MUST come before those for its parent. This is to facilitate
/// the integration with chalk-solve, which mildly puts constraints as such. See #13335 for its
/// motivation in detail.
pub(crate) type Binders<T> = chalk_ir::Binders<T>;
/// Interned list of generic arguments for an item. When an item has parent, the `Substitution` for
/// it contains generic arguments for both its parent and itself. See chalk's documentation for
/// details.
///
/// See `Binders` for the constraint on the ordering.
pub(crate) type Substitution = chalk_ir::Substitution<Interner>;
pub(crate) type GenericArg = chalk_ir::GenericArg<Interner>;
pub(crate) type GenericArgData = chalk_ir::GenericArgData<Interner>;

pub(crate) type Ty = chalk_ir::Ty<Interner>;
pub type TyKind = chalk_ir::TyKind<Interner>;
pub(crate) type TypeFlags = chalk_ir::TypeFlags;
pub(crate) type DynTy = chalk_ir::DynTy<Interner>;
pub(crate) type FnPointer = chalk_ir::FnPointer<Interner>;
pub(crate) use chalk_ir::FnSubst; // a re-export so we don't lose the tuple constructor

pub type AliasTy = chalk_ir::AliasTy<Interner>;

pub(crate) type ProjectionTy = chalk_ir::ProjectionTy<Interner>;
pub(crate) type OpaqueTy = chalk_ir::OpaqueTy<Interner>;

pub(crate) type Lifetime = chalk_ir::Lifetime<Interner>;
pub(crate) type LifetimeData = chalk_ir::LifetimeData<Interner>;
pub(crate) type LifetimeOutlives = chalk_ir::LifetimeOutlives<Interner>;

pub(crate) type ConstValue = chalk_ir::ConstValue<Interner>;

pub(crate) type Const = chalk_ir::Const<Interner>;
pub(crate) type ConstData = chalk_ir::ConstData<Interner>;
pub(crate) type ConcreteConst = chalk_ir::ConcreteConst<Interner>;

pub(crate) type TraitRef = chalk_ir::TraitRef<Interner>;
pub(crate) type QuantifiedWhereClause = Binders<WhereClause>;
pub(crate) type Canonical<T> = chalk_ir::Canonical<T>;

pub(crate) type ChalkTraitId = chalk_ir::TraitId<Interner>;
pub(crate) type QuantifiedWhereClauses = chalk_ir::QuantifiedWhereClauses<Interner>;

pub(crate) type FnSig = chalk_ir::FnSig<Interner>;

pub(crate) type InEnvironment<T> = chalk_ir::InEnvironment<T>;
pub type AliasEq = chalk_ir::AliasEq<Interner>;
pub type WhereClause = chalk_ir::WhereClause<Interner>;

pub(crate) type DomainGoal = chalk_ir::DomainGoal<Interner>;
pub(crate) type Goal = chalk_ir::Goal<Interner>;

pub(crate) type CanonicalVarKind = chalk_ir::CanonicalVarKind<Interner>;
pub(crate) type GoalData = chalk_ir::GoalData<Interner>;
pub(crate) type ProgramClause = chalk_ir::ProgramClause<Interner>;

/// A constant can have reference to other things. Memory map job is holding
/// the necessary bits of memory of the const eval session to keep the constant
/// meaningful.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum MemoryMap<'db> {
    #[default]
    Empty,
    Simple(Box<[u8]>),
    Complex(Box<ComplexMemoryMap<'db>>),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ComplexMemoryMap<'db> {
    memory: IndexMap<usize, Box<[u8]>, FxBuildHasher>,
    vtable: VTableMap<'db>,
}

impl ComplexMemoryMap<'_> {
    fn insert(&mut self, addr: usize, val: Box<[u8]>) {
        match self.memory.entry(addr) {
            Entry::Occupied(mut e) => {
                if e.get().len() < val.len() {
                    e.insert(val);
                }
            }
            Entry::Vacant(e) => {
                e.insert(val);
            }
        }
    }
}

impl<'db> MemoryMap<'db> {
    pub fn vtable_ty(&self, id: usize) -> Result<crate::next_solver::Ty<'db>, MirEvalError<'db>> {
        match self {
            MemoryMap::Empty | MemoryMap::Simple(_) => Err(MirEvalError::InvalidVTableId(id)),
            MemoryMap::Complex(cm) => cm.vtable.ty(id),
        }
    }

    fn simple(v: Box<[u8]>) -> Self {
        MemoryMap::Simple(v)
    }

    /// This functions convert each address by a function `f` which gets the byte intervals and assign an address
    /// to them. It is useful when you want to load a constant with a memory map in a new memory. You can pass an
    /// allocator function as `f` and it will return a mapping of old addresses to new addresses.
    fn transform_addresses(
        &self,
        mut f: impl FnMut(&[u8], usize) -> Result<usize, MirEvalError<'db>>,
    ) -> Result<FxHashMap<usize, usize>, MirEvalError<'db>> {
        let mut transform = |(addr, val): (&usize, &[u8])| {
            let addr = *addr;
            let align = if addr == 0 { 64 } else { (addr - (addr & (addr - 1))).min(64) };
            f(val, align).map(|it| (addr, it))
        };
        match self {
            MemoryMap::Empty => Ok(Default::default()),
            MemoryMap::Simple(m) => transform((&0, m)).map(|(addr, val)| {
                let mut map = FxHashMap::with_capacity_and_hasher(1, rustc_hash::FxBuildHasher);
                map.insert(addr, val);
                map
            }),
            MemoryMap::Complex(cm) => {
                cm.memory.iter().map(|(addr, val)| transform((addr, val))).collect()
            }
        }
    }

    fn get(&self, addr: usize, size: usize) -> Option<&[u8]> {
        if size == 0 {
            Some(&[])
        } else {
            match self {
                MemoryMap::Empty => Some(&[]),
                MemoryMap::Simple(m) if addr == 0 => m.get(0..size),
                MemoryMap::Simple(_) => None,
                MemoryMap::Complex(cm) => cm.memory.get(&addr)?.get(0..size),
            }
        }
    }
}

// FIXME(next-solver): add a lifetime to this
/// A concrete constant value
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstScalar {
    Bytes(Box<[u8]>, MemoryMap<'static>),
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    UnevaluatedConst(GeneralConstId, Substitution),
    /// Case of an unknown value that rustc might know but we don't
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    // https://github.com/rust-lang/rust-analyzer/pull/8813#issuecomment-840679177
    // https://rust-lang.zulipchat.com/#narrow/stream/144729-wg-traits/topic/Handling.20non.20evaluatable.20constants'.20equality/near/238386348
    Unknown,
}

impl Hash for ConstScalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        if let ConstScalar::Bytes(b, _) = self {
            b.hash(state)
        }
    }
}

/// A concrete constant value
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstScalarNs<'db> {
    Bytes(Box<[u8]>, MemoryMap<'db>),
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    UnevaluatedConst(GeneralConstId, Substitution),
    /// Case of an unknown value that rustc might know but we don't
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    // https://github.com/rust-lang/rust-analyzer/pull/8813#issuecomment-840679177
    // https://rust-lang.zulipchat.com/#narrow/stream/144729-wg-traits/topic/Handling.20non.20evaluatable.20constants'.20equality/near/238386348
    Unknown,
}

impl Hash for ConstScalarNs<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        if let ConstScalarNs::Bytes(b, _) = self {
            b.hash(state)
        }
    }
}

/// Return an index of a parameter in the generic type parameter list by it's id.
pub fn param_idx(db: &dyn HirDatabase, id: TypeOrConstParamId) -> Option<usize> {
    generics::generics(db, id.parent).type_or_const_param_idx(id)
}

pub(crate) fn wrap_empty_binders<T>(value: T) -> Binders<T>
where
    T: TypeFoldable<Interner> + HasInterner<Interner = Interner>,
{
    Binders::empty(Interner, value.shifted_in_from(Interner, DebruijnIndex::ONE))
}

pub(crate) fn make_single_type_binders<T: HasInterner<Interner = Interner>>(
    value: T,
) -> Binders<T> {
    Binders::new(
        chalk_ir::VariableKinds::from_iter(
            Interner,
            std::iter::once(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General)),
        ),
        value,
    )
}

pub(crate) fn make_binders<T: HasInterner<Interner = Interner>>(
    db: &dyn HirDatabase,
    generics: &Generics,
    value: T,
) -> Binders<T> {
    Binders::new(variable_kinds_from_iter(db, generics.iter_id()), value)
}

pub(crate) fn variable_kinds_from_iter(
    db: &dyn HirDatabase,
    iter: impl Iterator<Item = hir_def::GenericParamId>,
) -> VariableKinds<Interner> {
    VariableKinds::from_iter(
        Interner,
        iter.map(|x| match x {
            hir_def::GenericParamId::ConstParamId(id) => {
                chalk_ir::VariableKind::Const(db.const_param_ty(id))
            }
            hir_def::GenericParamId::TypeParamId(_) => {
                chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General)
            }
            hir_def::GenericParamId::LifetimeParamId(_) => chalk_ir::VariableKind::Lifetime,
        }),
    )
}

// FIXME: get rid of this, just replace it by FnPointer
/// A function signature as seen by type inference: Several parameter types and
/// one return type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallableSig {
    params_and_return: Arc<[Ty]>,
    is_varargs: bool,
    safety: Safety,
    abi: FnAbi,
}

has_interner!(CallableSig);

#[derive(Debug, Copy, Clone, Eq)]
pub enum FnAbi {
    Aapcs,
    AapcsUnwind,
    AvrInterrupt,
    AvrNonBlockingInterrupt,
    C,
    CCmseNonsecureCall,
    CCmseNonsecureEntry,
    CDecl,
    CDeclUnwind,
    CUnwind,
    Efiapi,
    Fastcall,
    FastcallUnwind,
    Msp430Interrupt,
    PtxKernel,
    RiscvInterruptM,
    RiscvInterruptS,
    Rust,
    RustCall,
    RustCold,
    RustIntrinsic,
    Stdcall,
    StdcallUnwind,
    System,
    SystemUnwind,
    Sysv64,
    Sysv64Unwind,
    Thiscall,
    ThiscallUnwind,
    Unadjusted,
    Vectorcall,
    VectorcallUnwind,
    Wasm,
    Win64,
    Win64Unwind,
    X86Interrupt,
    Unknown,
}

impl PartialEq for FnAbi {
    fn eq(&self, _other: &Self) -> bool {
        // FIXME: Proper equality breaks `coercion::two_closures_lub` test
        true
    }
}

impl Hash for FnAbi {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Required because of the FIXME above and due to us implementing `Eq`, without this
        // we would break the `Hash` + `Eq` contract
        core::mem::discriminant(&Self::Unknown).hash(state);
    }
}

impl FnAbi {
    #[rustfmt::skip]
    pub fn from_symbol(s: &Symbol) -> FnAbi {
        match s {
            s if *s == sym::aapcs_dash_unwind => FnAbi::AapcsUnwind,
            s if *s == sym::aapcs => FnAbi::Aapcs,
            s if *s == sym::avr_dash_interrupt => FnAbi::AvrInterrupt,
            s if *s == sym::avr_dash_non_dash_blocking_dash_interrupt => FnAbi::AvrNonBlockingInterrupt,
            s if *s == sym::C_dash_cmse_dash_nonsecure_dash_call => FnAbi::CCmseNonsecureCall,
            s if *s == sym::C_dash_cmse_dash_nonsecure_dash_entry => FnAbi::CCmseNonsecureEntry,
            s if *s == sym::C_dash_unwind => FnAbi::CUnwind,
            s if *s == sym::C => FnAbi::C,
            s if *s == sym::cdecl_dash_unwind => FnAbi::CDeclUnwind,
            s if *s == sym::cdecl => FnAbi::CDecl,
            s if *s == sym::efiapi => FnAbi::Efiapi,
            s if *s == sym::fastcall_dash_unwind => FnAbi::FastcallUnwind,
            s if *s == sym::fastcall => FnAbi::Fastcall,
            s if *s == sym::msp430_dash_interrupt => FnAbi::Msp430Interrupt,
            s if *s == sym::ptx_dash_kernel => FnAbi::PtxKernel,
            s if *s == sym::riscv_dash_interrupt_dash_m => FnAbi::RiscvInterruptM,
            s if *s == sym::riscv_dash_interrupt_dash_s => FnAbi::RiscvInterruptS,
            s if *s == sym::rust_dash_call => FnAbi::RustCall,
            s if *s == sym::rust_dash_cold => FnAbi::RustCold,
            s if *s == sym::rust_dash_intrinsic => FnAbi::RustIntrinsic,
            s if *s == sym::Rust => FnAbi::Rust,
            s if *s == sym::stdcall_dash_unwind => FnAbi::StdcallUnwind,
            s if *s == sym::stdcall => FnAbi::Stdcall,
            s if *s == sym::system_dash_unwind => FnAbi::SystemUnwind,
            s if *s == sym::system => FnAbi::System,
            s if *s == sym::sysv64_dash_unwind => FnAbi::Sysv64Unwind,
            s if *s == sym::sysv64 => FnAbi::Sysv64,
            s if *s == sym::thiscall_dash_unwind => FnAbi::ThiscallUnwind,
            s if *s == sym::thiscall => FnAbi::Thiscall,
            s if *s == sym::unadjusted => FnAbi::Unadjusted,
            s if *s == sym::vectorcall_dash_unwind => FnAbi::VectorcallUnwind,
            s if *s == sym::vectorcall => FnAbi::Vectorcall,
            s if *s == sym::wasm => FnAbi::Wasm,
            s if *s == sym::win64_dash_unwind => FnAbi::Win64Unwind,
            s if *s == sym::win64 => FnAbi::Win64,
            s if *s == sym::x86_dash_interrupt => FnAbi::X86Interrupt,
            _ => FnAbi::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FnAbi::Aapcs => "aapcs",
            FnAbi::AapcsUnwind => "aapcs-unwind",
            FnAbi::AvrInterrupt => "avr-interrupt",
            FnAbi::AvrNonBlockingInterrupt => "avr-non-blocking-interrupt",
            FnAbi::C => "C",
            FnAbi::CCmseNonsecureCall => "C-cmse-nonsecure-call",
            FnAbi::CCmseNonsecureEntry => "C-cmse-nonsecure-entry",
            FnAbi::CDecl => "C-decl",
            FnAbi::CDeclUnwind => "cdecl-unwind",
            FnAbi::CUnwind => "C-unwind",
            FnAbi::Efiapi => "efiapi",
            FnAbi::Fastcall => "fastcall",
            FnAbi::FastcallUnwind => "fastcall-unwind",
            FnAbi::Msp430Interrupt => "msp430-interrupt",
            FnAbi::PtxKernel => "ptx-kernel",
            FnAbi::RiscvInterruptM => "riscv-interrupt-m",
            FnAbi::RiscvInterruptS => "riscv-interrupt-s",
            FnAbi::Rust => "Rust",
            FnAbi::RustCall => "rust-call",
            FnAbi::RustCold => "rust-cold",
            FnAbi::RustIntrinsic => "rust-intrinsic",
            FnAbi::Stdcall => "stdcall",
            FnAbi::StdcallUnwind => "stdcall-unwind",
            FnAbi::System => "system",
            FnAbi::SystemUnwind => "system-unwind",
            FnAbi::Sysv64 => "sysv64",
            FnAbi::Sysv64Unwind => "sysv64-unwind",
            FnAbi::Thiscall => "thiscall",
            FnAbi::ThiscallUnwind => "thiscall-unwind",
            FnAbi::Unadjusted => "unadjusted",
            FnAbi::Vectorcall => "vectorcall",
            FnAbi::VectorcallUnwind => "vectorcall-unwind",
            FnAbi::Wasm => "wasm",
            FnAbi::Win64 => "win64",
            FnAbi::Win64Unwind => "win64-unwind",
            FnAbi::X86Interrupt => "x86-interrupt",
            FnAbi::Unknown => "unknown-abi",
        }
    }
}

/// A polymorphic function signature.
pub type PolyFnSig = Binders<CallableSig>;

impl CallableSig {
    pub fn from_params_and_return(
        params: impl Iterator<Item = Ty>,
        ret: Ty,
        is_varargs: bool,
        safety: Safety,
        abi: FnAbi,
    ) -> CallableSig {
        let mut params_and_return = Vec::with_capacity(params.size_hint().0 + 1);
        params_and_return.extend(params);
        params_and_return.push(ret);
        CallableSig { params_and_return: params_and_return.into(), is_varargs, safety, abi }
    }

    pub fn from_def(db: &dyn HirDatabase, def: FnDefId, substs: &Substitution) -> CallableSig {
        let callable_def = ToChalk::from_chalk(db, def);
        let interner = DbInterner::new_with(db, None, None);
        let args: crate::next_solver::GenericArgs<'_> = substs.to_nextsolver(interner);
        let sig = db.callable_item_signature(callable_def);
        sig.instantiate(interner, args).skip_binder().to_chalk(interner)
    }
    pub fn from_fn_ptr(fn_ptr: &FnPointer) -> CallableSig {
        CallableSig {
            // FIXME: what to do about lifetime params? -> return PolyFnSig
            params_and_return: Arc::from_iter(
                fn_ptr
                    .substitution
                    .clone()
                    .shifted_out_to(Interner, DebruijnIndex::ONE)
                    .expect("unexpected lifetime vars in fn ptr")
                    .0
                    .as_slice(Interner)
                    .iter()
                    .map(|arg| arg.assert_ty_ref(Interner).clone()),
            ),
            is_varargs: fn_ptr.sig.variadic,
            safety: fn_ptr.sig.safety,
            abi: fn_ptr.sig.abi,
        }
    }
    pub fn from_fn_sig_and_header<'db>(
        interner: DbInterner<'db>,
        sig: crate::next_solver::Binder<'db, rustc_type_ir::FnSigTys<DbInterner<'db>>>,
        header: rustc_type_ir::FnHeader<DbInterner<'db>>,
    ) -> CallableSig {
        CallableSig {
            // FIXME: what to do about lifetime params? -> return PolyFnSig
            params_and_return: Arc::from_iter(
                sig.skip_binder()
                    .inputs_and_output
                    .iter()
                    .map(|t| convert_ty_for_result(interner, t)),
            ),
            is_varargs: header.c_variadic,
            safety: match header.safety {
                next_solver::abi::Safety::Safe => chalk_ir::Safety::Safe,
                next_solver::abi::Safety::Unsafe => chalk_ir::Safety::Unsafe,
            },
            abi: header.abi,
        }
    }

    pub fn to_fn_ptr(&self) -> FnPointer {
        FnPointer {
            num_binders: 0,
            sig: FnSig { abi: self.abi, safety: self.safety, variadic: self.is_varargs },
            substitution: FnSubst(Substitution::from_iter(
                Interner,
                self.params_and_return.iter().cloned(),
            )),
        }
    }

    pub fn abi(&self) -> FnAbi {
        self.abi
    }

    pub fn params(&self) -> &[Ty] {
        &self.params_and_return[0..self.params_and_return.len() - 1]
    }

    pub fn ret(&self) -> &Ty {
        &self.params_and_return[self.params_and_return.len() - 1]
    }
}

impl TypeFoldable<Interner> for CallableSig {
    fn try_fold_with<E>(
        self,
        folder: &mut dyn chalk_ir::fold::FallibleTypeFolder<Interner, Error = E>,
        outer_binder: DebruijnIndex,
    ) -> Result<Self, E> {
        let vec = self.params_and_return.to_vec();
        let folded = vec.try_fold_with(folder, outer_binder)?;
        Ok(CallableSig {
            params_and_return: folded.into(),
            is_varargs: self.is_varargs,
            safety: self.safety,
            abi: self.abi,
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ImplTraitId {
    ReturnTypeImplTrait(hir_def::FunctionId, ImplTraitIdx), // FIXME(next-solver): Should be crate::nextsolver::ImplTraitIdx.
    TypeAliasImplTrait(hir_def::TypeAliasId, ImplTraitIdx),
    AsyncBlockTypeImplTrait(hir_def::DefWithBodyId, ExprId),
}

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct ImplTraits {
    pub(crate) impl_traits: Arena<ImplTrait>,
}

has_interner!(ImplTraits);

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct ImplTrait {
    pub(crate) bounds: Binders<Vec<QuantifiedWhereClause>>,
}

pub type ImplTraitIdx = Idx<ImplTrait>;

pub fn static_lifetime() -> Lifetime {
    LifetimeData::Static.intern(Interner)
}

pub fn error_lifetime() -> Lifetime {
    LifetimeData::Error.intern(Interner)
}

pub(crate) fn fold_free_vars<T: HasInterner<Interner = Interner> + TypeFoldable<Interner>>(
    t: T,
    for_ty: impl FnMut(BoundVar, DebruijnIndex) -> Ty,
    for_const: impl FnMut(Ty, BoundVar, DebruijnIndex) -> Const,
) -> T {
    use chalk_ir::fold::TypeFolder;

    #[derive(chalk_derive::FallibleTypeFolder)]
    #[has_interner(Interner)]
    struct FreeVarFolder<
        F1: FnMut(BoundVar, DebruijnIndex) -> Ty,
        F2: FnMut(Ty, BoundVar, DebruijnIndex) -> Const,
    >(F1, F2);
    impl<F1: FnMut(BoundVar, DebruijnIndex) -> Ty, F2: FnMut(Ty, BoundVar, DebruijnIndex) -> Const>
        TypeFolder<Interner> for FreeVarFolder<F1, F2>
    {
        fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn fold_free_var_ty(&mut self, bound_var: BoundVar, outer_binder: DebruijnIndex) -> Ty {
            self.0(bound_var, outer_binder)
        }

        fn fold_free_var_const(
            &mut self,
            ty: Ty,
            bound_var: BoundVar,
            outer_binder: DebruijnIndex,
        ) -> Const {
            self.1(ty, bound_var, outer_binder)
        }
    }
    t.fold_with(&mut FreeVarFolder(for_ty, for_const), DebruijnIndex::INNERMOST)
}

/// 'Canonicalizes' the `t` by replacing any errors with new variables. Also
/// ensures there are no unbound variables or inference variables anywhere in
/// the `t`.
pub fn replace_errors_with_variables<'db, T>(
    interner: DbInterner<'db>,
    t: &T,
) -> crate::next_solver::Canonical<'db, T>
where
    T: rustc_type_ir::TypeFoldable<DbInterner<'db>> + Clone,
{
    use rustc_type_ir::{FallibleTypeFolder, TypeSuperFoldable};
    struct ErrorReplacer<'db> {
        interner: DbInterner<'db>,
        vars: Vec<crate::next_solver::CanonicalVarKind<'db>>,
        binder: rustc_type_ir::DebruijnIndex,
    }
    impl<'db> FallibleTypeFolder<DbInterner<'db>> for ErrorReplacer<'db> {
        #[cfg(debug_assertions)]
        type Error = ();
        #[cfg(not(debug_assertions))]
        type Error = std::convert::Infallible;

        fn cx(&self) -> DbInterner<'db> {
            self.interner
        }

        fn try_fold_binder<T>(
            &mut self,
            t: crate::next_solver::Binder<'db, T>,
        ) -> Result<crate::next_solver::Binder<'db, T>, Self::Error>
        where
            T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
        {
            self.binder.shift_in(1);
            let result = t.try_super_fold_with(self);
            self.binder.shift_out(1);
            result
        }

        fn try_fold_ty(
            &mut self,
            t: crate::next_solver::Ty<'db>,
        ) -> Result<crate::next_solver::Ty<'db>, Self::Error> {
            if !t.has_type_flags(
                rustc_type_ir::TypeFlags::HAS_ERROR
                    | rustc_type_ir::TypeFlags::HAS_TY_INFER
                    | rustc_type_ir::TypeFlags::HAS_CT_INFER
                    | rustc_type_ir::TypeFlags::HAS_RE_INFER,
            ) {
                return Ok(t);
            }

            #[cfg(debug_assertions)]
            let error = || Err(());
            #[cfg(not(debug_assertions))]
            let error = || Ok(crate::next_solver::Ty::new_error(self.interner, ErrorGuaranteed));

            match t.kind() {
                crate::next_solver::TyKind::Error(_) => {
                    let var = rustc_type_ir::BoundVar::from_usize(self.vars.len());
                    self.vars.push(crate::next_solver::CanonicalVarKind::Ty {
                        ui: rustc_type_ir::UniverseIndex::ZERO,
                        sub_root: var,
                    });
                    Ok(crate::next_solver::Ty::new_bound(
                        self.interner,
                        self.binder,
                        crate::next_solver::BoundTy {
                            var,
                            kind: crate::next_solver::BoundTyKind::Anon,
                        },
                    ))
                }
                crate::next_solver::TyKind::Infer(_) => error(),
                crate::next_solver::TyKind::Bound(index, _) if index > self.binder => error(),
                _ => t.try_super_fold_with(self),
            }
        }

        fn try_fold_const(
            &mut self,
            ct: crate::next_solver::Const<'db>,
        ) -> Result<crate::next_solver::Const<'db>, Self::Error> {
            if !ct.has_type_flags(
                rustc_type_ir::TypeFlags::HAS_ERROR
                    | rustc_type_ir::TypeFlags::HAS_TY_INFER
                    | rustc_type_ir::TypeFlags::HAS_CT_INFER
                    | rustc_type_ir::TypeFlags::HAS_RE_INFER,
            ) {
                return Ok(ct);
            }

            #[cfg(debug_assertions)]
            let error = || Err(());
            #[cfg(not(debug_assertions))]
            let error = || Ok(crate::next_solver::Const::error(self.interner));

            match ct.kind() {
                crate::next_solver::ConstKind::Error(_) => {
                    let var = rustc_type_ir::BoundVar::from_usize(self.vars.len());
                    self.vars.push(crate::next_solver::CanonicalVarKind::Const(
                        rustc_type_ir::UniverseIndex::ZERO,
                    ));
                    Ok(crate::next_solver::Const::new_bound(
                        self.interner,
                        self.binder,
                        crate::next_solver::BoundConst { var },
                    ))
                }
                crate::next_solver::ConstKind::Infer(_) => error(),
                crate::next_solver::ConstKind::Bound(index, _) if index > self.binder => error(),
                _ => ct.try_super_fold_with(self),
            }
        }

        fn try_fold_region(
            &mut self,
            region: crate::next_solver::Region<'db>,
        ) -> Result<crate::next_solver::Region<'db>, Self::Error> {
            #[cfg(debug_assertions)]
            let error = || Err(());
            #[cfg(not(debug_assertions))]
            let error = || Ok(crate::next_solver::Region::error(self.interner));

            match region.kind() {
                crate::next_solver::RegionKind::ReError(_) => {
                    let var = rustc_type_ir::BoundVar::from_usize(self.vars.len());
                    self.vars.push(crate::next_solver::CanonicalVarKind::Region(
                        rustc_type_ir::UniverseIndex::ZERO,
                    ));
                    Ok(crate::next_solver::Region::new_bound(
                        self.interner,
                        self.binder,
                        crate::next_solver::BoundRegion {
                            var,
                            kind: crate::next_solver::BoundRegionKind::Anon,
                        },
                    ))
                }
                crate::next_solver::RegionKind::ReVar(_) => error(),
                crate::next_solver::RegionKind::ReBound(index, _) if index > self.binder => error(),
                _ => Ok(region),
            }
        }
    }

    let mut error_replacer =
        ErrorReplacer { vars: Vec::new(), binder: rustc_type_ir::DebruijnIndex::ZERO, interner };
    let value = match t.clone().try_fold_with(&mut error_replacer) {
        Ok(t) => t,
        Err(_) => panic!("Encountered unbound or inference vars in {t:?}"),
    };
    crate::next_solver::Canonical {
        value,
        max_universe: rustc_type_ir::UniverseIndex::ZERO,
        variables: crate::next_solver::CanonicalVars::new_from_iter(interner, error_replacer.vars),
    }
}

pub fn callable_sig_from_fn_trait<'db>(
    self_ty: crate::next_solver::Ty<'db>,
    trait_env: Arc<TraitEnvironment<'db>>,
    db: &'db dyn HirDatabase,
) -> Option<(FnTrait, crate::next_solver::PolyFnSig<'db>)> {
    let krate = trait_env.krate;
    let fn_once_trait = FnTrait::FnOnce.get_id(db, krate)?;
    let output_assoc_type = fn_once_trait
        .trait_items(db)
        .associated_type_by_name(&Name::new_symbol_root(sym::Output))?;

    let mut table = InferenceTable::new(db, trait_env.clone());
    let b = TyBuilder::trait_ref(db, fn_once_trait);
    if b.remaining() != 2 {
        return None;
    }

    // Register two obligations:
    // - Self: FnOnce<?args_ty>
    // - <Self as FnOnce<?args_ty>>::Output == ?ret_ty
    let args_ty = table.next_ty_var();
    let args = [self_ty, args_ty];
    let trait_ref = crate::next_solver::TraitRef::new(table.interner(), fn_once_trait.into(), args);
    let projection = crate::next_solver::Ty::new_alias(
        table.interner(),
        rustc_type_ir::AliasTyKind::Projection,
        crate::next_solver::AliasTy::new(table.interner(), output_assoc_type.into(), args),
    );

    let pred = crate::next_solver::Predicate::upcast_from(trait_ref, table.interner());
    if !table.try_obligation(pred).no_solution() {
        table.register_obligation(pred);
        let return_ty = table.normalize_alias_ty(projection);
        for fn_x in [FnTrait::Fn, FnTrait::FnMut, FnTrait::FnOnce] {
            let fn_x_trait = fn_x.get_id(db, krate)?;
            let trait_ref =
                crate::next_solver::TraitRef::new(table.interner(), fn_x_trait.into(), args);
            if !table
                .try_obligation(crate::next_solver::Predicate::upcast_from(
                    trait_ref,
                    table.interner(),
                ))
                .no_solution()
            {
                let ret_ty = table.resolve_completely(return_ty);
                let args_ty = table.resolve_completely(args_ty);
                let crate::next_solver::TyKind::Tuple(params) = args_ty.kind() else {
                    return None;
                };
                let inputs_and_output = crate::next_solver::Tys::new_from_iter(
                    table.interner(),
                    params.iter().chain(std::iter::once(ret_ty)),
                );

                return Some((
                    fn_x,
                    crate::next_solver::Binder::dummy(crate::next_solver::FnSig {
                        inputs_and_output,
                        c_variadic: false,
                        safety: crate::next_solver::abi::Safety::Safe,
                        abi: FnAbi::RustCall,
                    }),
                ));
            }
        }
        unreachable!("It should at least implement FnOnce at this point");
    } else {
        None
    }
}

struct ParamCollector {
    params: FxHashSet<TypeOrConstParamId>,
}

impl<'db> rustc_type_ir::TypeVisitor<DbInterner<'db>> for ParamCollector {
    type Result = ();

    fn visit_ty(&mut self, ty: crate::next_solver::Ty<'db>) -> Self::Result {
        if let crate::next_solver::TyKind::Param(param) = ty.kind() {
            self.params.insert(param.id.into());
        }

        ty.super_visit_with(self);
    }

    fn visit_const(&mut self, konst: crate::next_solver::Const<'db>) -> Self::Result {
        if let crate::next_solver::ConstKind::Param(param) = konst.kind() {
            self.params.insert(param.id.into());
        }

        konst.super_visit_with(self);
    }
}

/// Returns unique params for types and consts contained in `value`.
pub fn collect_params<'db, T>(value: &T) -> Vec<TypeOrConstParamId>
where
    T: ?Sized + rustc_type_ir::TypeVisitable<DbInterner<'db>>,
{
    let mut collector = ParamCollector { params: FxHashSet::default() };
    value.visit_with(&mut collector);
    Vec::from_iter(collector.params)
}

pub fn known_const_to_ast<'db>(
    konst: crate::next_solver::Const<'db>,
    db: &'db dyn HirDatabase,
    display_target: DisplayTarget,
) -> Option<ConstArg> {
    Some(make::expr_const_value(konst.display(db, display_target).to_string().as_str()))
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum DeclOrigin {
    LetExpr,
    /// from `let x = ..`
    LocalDecl {
        has_else: bool,
    },
}

/// Provides context for checking patterns in declarations. More specifically this
/// allows us to infer array types if the pattern is irrefutable and allows us to infer
/// the size of the array. See issue rust-lang/rust#76342.
#[derive(Debug, Copy, Clone)]
pub(crate) struct DeclContext {
    pub(crate) origin: DeclOrigin,
}

pub fn setup_tracing() -> Option<tracing::subscriber::DefaultGuard> {
    use std::env;
    use std::sync::LazyLock;
    use tracing_subscriber::{Registry, layer::SubscriberExt};
    use tracing_tree::HierarchicalLayer;

    static ENABLE: LazyLock<bool> = LazyLock::new(|| env::var("CHALK_DEBUG").is_ok());
    if !*ENABLE {
        return None;
    }

    let filter: tracing_subscriber::filter::Targets =
        env::var("CHALK_DEBUG").ok().and_then(|it| it.parse().ok()).unwrap_or_default();
    let layer = HierarchicalLayer::default()
        .with_indent_lines(true)
        .with_ansi(false)
        .with_indent_amount(2)
        .with_writer(std::io::stderr);
    let subscriber = Registry::default().with(filter).with(layer);
    Some(tracing::subscriber::set_default(subscriber))
}
