//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
// It's useful to refer to code that is private in doc comments.
#![allow(rustdoc::private_intra_doc_links)]

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

pub mod builtin_derive;
mod infer;
mod inhabitedness;
mod lower;
pub mod next_solver;
mod opaques;
mod specialization;
mod target_feature;
mod utils;
mod variance;

pub mod autoderef;
pub mod consteval;
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
pub mod upvars;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod tests;

use std::hash::Hash;

use hir_def::{CallableDefId, TypeOrConstParamId, type_ref::Rawness};
use hir_expand::name::Name;
use indexmap::{IndexMap, map::Entry};
use intern::{Symbol, sym};
use macros::GenericTypeVisitable;
use mir::{MirEvalError, VTableMap};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use rustc_type_ir::{
    BoundVarIndexKind, TypeSuperVisitable, TypeVisitableExt, UpcastFrom,
    inherent::{IntoKind, Ty as _},
};
use syntax::ast::{ConstArg, make};
use traits::FnTrait;

use crate::{
    db::HirDatabase,
    display::{DisplayTarget, HirDisplay},
    infer::unify::InferenceTable,
    next_solver::{
        AliasTy, Binder, BoundConst, BoundRegion, BoundRegionKind, BoundTy, BoundTyKind, Canonical,
        CanonicalVarKind, CanonicalVars, Const, ConstKind, DbInterner, FnSig, GenericArgs,
        PolyFnSig, Predicate, Region, RegionKind, TraitRef, Ty, TyKind, Tys, abi,
    },
};

pub use autoderef::autoderef;
pub use infer::{
    Adjust, Adjustment, AutoBorrow, BindingMode, InferenceDiagnostic, InferenceResult,
    InferenceTyDiagnosticSource, OverloadedDeref, PointerCast,
    cast::CastError,
    closure::analysis::{CaptureKind, CapturedItem},
    could_coerce, could_unify, could_unify_deeply, infer_query_with_inspect,
};
pub use lower::{
    GenericPredicates, ImplTraits, LifetimeElisionKind, TyDefId, TyLoweringContext, ValueTyDefId,
    associated_type_shorthand_candidates, diagnostics::*,
};
pub use next_solver::interner::{attach_db, attach_db_allow_change, with_attached_db};
pub use target_feature::TargetFeatures;
pub use traits::{ParamEnvAndCrate, check_orphan_rules};
pub use utils::{
    TargetFeatureIsSafeInTarget, Unsafety, all_super_traits, direct_super_traits,
    is_fn_unsafe_to_call, target_feature_is_safe_in_target,
};

/// A constant can have reference to other things. Memory map job is holding
/// the necessary bits of memory of the const eval session to keep the constant
/// meaningful.
#[derive(Debug, Default, Clone, PartialEq, Eq, GenericTypeVisitable)]
pub enum MemoryMap<'db> {
    #[default]
    Empty,
    Simple(Box<[u8]>),
    Complex(Box<ComplexMemoryMap<'db>>),
}

#[derive(Debug, Default, Clone, PartialEq, Eq, GenericTypeVisitable)]
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
    pub fn vtable_ty(&self, id: usize) -> Result<Ty<'db>, MirEvalError> {
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
        mut f: impl FnMut(&[u8], usize) -> Result<usize, MirEvalError>,
    ) -> Result<FxHashMap<usize, usize>, MirEvalError> {
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

/// Return an index of a parameter in the generic type parameter list by it's id.
pub fn param_idx(db: &dyn HirDatabase, id: TypeOrConstParamId) -> Option<usize> {
    generics::generics(db, id.parent).type_or_const_param_idx(id)
}

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
    RustPreserveNone,
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
            s if *s == sym::rust_dash_preserve_dash_none => FnAbi::RustPreserveNone,
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
            FnAbi::RustPreserveNone => "rust-preserve-none",
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

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ImplTraitId {
    ReturnTypeImplTrait(hir_def::FunctionId, next_solver::ImplTraitIdx),
    TypeAliasImplTrait(hir_def::TypeAliasId, next_solver::ImplTraitIdx),
}

/// 'Canonicalizes' the `t` by replacing any errors with new variables. Also
/// ensures there are no unbound variables or inference variables anywhere in
/// the `t`.
pub fn replace_errors_with_variables<'db, T>(interner: DbInterner<'db>, t: &T) -> Canonical<'db, T>
where
    T: rustc_type_ir::TypeFoldable<DbInterner<'db>> + Clone,
{
    use rustc_type_ir::{FallibleTypeFolder, TypeSuperFoldable};
    struct ErrorReplacer<'db> {
        interner: DbInterner<'db>,
        vars: Vec<CanonicalVarKind<'db>>,
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

        fn try_fold_binder<T>(&mut self, t: Binder<'db, T>) -> Result<Binder<'db, T>, Self::Error>
        where
            T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
        {
            self.binder.shift_in(1);
            let result = t.try_super_fold_with(self);
            self.binder.shift_out(1);
            result
        }

        fn try_fold_ty(&mut self, t: Ty<'db>) -> Result<Ty<'db>, Self::Error> {
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
            let error = || Ok(Ty::new_error(self.interner, crate::next_solver::ErrorGuaranteed));

            match t.kind() {
                TyKind::Error(_) => {
                    let var = rustc_type_ir::BoundVar::from_usize(self.vars.len());
                    self.vars.push(CanonicalVarKind::Ty {
                        ui: rustc_type_ir::UniverseIndex::ZERO,
                        sub_root: var,
                    });
                    Ok(Ty::new_bound(
                        self.interner,
                        self.binder,
                        BoundTy { var, kind: BoundTyKind::Anon },
                    ))
                }
                TyKind::Infer(_) => error(),
                TyKind::Bound(BoundVarIndexKind::Bound(index), _) if index > self.binder => error(),
                _ => t.try_super_fold_with(self),
            }
        }

        fn try_fold_const(&mut self, ct: Const<'db>) -> Result<Const<'db>, Self::Error> {
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
            let error = || Ok(Const::error(self.interner));

            match ct.kind() {
                ConstKind::Error(_) => {
                    let var = rustc_type_ir::BoundVar::from_usize(self.vars.len());
                    self.vars.push(CanonicalVarKind::Const(rustc_type_ir::UniverseIndex::ZERO));
                    Ok(Const::new_bound(self.interner, self.binder, BoundConst { var }))
                }
                ConstKind::Infer(_) => error(),
                ConstKind::Bound(BoundVarIndexKind::Bound(index), _) if index > self.binder => {
                    error()
                }
                _ => ct.try_super_fold_with(self),
            }
        }

        fn try_fold_region(&mut self, region: Region<'db>) -> Result<Region<'db>, Self::Error> {
            #[cfg(debug_assertions)]
            let error = || Err(());
            #[cfg(not(debug_assertions))]
            let error = || Ok(Region::error(self.interner));

            match region.kind() {
                RegionKind::ReError(_) => {
                    let var = rustc_type_ir::BoundVar::from_usize(self.vars.len());
                    self.vars.push(CanonicalVarKind::Region(rustc_type_ir::UniverseIndex::ZERO));
                    Ok(Region::new_bound(
                        self.interner,
                        self.binder,
                        BoundRegion { var, kind: BoundRegionKind::Anon },
                    ))
                }
                RegionKind::ReVar(_) => error(),
                RegionKind::ReBound(BoundVarIndexKind::Bound(index), _) if index > self.binder => {
                    error()
                }
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
    Canonical {
        value,
        max_universe: rustc_type_ir::UniverseIndex::ZERO,
        variables: CanonicalVars::new_from_slice(&error_replacer.vars),
    }
}

/// To be used from `hir` only.
pub fn callable_sig_from_fn_trait<'db>(
    self_ty: Ty<'db>,
    trait_env: ParamEnvAndCrate<'db>,
    db: &'db dyn HirDatabase,
) -> Option<(FnTrait, PolyFnSig<'db>)> {
    let mut table = InferenceTable::new(db, trait_env.param_env, trait_env.krate, None);
    let lang_items = table.interner().lang_items();

    let fn_once_trait = FnTrait::FnOnce.get_id(lang_items)?;
    let output_assoc_type = fn_once_trait
        .trait_items(db)
        .associated_type_by_name(&Name::new_symbol_root(sym::Output))?;

    // Register two obligations:
    // - Self: FnOnce<?args_ty>
    // - <Self as FnOnce<?args_ty>>::Output == ?ret_ty
    let args_ty = table.next_ty_var();
    let args = GenericArgs::new_from_slice(&[self_ty.into(), args_ty.into()]);
    let trait_ref = TraitRef::new_from_args(table.interner(), fn_once_trait.into(), args);
    let projection = Ty::new_alias(
        table.interner(),
        rustc_type_ir::AliasTyKind::Projection,
        AliasTy::new_from_args(table.interner(), output_assoc_type.into(), args),
    );

    let pred = Predicate::upcast_from(trait_ref, table.interner());
    if !table.try_obligation(pred).no_solution() {
        table.register_obligation(pred);
        let return_ty = table.normalize_alias_ty(projection);
        for fn_x in [FnTrait::Fn, FnTrait::FnMut, FnTrait::FnOnce] {
            let fn_x_trait = fn_x.get_id(lang_items)?;
            let trait_ref = TraitRef::new_from_args(table.interner(), fn_x_trait.into(), args);
            if !table
                .try_obligation(Predicate::upcast_from(trait_ref, table.interner()))
                .no_solution()
            {
                let ret_ty = table.resolve_completely(return_ty);
                let args_ty = table.resolve_completely(args_ty);
                let TyKind::Tuple(params) = args_ty.kind() else {
                    return None;
                };
                let inputs_and_output = Tys::new_from_iter(
                    table.interner(),
                    params.iter().chain(std::iter::once(ret_ty)),
                );

                return Some((
                    fn_x,
                    Binder::dummy(FnSig {
                        inputs_and_output,
                        c_variadic: false,
                        safety: abi::Safety::Safe,
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

    fn visit_ty(&mut self, ty: Ty<'db>) -> Self::Result {
        if let TyKind::Param(param) = ty.kind() {
            self.params.insert(param.id.into());
        }

        ty.super_visit_with(self);
    }

    fn visit_const(&mut self, konst: Const<'db>) -> Self::Result {
        if let ConstKind::Param(param) = konst.kind() {
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

struct TypeInferenceVarCollector<'db> {
    type_inference_vars: Vec<Ty<'db>>,
}

impl<'db> rustc_type_ir::TypeVisitor<DbInterner<'db>> for TypeInferenceVarCollector<'db> {
    type Result = ();

    fn visit_ty(&mut self, ty: Ty<'db>) -> Self::Result {
        use crate::rustc_type_ir::Flags;
        if ty.is_ty_var() {
            self.type_inference_vars.push(ty);
        } else if ty.flags().intersects(rustc_type_ir::TypeFlags::HAS_TY_INFER) {
            ty.super_visit_with(self);
        } else {
            // Fast path: don't visit inner types (e.g. generic arguments) when `flags` indicate
            // that there are no placeholders.
        }
    }
}

pub fn collect_type_inference_vars<'db, T>(value: &T) -> Vec<Ty<'db>>
where
    T: ?Sized + rustc_type_ir::TypeVisitable<DbInterner<'db>>,
{
    let mut collector = TypeInferenceVarCollector { type_inference_vars: vec![] };
    value.visit_with(&mut collector);
    collector.type_inference_vars
}

pub fn known_const_to_ast<'db>(
    konst: Const<'db>,
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
