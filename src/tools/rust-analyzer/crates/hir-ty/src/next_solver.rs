//! Things relevant to the next trait solver.

// Note: in interned types defined in this module, we generally treat the lifetime as advisory
// and transmute it as needed. This is because no real memory unsafety can be caused from an
// incorrect lifetime here.

pub mod abi;
mod binder;
mod consts;
mod def_id;
pub mod fold;
pub mod format_proof_tree;
pub mod fulfill;
mod generic_arg;
pub mod generics;
pub mod infer;
pub(crate) mod inspect;
pub mod interner;
mod ir_print;
pub mod normalize;
pub mod obligation_ctxt;
mod opaques;
pub mod predicate;
mod region;
mod solver;
mod structural_normalize;
mod ty;
pub mod util;

use std::{mem::ManuallyDrop, sync::OnceLock};

pub use binder::*;
pub use consts::*;
pub use def_id::*;
pub use generic_arg::*;
pub use interner::*;
pub use opaques::*;
pub use predicate::*;
pub use region::*;
pub use solver::*;
pub use ty::*;

use crate::db::HirDatabase;
pub use crate::lower::ImplTraitIdx;
pub use rustc_ast_ir::Mutability;

pub type Binder<'db, T> = rustc_type_ir::Binder<DbInterner<'db>, T>;
pub type EarlyBinder<'db, T> = rustc_type_ir::EarlyBinder<DbInterner<'db>, T>;
pub type Canonical<'db, T> = rustc_type_ir::Canonical<DbInterner<'db>, T>;
pub type CanonicalVarValues<'db> = rustc_type_ir::CanonicalVarValues<DbInterner<'db>>;
pub type CanonicalVarKind<'db> = rustc_type_ir::CanonicalVarKind<DbInterner<'db>>;
pub type CanonicalQueryInput<'db, V> = rustc_type_ir::CanonicalQueryInput<DbInterner<'db>, V>;
pub type AliasTy<'db> = rustc_type_ir::AliasTy<DbInterner<'db>>;
pub type FnSig<'db> = rustc_type_ir::FnSig<DbInterner<'db>>;
pub type PolyFnSig<'db> = Binder<'db, rustc_type_ir::FnSig<DbInterner<'db>>>;
pub type TypingMode<'db> = rustc_type_ir::TypingMode<DbInterner<'db>>;
pub type TypeError<'db> = rustc_type_ir::error::TypeError<DbInterner<'db>>;
pub type QueryResult<'db> = rustc_type_ir::solve::QueryResult<DbInterner<'db>>;
pub type FxIndexMap<K, V> = rustc_type_ir::data_structures::IndexMap<K, V>;

pub struct DefaultTypes<'db> {
    pub usize: Ty<'db>,
    pub u8: Ty<'db>,
    pub u16: Ty<'db>,
    pub u32: Ty<'db>,
    pub u64: Ty<'db>,
    pub u128: Ty<'db>,
    pub isize: Ty<'db>,
    pub i8: Ty<'db>,
    pub i16: Ty<'db>,
    pub i32: Ty<'db>,
    pub i64: Ty<'db>,
    pub i128: Ty<'db>,
    pub f16: Ty<'db>,
    pub f32: Ty<'db>,
    pub f64: Ty<'db>,
    pub f128: Ty<'db>,
    pub unit: Ty<'db>,
    pub bool: Ty<'db>,
    pub char: Ty<'db>,
    pub str: Ty<'db>,
    pub never: Ty<'db>,
    pub error: Ty<'db>,
    /// `&'static str`
    pub static_str_ref: Ty<'db>,
    /// `*mut ()`
    pub mut_unit_ptr: Ty<'db>,
}

pub struct DefaultConsts<'db> {
    pub error: Const<'db>,
}

pub struct DefaultRegions<'db> {
    pub error: Region<'db>,
    pub statik: Region<'db>,
    pub erased: Region<'db>,
}

pub struct DefaultEmpty<'db> {
    pub tys: Tys<'db>,
    pub generic_args: GenericArgs<'db>,
    pub bound_var_kinds: BoundVarKinds<'db>,
    pub canonical_vars: CanonicalVars<'db>,
    pub variances: VariancesOf<'db>,
    pub pat_list: PatList<'db>,
    pub predefined_opaques: PredefinedOpaques<'db>,
    pub def_ids: SolverDefIds<'db>,
    pub bound_existential_predicates: BoundExistentialPredicates<'db>,
    pub clauses: Clauses<'db>,
    pub region_assumptions: RegionAssumptions<'db>,
}

pub struct DefaultAny<'db> {
    pub types: DefaultTypes<'db>,
    pub consts: DefaultConsts<'db>,
    pub regions: DefaultRegions<'db>,
    pub empty: DefaultEmpty<'db>,
    /// `[Invariant]`
    pub one_invariant: VariancesOf<'db>,
    /// `[Covariant]`
    pub one_covariant: VariancesOf<'db>,
    /// `for<'env>`
    pub coroutine_captures_by_ref_bound_var_kinds: BoundVarKinds<'db>,
}

impl std::fmt::Debug for DefaultAny<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultAny").finish_non_exhaustive()
    }
}

#[inline]
pub fn default_types<'a, 'db>(db: &'db dyn HirDatabase) -> &'a DefaultAny<'db> {
    static TYPES: OnceLock<DefaultAny<'static>> = OnceLock::new();

    let interner = DbInterner::new_no_crate(db);
    TYPES.get_or_init(|| {
        let create_ty = |kind| {
            let ty = Ty::new(interner, kind);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_const = |kind| {
            let ty = Const::new(interner, kind);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_region = |kind| {
            let ty = Region::new(interner, kind);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_generic_args = |slice| {
            let ty = GenericArgs::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_bound_var_kinds = |slice| {
            let ty = BoundVarKinds::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_canonical_vars = |slice| {
            let ty = CanonicalVars::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_variances_of = |slice| {
            let ty = VariancesOf::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_pat_list = |slice| {
            let ty = PatList::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_predefined_opaques = |slice| {
            let ty = PredefinedOpaques::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_solver_def_ids = |slice| {
            let ty = SolverDefIds::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_bound_existential_predicates = |slice| {
            let ty = BoundExistentialPredicates::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_clauses = |slice| {
            let ty = Clauses::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_region_assumptions = |slice| {
            let ty = RegionAssumptions::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };
        let create_tys = |slice| {
            let ty = Tys::new_from_slice(slice);
            // We need to increase the refcount (forever), so that the types won't be freed.
            let ty = ManuallyDrop::new(ty.store());
            ty.as_ref()
        };

        let str = create_ty(TyKind::Str);
        let statik = create_region(RegionKind::ReStatic);
        let empty_tys = create_tys(&[]);
        let unit = create_ty(TyKind::Tuple(empty_tys));
        DefaultAny {
            types: DefaultTypes {
                usize: create_ty(TyKind::Uint(rustc_ast_ir::UintTy::Usize)),
                u8: create_ty(TyKind::Uint(rustc_ast_ir::UintTy::U8)),
                u16: create_ty(TyKind::Uint(rustc_ast_ir::UintTy::U16)),
                u32: create_ty(TyKind::Uint(rustc_ast_ir::UintTy::U32)),
                u64: create_ty(TyKind::Uint(rustc_ast_ir::UintTy::U64)),
                u128: create_ty(TyKind::Uint(rustc_ast_ir::UintTy::U128)),
                isize: create_ty(TyKind::Int(rustc_ast_ir::IntTy::Isize)),
                i8: create_ty(TyKind::Int(rustc_ast_ir::IntTy::I8)),
                i16: create_ty(TyKind::Int(rustc_ast_ir::IntTy::I16)),
                i32: create_ty(TyKind::Int(rustc_ast_ir::IntTy::I32)),
                i64: create_ty(TyKind::Int(rustc_ast_ir::IntTy::I64)),
                i128: create_ty(TyKind::Int(rustc_ast_ir::IntTy::I128)),
                f16: create_ty(TyKind::Float(rustc_ast_ir::FloatTy::F16)),
                f32: create_ty(TyKind::Float(rustc_ast_ir::FloatTy::F32)),
                f64: create_ty(TyKind::Float(rustc_ast_ir::FloatTy::F64)),
                f128: create_ty(TyKind::Float(rustc_ast_ir::FloatTy::F128)),
                unit,
                bool: create_ty(TyKind::Bool),
                char: create_ty(TyKind::Char),
                str,
                never: create_ty(TyKind::Never),
                error: create_ty(TyKind::Error(ErrorGuaranteed)),
                static_str_ref: create_ty(TyKind::Ref(statik, str, rustc_ast_ir::Mutability::Not)),
                mut_unit_ptr: create_ty(TyKind::RawPtr(unit, rustc_ast_ir::Mutability::Mut)),
            },
            consts: DefaultConsts { error: create_const(ConstKind::Error(ErrorGuaranteed)) },
            regions: DefaultRegions {
                error: create_region(RegionKind::ReError(ErrorGuaranteed)),
                statik,
                erased: create_region(RegionKind::ReErased),
            },
            empty: DefaultEmpty {
                tys: empty_tys,
                generic_args: create_generic_args(&[]),
                bound_var_kinds: create_bound_var_kinds(&[]),
                canonical_vars: create_canonical_vars(&[]),
                variances: create_variances_of(&[]),
                pat_list: create_pat_list(&[]),
                predefined_opaques: create_predefined_opaques(&[]),
                def_ids: create_solver_def_ids(&[]),
                bound_existential_predicates: create_bound_existential_predicates(&[]),
                clauses: create_clauses(&[]),
                region_assumptions: create_region_assumptions(&[]),
            },
            one_invariant: create_variances_of(&[rustc_type_ir::Variance::Invariant]),
            one_covariant: create_variances_of(&[rustc_type_ir::Variance::Covariant]),
            coroutine_captures_by_ref_bound_var_kinds: create_bound_var_kinds(&[
                BoundVarKind::Region(BoundRegionKind::ClosureEnv),
            ]),
        }
    })
}
