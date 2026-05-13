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
mod generics;
mod infer;
mod inhabitedness;
mod lower;
pub mod next_solver;
mod opaques;
mod representability;
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
pub mod lang_items;
pub mod layout;
pub mod method_resolution;
pub mod mir;
pub mod primitive;
pub mod solver_errors;
pub mod traits;
pub mod upvars;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod tests;

use std::{hash::Hash, ops::ControlFlow};

use hir_def::{
    CallableDefId, ConstId, DefWithBodyId, EnumVariantId, ExpressionStoreOwnerId, FunctionId,
    GenericDefId, HasModule, LifetimeParamId, ModuleId, StaticId, TypeAliasId, TypeOrConstParamId,
    TypeParamId,
    db::DefDatabase,
    expr_store::{Body, ExpressionStore},
    hir::{BindingId, ExprId, ExprOrPatId, PatId},
    resolver::{HasResolver, Resolver, TypeNs},
    type_ref::{Rawness, TypeRefId},
};
use hir_expand::name::Name;
use indexmap::{IndexMap, map::Entry};
use macros::GenericTypeVisitable;
use mir::{MirEvalError, VTableMap};
use rustc_abi::ExternAbi;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use rustc_type_ir::{
    BoundVarIndexKind, TypeSuperVisitable, TypeVisitableExt,
    inherent::{IntoKind, Ty as _},
};
use stdx::impl_from;
use syntax::ast::{ConstArg, make};
use traits::FnTrait;

use crate::{
    db::{AnonConstId, HirDatabase},
    display::HirDisplay,
    lower::SupertraitsInfo,
    next_solver::{
        AliasTy, Binder, BoundConst, BoundRegion, BoundRegionKind, BoundTy, BoundTyKind, Canonical,
        CanonicalVarKind, CanonicalVarKinds, ClauseKind, Const, ConstKind, DbInterner, GenericArgs,
        PolyFnSig, Region, RegionKind, TraitRef, Ty, TyKind, TypingMode,
        abi::Safety,
        infer::{
            DbInternerInferExt,
            traits::{Obligation, ObligationCause},
        },
        obligation_ctxt::ObligationCtxt,
    },
};

pub use autoderef::autoderef;
pub use infer::{
    Adjust, Adjustment, AutoBorrow, BindingMode, ByRef, InferenceDiagnostic, InferenceResult,
    InferenceTyDiagnosticSource, OverloadedDeref, PointerCast, cast::CastError, could_coerce,
    could_unify, could_unify_deeply, infer_query_with_inspect,
};
pub use lower::{
    GenericDefaults, GenericDefaultsRef, GenericPredicates, ImplTraits, LifetimeElisionKind,
    TyDefId, TyLoweringContext, TyLoweringInferVarsCtx, TyLoweringResult, ValueTyDefId,
    diagnostics::*,
};
pub use next_solver::interner::{attach_db, attach_db_allow_change, with_attached_db};
pub use target_feature::TargetFeatures;
pub use traits::{ParamEnvAndCrate, check_orphan_rules};
pub use utils::{
    TargetFeatureIsSafeInTarget, Unsafety, all_super_traits, direct_super_traits,
    is_fn_unsafe_to_call, target_feature_is_safe_in_target,
};

pub mod closure_analysis {
    pub use crate::infer::{
        CaptureInfo, CaptureSourceStack, CapturedPlace, ClosureData, UpvarCapture,
        closure::analysis::{
            BorrowKind,
            expr_use_visitor::{FakeReadCause, Place, PlaceBase, Projection, ProjectionKind},
        },
    };
}

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
pub fn type_or_const_param_idx(db: &dyn HirDatabase, id: TypeOrConstParamId) -> u32 {
    generics::generics(db, id.parent).type_or_const_param_idx(id)
}

pub fn lifetime_param_idx(db: &dyn HirDatabase, id: LifetimeParamId) -> u32 {
    generics::generics(db, id.parent).lifetime_param_idx(id)
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
                    Ok(Const::new_bound(self.interner, self.binder, BoundConst::new(var)))
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
        var_kinds: CanonicalVarKinds::new_from_slice(&error_replacer.vars),
    }
}

/// To be used from `hir` only.
pub fn associated_type_shorthand_candidates(
    db: &dyn HirDatabase,
    def: GenericDefId,
    res: TypeNs,
    mut cb: impl FnMut(&Name, TypeAliasId) -> bool,
) -> Option<TypeAliasId> {
    let interner = DbInterner::new_no_crate(db);
    let (def, param) = match res {
        TypeNs::GenericParam(param) => (def, param),
        TypeNs::SelfType(impl_) => {
            let impl_trait = db.impl_trait(impl_)?.skip_binder().def_id.0;
            let param = TypeParamId::trait_self(impl_trait);
            (impl_trait.into(), param)
        }
        _ => return None,
    };

    let mut dedup_map = FxHashSet::default();
    let param_ty = Ty::new_param(interner, param, type_or_const_param_idx(db, param.into()));
    // We use the ParamEnv and not the predicates because the ParamEnv elaborates bounds.
    let param_env = db.trait_environment(ExpressionStoreOwnerId::from(def));
    for clause in param_env.clauses {
        let ClauseKind::Trait(trait_clause) = clause.kind().skip_binder() else { continue };
        if trait_clause.self_ty() != param_ty {
            continue;
        }
        let trait_id = trait_clause.def_id().0;
        dedup_map.extend(
            SupertraitsInfo::query(db, trait_id)
                .defined_assoc_types
                .iter()
                .map(|(name, id)| (name, *id)),
        );
    }

    dedup_map
        .into_iter()
        .try_for_each(
            |(name, id)| {
                if cb(name, id) { ControlFlow::Break(id) } else { ControlFlow::Continue(()) }
            },
        )
        .break_value()
}

/// To be used from `hir` only.
pub fn callable_sig_from_fn_trait<'db>(
    self_ty: Ty<'db>,
    param_env: ParamEnvAndCrate<'db>,
    db: &'db dyn HirDatabase,
) -> Option<(FnTrait, PolyFnSig<'db>)> {
    let ParamEnvAndCrate { param_env, krate } = param_env;
    let interner = DbInterner::new_with(db, krate);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let lang_items = interner.lang_items();
    let cause = ObligationCause::dummy();

    let impls_trait = |trait_: FnTrait| {
        let mut ocx = ObligationCtxt::new(&infcx);
        let tupled_args = infcx.next_ty_var(Span::Dummy);
        let args = GenericArgs::new_from_slice(&[self_ty.into(), tupled_args.into()]);
        let trait_id = trait_.get_id(lang_items)?;
        let trait_ref = TraitRef::new_from_args(interner, trait_id.into(), args);
        let obligation = Obligation::new(interner, cause, param_env, trait_ref);
        ocx.register_obligation(obligation);
        if !ocx.try_evaluate_obligations().is_empty() {
            return None;
        }
        let tupled_args =
            infcx.resolve_vars_if_possible(tupled_args).replace_infer_with_error(interner);
        if tupled_args.is_tuple() { Some(tupled_args) } else { None }
    };

    let (trait_, args) = 'find_trait: {
        for trait_ in [FnTrait::Fn, FnTrait::FnMut, FnTrait::FnOnce] {
            if let Some(args) = impls_trait(trait_) {
                break 'find_trait (trait_, args);
            }
        }
        return None;
    };

    let output_assoc_type = lang_items.FnOnceOutput?;
    let output_projection = Ty::new_alias(
        interner,
        AliasTy::new(
            interner,
            rustc_type_ir::Projection { def_id: output_assoc_type.into() },
            [self_ty, args],
        ),
    );
    let mut ocx = ObligationCtxt::new(&infcx);
    let ret = ocx.structurally_normalize_ty(&cause, param_env, output_projection).ok()?;
    let ret = ret.replace_infer_with_error(interner);

    let sig = Binder::dummy(interner.mk_fn_sig(
        args.tuple_fields(),
        ret,
        false,
        Safety::Safe,
        ExternAbi::Rust,
    ));
    Some((trait_, sig))
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

pub fn known_const_to_ast<'db>(
    konst: Const<'db>,
    db: &'db dyn HirDatabase,
    target_module: ModuleId,
) -> Option<ConstArg> {
    Some(make::expr_const_value(
        &konst.display_source_code(db, target_module, true).unwrap_or_else(|_| "_".to_owned()),
    ))
}

/// A `Span` represents some location in lowered code - a type, expression or pattern.
///
/// It has no meaning outside its body therefore it should not exit the pass it was created in
/// (e.g. inference). It is usually associated with a solver obligation or an infer var, which
/// should also not cross the pass they were created in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Span {
    ExprId(ExprId),
    PatId(PatId),
    BindingId(BindingId),
    TypeRefId(TypeRefId),
    /// An unimportant location. Errors on this will be suppressed.
    Dummy,
}
impl_from!(ExprId, PatId, BindingId, TypeRefId for Span);

impl From<ExprOrPatId> for Span {
    fn from(value: ExprOrPatId) -> Self {
        match value {
            ExprOrPatId::ExprId(idx) => idx.into(),
            ExprOrPatId::PatId(idx) => idx.into(),
        }
    }
}

impl Span {
    pub(crate) fn pick_best(a: Span, b: Span) -> Span {
        // We prefer dummy spans to minimize the risk of false errors.
        if b.is_dummy() { b } else { a }
    }

    #[inline]
    pub fn is_dummy(&self) -> bool {
        matches!(self, Self::Dummy)
    }
}

/// A [`DefWithBodyId`], or an anon const.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, salsa::Supertype)]
pub enum InferBodyId {
    DefWithBodyId(DefWithBodyId),
    AnonConstId(AnonConstId),
}
impl_from!(DefWithBodyId(FunctionId, ConstId, StaticId), AnonConstId for InferBodyId);
impl From<EnumVariantId> for InferBodyId {
    fn from(id: EnumVariantId) -> Self {
        InferBodyId::DefWithBodyId(DefWithBodyId::VariantId(id))
    }
}

impl HasModule for InferBodyId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match self {
            InferBodyId::DefWithBodyId(id) => id.module(db),
            InferBodyId::AnonConstId(id) => id.module(db),
        }
    }
}

impl HasResolver for InferBodyId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        match self {
            InferBodyId::DefWithBodyId(id) => id.resolver(db),
            InferBodyId::AnonConstId(id) => id.resolver(db),
        }
    }
}

impl InferBodyId {
    pub fn expression_store_owner(self, db: &dyn HirDatabase) -> ExpressionStoreOwnerId {
        match self {
            InferBodyId::DefWithBodyId(id) => id.into(),
            InferBodyId::AnonConstId(id) => id.loc(db).owner,
        }
    }

    pub fn generic_def(self, db: &dyn HirDatabase) -> GenericDefId {
        match self {
            InferBodyId::DefWithBodyId(id) => id.generic_def(db),
            InferBodyId::AnonConstId(id) => id.loc(db).owner.generic_def(db),
        }
    }

    #[inline]
    pub fn as_function(self) -> Option<FunctionId> {
        match self {
            InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(it)) => Some(it),
            _ => None,
        }
    }

    #[inline]
    pub fn as_variant(self) -> Option<EnumVariantId> {
        match self {
            InferBodyId::DefWithBodyId(DefWithBodyId::VariantId(it)) => Some(it),
            _ => None,
        }
    }

    pub fn store_and_root_expr(self, db: &dyn HirDatabase) -> (&ExpressionStore, ExprId) {
        match self {
            InferBodyId::DefWithBodyId(id) => {
                let body = Body::of(db, id);
                (body, body.root_expr())
            }
            InferBodyId::AnonConstId(id) => {
                let loc = id.loc(db);
                let store = ExpressionStore::of(db, loc.owner);
                (store, loc.expr)
            }
        }
    }
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
