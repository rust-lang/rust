//! Trait solving using next trait solver.

use std::hash::Hash;

use base_db::Crate;
use hir_def::{
    AdtId, AssocItemId, HasModule, ImplId, Lookup, TraitId,
    lang_item::LangItems,
    nameres::DefMap,
    signatures::{ConstFlags, EnumFlags, FnFlags, StructFlags, TraitFlags, TypeAliasFlags},
};
use hir_expand::name::Name;
use intern::sym;
use rustc_next_trait_solver::solve::{HasChanged, SolverDelegateEvalExt};
use rustc_type_ir::{
    TypingMode,
    inherent::{AdtDef, BoundExistentialPredicates, IntoKind, Span as _},
    solve::Certainty,
};

use crate::{
    db::HirDatabase,
    next_solver::{
        Canonical, DbInterner, GenericArgs, Goal, ParamEnv, Predicate, SolverContext, Span, Ty,
        TyKind,
        infer::{DbInternerInferExt, InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

/// Type for `hir`, because commonly we want both param env and a crate in an exported API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamEnvAndCrate<'db> {
    pub param_env: ParamEnv<'db>,
    pub krate: Crate,
}

/// This should be used in `hir` only.
pub fn structurally_normalize_ty<'db>(
    infcx: &InferCtxt<'db>,
    ty: Ty<'db>,
    env: ParamEnv<'db>,
) -> Ty<'db> {
    let TyKind::Alias(..) = ty.kind() else { return ty };
    let mut ocx = ObligationCtxt::new(infcx);
    let ty = ocx.structurally_normalize_ty(&ObligationCause::dummy(), env, ty).unwrap_or(ty);
    ty.replace_infer_with_error(infcx.interner)
}

#[derive(Clone, Debug, PartialEq)]
pub enum NextTraitSolveResult {
    Certain,
    Uncertain,
    NoSolution,
}

impl NextTraitSolveResult {
    pub fn no_solution(&self) -> bool {
        matches!(self, NextTraitSolveResult::NoSolution)
    }

    pub fn certain(&self) -> bool {
        matches!(self, NextTraitSolveResult::Certain)
    }

    pub fn uncertain(&self) -> bool {
        matches!(self, NextTraitSolveResult::Uncertain)
    }
}

pub fn next_trait_solve_canonical_in_ctxt<'db>(
    infer_ctxt: &InferCtxt<'db>,
    goal: Canonical<'db, Goal<'db, Predicate<'db>>>,
) -> NextTraitSolveResult {
    infer_ctxt.probe(|_| {
        let context = <&SolverContext<'db>>::from(infer_ctxt);

        tracing::info!(?goal);

        let (goal, var_values) = context.instantiate_canonical(&goal);
        tracing::info!(?var_values);

        let res = context.evaluate_root_goal(goal, Span::dummy(), None);

        let res = res.map(|r| (r.has_changed, r.certainty));

        tracing::debug!("solve_nextsolver({:?}) => {:?}", goal, res);

        match res {
            Err(_) => NextTraitSolveResult::NoSolution,
            Ok((_, Certainty::Yes)) => NextTraitSolveResult::Certain,
            Ok((_, Certainty::Maybe { .. })) => NextTraitSolveResult::Uncertain,
        }
    })
}

/// Solve a trait goal using next trait solver.
pub fn next_trait_solve_in_ctxt<'db, 'a>(
    infer_ctxt: &'a InferCtxt<'db>,
    goal: Goal<'db, Predicate<'db>>,
) -> Result<(HasChanged, Certainty), rustc_type_ir::solve::NoSolution> {
    tracing::info!(?goal);

    let context = <&SolverContext<'db>>::from(infer_ctxt);

    let res = context.evaluate_root_goal(goal, Span::dummy(), None);

    let res = res.map(|r| (r.has_changed, r.certainty));

    tracing::debug!("solve_nextsolver({:?}) => {:?}", goal, res);

    res
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, salsa::Update)]
pub enum FnTrait {
    // Warning: Order is important. If something implements `x` it should also implement
    // `y` if `y <= x`.
    FnOnce,
    FnMut,
    Fn,

    AsyncFnOnce,
    AsyncFnMut,
    AsyncFn,
}

impl FnTrait {
    pub fn method_name(self) -> Name {
        match self {
            FnTrait::FnOnce => Name::new_symbol_root(sym::call_once),
            FnTrait::FnMut => Name::new_symbol_root(sym::call_mut),
            FnTrait::Fn => Name::new_symbol_root(sym::call),
            FnTrait::AsyncFnOnce => Name::new_symbol_root(sym::async_call_once),
            FnTrait::AsyncFnMut => Name::new_symbol_root(sym::async_call_mut),
            FnTrait::AsyncFn => Name::new_symbol_root(sym::async_call),
        }
    }

    pub fn get_id(self, lang_items: &LangItems) -> Option<TraitId> {
        match self {
            FnTrait::FnOnce => lang_items.FnOnce,
            FnTrait::FnMut => lang_items.FnMut,
            FnTrait::Fn => lang_items.Fn,
            FnTrait::AsyncFnOnce => lang_items.AsyncFnOnce,
            FnTrait::AsyncFnMut => lang_items.AsyncFnMut,
            FnTrait::AsyncFn => lang_items.AsyncFn,
        }
    }
}

/// This should not be used in `hir-ty`, only in `hir`.
pub fn implements_trait_unique<'db>(
    ty: Ty<'db>,
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    trait_: TraitId,
) -> bool {
    implements_trait_unique_impl(db, env, trait_, &mut |infcx| {
        infcx.fill_rest_fresh_args(trait_.into(), [ty.into()])
    })
}

/// This should not be used in `hir-ty`, only in `hir`.
pub fn implements_trait_unique_with_args<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    trait_: TraitId,
    args: GenericArgs<'db>,
) -> bool {
    implements_trait_unique_impl(db, env, trait_, &mut |_| args)
}

fn implements_trait_unique_impl<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    trait_: TraitId,
    create_args: &mut dyn FnMut(&InferCtxt<'db>) -> GenericArgs<'db>,
) -> bool {
    let interner = DbInterner::new_with(db, env.krate);
    // FIXME(next-solver): I believe this should be `PostAnalysis`.
    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());

    let args = create_args(&infcx);
    let trait_ref = rustc_type_ir::TraitRef::new_from_args(interner, trait_.into(), args);
    let goal = Goal::new(interner, env.param_env, trait_ref);

    let result = crate::traits::next_trait_solve_in_ctxt(&infcx, goal);
    matches!(result, Ok((_, Certainty::Yes)))
}

pub fn is_inherent_impl_coherent(db: &dyn HirDatabase, def_map: &DefMap, impl_id: ImplId) -> bool {
    let self_ty = db.impl_self_ty(impl_id).instantiate_identity();
    let self_ty = self_ty.kind();
    let impl_allowed = match self_ty {
        TyKind::Tuple(_)
        | TyKind::FnDef(_, _)
        | TyKind::Array(_, _)
        | TyKind::Never
        | TyKind::RawPtr(_, _)
        | TyKind::Ref(_, _, _)
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_) => def_map.is_rustc_coherence_is_core(),

        TyKind::Adt(adt_def, _) => adt_def.def_id().0.module(db).krate(db) == def_map.krate(),
        TyKind::Dynamic(it, _) => it
            .principal_def_id()
            .is_some_and(|trait_id| trait_id.0.module(db).krate(db) == def_map.krate()),

        _ => true,
    };
    impl_allowed || {
        let rustc_has_incoherent_inherent_impls = match self_ty {
            TyKind::Tuple(_)
            | TyKind::FnDef(_, _)
            | TyKind::Array(_, _)
            | TyKind::Never
            | TyKind::RawPtr(_, _)
            | TyKind::Ref(_, _, _)
            | TyKind::Slice(_)
            | TyKind::Str
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_) => true,

            TyKind::Adt(adt_def, _) => match adt_def.def_id().0 {
                hir_def::AdtId::StructId(id) => db
                    .struct_signature(id)
                    .flags
                    .contains(StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
                hir_def::AdtId::UnionId(id) => db
                    .union_signature(id)
                    .flags
                    .contains(StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
                hir_def::AdtId::EnumId(it) => db
                    .enum_signature(it)
                    .flags
                    .contains(EnumFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
            },
            TyKind::Dynamic(it, _) => it.principal_def_id().is_some_and(|trait_id| {
                db.trait_signature(trait_id.0)
                    .flags
                    .contains(TraitFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS)
            }),

            _ => false,
        };
        let items = impl_id.impl_items(db);
        rustc_has_incoherent_inherent_impls
            && !items.items.is_empty()
            && items.items.iter().all(|&(_, assoc)| match assoc {
                AssocItemId::FunctionId(it) => {
                    db.function_signature(it).flags.contains(FnFlags::RUSTC_ALLOW_INCOHERENT_IMPL)
                }
                AssocItemId::ConstId(it) => {
                    db.const_signature(it).flags.contains(ConstFlags::RUSTC_ALLOW_INCOHERENT_IMPL)
                }
                AssocItemId::TypeAliasId(it) => db
                    .type_alias_signature(it)
                    .flags
                    .contains(TypeAliasFlags::RUSTC_ALLOW_INCOHERENT_IMPL),
            })
    }
}

/// Checks whether the impl satisfies the orphan rules.
///
/// Given `impl<P1..=Pn> Trait<T1..=Tn> for T0`, an `impl`` is valid only if at least one of the following is true:
/// - Trait is a local trait
/// - All of
///   - At least one of the types `T0..=Tn`` must be a local type. Let `Ti`` be the first such type.
///   - No uncovered type parameters `P1..=Pn` may appear in `T0..Ti`` (excluding `Ti`)
pub fn check_orphan_rules<'db>(db: &'db dyn HirDatabase, impl_: ImplId) -> bool {
    let Some(impl_trait) = db.impl_trait(impl_) else {
        // not a trait impl
        return true;
    };

    let local_crate = impl_.lookup(db).container.krate(db);
    let is_local = |tgt_crate| tgt_crate == local_crate;

    let trait_ref = impl_trait.instantiate_identity();
    let trait_id = trait_ref.def_id.0;
    if is_local(trait_id.module(db).krate(db)) {
        // trait to be implemented is local
        return true;
    }

    let unwrap_fundamental = |mut ty: Ty<'db>| {
        // Unwrap all layers of fundamental types with a loop.
        loop {
            match ty.kind() {
                TyKind::Ref(_, referenced, _) => ty = referenced,
                TyKind::Adt(adt_def, subs) => {
                    let AdtId::StructId(s) = adt_def.def_id().0 else {
                        break ty;
                    };
                    let struct_signature = db.struct_signature(s);
                    if struct_signature.flags.contains(StructFlags::FUNDAMENTAL) {
                        let next = subs.types().next();
                        match next {
                            Some(it) => ty = it,
                            None => break ty,
                        }
                    } else {
                        break ty;
                    }
                }
                _ => break ty,
            }
        }
    };
    //   - At least one of the types `T0..=Tn`` must be a local type. Let `Ti`` be the first such type.

    // FIXME: param coverage
    //   - No uncovered type parameters `P1..=Pn` may appear in `T0..Ti`` (excluding `Ti`)
    let is_not_orphan = trait_ref.args.types().any(|ty| match unwrap_fundamental(ty).kind() {
        TyKind::Adt(adt_def, _) => is_local(adt_def.def_id().0.module(db).krate(db)),
        TyKind::Error(_) => true,
        TyKind::Dynamic(it, _) => {
            it.principal_def_id().is_some_and(|trait_id| is_local(trait_id.0.module(db).krate(db)))
        }
        _ => false,
    });
    #[allow(clippy::let_and_return)]
    is_not_orphan
}
