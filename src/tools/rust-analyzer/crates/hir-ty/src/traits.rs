//! Trait solving using next trait solver.

use core::fmt;
use std::hash::Hash;

use base_db::Crate;
use hir_def::{BlockId, TraitId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use rustc_next_trait_solver::solve::{HasChanged, SolverDelegateEvalExt};
use rustc_type_ir::{
    TypingMode,
    inherent::{IntoKind, Span as _},
    solve::Certainty,
};
use triomphe::Arc;

use crate::{
    db::HirDatabase,
    next_solver::{
        Canonical, DbInterner, GenericArgs, Goal, ParamEnv, Predicate, SolverContext, Span, Ty,
        TyKind,
        infer::{DbInternerInferExt, InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

/// A set of clauses that we assume to be true. E.g. if we are inside this function:
/// ```rust
/// fn foo<T: Default>(t: T) {}
/// ```
/// we assume that `T: Default`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitEnvironment<'db> {
    pub krate: Crate,
    pub block: Option<BlockId>,
    // FIXME make this a BTreeMap
    traits_from_clauses: Box<[(Ty<'db>, TraitId)]>,
    pub env: ParamEnv<'db>,
}

impl<'db> TraitEnvironment<'db> {
    pub fn empty(krate: Crate) -> Arc<Self> {
        Arc::new(TraitEnvironment {
            krate,
            block: None,
            traits_from_clauses: Box::default(),
            env: ParamEnv::empty(),
        })
    }

    pub fn new(
        krate: Crate,
        block: Option<BlockId>,
        traits_from_clauses: Box<[(Ty<'db>, TraitId)]>,
        env: ParamEnv<'db>,
    ) -> Arc<Self> {
        Arc::new(TraitEnvironment { krate, block, traits_from_clauses, env })
    }

    // pub fn with_block(self: &mut Arc<Self>, block: BlockId) {
    pub fn with_block(this: &mut Arc<Self>, block: BlockId) {
        Arc::make_mut(this).block = Some(block);
    }

    pub fn traits_in_scope_from_clauses(&self, ty: Ty<'db>) -> impl Iterator<Item = TraitId> + '_ {
        self.traits_from_clauses
            .iter()
            .filter_map(move |(self_ty, trait_id)| (*self_ty == ty).then_some(*trait_id))
    }
}

/// This should be used in `hir` only.
pub fn structurally_normalize_ty<'db>(
    infcx: &InferCtxt<'db>,
    ty: Ty<'db>,
    env: Arc<TraitEnvironment<'db>>,
) -> Ty<'db> {
    let TyKind::Alias(..) = ty.kind() else { return ty };
    let mut ocx = ObligationCtxt::new(infcx);
    let ty = ocx.structurally_normalize_ty(&ObligationCause::dummy(), env.env, ty).unwrap_or(ty);
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl fmt::Display for FnTrait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FnTrait::FnOnce => write!(f, "FnOnce"),
            FnTrait::FnMut => write!(f, "FnMut"),
            FnTrait::Fn => write!(f, "Fn"),
            FnTrait::AsyncFnOnce => write!(f, "AsyncFnOnce"),
            FnTrait::AsyncFnMut => write!(f, "AsyncFnMut"),
            FnTrait::AsyncFn => write!(f, "AsyncFn"),
        }
    }
}

impl FnTrait {
    pub const fn function_name(&self) -> &'static str {
        match self {
            FnTrait::FnOnce => "call_once",
            FnTrait::FnMut => "call_mut",
            FnTrait::Fn => "call",
            FnTrait::AsyncFnOnce => "async_call_once",
            FnTrait::AsyncFnMut => "async_call_mut",
            FnTrait::AsyncFn => "async_call",
        }
    }

    const fn lang_item(self) -> LangItem {
        match self {
            FnTrait::FnOnce => LangItem::FnOnce,
            FnTrait::FnMut => LangItem::FnMut,
            FnTrait::Fn => LangItem::Fn,
            FnTrait::AsyncFnOnce => LangItem::AsyncFnOnce,
            FnTrait::AsyncFnMut => LangItem::AsyncFnMut,
            FnTrait::AsyncFn => LangItem::AsyncFn,
        }
    }

    pub const fn from_lang_item(lang_item: LangItem) -> Option<Self> {
        match lang_item {
            LangItem::FnOnce => Some(FnTrait::FnOnce),
            LangItem::FnMut => Some(FnTrait::FnMut),
            LangItem::Fn => Some(FnTrait::Fn),
            LangItem::AsyncFnOnce => Some(FnTrait::AsyncFnOnce),
            LangItem::AsyncFnMut => Some(FnTrait::AsyncFnMut),
            LangItem::AsyncFn => Some(FnTrait::AsyncFn),
            _ => None,
        }
    }

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

    pub fn get_id(self, db: &dyn HirDatabase, krate: Crate) -> Option<TraitId> {
        self.lang_item().resolve_trait(db, krate)
    }
}

/// This should not be used in `hir-ty`, only in `hir`.
pub fn implements_trait_unique<'db>(
    ty: Ty<'db>,
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    trait_: TraitId,
) -> bool {
    implements_trait_unique_impl(db, env, trait_, &mut |infcx| {
        infcx.fill_rest_fresh_args(trait_.into(), [ty.into()])
    })
}

/// This should not be used in `hir-ty`, only in `hir`.
pub fn implements_trait_unique_with_args<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    trait_: TraitId,
    args: GenericArgs<'db>,
) -> bool {
    implements_trait_unique_impl(db, env, trait_, &mut |_| args)
}

fn implements_trait_unique_impl<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    trait_: TraitId,
    create_args: &mut dyn FnMut(&InferCtxt<'db>) -> GenericArgs<'db>,
) -> bool {
    let interner = DbInterner::new_with(db, Some(env.krate), env.block);
    // FIXME(next-solver): I believe this should be `PostAnalysis`.
    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());

    let args = create_args(&infcx);
    let trait_ref = rustc_type_ir::TraitRef::new_from_args(interner, trait_.into(), args);
    let goal = Goal::new(interner, env.env, trait_ref);

    let result = crate::traits::next_trait_solve_in_ctxt(&infcx, goal);
    matches!(result, Ok((_, Certainty::Yes)))
}
