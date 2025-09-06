//! Normalization within a next-solver infer context.

use std::fmt::Debug;

use rustc_next_trait_solver::placeholder::BoundVarReplacer;
use rustc_type_ir::{
    AliasRelationDirection, FallibleTypeFolder, Flags, Interner, TermKind, TypeFoldable,
    TypeFolder, TypeSuperFoldable, TypeVisitableExt, UniverseIndex,
    inherent::{IntoKind, Span as _, Term as _},
};

use crate::next_solver::{
    Binder, Const, ConstKind, DbInterner, Goal, ParamEnv, Predicate, PredicateKind, Span, Term, Ty,
    TyKind, TypingMode,
    fulfill::{FulfillmentCtxt, NextSolverError},
    infer::{
        DbInternerInferExt, InferCtxt,
        at::At,
        traits::{Obligation, ObligationCause},
    },
    util::PlaceholderReplacer,
};

pub fn normalize<'db, T>(interner: DbInterner<'db>, param_env: ParamEnv<'db>, value: T) -> T
where
    T: TypeFoldable<DbInterner<'db>>,
{
    let infer_ctxt = interner.infer_ctxt().build(TypingMode::non_body_analysis());
    let cause = ObligationCause::dummy();
    deeply_normalize(infer_ctxt.at(&cause, param_env), value.clone()).unwrap_or(value)
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
pub fn deeply_normalize<'db, T>(at: At<'_, 'db>, value: T) -> Result<T, Vec<NextSolverError<'db>>>
where
    T: TypeFoldable<DbInterner<'db>>,
{
    assert!(!value.has_escaping_bound_vars());
    deeply_normalize_with_skipped_universes(at, value, vec![])
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
///
/// Additionally takes a list of universes which represents the binders which have been
/// entered before passing `value` to the function. This is currently needed for
/// `normalize_erasing_regions`, which skips binders as it walks through a type.
pub fn deeply_normalize_with_skipped_universes<'db, T>(
    at: At<'_, 'db>,
    value: T,
    universes: Vec<Option<UniverseIndex>>,
) -> Result<T, Vec<NextSolverError<'db>>>
where
    T: TypeFoldable<DbInterner<'db>>,
{
    let (value, coroutine_goals) =
        deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
            at, value, universes,
        )?;
    assert_eq!(coroutine_goals, vec![]);

    Ok(value)
}

/// Deeply normalize all aliases in `value`. This does not handle inference and expects
/// its input to be already fully resolved.
///
/// Additionally takes a list of universes which represents the binders which have been
/// entered before passing `value` to the function. This is currently needed for
/// `normalize_erasing_regions`, which skips binders as it walks through a type.
///
/// This returns a set of stalled obligations involving coroutines if the typing mode of
/// the underlying infcx has any stalled coroutine def ids.
pub fn deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals<'db, T>(
    at: At<'_, 'db>,
    value: T,
    universes: Vec<Option<UniverseIndex>>,
) -> Result<(T, Vec<Goal<'db, Predicate<'db>>>), Vec<NextSolverError<'db>>>
where
    T: TypeFoldable<DbInterner<'db>>,
{
    let fulfill_cx = FulfillmentCtxt::new(at.infcx);
    let mut folder = NormalizationFolder { at, fulfill_cx, depth: 0, universes };
    let value = value.try_fold_with(&mut folder)?;
    let errors = folder.fulfill_cx.select_all_or_error(at.infcx);
    if errors.is_empty() { Ok((value, vec![])) } else { Err(errors) }
}

struct NormalizationFolder<'me, 'db> {
    at: At<'me, 'db>,
    fulfill_cx: FulfillmentCtxt<'db>,
    depth: usize,
    universes: Vec<Option<UniverseIndex>>,
}

impl<'db> NormalizationFolder<'_, 'db> {
    fn normalize_alias_term(
        &mut self,
        alias_term: Term<'db>,
    ) -> Result<Term<'db>, Vec<NextSolverError<'db>>> {
        let infcx = self.at.infcx;
        let tcx = infcx.interner;
        let recursion_limit = tcx.recursion_limit();
        if self.depth > recursion_limit {
            return Err(vec![]);
        }

        self.depth += 1;

        let infer_term = infcx.next_term_var_of_kind(alias_term);
        let obligation = Obligation::new(
            tcx,
            self.at.cause.clone(),
            self.at.param_env,
            PredicateKind::AliasRelate(alias_term, infer_term, AliasRelationDirection::Equate),
        );

        self.fulfill_cx.register_predicate_obligation(infcx, obligation);
        self.select_all_and_stall_coroutine_predicates()?;

        // Alias is guaranteed to be fully structurally resolved,
        // so we can super fold here.
        let term = infcx.resolve_vars_if_possible(infer_term);
        // super-folding the `term` will directly fold the `Ty` or `Const` so
        // we have to match on the term and super-fold them manually.
        let result = match term.kind() {
            TermKind::Ty(ty) => ty.try_super_fold_with(self)?.into(),
            TermKind::Const(ct) => ct.try_super_fold_with(self)?.into(),
        };
        self.depth -= 1;
        Ok(result)
    }

    fn select_all_and_stall_coroutine_predicates(
        &mut self,
    ) -> Result<(), Vec<NextSolverError<'db>>> {
        let errors = self.fulfill_cx.select_where_possible(self.at.infcx);
        if !errors.is_empty() {
            return Err(errors);
        }

        let errors = self.fulfill_cx.collect_remaining_errors(self.at.infcx);
        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(())
    }
}

impl<'db> FallibleTypeFolder<DbInterner<'db>> for NormalizationFolder<'_, 'db> {
    type Error = Vec<NextSolverError<'db>>;

    fn cx(&self) -> DbInterner<'db> {
        self.at.infcx.interner
    }

    fn try_fold_binder<T: TypeFoldable<DbInterner<'db>>>(
        &mut self,
        t: Binder<'db, T>,
    ) -> Result<Binder<'db, T>, Self::Error> {
        self.universes.push(None);
        let t = t.try_super_fold_with(self)?;
        self.universes.pop();
        Ok(t)
    }

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn try_fold_ty(&mut self, ty: Ty<'db>) -> Result<Ty<'db>, Self::Error> {
        let infcx = self.at.infcx;
        debug_assert_eq!(ty, infcx.shallow_resolve(ty));
        if !ty.has_aliases() {
            return Ok(ty);
        }

        let TyKind::Alias(..) = ty.kind() else { return ty.try_super_fold_with(self) };

        if ty.has_escaping_bound_vars() {
            let (ty, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, ty);
            let result = self.normalize_alias_term(ty.into())?.expect_type();
            Ok(PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                result,
            ))
        } else {
            Ok(self.normalize_alias_term(ty.into())?.expect_type())
        }
    }

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn try_fold_const(&mut self, ct: Const<'db>) -> Result<Const<'db>, Self::Error> {
        let infcx = self.at.infcx;
        debug_assert_eq!(ct, infcx.shallow_resolve_const(ct));
        if !ct.has_aliases() {
            return Ok(ct);
        }

        let ConstKind::Unevaluated(..) = ct.kind() else { return ct.try_super_fold_with(self) };

        if ct.has_escaping_bound_vars() {
            let (ct, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, ct);
            let result = self.normalize_alias_term(ct.into())?.expect_const();
            Ok(PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                result,
            ))
        } else {
            Ok(self.normalize_alias_term(ct.into())?.expect_const())
        }
    }
}

// Deeply normalize a value and return it
pub(crate) fn deeply_normalize_for_diagnostics<'db, T: TypeFoldable<DbInterner<'db>>>(
    infcx: &InferCtxt<'db>,
    param_env: ParamEnv<'db>,
    t: T,
) -> T {
    t.fold_with(&mut DeeplyNormalizeForDiagnosticsFolder {
        at: infcx.at(&ObligationCause::dummy(), param_env),
    })
}

struct DeeplyNormalizeForDiagnosticsFolder<'a, 'db> {
    at: At<'a, 'db>,
}

impl<'db> TypeFolder<DbInterner<'db>> for DeeplyNormalizeForDiagnosticsFolder<'_, 'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.at.infcx.interner
    }

    fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        let infcx = self.at.infcx;
        let result: Result<_, Vec<NextSolverError<'db>>> = infcx.commit_if_ok(|_| {
            deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                self.at,
                ty,
                vec![None; ty.outer_exclusive_binder().as_usize()],
            )
        });
        match result {
            Ok((ty, _)) => ty,
            Err(_) => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        let infcx = self.at.infcx;
        let result: Result<_, Vec<NextSolverError<'db>>> = infcx.commit_if_ok(|_| {
            deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                self.at,
                ct,
                vec![None; ct.outer_exclusive_binder().as_usize()],
            )
        });
        match result {
            Ok((ct, _)) => ct,
            Err(_) => ct.super_fold_with(self),
        }
    }
}
