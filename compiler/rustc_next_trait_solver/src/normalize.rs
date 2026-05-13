use std::fmt::Debug;
use std::marker::PhantomData;

use rustc_type_ir::data_structures::ensure_sufficient_stack;
use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::Goal;
use rustc_type_ir::{
    self as ty, Binder, FallibleTypeFolder, InferConst, InferCtxtLike, InferTy, Interner,
    TypeFoldable, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor, UniverseIndex,
};
use tracing::instrument;

use crate::placeholder::{BoundVarReplacer, PlaceholderReplacer};

/// This folder normalizes value and collects ambiguous goals.
///
/// Note that for ambiguous alias which contains escaping bound vars,
/// we just return the original alias and don't collect the ambiguous goal.
pub struct NormalizationFolder<'a, Infcx, I, F, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
    universes: Vec<Option<UniverseIndex>>,
    stalled_goals: Vec<Goal<I, I::Predicate>>,
    normalize: F,
    _error: PhantomData<E>,
}

#[derive(PartialEq, Eq)]
enum HasEscapingBoundVars {
    Yes,
    No,
}

#[derive(PartialEq, Eq)]
pub enum NormalizationScope {
    All,
    AmbiguousAlias,
}

/// Finds the max universe present in infer vars.
struct MaxUniverse<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
    max_universe: ty::UniverseIndex,
}

impl<'a, Infcx, I> MaxUniverse<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn new(infcx: &'a Infcx) -> Self {
        MaxUniverse { infcx, max_universe: ty::UniverseIndex::ROOT }
    }

    fn max_universe(self) -> ty::UniverseIndex {
        self.max_universe
    }
}

impl<'a, Infcx, I> TypeVisitor<I> for MaxUniverse<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    type Result = ();

    fn visit_ty(&mut self, t: I::Ty) {
        if !t.has_infer() {
            return;
        }

        if let ty::Infer(InferTy::TyVar(vid)) = t.kind() {
            // We shallow resolved the infer var before.
            // So it should be a unresolved infer var with an universe.
            self.max_universe = self.max_universe.max(self.infcx.universe_of_ty(vid).unwrap());
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: I::Const) {
        if !c.has_infer() {
            return;
        }

        if let ty::ConstKind::Infer(InferConst::Var(vid)) = c.kind() {
            // We shallow resolved the infer var before.
            // So it should be a unresolved infer var with an universe.
            self.max_universe = self.max_universe.max(self.infcx.universe_of_ct(vid).unwrap());
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: I::Region) {
        if let ty::ReVar(vid) = r.kind() {
            self.max_universe = self.max_universe.max(self.infcx.universe_of_lt(vid).unwrap());
        }
    }
}

#[derive(PartialEq, Eq)]
enum NeedRenormalization {
    Yes,
    No,
}

impl<'a, Infcx, I, F, E> NormalizationFolder<'a, Infcx, I, F, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(I::Term) -> Result<(I::Term, Option<Goal<I, I::Predicate>>), E>,
    E: Debug,
{
    pub fn new(
        infcx: &'a Infcx,
        universes: Vec<Option<UniverseIndex>>,
        stalled_goals: Vec<Goal<I, I::Predicate>>,
        normalize: F,
    ) -> Self {
        Self { infcx, universes, stalled_goals, normalize, _error: PhantomData }
    }

    pub fn stalled_goals(self) -> Vec<Goal<I, I::Predicate>> {
        self.stalled_goals
    }

    fn normalize_alias_term(
        &mut self,
        alias_term: I::Term,
        has_escaping: HasEscapingBoundVars,
    ) -> Result<(I::Term, NeedRenormalization), E> {
        let current_universe = self.infcx.universe();
        self.infcx.create_next_universe();

        let (normalized, ambig_goal) = (self.normalize)(alias_term)?;

        // Return ambiguous higher ranked alias as is, if
        //   - it contains escaping vars, and
        //   - the normalized term contains infer vars newly created
        //     in the normalization above.
        // The problem is that they may be resolved to types
        // referencing the temporary placeholders.
        //
        // We can normalize the ambiguous alias again after the binder is instantiated.
        if ambig_goal.is_some() && has_escaping == HasEscapingBoundVars::Yes {
            let mut visitor = MaxUniverse::new(self.infcx);
            normalized.visit_with(&mut visitor);
            let max_universe = visitor.max_universe();
            if current_universe.cannot_name(max_universe) {
                return Ok((alias_term, NeedRenormalization::Yes));
            }
        }

        self.stalled_goals.extend(ambig_goal);
        Ok((normalized, NeedRenormalization::No))
    }
}

impl<'a, Infcx, I, F, E> FallibleTypeFolder<I> for NormalizationFolder<'a, Infcx, I, F, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(I::Term) -> Result<(I::Term, Option<Goal<I, I::Predicate>>), E>,
    E: Debug,
{
    type Error = E;

    fn cx(&self) -> I {
        self.infcx.cx()
    }

    fn try_fold_binder<T: TypeFoldable<I>>(
        &mut self,
        t: Binder<I, T>,
    ) -> Result<Binder<I, T>, Self::Error> {
        self.universes.push(None);
        let t = t.try_super_fold_with(self)?;
        self.universes.pop();
        Ok(t)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn try_fold_ty(&mut self, ty: I::Ty) -> Result<I::Ty, Self::Error> {
        let infcx = self.infcx;
        if !ty.has_aliases() {
            return Ok(ty);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ty = ty.try_super_fold_with(self)?;
        let ty::Alias(alias_ty) = ty.kind() else { return Ok(ty) };

        if ty.has_escaping_bound_vars() {
            let (ty, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, ty);
            let (normalized_term, need_renormalization) = ensure_sufficient_stack(|| {
                self.normalize_alias_term(ty.into(), HasEscapingBoundVars::Yes)
            })?;
            let normalized_ty = PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                normalized_term.expect_ty(),
            );
            if need_renormalization == NeedRenormalization::Yes {
                Ok(I::Ty::new_alias(
                    self.cx(),
                    ty::AliasTy::new_from_args(
                        self.cx(),
                        ty::AliasTyKind::Ambiguous,
                        self.cx().mk_args(&[normalized_ty.into()]),
                    ),
                ))
            } else {
                Ok(normalized_ty)
            }
        } else {
            let alias =
                if let ty::Ambiguous = alias_ty.kind { alias_ty.args.type_at(0) } else { ty };
            Ok(ensure_sufficient_stack(|| {
                self.normalize_alias_term(alias.into(), HasEscapingBoundVars::No)
            })?
            .0
            .expect_ty())
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn try_fold_const(&mut self, ct: I::Const) -> Result<I::Const, Self::Error> {
        let infcx = self.infcx;
        if !ct.has_aliases() {
            return Ok(ct);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ct = ct.try_super_fold_with(self)?;
        let ty::ConstKind::Unevaluated(..) = ct.kind() else { return Ok(ct) };

        if ct.has_escaping_bound_vars() {
            let (ct, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, ct);
            let result = ensure_sufficient_stack(|| {
                self.normalize_alias_term(ct.into(), HasEscapingBoundVars::Yes)
            })?
            .0
            .expect_const();
            Ok(PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                result,
            ))
        } else {
            Ok(ensure_sufficient_stack(|| {
                self.normalize_alias_term(ct.into(), HasEscapingBoundVars::No)
            })?
            .0
            .expect_const())
        }
    }
}

// Only handle ambiguous alias in the outmost instantiated binder.
pub struct BinderRenormalizer<'a, Infcx, I, F, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
    stalled_goals: Vec<Goal<I, I::Predicate>>,
    normalize: F,
    _error: PhantomData<E>,
}

impl<'a, Infcx, I, F, E> BinderRenormalizer<'a, Infcx, I, F, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(I::Term) -> Result<(I::Term, Option<Goal<I, I::Predicate>>), E>,
    E: Debug,
{
    pub fn new(infcx: &'a Infcx, stalled_goals: Vec<Goal<I, I::Predicate>>, normalize: F) -> Self {
        Self { infcx, stalled_goals, normalize, _error: PhantomData }
    }

    pub fn stalled_goals(self) -> Vec<Goal<I, I::Predicate>> {
        self.stalled_goals
    }

    fn normalize_alias_term(&mut self, alias_term: I::Term) -> Result<I::Term, E> {
        let (normalized, ambig_goal) = (self.normalize)(alias_term)?;

        self.stalled_goals.extend(ambig_goal);
        Ok(normalized)
    }
}

impl<'a, Infcx, I, F, E> FallibleTypeFolder<I> for BinderRenormalizer<'a, Infcx, I, F, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(I::Term) -> Result<(I::Term, Option<Goal<I, I::Predicate>>), E>,
    E: Debug,
{
    type Error = E;

    fn cx(&self) -> I {
        self.infcx.cx()
    }

    fn try_fold_binder<T: TypeFoldable<I>>(
        &mut self,
        t: Binder<I, T>,
    ) -> Result<Binder<I, T>, Self::Error> {
        // We don't look into nested binders since they're not instantiated.
        // Thus we assume we won't meet escaping bound vars as well.
        Ok(t)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn try_fold_ty(&mut self, ty: I::Ty) -> Result<I::Ty, Self::Error> {
        debug_assert!(!ty.has_escaping_bound_vars());
        if !ty.has_ambiguous_aliases() {
            return Ok(ty);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ty = ty.try_super_fold_with(self)?;
        let ty::Alias(ty::AliasTy { kind: ty::AliasTyKind::Ambiguous, args, .. }) = ty.kind()
        else {
            return Ok(ty);
        };

        let original_alias = args.type_at(0);
        Ok(ensure_sufficient_stack(|| self.normalize_alias_term(original_alias.into()))?
            .expect_ty())
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn try_fold_const(&mut self, ct: I::Const) -> Result<I::Const, Self::Error> {
        debug_assert!(!ct.has_escaping_bound_vars());
        if !ct.has_ambiguous_aliases() {
            return Ok(ct);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ct = ct.try_super_fold_with(self)?;
        let ty::ConstKind::Unevaluated(..) = ct.kind() else { return Ok(ct) };
        // FIXME: add an new `Ambiguous` kind to `UnevaluatedConst` as well.
        // As a field or a new kind on `ConstKind`?
        Ok(ct)
    }
}
