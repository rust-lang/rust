use std::fmt::Debug;

use rustc_type_ir::inherent::*;
use rustc_type_ir::{
    self as ty, AliasTerm, Binder, Const, FallibleTypeFolder, InferCtxtLike, Interner,
    PredicateProxy, TypeFoldable, TypeSuperFoldable, TypeVisitableExt, UniverseIndex,
};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::placeholder::{BoundVarReplacer, PlaceholderReplacer};

/// This folder normalizes value and collects ambiguous goals.
///
/// Note that for ambiguous alias which contains escaping bound vars,
/// we just return the original alias and don't collect the ambiguous goal.
pub struct NormalizationFolder<'a, D, I, F>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    delegate: &'a D,
    universes: Vec<Option<UniverseIndex>>,
    depth: usize,
    normalize: F,
}

#[derive(PartialEq, Eq)]
pub enum NormalizationWasAmbiguous {
    Yes,
    No,
}

impl<'a, D, I, F, E> NormalizationFolder<'a, D, I, F>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    F: FnMut(AliasTerm<I>) -> Result<(I::Term, NormalizationWasAmbiguous), E>,
    E: Debug,
{
    pub fn new(delegate: &'a D, universes: Vec<Option<UniverseIndex>>, normalize: F) -> Self {
        Self { delegate, universes, depth: 0, normalize }
    }

    fn normalize_alias_term(&mut self, alias_term: AliasTerm<I>) -> Result<Option<I::Term>, E> {
        let Some(normalized) = self.normalize_inner(alias_term)? else {
            return Ok(None);
        };

        // The solver can return a term which still contains non-rigid aliases with escaping
        // bound vars, as the generalizer does not replace those with inference vars. In this
        // case, also want those aliases to be normalized.
        //
        // This does not terminate if an alias effectively normalizes to itself. This should
        // be incredibly rare, so we simply abort compilation at the recursion limit.
        if !normalized.has_non_rigid_aliases() {
            return Ok(Some(normalized));
        }
        if self.depth >= self.cx().recursion_limit() {
            self.delegate.report_normalization_overflow(alias_term);
        }
        self.depth += 1;
        let normalized = normalized.try_fold_with(self);
        self.depth -= 1;
        normalized.map(Some)
    }

    fn normalize_inner(&mut self, alias_term: AliasTerm<I>) -> Result<Option<I::Term>, E> {
        let infcx = &**self.delegate;
        let normalized = if alias_term.has_escaping_bound_vars() {
            let (replaced, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, alias_term);
            let (normalized, normalization_was_ambiguous) = (self.normalize)(replaced)?;

            // Return ambiguous higher ranked alias as is, if the normalized term
            // contains infer vars which may mention temporary placeholders after
            // we've already mapped them back to bound vars.
            //
            // We can normalize the ambiguous alias again after the binder is instantiated.
            if normalization_was_ambiguous == NormalizationWasAmbiguous::Yes {
                let max_universe = ty::max_universe_of_infer_vars(infcx, normalized);
                if max_universe.can_name(self.universes.first().unwrap().unwrap()) {
                    return Ok(None);
                }
            }

            PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                normalized,
            )
        } else {
            (self.normalize)(alias_term)?.0
        };

        Ok(Some(normalized))
    }
}

impl<'a, D, I, F, E> FallibleTypeFolder<I> for NormalizationFolder<'a, D, I, F>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    F: FnMut(AliasTerm<I>) -> Result<(I::Term, NormalizationWasAmbiguous), E>,
    E: Debug,
{
    type Error = E;

    fn cx(&self) -> I {
        self.delegate.cx()
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
        let infcx = &**self.delegate;
        let original = ty;

        if !self.cx().renormalize_rigid_aliases() && !ty.has_non_rigid_aliases() {
            return Ok(ty);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ty = ty.try_super_fold_with(self)?;
        let ty::Alias(orig_is_rigid, alias_ty) = ty.kind() else { return Ok(ty) };
        // We support ambiguous aliases inside rigid alias. So we still recognize
        // the rigidness of the outer alias.
        if !self.cx().renormalize_rigid_aliases() && orig_is_rigid == ty::IsRigid::Yes {
            return Ok(ty);
        }

        let Some(normalized) = self.normalize_alias_term(alias_ty.into())? else {
            return Ok(ty);
        };
        let normalized = normalized.expect_ty();

        if self.cx().renormalize_rigid_aliases() && orig_is_rigid == ty::IsRigid::Yes {
            // find out missing typing env change.
            let original = infcx.deeply_resolve_via_unification_table(original);
            let normalized = infcx.deeply_resolve_via_unification_table(normalized);
            assert_eq!(original, normalized, "rigid alias is further normalized");
        }
        Ok(normalized)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn try_fold_const(&mut self, ct: Const<I>) -> Result<Const<I>, Self::Error> {
        let infcx = &**self.delegate;
        let original = ct;

        if !self.cx().renormalize_rigid_aliases() && !ct.has_non_rigid_aliases() {
            return Ok(ct);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ct = ct.try_super_fold_with(self)?;
        let ty::ConstKind::Alias(orig_is_rigid, alias_const) = ct.kind() else { return Ok(ct) };
        // We support ambiguous aliases inside rigid alias. So we still recognize
        // the rigidness of the outer alias.
        if !self.cx().renormalize_rigid_aliases() && orig_is_rigid == ty::IsRigid::Yes {
            return Ok(ct);
        }

        let Some(normalized) = self.normalize_alias_term(alias_const.into())? else {
            return Ok(ct);
        };
        let normalized = normalized.expect_const();

        if self.cx().renormalize_rigid_aliases() && orig_is_rigid == ty::IsRigid::Yes {
            // find out missing typing env change.
            let original = infcx.deeply_resolve_via_unification_table(original);
            let normalized = infcx.deeply_resolve_via_unification_table(normalized);
            assert_eq!(original, normalized, "rigid alias is further normalized");
        }

        Ok(normalized)
    }

    fn try_fold_predicate<P: PredicateProxy<I>>(&mut self, p: P) -> Result<P, Self::Error> {
        if p.allow_normalization() { p.try_super_fold_with(self) } else { Ok(p) }
    }
}
