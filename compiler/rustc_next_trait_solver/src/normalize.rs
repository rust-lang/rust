use std::fmt::Debug;

use rustc_data_structures::sso::{SsoHashMap, SsoHashSet};
use rustc_type_ir::data_structures::ensure_sufficient_stack;
use rustc_type_ir::inherent::*;
use rustc_type_ir::{
    self as ty, AliasTerm, Binder, FallibleTypeFolder, InferConst, InferCtxtLike, InferTy,
    Interner, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor, UniverseIndex,
};
use tracing::instrument;

use crate::placeholder::{BoundVarReplacer, PlaceholderReplacer};

/// This folder normalizes value and collects ambiguous goals.
///
/// Note that for ambiguous alias which contains escaping bound vars,
/// we just return the original alias.
pub struct NormalizationFolder<'a, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
    universes: Vec<Option<UniverseIndex>>,
    normalize: F,
    cache: SsoHashMap<I::Ty, I::Ty>,
}

#[derive(PartialEq, Eq, Debug)]
pub enum NormalizationWasAmbiguous {
    Yes,
    No,
}

#[derive(PartialEq, Eq, Debug)]
enum HasEscapingBoundVars {
    Yes,
    No,
}

/// Finds the max universe present in infer vars.
struct MaxUniverse<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
    max_universe: ty::UniverseIndex,
    cache: SsoHashSet<I::Ty>,
}

impl<'a, Infcx, I> MaxUniverse<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn new(infcx: &'a Infcx) -> Self {
        MaxUniverse { infcx, max_universe: ty::UniverseIndex::ROOT, cache: Default::default() }
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

        if self.cache.insert(t) {
            t.super_visit_with(self)
        }
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

impl<'a, Infcx, I, F, E> NormalizationFolder<'a, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(AliasTerm<I>) -> Result<(I::Term, NormalizationWasAmbiguous), E>,
{
    pub fn new(infcx: &'a Infcx, universes: Vec<Option<UniverseIndex>>, normalize: F) -> Self {
        Self { infcx, universes, normalize, cache: Default::default() }
    }

    fn normalize_alias_term(
        &mut self,
        alias_term: AliasTerm<I>,
        has_escaping: HasEscapingBoundVars,
    ) -> Result<Option<I::Term>, E> {
        let current_universe = self.infcx.universe();
        self.infcx.create_next_universe();

        let (normalized, normalization_was_ambiguous) = (self.normalize)(alias_term)?;

        // Return ambiguous higher ranked alias as is, if
        //   - it contains escaping vars, and
        //   - the normalized term contains infer vars newly created
        //     in the normalization above.
        // The problem is that they may be resolved to types
        // referencing the temporary placeholders.
        //
        // We can normalize the ambiguous alias again after the binder is instantiated.
        if NormalizationWasAmbiguous::Yes == normalization_was_ambiguous
            && has_escaping == HasEscapingBoundVars::Yes
        {
            let mut visitor = MaxUniverse::new(self.infcx);
            normalized.visit_with(&mut visitor);
            let max_universe = visitor.max_universe();
            if current_universe.cannot_name(max_universe) {
                return Ok(None);
            }
        }

        Ok(Some(normalized))
    }
}

impl<'a, Infcx, I, F, E> FallibleTypeFolder<I> for NormalizationFolder<'a, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(AliasTerm<I>) -> Result<(I::Term, NormalizationWasAmbiguous), E>,
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

        if !self.cx().renormalize_rigid_aliases() && !ty.has_non_rigid_aliases() {
            return Ok(ty);
        }

        if let Some(ty) = self.cache.get(&ty) {
            return Ok(*ty);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let folded_ty = ty.try_super_fold_with(self)?;
        let ty::Alias(alias_ty) = folded_ty.kind() else { return Ok(folded_ty) };
        // We support ambiguous aliases inside rigid alias. So we still recognize
        // the rigidness of the outer alias.
        if !self.cx().renormalize_rigid_aliases() && alias_ty.is_rigid == ty::IsRigid::Yes {
            return Ok(folded_ty);
        }

        let result = if alias_ty.has_escaping_bound_vars() {
            let (alias_ty, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, alias_ty);
            match ensure_sufficient_stack(|| {
                self.normalize_alias_term(alias_ty.into(), HasEscapingBoundVars::Yes)
            })? {
                Some(result) => PlaceholderReplacer::replace_placeholders(
                    infcx,
                    mapped_regions,
                    mapped_types,
                    mapped_consts,
                    &self.universes,
                    result.expect_ty(),
                ),
                None => folded_ty,
            }
        } else {
            match ensure_sufficient_stack(|| {
                self.normalize_alias_term(alias_ty.into(), HasEscapingBoundVars::No)
            })? {
                Some(term) => term.expect_ty(),
                None => folded_ty,
            }
        };

        assert!(self.cache.insert(ty, result).is_none(), "{ty:?} {result:?} {:?}", self.cache);

        if self.cx().renormalize_rigid_aliases()
            && let ty::Alias(alias) = ty.kind()
            && alias.is_rigid == ty::IsRigid::Yes
        {
            // find out missing typing env change.
            let original = crate::resolve::eager_resolve_vars_with_infcx(infcx, ty);
            let normalized = crate::resolve::eager_resolve_vars_with_infcx(infcx, result);
            assert_eq!(original, normalized, "rigid alias is further normalized");
        }
        Ok(result)
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn try_fold_const(&mut self, ct: I::Const) -> Result<I::Const, Self::Error> {
        let infcx = self.infcx;
        if !self.cx().renormalize_rigid_aliases() && !ct.has_non_rigid_aliases() {
            return Ok(ct);
        }

        // With eager normalization, we should normalize the args of alias before
        // normalizing the alias itself.
        let ct = ct.try_super_fold_with(self)?;
        let ty::ConstKind::Unevaluated(uv) = ct.kind() else { return Ok(ct) };
        // We support ambiguous aliases inside rigid alias. So we still recognize
        // the rigidness of the outer alias.
        if !self.cx().renormalize_rigid_aliases() && uv.is_rigid == ty::IsRigid::Yes {
            return Ok(ct);
        }

        if ct.has_escaping_bound_vars() {
            let (uv, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, uv);
            match ensure_sufficient_stack(|| {
                self.normalize_alias_term(uv.into(), HasEscapingBoundVars::Yes)
            })? {
                Some(result) => Ok(PlaceholderReplacer::replace_placeholders(
                    infcx,
                    mapped_regions,
                    mapped_types,
                    mapped_consts,
                    &self.universes,
                    result.expect_const(),
                )),
                None => Ok(ct),
            }
        } else {
            match ensure_sufficient_stack(|| {
                self.normalize_alias_term(uv.into(), HasEscapingBoundVars::No)
            })? {
                Some(term) => Ok(term.expect_const()),
                None => Ok(ct),
            }
        }
    }

    fn try_fold_predicate(&mut self, p: I::Predicate) -> Result<I::Predicate, Self::Error> {
        if p.allow_normalization() { p.try_super_fold_with(self) } else { Ok(p) }
    }
}
