//! In certain situations, rust automatically inserts derefs as necessary: for
//! example, field accesses `foo.bar` still work when `foo` is actually a
//! reference to a type with the field `bar`. This is an approximation of the
//! logic in rustc (which lives in librustc_typeck/check/autoderef.rs).

use std::iter::successors;

use base_db::CrateId;
use chalk_ir::{cast::Cast, fold::Fold, interner::HasInterner, VariableKind};
use hir_def::lang_item::LangItemTarget;
use hir_expand::name::name;
use limit::Limit;
use syntax::SmolStr;
use tracing::{info, warn};

use crate::{
    db::HirDatabase, static_lifetime, AliasEq, AliasTy, BoundVar, Canonical, CanonicalVarKinds,
    ConstrainedSubst, DebruijnIndex, Environment, Guidance, InEnvironment, Interner,
    ProjectionTyExt, Solution, Substitution, Ty, TyBuilder, TyKind,
};

static AUTODEREF_RECURSION_LIMIT: Limit = Limit::new(10);

pub(crate) enum AutoderefKind {
    Builtin,
    Overloaded,
}

pub(crate) struct Autoderef<'db> {
    db: &'db dyn HirDatabase,
    ty: Canonical<Ty>,
    at_start: bool,
    krate: Option<CrateId>,
    environment: Environment,
    steps: Vec<(AutoderefKind, Ty)>,
}

impl<'db> Autoderef<'db> {
    pub(crate) fn new(
        db: &'db dyn HirDatabase,
        krate: Option<CrateId>,
        ty: InEnvironment<Canonical<Ty>>,
    ) -> Self {
        let InEnvironment { goal: ty, environment } = ty;
        Autoderef { db, ty, at_start: true, environment, krate, steps: Vec::new() }
    }

    pub(crate) fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub(crate) fn steps(&self) -> &[(AutoderefKind, chalk_ir::Ty<Interner>)] {
        &self.steps
    }

    pub(crate) fn final_ty(&self) -> Ty {
        self.ty.value.clone()
    }
}

impl Iterator for Autoderef<'_> {
    type Item = (Canonical<Ty>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.at_start {
            self.at_start = false;
            return Some((self.ty.clone(), 0));
        }

        if AUTODEREF_RECURSION_LIMIT.check(self.steps.len() + 1).is_err() {
            return None;
        }

        let (kind, new_ty) = if let Some(derefed) = builtin_deref(&self.ty.value) {
            (
                AutoderefKind::Builtin,
                Canonical { value: derefed.clone(), binders: self.ty.binders.clone() },
            )
        } else {
            (
                AutoderefKind::Overloaded,
                deref_by_trait(
                    self.db,
                    self.krate?,
                    InEnvironment { goal: &self.ty, environment: self.environment.clone() },
                )?,
            )
        };

        self.steps.push((kind, self.ty.value.clone()));
        self.ty = new_ty;

        Some((self.ty.clone(), self.step_count()))
    }
}

// FIXME: replace uses of this with Autoderef above
pub fn autoderef<'a>(
    db: &'a dyn HirDatabase,
    krate: Option<CrateId>,
    ty: InEnvironment<Canonical<Ty>>,
) -> impl Iterator<Item = Canonical<Ty>> + 'a {
    let InEnvironment { goal: ty, environment } = ty;
    successors(Some(ty), move |ty| {
        deref(db, krate?, InEnvironment { goal: ty, environment: environment.clone() })
    })
    .take(AUTODEREF_RECURSION_LIMIT.inner())
}

pub(crate) fn deref(
    db: &dyn HirDatabase,
    krate: CrateId,
    ty: InEnvironment<&Canonical<Ty>>,
) -> Option<Canonical<Ty>> {
    let _p = profile::span("deref");
    match builtin_deref(&ty.goal.value) {
        Some(derefed) => {
            Some(Canonical { value: derefed.clone(), binders: ty.goal.binders.clone() })
        }
        None => deref_by_trait(db, krate, ty),
    }
}

fn builtin_deref(ty: &Ty) -> Option<&Ty> {
    match ty.kind(Interner) {
        TyKind::Ref(.., ty) => Some(ty),
        TyKind::Raw(.., ty) => Some(ty),
        _ => None,
    }
}

fn deref_by_trait(
    db: &dyn HirDatabase,
    krate: CrateId,
    ty: InEnvironment<&Canonical<Ty>>,
) -> Option<Canonical<Ty>> {
    let _p = profile::span("deref_by_trait");
    let deref_trait = match db.lang_item(krate, SmolStr::new_inline("deref"))? {
        LangItemTarget::TraitId(it) => it,
        _ => return None,
    };
    let target = db.trait_data(deref_trait).associated_type_by_name(&name![Target])?;

    let projection = {
        let b = TyBuilder::assoc_type_projection(db, target);
        if b.remaining() != 1 {
            // the Target type + Deref trait should only have one generic parameter,
            // namely Deref's Self type
            return None;
        }
        b.push(ty.goal.value.clone()).build()
    };

    // FIXME make the Canonical / bound var handling nicer

    // Check that the type implements Deref at all
    let trait_ref = projection.trait_ref(db);
    let implements_goal = Canonical {
        binders: ty.goal.binders.clone(),
        value: InEnvironment {
            goal: trait_ref.cast(Interner),
            environment: ty.environment.clone(),
        },
    };
    if db.trait_solve(krate, implements_goal).is_none() {
        return None;
    }

    // Now do the assoc type projection
    let alias_eq = AliasEq {
        alias: AliasTy::Projection(projection),
        ty: TyKind::BoundVar(BoundVar::new(
            DebruijnIndex::INNERMOST,
            ty.goal.binders.len(Interner),
        ))
        .intern(Interner),
    };

    let in_env = InEnvironment { goal: alias_eq.cast(Interner), environment: ty.environment };

    let canonical = Canonical {
        value: in_env,
        binders: CanonicalVarKinds::from_iter(
            Interner,
            ty.goal.binders.iter(Interner).cloned().chain(Some(chalk_ir::WithKind::new(
                VariableKind::Ty(chalk_ir::TyVariableKind::General),
                chalk_ir::UniverseIndex::ROOT,
            ))),
        ),
    };

    let solution = db.trait_solve(krate, canonical)?;

    match &solution {
        Solution::Unique(Canonical { value: ConstrainedSubst { subst, .. }, binders })
        | Solution::Ambig(Guidance::Definite(Canonical { value: subst, binders })) => {
            // FIXME: vars may contain solutions for any inference variables
            // that happened to be inside ty. To correctly handle these, we
            // would have to pass the solution up to the inference context, but
            // that requires a larger refactoring (especially if the deref
            // happens during method resolution). So for the moment, we just
            // check that we're not in the situation where we would actually
            // need to handle the values of the additional variables, i.e.
            // they're just being 'passed through'. In the 'standard' case where
            // we have `impl<T> Deref for Foo<T> { Target = T }`, that should be
            // the case.

            // FIXME: if the trait solver decides to truncate the type, these
            // assumptions will be broken. We would need to properly introduce
            // new variables in that case

            for i in 1..binders.len(Interner) {
                if subst.at(Interner, i - 1).assert_ty_ref(Interner).kind(Interner)
                    != &TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, i - 1))
                {
                    warn!("complex solution for derefing {:?}: {:?}, ignoring", ty.goal, solution);
                    return None;
                }
            }
            // FIXME: we remove lifetime variables here since they can confuse
            // the method resolution code later
            Some(fixup_lifetime_variables(Canonical {
                value: subst.at(Interner, subst.len(Interner) - 1).assert_ty_ref(Interner).clone(),
                binders: binders.clone(),
            }))
        }
        Solution::Ambig(_) => {
            info!("Ambiguous solution for derefing {:?}: {:?}", ty.goal, solution);
            None
        }
    }
}

fn fixup_lifetime_variables<T: Fold<Interner, Result = T> + HasInterner<Interner = Interner>>(
    c: Canonical<T>,
) -> Canonical<T> {
    // Removes lifetime variables from the Canonical, replacing them by static lifetimes.
    let mut i = 0;
    let subst = Substitution::from_iter(
        Interner,
        c.binders.iter(Interner).map(|vk| match vk.kind {
            VariableKind::Ty(_) => {
                let index = i;
                i += 1;
                BoundVar::new(DebruijnIndex::INNERMOST, index).to_ty(Interner).cast(Interner)
            }
            VariableKind::Lifetime => static_lifetime().cast(Interner),
            VariableKind::Const(_) => unimplemented!(),
        }),
    );
    let binders = CanonicalVarKinds::from_iter(
        Interner,
        c.binders.iter(Interner).filter(|vk| match vk.kind {
            VariableKind::Ty(_) => true,
            VariableKind::Lifetime => false,
            VariableKind::Const(_) => true,
        }),
    );
    let value = subst.apply(c.value, Interner);
    Canonical { binders, value }
}
