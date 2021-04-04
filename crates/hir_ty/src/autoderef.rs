//! In certain situations, rust automatically inserts derefs as necessary: for
//! example, field accesses `foo.bar` still work when `foo` is actually a
//! reference to a type with the field `bar`. This is an approximation of the
//! logic in rustc (which lives in librustc_typeck/check/autoderef.rs).

use std::iter::successors;

use base_db::CrateId;
use chalk_ir::cast::Cast;
use hir_def::lang_item::LangItemTarget;
use hir_expand::name::name;
use log::{info, warn};

use crate::{
    db::HirDatabase, AliasEq, AliasTy, BoundVar, Canonical, CanonicalVarKinds, DebruijnIndex,
    InEnvironment, Interner, Solution, Ty, TyBuilder, TyKind,
};

const AUTODEREF_RECURSION_LIMIT: usize = 10;

pub fn autoderef<'a>(
    db: &'a dyn HirDatabase,
    krate: Option<CrateId>,
    ty: InEnvironment<Canonical<Ty>>,
) -> impl Iterator<Item = Canonical<Ty>> + 'a {
    let InEnvironment { goal: ty, environment } = ty;
    successors(Some(ty), move |ty| {
        deref(db, krate?, InEnvironment { goal: ty, environment: environment.clone() })
    })
    .take(AUTODEREF_RECURSION_LIMIT)
}

pub(crate) fn deref(
    db: &dyn HirDatabase,
    krate: CrateId,
    ty: InEnvironment<&Canonical<Ty>>,
) -> Option<Canonical<Ty>> {
    if let Some(derefed) = ty.goal.value.builtin_deref() {
        Some(Canonical { value: derefed, binders: ty.goal.binders.clone() })
    } else {
        deref_by_trait(db, krate, ty)
    }
}

fn deref_by_trait(
    db: &dyn HirDatabase,
    krate: CrateId,
    ty: InEnvironment<&Canonical<Ty>>,
) -> Option<Canonical<Ty>> {
    let deref_trait = match db.lang_item(krate, "deref".into())? {
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
            goal: trait_ref.cast(&Interner),
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
            ty.goal.binders.len(&Interner),
        ))
        .intern(&Interner),
    };

    let in_env = InEnvironment { goal: alias_eq.cast(&Interner), environment: ty.environment };

    let canonical = Canonical {
        value: in_env,
        binders: CanonicalVarKinds::from_iter(
            &Interner,
            ty.goal.binders.iter(&Interner).cloned().chain(Some(chalk_ir::WithKind::new(
                chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
                chalk_ir::UniverseIndex::ROOT,
            ))),
        ),
    };

    let solution = db.trait_solve(krate, canonical)?;

    match &solution {
        Solution::Unique(vars) => {
            // FIXME: vars may contain solutions for any inference variables
            // that happened to be inside ty. To correctly handle these, we
            // would have to pass the solution up to the inference context, but
            // that requires a larger refactoring (especially if the deref
            // happens during method resolution). So for the moment, we just
            // check that we're not in the situation we're we would actually
            // need to handle the values of the additional variables, i.e.
            // they're just being 'passed through'. In the 'standard' case where
            // we have `impl<T> Deref for Foo<T> { Target = T }`, that should be
            // the case.

            // FIXME: if the trait solver decides to truncate the type, these
            // assumptions will be broken. We would need to properly introduce
            // new variables in that case

            for i in 1..vars.0.binders.len(&Interner) {
                if vars.0.value.at(&Interner, i - 1).assert_ty_ref(&Interner).kind(&Interner)
                    != &TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, i - 1))
                {
                    warn!("complex solution for derefing {:?}: {:?}, ignoring", ty.goal, solution);
                    return None;
                }
            }
            Some(Canonical {
                value: vars
                    .0
                    .value
                    .at(&Interner, vars.0.value.len(&Interner) - 1)
                    .assert_ty_ref(&Interner)
                    .clone(),
                binders: vars.0.binders.clone(),
            })
        }
        Solution::Ambig(_) => {
            info!("Ambiguous solution for derefing {:?}: {:?}", ty.goal, solution);
            None
        }
    }
}
