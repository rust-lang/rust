//! In certain situations, rust automatically inserts derefs as necessary: for
//! example, field accesses `foo.bar` still work when `foo` is actually a
//! reference to a type with the field `bar`. This is an approximation of the
//! logic in rustc (which lives in librustc_typeck/check/autoderef.rs).

use std::iter::successors;

use log::info;

use crate::{HirDatabase, Name, Resolver};
use super::{traits::Solution, Ty, Canonical};

pub(crate) fn autoderef<'a>(
    db: &'a impl HirDatabase,
    resolver: &'a Resolver,
    ty: Canonical<Ty>,
) -> impl Iterator<Item = Canonical<Ty>> + 'a {
    successors(Some(ty), move |ty| deref(db, resolver, ty))
}

pub(crate) fn deref(
    db: &impl HirDatabase,
    resolver: &Resolver,
    ty: &Canonical<Ty>,
) -> Option<Canonical<Ty>> {
    if let Some(derefed) = ty.value.builtin_deref() {
        Some(Canonical { value: derefed, num_vars: ty.num_vars })
    } else {
        deref_by_trait(db, resolver, ty)
    }
}

fn deref_by_trait(
    db: &impl HirDatabase,
    resolver: &Resolver,
    ty: &Canonical<Ty>,
) -> Option<Canonical<Ty>> {
    let krate = resolver.krate()?;
    let deref_trait = match db.lang_item(krate, "deref".into())? {
        crate::lang_item::LangItemTarget::Trait(t) => t,
        _ => return None,
    };
    let target = deref_trait.associated_type_by_name(db, Name::target())?;

    // FIXME we should check that Deref has no type parameters, because we assume it below

    // FIXME make the Canonical handling nicer
    // TODO shift inference variables in ty

    let projection = super::traits::ProjectionPredicate {
        ty: Ty::Bound(0),
        projection_ty: super::ProjectionTy {
            associated_ty: target,
            parameters: vec![ty.value.clone()].into(),
        },
    };

    let canonical = super::Canonical { num_vars: 1 + ty.num_vars, value: projection };

    let solution = db.normalize(krate, canonical)?;

    match &solution {
        Solution::Unique(vars) => {
            Some(Canonical { value: vars.0.value[0].clone(), num_vars: vars.0.num_vars })
        }
        Solution::Ambig(_) => {
            info!("Ambiguous solution for deref: {:?}", solution);
            None
        }
    }
}
