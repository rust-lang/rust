//! HirDisplay implementations for various hir types.
use hir_def::{
    generics::{TypeParamProvenance, WherePredicate, WherePredicateTypeTarget},
    type_ref::{TypeBound, TypeRef},
    GenericDefId,
};
use hir_ty::display::{
    write_bounds_like_dyn_trait_with_prefix, write_visibility, HirDisplay, HirDisplayError,
    HirFormatter,
};
use syntax::ast::{self, NameOwner};

use crate::{Function, HasVisibility, Substs, Type, TypeParam};

impl HirDisplay for Function {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        let data = f.db.function_data(self.id);
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        let qual = &data.qualifier;
        if qual.is_default {
            write!(f, "default ")?;
        }
        if qual.is_const {
            write!(f, "const ")?;
        }
        if qual.is_async {
            write!(f, "async ")?;
        }
        if qual.is_unsafe {
            write!(f, "unsafe ")?;
        }
        if let Some(abi) = &qual.abi {
            // FIXME: String escape?
            write!(f, "extern \"{}\" ", abi)?;
        }
        write!(f, "fn {}", data.name)?;

        write_generic_params(GenericDefId::FunctionId(self.id), f)?;

        write!(f, "(")?;

        let write_self_param = |ty: &TypeRef, f: &mut HirFormatter| match ty {
            TypeRef::Path(p) if p.is_self_type() => write!(f, "self"),
            TypeRef::Reference(inner, lifetime, mut_) if matches!(&**inner,TypeRef::Path(p) if p.is_self_type()) =>
            {
                write!(f, "&")?;
                if let Some(lifetime) = lifetime {
                    write!(f, "{} ", lifetime.name)?;
                }
                if let hir_def::type_ref::Mutability::Mut = mut_ {
                    write!(f, "mut ")?;
                }
                write!(f, "self")
            }
            _ => {
                write!(f, "self: ")?;
                ty.hir_fmt(f)
            }
        };

        let mut first = true;
        for (param, type_ref) in self.assoc_fn_params(f.db).into_iter().zip(&data.params) {
            if !first {
                write!(f, ", ")?;
            } else {
                first = false;
                if data.has_self_param {
                    write_self_param(type_ref, f)?;
                    continue;
                }
            }
            match param.pattern_source(f.db) {
                Some(ast::Pat::IdentPat(p)) if p.name().is_some() => {
                    write!(f, "{}: ", p.name().unwrap())?
                }
                _ => write!(f, "_: ")?,
            }
            // FIXME: Use resolved `param.ty` or raw `type_ref`?
            // The former will ignore lifetime arguments currently.
            type_ref.hir_fmt(f)?;
        }
        write!(f, ")")?;

        // `FunctionData::ret_type` will be `::core::future::Future<Output = ...>` for async fns.
        // Use ugly pattern match to strip the Future trait.
        // Better way?
        let ret_type = if !qual.is_async {
            &data.ret_type
        } else {
            match &data.ret_type {
                TypeRef::ImplTrait(bounds) => match &bounds[0] {
                    TypeBound::Path(path) => {
                        path.segments().iter().last().unwrap().args_and_bindings.unwrap().bindings
                            [0]
                        .type_ref
                        .as_ref()
                        .unwrap()
                    }
                    _ => panic!("Async fn ret_type should be impl Future"),
                },
                _ => panic!("Async fn ret_type should be impl Future"),
            }
        };

        match ret_type {
            TypeRef::Tuple(tup) if tup.is_empty() => {}
            ty => {
                write!(f, " -> ")?;
                ty.hir_fmt(f)?;
            }
        }

        write_where_clause(GenericDefId::FunctionId(self.id), f)?;

        Ok(())
    }
}

impl HirDisplay for Type {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        self.ty.value.hir_fmt(f)
    }
}

impl HirDisplay for TypeParam {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db))?;
        let bounds = f.db.generic_predicates_for_param(self.id);
        let substs = Substs::type_params(f.db, self.id.parent);
        let predicates = bounds.iter().cloned().map(|b| b.subst(&substs)).collect::<Vec<_>>();
        if !(predicates.is_empty() || f.omit_verbose_types()) {
            write_bounds_like_dyn_trait_with_prefix(":", &predicates, f)?;
        }
        Ok(())
    }
}

fn write_generic_params(def: GenericDefId, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
    let params = f.db.generic_params(def);
    if params.lifetimes.is_empty() && params.types.is_empty() && params.consts.is_empty() {
        return Ok(());
    }
    write!(f, "<")?;

    let mut first = true;
    let mut delim = |f: &mut HirFormatter| {
        if first {
            first = false;
            Ok(())
        } else {
            write!(f, ", ")
        }
    };
    for (_, lifetime) in params.lifetimes.iter() {
        delim(f)?;
        write!(f, "{}", lifetime.name)?;
    }
    for (_, ty) in params.types.iter() {
        if ty.provenance != TypeParamProvenance::TypeParamList {
            continue;
        }
        if let Some(name) = &ty.name {
            delim(f)?;
            write!(f, "{}", name)?;
            if let Some(default) = &ty.default {
                write!(f, " = ")?;
                default.hir_fmt(f)?;
            }
        }
    }
    for (_, konst) in params.consts.iter() {
        delim(f)?;
        write!(f, "const {}: ", konst.name)?;
        konst.ty.hir_fmt(f)?;
    }

    write!(f, ">")?;
    Ok(())
}

fn write_where_clause(def: GenericDefId, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
    let params = f.db.generic_params(def);
    if params.where_predicates.is_empty() {
        return Ok(());
    }

    let write_target = |target: &WherePredicateTypeTarget, f: &mut HirFormatter| match target {
        WherePredicateTypeTarget::TypeRef(ty) => ty.hir_fmt(f),
        WherePredicateTypeTarget::TypeParam(id) => match &params.types[*id].name {
            Some(name) => write!(f, "{}", name),
            None => write!(f, "{{unnamed}}"),
        },
    };

    write!(f, "\nwhere")?;

    for (pred_idx, pred) in params.where_predicates.iter().enumerate() {
        let prev_pred =
            if pred_idx == 0 { None } else { Some(&params.where_predicates[pred_idx - 1]) };

        let new_predicate = |f: &mut HirFormatter| {
            write!(f, "{}", if pred_idx == 0 { "\n    " } else { ",\n    " })
        };

        match pred {
            WherePredicate::TypeBound { target, bound } => {
                if matches!(prev_pred, Some(WherePredicate::TypeBound { target: target_, .. }) if target_ == target)
                {
                    write!(f, " + ")?;
                } else {
                    new_predicate(f)?;
                    write_target(target, f)?;
                    write!(f, ": ")?;
                }
                bound.hir_fmt(f)?;
            }
            WherePredicate::Lifetime { target, bound } => {
                if matches!(prev_pred, Some(WherePredicate::Lifetime { target: target_, .. }) if target_ == target)
                {
                    write!(f, " + {}", bound.name)?;
                } else {
                    new_predicate(f)?;
                    write!(f, "{}: {}", target.name, bound.name)?;
                }
            }
            WherePredicate::ForLifetime { lifetimes, target, bound } => {
                if matches!(
                    prev_pred,
                    Some(WherePredicate::ForLifetime { lifetimes: lifetimes_, target: target_, .. })
                    if lifetimes_ == lifetimes && target_ == target,
                ) {
                    write!(f, " + ")?;
                } else {
                    new_predicate(f)?;
                    write!(f, "for<")?;
                    for (idx, lifetime) in lifetimes.iter().enumerate() {
                        if idx != 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", lifetime)?;
                    }
                    write!(f, "> ")?;
                    write_target(target, f)?;
                    write!(f, ": ")?;
                }
                bound.hir_fmt(f)?;
            }
        }
    }

    // End of final predicate. There must be at least one predicate here.
    write!(f, ",")?;

    Ok(())
}
