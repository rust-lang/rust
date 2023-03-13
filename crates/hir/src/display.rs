//! HirDisplay implementations for various hir types.
use hir_def::{
    adt::VariantData,
    generics::{
        TypeOrConstParamData, TypeParamProvenance, WherePredicate, WherePredicateTypeTarget,
    },
    lang_item::LangItem,
    type_ref::{TypeBound, TypeRef},
    AdtId, GenericDefId,
};
use hir_ty::{
    display::{
        write_bounds_like_dyn_trait_with_prefix, write_visibility, HirDisplay, HirDisplayError,
        HirFormatter, SizedByDefault,
    },
    Interner, TraitRefExt, WhereClause,
};

use crate::{
    Adt, AsAssocItem, AssocItemContainer, Const, ConstParam, Enum, Field, Function, GenericParam,
    HasCrate, HasVisibility, LifetimeParam, Macro, Module, Static, Struct, Trait, TraitAlias,
    TyBuilder, Type, TypeAlias, TypeOrConstParam, TypeParam, Union, Variant,
};

impl HirDisplay for Function {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let db = f.db;
        let data = db.function_data(self.id);
        let container = self.as_assoc_item(db).map(|it| it.container(db));
        let mut module = self.module(db);
        if let Some(AssocItemContainer::Impl(_)) = container {
            // Block-local impls are "hoisted" to the nearest (non-block) module.
            module = module.nearest_non_block_module(db);
        }
        let module_id = module.id;
        write_visibility(module_id, self.visibility(db), f)?;
        if data.has_default_kw() {
            f.write_str("default ")?;
        }
        if data.has_const_kw() {
            f.write_str("const ")?;
        }
        if data.has_async_kw() {
            f.write_str("async ")?;
        }
        if self.is_unsafe_to_call(db) {
            f.write_str("unsafe ")?;
        }
        if let Some(abi) = &data.abi {
            // FIXME: String escape?
            write!(f, "extern \"{}\" ", &**abi)?;
        }
        write!(f, "fn {}", data.name)?;

        write_generic_params(GenericDefId::FunctionId(self.id), f)?;

        f.write_char('(')?;

        let write_self_param = |ty: &TypeRef, f: &mut HirFormatter<'_>| match ty {
            TypeRef::Path(p) if p.is_self_type() => f.write_str("self"),
            TypeRef::Reference(inner, lifetime, mut_) if matches!(&**inner, TypeRef::Path(p) if p.is_self_type()) =>
            {
                f.write_char('&')?;
                if let Some(lifetime) = lifetime {
                    write!(f, "{} ", lifetime.name)?;
                }
                if let hir_def::type_ref::Mutability::Mut = mut_ {
                    f.write_str("mut ")?;
                }
                f.write_str("self")
            }
            _ => {
                f.write_str("self: ")?;
                ty.hir_fmt(f)
            }
        };

        let mut first = true;
        for (name, type_ref) in &data.params {
            if !first {
                f.write_str(", ")?;
            } else {
                first = false;
                if data.has_self_param() {
                    write_self_param(type_ref, f)?;
                    continue;
                }
            }
            match name {
                Some(name) => write!(f, "{name}: ")?,
                None => f.write_str("_: ")?,
            }
            // FIXME: Use resolved `param.ty` or raw `type_ref`?
            // The former will ignore lifetime arguments currently.
            type_ref.hir_fmt(f)?;
        }

        if data.is_varargs() {
            f.write_str(", ...")?;
        }

        f.write_char(')')?;

        // `FunctionData::ret_type` will be `::core::future::Future<Output = ...>` for async fns.
        // Use ugly pattern match to strip the Future trait.
        // Better way?
        let ret_type = if !data.has_async_kw() {
            &data.ret_type
        } else {
            match &*data.ret_type {
                TypeRef::ImplTrait(bounds) => match bounds[0].as_ref() {
                    TypeBound::Path(path, _) => {
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
                f.write_str(" -> ")?;
                ty.hir_fmt(f)?;
            }
        }

        write_where_clause(GenericDefId::FunctionId(self.id), f)?;

        Ok(())
    }
}

impl HirDisplay for Adt {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            Adt::Struct(it) => it.hir_fmt(f),
            Adt::Union(it) => it.hir_fmt(f),
            Adt::Enum(it) => it.hir_fmt(f),
        }
    }
}

impl HirDisplay for Struct {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        f.write_str("struct ")?;
        write!(f, "{}", self.name(f.db))?;
        let def_id = GenericDefId::AdtId(AdtId::StructId(self.id));
        write_generic_params(def_id, f)?;
        write_where_clause(def_id, f)?;
        Ok(())
    }
}

impl HirDisplay for Enum {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        f.write_str("enum ")?;
        write!(f, "{}", self.name(f.db))?;
        let def_id = GenericDefId::AdtId(AdtId::EnumId(self.id));
        write_generic_params(def_id, f)?;
        write_where_clause(def_id, f)?;
        Ok(())
    }
}

impl HirDisplay for Union {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        f.write_str("union ")?;
        write!(f, "{}", self.name(f.db))?;
        let def_id = GenericDefId::AdtId(AdtId::UnionId(self.id));
        write_generic_params(def_id, f)?;
        write_where_clause(def_id, f)?;
        Ok(())
    }
}

impl HirDisplay for Field {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.parent.module(f.db).id, self.visibility(f.db), f)?;
        write!(f, "{}: ", self.name(f.db))?;
        self.ty(f.db).hir_fmt(f)
    }
}

impl HirDisplay for Variant {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db))?;
        let data = self.variant_data(f.db);
        match &*data {
            VariantData::Unit => {}
            VariantData::Tuple(fields) => {
                f.write_char('(')?;
                let mut first = true;
                for (_, field) in fields.iter() {
                    if first {
                        first = false;
                    } else {
                        f.write_str(", ")?;
                    }
                    // Enum variant fields must be pub.
                    field.type_ref.hir_fmt(f)?;
                }
                f.write_char(')')?;
            }
            VariantData::Record(fields) => {
                f.write_str(" {")?;
                let mut first = true;
                for (_, field) in fields.iter() {
                    if first {
                        first = false;
                        f.write_char(' ')?;
                    } else {
                        f.write_str(", ")?;
                    }
                    // Enum variant fields must be pub.
                    write!(f, "{}: ", field.name)?;
                    field.type_ref.hir_fmt(f)?;
                }
                f.write_str(" }")?;
            }
        }
        Ok(())
    }
}

impl HirDisplay for Type {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        self.ty.hir_fmt(f)
    }
}

impl HirDisplay for GenericParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            GenericParam::TypeParam(it) => it.hir_fmt(f),
            GenericParam::ConstParam(it) => it.hir_fmt(f),
            GenericParam::LifetimeParam(it) => it.hir_fmt(f),
        }
    }
}

impl HirDisplay for TypeOrConstParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self.split(f.db) {
            either::Either::Left(x) => x.hir_fmt(f),
            either::Either::Right(x) => x.hir_fmt(f),
        }
    }
}

impl HirDisplay for TypeParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db))?;
        if f.omit_verbose_types() {
            return Ok(());
        }

        let bounds = f.db.generic_predicates_for_param(self.id.parent(), self.id.into(), None);
        let substs = TyBuilder::placeholder_subst(f.db, self.id.parent());
        let predicates: Vec<_> =
            bounds.iter().cloned().map(|b| b.substitute(Interner, &substs)).collect();
        let krate = self.id.parent().krate(f.db).id;
        let sized_trait =
            f.db.lang_item(krate, LangItem::Sized).and_then(|lang_item| lang_item.as_trait());
        let has_only_sized_bound = predicates.iter().all(move |pred| match pred.skip_binders() {
            WhereClause::Implemented(it) => Some(it.hir_trait_id()) == sized_trait,
            _ => false,
        });
        let has_only_not_sized_bound = predicates.is_empty();
        if !has_only_sized_bound || has_only_not_sized_bound {
            let default_sized = SizedByDefault::Sized { anchor: krate };
            write_bounds_like_dyn_trait_with_prefix(f, ":", &predicates, default_sized)?;
        }
        Ok(())
    }
}

impl HirDisplay for LifetimeParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db))
    }
}

impl HirDisplay for ConstParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "const {}: ", self.name(f.db))?;
        self.ty(f.db).hir_fmt(f)
    }
}

fn write_generic_params(
    def: GenericDefId,
    f: &mut HirFormatter<'_>,
) -> Result<(), HirDisplayError> {
    let params = f.db.generic_params(def);
    if params.lifetimes.is_empty()
        && params.type_or_consts.iter().all(|x| x.1.const_param().is_none())
        && params
            .type_or_consts
            .iter()
            .filter_map(|x| x.1.type_param())
            .all(|param| !matches!(param.provenance, TypeParamProvenance::TypeParamList))
    {
        return Ok(());
    }
    f.write_char('<')?;

    let mut first = true;
    let mut delim = |f: &mut HirFormatter<'_>| {
        if first {
            first = false;
            Ok(())
        } else {
            f.write_str(", ")
        }
    };
    for (_, lifetime) in params.lifetimes.iter() {
        delim(f)?;
        write!(f, "{}", lifetime.name)?;
    }
    for (_, ty) in params.type_or_consts.iter() {
        if let Some(name) = &ty.name() {
            match ty {
                TypeOrConstParamData::TypeParamData(ty) => {
                    if ty.provenance != TypeParamProvenance::TypeParamList {
                        continue;
                    }
                    delim(f)?;
                    write!(f, "{name}")?;
                    if let Some(default) = &ty.default {
                        f.write_str(" = ")?;
                        default.hir_fmt(f)?;
                    }
                }
                TypeOrConstParamData::ConstParamData(c) => {
                    delim(f)?;
                    write!(f, "const {name}: ")?;
                    c.ty.hir_fmt(f)?;
                }
            }
        }
    }

    f.write_char('>')?;
    Ok(())
}

fn write_where_clause(def: GenericDefId, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
    let params = f.db.generic_params(def);

    // unnamed type targets are displayed inline with the argument itself, e.g. `f: impl Y`.
    let is_unnamed_type_target = |target: &WherePredicateTypeTarget| match target {
        WherePredicateTypeTarget::TypeRef(_) => false,
        WherePredicateTypeTarget::TypeOrConstParam(id) => {
            params.type_or_consts[*id].name().is_none()
        }
    };

    let has_displayable_predicate = params
        .where_predicates
        .iter()
        .any(|pred| {
            !matches!(pred, WherePredicate::TypeBound { target, .. } if is_unnamed_type_target(target))
        });

    if !has_displayable_predicate {
        return Ok(());
    }

    let write_target = |target: &WherePredicateTypeTarget, f: &mut HirFormatter<'_>| match target {
        WherePredicateTypeTarget::TypeRef(ty) => ty.hir_fmt(f),
        WherePredicateTypeTarget::TypeOrConstParam(id) => {
            match &params.type_or_consts[*id].name() {
                Some(name) => write!(f, "{name}"),
                None => f.write_str("{unnamed}"),
            }
        }
    };

    f.write_str("\nwhere")?;

    for (pred_idx, pred) in params.where_predicates.iter().enumerate() {
        let prev_pred =
            if pred_idx == 0 { None } else { Some(&params.where_predicates[pred_idx - 1]) };

        let new_predicate = |f: &mut HirFormatter<'_>| {
            f.write_str(if pred_idx == 0 { "\n    " } else { ",\n    " })
        };

        match pred {
            WherePredicate::TypeBound { target, .. } if is_unnamed_type_target(target) => {}
            WherePredicate::TypeBound { target, bound } => {
                if matches!(prev_pred, Some(WherePredicate::TypeBound { target: target_, .. }) if target_ == target)
                {
                    f.write_str(" + ")?;
                } else {
                    new_predicate(f)?;
                    write_target(target, f)?;
                    f.write_str(": ")?;
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
                    f.write_str(" + ")?;
                } else {
                    new_predicate(f)?;
                    f.write_str("for<")?;
                    for (idx, lifetime) in lifetimes.iter().enumerate() {
                        if idx != 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{lifetime}")?;
                    }
                    f.write_str("> ")?;
                    write_target(target, f)?;
                    f.write_str(": ")?;
                }
                bound.hir_fmt(f)?;
            }
        }
    }

    // End of final predicate. There must be at least one predicate here.
    f.write_char(',')?;

    Ok(())
}

impl HirDisplay for Const {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let db = f.db;
        let container = self.as_assoc_item(db).map(|it| it.container(db));
        let mut module = self.module(db);
        if let Some(AssocItemContainer::Impl(_)) = container {
            // Block-local impls are "hoisted" to the nearest (non-block) module.
            module = module.nearest_non_block_module(db);
        }
        write_visibility(module.id, self.visibility(db), f)?;
        let data = db.const_data(self.id);
        f.write_str("const ")?;
        match &data.name {
            Some(name) => write!(f, "{name}: ")?,
            None => f.write_str("_: ")?,
        }
        data.type_ref.hir_fmt(f)?;
        Ok(())
    }
}

impl HirDisplay for Static {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        let data = f.db.static_data(self.id);
        f.write_str("static ")?;
        if data.mutable {
            f.write_str("mut ")?;
        }
        write!(f, "{}: ", &data.name)?;
        data.type_ref.hir_fmt(f)?;
        Ok(())
    }
}

impl HirDisplay for Trait {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        let data = f.db.trait_data(self.id);
        if data.is_unsafe {
            f.write_str("unsafe ")?;
        }
        if data.is_auto {
            f.write_str("auto ")?;
        }
        write!(f, "trait {}", data.name)?;
        let def_id = GenericDefId::TraitId(self.id);
        write_generic_params(def_id, f)?;
        write_where_clause(def_id, f)?;
        Ok(())
    }
}

impl HirDisplay for TraitAlias {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        let data = f.db.trait_alias_data(self.id);
        write!(f, "trait {}", data.name)?;
        let def_id = GenericDefId::TraitAliasId(self.id);
        write_generic_params(def_id, f)?;
        f.write_str(" = ")?;
        // FIXME: Currently we lower every bounds in a trait alias as a trait bound on `Self` i.e.
        // `trait Foo = Bar` is stored and displayed as `trait Foo = where Self: Bar`, which might
        // be less readable.
        write_where_clause(def_id, f)?;
        Ok(())
    }
}

impl HirDisplay for TypeAlias {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        let data = f.db.type_alias_data(self.id);
        write!(f, "type {}", data.name)?;
        let def_id = GenericDefId::TypeAliasId(self.id);
        write_generic_params(def_id, f)?;
        write_where_clause(def_id, f)?;
        if !data.bounds.is_empty() {
            f.write_str(": ")?;
            f.write_joined(&data.bounds, " + ")?;
        }
        if let Some(ty) = &data.type_ref {
            f.write_str(" = ")?;
            ty.hir_fmt(f)?;
        }
        Ok(())
    }
}

impl HirDisplay for Module {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        // FIXME: Module doesn't have visibility saved in data.
        match self.name(f.db) {
            Some(name) => write!(f, "mod {name}"),
            None if self.is_crate_root(f.db) => match self.krate(f.db).display_name(f.db) {
                Some(name) => write!(f, "extern crate {name}"),
                None => f.write_str("extern crate {unknown}"),
            },
            None => f.write_str("mod {unnamed}"),
        }
    }
}

impl HirDisplay for Macro {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self.id {
            hir_def::MacroId::Macro2Id(_) => f.write_str("macro"),
            hir_def::MacroId::MacroRulesId(_) => f.write_str("macro_rules!"),
            hir_def::MacroId::ProcMacroId(_) => f.write_str("proc_macro"),
        }?;
        write!(f, " {}", self.name(f.db))
    }
}
