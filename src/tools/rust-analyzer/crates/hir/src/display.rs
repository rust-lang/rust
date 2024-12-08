//! HirDisplay implementations for various hir types.
use either::Either;
use hir_def::{
    data::adt::{StructKind, VariantData},
    generics::{
        GenericParams, TypeOrConstParamData, TypeParamProvenance, WherePredicate,
        WherePredicateTypeTarget,
    },
    lang_item::LangItem,
    type_ref::{TypeBound, TypeRef},
    AdtId, GenericDefId,
};
use hir_ty::{
    display::{
        hir_display_with_types_map, write_bounds_like_dyn_trait_with_prefix, write_visibility,
        HirDisplay, HirDisplayError, HirDisplayWithTypesMap, HirFormatter, SizedByDefault,
    },
    AliasEq, AliasTy, Interner, ProjectionTyExt, TraitRefExt, TyKind, WhereClause,
};
use itertools::Itertools;

use crate::{
    Adt, AsAssocItem, AssocItem, AssocItemContainer, Const, ConstParam, Enum, ExternCrateDecl,
    Field, Function, GenericParam, HasCrate, HasVisibility, Impl, LifetimeParam, Macro, Module,
    SelfParam, Static, Struct, Trait, TraitAlias, TupleField, TyBuilder, Type, TypeAlias,
    TypeOrConstParam, TypeParam, Union, Variant,
};

impl HirDisplay for Function {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let db = f.db;
        let data = db.function_data(self.id);
        let container = self.as_assoc_item(db).map(|it| it.container(db));
        let mut module = self.module(db);

        // Write container (trait or impl)
        let container_params = match container {
            Some(AssocItemContainer::Trait(trait_)) => {
                let params = f.db.generic_params(trait_.id.into());
                if f.show_container_bounds() && !params.is_empty() {
                    write_trait_header(&trait_, f)?;
                    f.write_char('\n')?;
                    has_disaplayable_predicates(&params).then_some(params)
                } else {
                    None
                }
            }
            Some(AssocItemContainer::Impl(impl_)) => {
                let params = f.db.generic_params(impl_.id.into());
                if f.show_container_bounds() && !params.is_empty() {
                    write_impl_header(&impl_, f)?;
                    f.write_char('\n')?;
                    has_disaplayable_predicates(&params).then_some(params)
                } else {
                    None
                }
            }
            None => None,
        };

        // Write signature of the function

        // Block-local impls are "hoisted" to the nearest (non-block) module.
        if let Some(AssocItemContainer::Impl(_)) = container {
            module = module.nearest_non_block_module(db);
        }
        let module_id = module.id;

        write_visibility(module_id, self.visibility(db), f)?;

        if data.is_default() {
            f.write_str("default ")?;
        }
        if data.is_const() {
            f.write_str("const ")?;
        }
        if data.is_async() {
            f.write_str("async ")?;
        }
        if self.is_unsafe_to_call(db) {
            f.write_str("unsafe ")?;
        }
        if let Some(abi) = &data.abi {
            write!(f, "extern \"{}\" ", abi.as_str())?;
        }
        write!(f, "fn {}", data.name.display(f.db.upcast(), f.edition()))?;

        write_generic_params(GenericDefId::FunctionId(self.id), f)?;

        f.write_char('(')?;

        let mut first = true;
        let mut skip_self = 0;
        if let Some(self_param) = self.self_param(db) {
            self_param.hir_fmt(f)?;
            first = false;
            skip_self = 1;
        }

        // FIXME: Use resolved `param.ty` once we no longer discard lifetimes
        let body = db.body(self.id.into());
        for (type_ref, param) in data.params.iter().zip(self.assoc_fn_params(db)).skip(skip_self) {
            if !first {
                f.write_str(", ")?;
            } else {
                first = false;
            }

            let pat_id = body.params[param.idx - body.self_param.is_some() as usize];
            let pat_str =
                body.pretty_print_pat(db.upcast(), self.id.into(), pat_id, true, f.edition());
            f.write_str(&pat_str)?;

            f.write_str(": ")?;
            type_ref.hir_fmt(f, &data.types_map)?;
        }

        if data.is_varargs() {
            if !first {
                f.write_str(", ")?;
            }
            f.write_str("...")?;
        }

        f.write_char(')')?;

        // `FunctionData::ret_type` will be `::core::future::Future<Output = ...>` for async fns.
        // Use ugly pattern match to strip the Future trait.
        // Better way?
        let ret_type = if !data.is_async() {
            Some(data.ret_type)
        } else {
            match &data.types_map[data.ret_type] {
                TypeRef::ImplTrait(bounds) => match &bounds[0] {
                    TypeBound::Path(path, _) => Some(
                        *path.segments().iter().last().unwrap().args_and_bindings.unwrap().bindings
                            [0]
                        .type_ref
                        .as_ref()
                        .unwrap(),
                    ),
                    _ => None,
                },
                _ => None,
            }
        };

        if let Some(ret_type) = ret_type {
            match &data.types_map[ret_type] {
                TypeRef::Tuple(tup) if tup.is_empty() => {}
                _ => {
                    f.write_str(" -> ")?;
                    ret_type.hir_fmt(f, &data.types_map)?;
                }
            }
        }

        // Write where clauses
        let has_written_where = write_where_clause(GenericDefId::FunctionId(self.id), f)?;
        if let Some(container_params) = container_params {
            if !has_written_where {
                f.write_str("\nwhere")?;
            }
            let container_name = match container.unwrap() {
                AssocItemContainer::Trait(_) => "trait",
                AssocItemContainer::Impl(_) => "impl",
            };
            write!(f, "\n    // Bounds from {container_name}:",)?;
            write_where_predicates(&container_params, f)?;
        }
        Ok(())
    }
}

fn write_impl_header(impl_: &Impl, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
    let db = f.db;

    f.write_str("impl")?;
    let def_id = GenericDefId::ImplId(impl_.id);
    write_generic_params(def_id, f)?;

    if let Some(trait_) = impl_.trait_(db) {
        let trait_data = db.trait_data(trait_.id);
        write!(f, " {} for", trait_data.name.display(db.upcast(), f.edition()))?;
    }

    f.write_char(' ')?;
    impl_.self_ty(db).hir_fmt(f)?;

    Ok(())
}

impl HirDisplay for SelfParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let data = f.db.function_data(self.func);
        let param = *data.params.first().unwrap();
        match &data.types_map[param] {
            TypeRef::Path(p) if p.is_self_type() => f.write_str("self"),
            TypeRef::Reference(ref_) if matches!(&data.types_map[ref_.ty], TypeRef::Path(p) if p.is_self_type()) =>
            {
                f.write_char('&')?;
                if let Some(lifetime) = &ref_.lifetime {
                    write!(f, "{} ", lifetime.name.display(f.db.upcast(), f.edition()))?;
                }
                if let hir_def::type_ref::Mutability::Mut = ref_.mutability {
                    f.write_str("mut ")?;
                }
                f.write_str("self")
            }
            _ => {
                f.write_str("self: ")?;
                param.hir_fmt(f, &data.types_map)
            }
        }
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
        let module_id = self.module(f.db).id;
        // FIXME: Render repr if its set explicitly?
        write_visibility(module_id, self.visibility(f.db), f)?;
        f.write_str("struct ")?;
        write!(f, "{}", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        let def_id = GenericDefId::AdtId(AdtId::StructId(self.id));
        write_generic_params(def_id, f)?;

        let variant_data = self.variant_data(f.db);
        match variant_data.kind() {
            StructKind::Tuple => {
                f.write_char('(')?;
                let mut it = variant_data.fields().iter().peekable();

                while let Some((id, _)) = it.next() {
                    let field = Field { parent: (*self).into(), id };
                    write_visibility(module_id, field.visibility(f.db), f)?;
                    field.ty(f.db).hir_fmt(f)?;
                    if it.peek().is_some() {
                        f.write_str(", ")?;
                    }
                }

                f.write_char(')')?;
                write_where_clause(def_id, f)?;
            }
            StructKind::Record => {
                let has_where_clause = write_where_clause(def_id, f)?;
                if let Some(limit) = f.entity_limit {
                    write_fields(&self.fields(f.db), has_where_clause, limit, false, f)?;
                }
            }
            StructKind::Unit => _ = write_where_clause(def_id, f)?,
        }

        Ok(())
    }
}

impl HirDisplay for Enum {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        f.write_str("enum ")?;
        write!(f, "{}", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        let def_id = GenericDefId::AdtId(AdtId::EnumId(self.id));
        write_generic_params(def_id, f)?;

        let has_where_clause = write_where_clause(def_id, f)?;
        if let Some(limit) = f.entity_limit {
            write_variants(&self.variants(f.db), has_where_clause, limit, f)?;
        }

        Ok(())
    }
}

impl HirDisplay for Union {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        f.write_str("union ")?;
        write!(f, "{}", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        let def_id = GenericDefId::AdtId(AdtId::UnionId(self.id));
        write_generic_params(def_id, f)?;

        let has_where_clause = write_where_clause(def_id, f)?;
        if let Some(limit) = f.entity_limit {
            write_fields(&self.fields(f.db), has_where_clause, limit, false, f)?;
        }
        Ok(())
    }
}

fn write_fields(
    fields: &[Field],
    has_where_clause: bool,
    limit: usize,
    in_line: bool,
    f: &mut HirFormatter<'_>,
) -> Result<(), HirDisplayError> {
    let count = fields.len().min(limit);
    let (indent, separator) = if in_line { ("", ' ') } else { ("    ", '\n') };
    f.write_char(if !has_where_clause { ' ' } else { separator })?;
    if count == 0 {
        f.write_str(if fields.is_empty() { "{}" } else { "{ /* … */ }" })?;
    } else {
        f.write_char('{')?;

        if !fields.is_empty() {
            f.write_char(separator)?;
            for field in &fields[..count] {
                f.write_str(indent)?;
                field.hir_fmt(f)?;
                write!(f, ",{separator}")?;
            }

            if fields.len() > count {
                write!(f, "{indent}/* … */{separator}")?;
            }
        }

        f.write_str("}")?;
    }

    Ok(())
}

fn write_variants(
    variants: &[Variant],
    has_where_clause: bool,
    limit: usize,
    f: &mut HirFormatter<'_>,
) -> Result<(), HirDisplayError> {
    let count = variants.len().min(limit);
    f.write_char(if !has_where_clause { ' ' } else { '\n' })?;
    if count == 0 {
        let variants = if variants.is_empty() { "{}" } else { "{ /* … */ }" };
        f.write_str(variants)?;
    } else {
        f.write_str("{\n")?;
        for variant in &variants[..count] {
            write!(f, "    {}", variant.name(f.db).display(f.db.upcast(), f.edition()))?;
            match variant.kind(f.db) {
                StructKind::Tuple => {
                    let fields_str =
                        if variant.fields(f.db).is_empty() { "()" } else { "( /* … */ )" };
                    f.write_str(fields_str)?;
                }
                StructKind::Record => {
                    let fields_str =
                        if variant.fields(f.db).is_empty() { " {}" } else { " { /* … */ }" };
                    f.write_str(fields_str)?;
                }
                StructKind::Unit => {}
            }
            f.write_str(",\n")?;
        }

        if variants.len() > count {
            f.write_str("    /* … */\n")?;
        }
        f.write_str("}")?;
    }

    Ok(())
}

impl HirDisplay for Field {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.parent.module(f.db).id, self.visibility(f.db), f)?;
        write!(f, "{}: ", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        self.ty(f.db).hir_fmt(f)
    }
}

impl HirDisplay for TupleField {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "pub {}: ", self.name().display(f.db.upcast(), f.edition()))?;
        self.ty(f.db).hir_fmt(f)
    }
}

impl HirDisplay for Variant {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        let data = self.variant_data(f.db);
        match &*data {
            VariantData::Unit => {}
            VariantData::Tuple { fields, types_map } => {
                f.write_char('(')?;
                let mut first = true;
                for (_, field) in fields.iter() {
                    if first {
                        first = false;
                    } else {
                        f.write_str(", ")?;
                    }
                    // Enum variant fields must be pub.
                    field.type_ref.hir_fmt(f, types_map)?;
                }
                f.write_char(')')?;
            }
            VariantData::Record { .. } => {
                if let Some(limit) = f.entity_limit {
                    write_fields(&self.fields(f.db), false, limit, true, f)?;
                }
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

impl HirDisplay for ExternCrateDecl {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        f.write_str("extern crate ")?;
        write!(f, "{}", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        if let Some(alias) = self.alias(f.db) {
            write!(f, " as {}", alias.display(f.edition()))?;
        }
        Ok(())
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
            either::Either::Left(it) => it.hir_fmt(f),
            either::Either::Right(it) => it.hir_fmt(f),
        }
    }
}

impl HirDisplay for TypeParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let params = f.db.generic_params(self.id.parent());
        let param_data = &params[self.id.local_id()];
        let substs = TyBuilder::placeholder_subst(f.db, self.id.parent());
        let krate = self.id.parent().krate(f.db).id;
        let ty =
            TyKind::Placeholder(hir_ty::to_placeholder_idx(f.db, self.id.into())).intern(Interner);
        let predicates = f.db.generic_predicates(self.id.parent());
        let predicates = predicates
            .iter()
            .cloned()
            .map(|pred| pred.substitute(Interner, &substs))
            .filter(|wc| match wc.skip_binders() {
                WhereClause::Implemented(tr) => tr.self_type_parameter(Interner) == ty,
                WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(proj), ty: _ }) => {
                    proj.self_type_parameter(f.db) == ty
                }
                WhereClause::AliasEq(_) => false,
                WhereClause::TypeOutlives(to) => to.ty == ty,
                WhereClause::LifetimeOutlives(_) => false,
            })
            .collect::<Vec<_>>();

        match param_data {
            TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                TypeParamProvenance::TypeParamList | TypeParamProvenance::TraitSelf => {
                    write!(f, "{}", p.name.clone().unwrap().display(f.db.upcast(), f.edition()))?
                }
                TypeParamProvenance::ArgumentImplTrait => {
                    return write_bounds_like_dyn_trait_with_prefix(
                        f,
                        "impl",
                        Either::Left(&ty),
                        &predicates,
                        SizedByDefault::Sized { anchor: krate },
                    );
                }
            },
            TypeOrConstParamData::ConstParamData(p) => {
                write!(f, "{}", p.name.display(f.db.upcast(), f.edition()))?;
            }
        }

        if f.omit_verbose_types() {
            return Ok(());
        }

        let sized_trait =
            f.db.lang_item(krate, LangItem::Sized).and_then(|lang_item| lang_item.as_trait());
        let has_only_sized_bound = predicates.iter().all(move |pred| match pred.skip_binders() {
            WhereClause::Implemented(it) => Some(it.hir_trait_id()) == sized_trait,
            _ => false,
        });
        let has_only_not_sized_bound = predicates.is_empty();
        if !has_only_sized_bound || has_only_not_sized_bound {
            let default_sized = SizedByDefault::Sized { anchor: krate };
            write_bounds_like_dyn_trait_with_prefix(
                f,
                ":",
                Either::Left(
                    &hir_ty::TyKind::Placeholder(hir_ty::to_placeholder_idx(f.db, self.id.into()))
                        .intern(Interner),
                ),
                &predicates,
                default_sized,
            )?;
        }
        Ok(())
    }
}

impl HirDisplay for LifetimeParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db).display(f.db.upcast(), f.edition()))
    }
}

impl HirDisplay for ConstParam {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "const {}: ", self.name(f.db).display(f.db.upcast(), f.edition()))?;
        self.ty(f.db).hir_fmt(f)
    }
}

fn write_generic_params(
    def: GenericDefId,
    f: &mut HirFormatter<'_>,
) -> Result<(), HirDisplayError> {
    let params = f.db.generic_params(def);
    if params.iter_lt().next().is_none()
        && params.iter_type_or_consts().all(|it| it.1.const_param().is_none())
        && params
            .iter_type_or_consts()
            .filter_map(|it| it.1.type_param())
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
    for (_, lifetime) in params.iter_lt() {
        delim(f)?;
        write!(f, "{}", lifetime.name.display(f.db.upcast(), f.edition()))?;
    }
    for (_, ty) in params.iter_type_or_consts() {
        if let Some(name) = &ty.name() {
            match ty {
                TypeOrConstParamData::TypeParamData(ty) => {
                    if ty.provenance != TypeParamProvenance::TypeParamList {
                        continue;
                    }
                    delim(f)?;
                    write!(f, "{}", name.display(f.db.upcast(), f.edition()))?;
                    if let Some(default) = &ty.default {
                        f.write_str(" = ")?;
                        default.hir_fmt(f, &params.types_map)?;
                    }
                }
                TypeOrConstParamData::ConstParamData(c) => {
                    delim(f)?;
                    write!(f, "const {}: ", name.display(f.db.upcast(), f.edition()))?;
                    c.ty.hir_fmt(f, &params.types_map)?;

                    if let Some(default) = &c.default {
                        f.write_str(" = ")?;
                        write!(f, "{}", default.display(f.db.upcast(), f.edition()))?;
                    }
                }
            }
        }
    }

    f.write_char('>')?;
    Ok(())
}

fn write_where_clause(
    def: GenericDefId,
    f: &mut HirFormatter<'_>,
) -> Result<bool, HirDisplayError> {
    let params = f.db.generic_params(def);
    if !has_disaplayable_predicates(&params) {
        return Ok(false);
    }

    f.write_str("\nwhere")?;
    write_where_predicates(&params, f)?;

    Ok(true)
}

fn has_disaplayable_predicates(params: &GenericParams) -> bool {
    params.where_predicates().any(|pred| {
        !matches!(
            pred,
            WherePredicate::TypeBound { target: WherePredicateTypeTarget::TypeOrConstParam(id), .. }
            if params[*id].name().is_none()
        )
    })
}

fn write_where_predicates(
    params: &GenericParams,
    f: &mut HirFormatter<'_>,
) -> Result<(), HirDisplayError> {
    use WherePredicate::*;

    // unnamed type targets are displayed inline with the argument itself, e.g. `f: impl Y`.
    let is_unnamed_type_target = |params: &GenericParams, target: &WherePredicateTypeTarget| {
        matches!(target,
            WherePredicateTypeTarget::TypeOrConstParam(id) if params[*id].name().is_none()
        )
    };

    let write_target = |target: &WherePredicateTypeTarget, f: &mut HirFormatter<'_>| match target {
        WherePredicateTypeTarget::TypeRef(ty) => ty.hir_fmt(f, &params.types_map),
        WherePredicateTypeTarget::TypeOrConstParam(id) => match params[*id].name() {
            Some(name) => write!(f, "{}", name.display(f.db.upcast(), f.edition())),
            None => f.write_str("{unnamed}"),
        },
    };

    let check_same_target = |pred1: &WherePredicate, pred2: &WherePredicate| match (pred1, pred2) {
        (TypeBound { target: t1, .. }, TypeBound { target: t2, .. }) => t1 == t2,
        (Lifetime { target: t1, .. }, Lifetime { target: t2, .. }) => t1 == t2,
        (
            ForLifetime { lifetimes: l1, target: t1, .. },
            ForLifetime { lifetimes: l2, target: t2, .. },
        ) => l1 == l2 && t1 == t2,
        _ => false,
    };

    let mut iter = params.where_predicates().peekable();
    while let Some(pred) = iter.next() {
        if matches!(pred, TypeBound { target, .. } if is_unnamed_type_target(params, target)) {
            continue;
        }

        f.write_str("\n    ")?;
        match pred {
            TypeBound { target, bound } => {
                write_target(target, f)?;
                f.write_str(": ")?;
                bound.hir_fmt(f, &params.types_map)?;
            }
            Lifetime { target, bound } => {
                let target = target.name.display(f.db.upcast(), f.edition());
                let bound = bound.name.display(f.db.upcast(), f.edition());
                write!(f, "{target}: {bound}")?;
            }
            ForLifetime { lifetimes, target, bound } => {
                let lifetimes =
                    lifetimes.iter().map(|it| it.display(f.db.upcast(), f.edition())).join(", ");
                write!(f, "for<{lifetimes}> ")?;
                write_target(target, f)?;
                f.write_str(": ")?;
                bound.hir_fmt(f, &params.types_map)?;
            }
        }

        while let Some(nxt) = iter.next_if(|nxt| check_same_target(pred, nxt)) {
            f.write_str(" + ")?;
            match nxt {
                TypeBound { bound, .. } | ForLifetime { bound, .. } => {
                    bound.hir_fmt(f, &params.types_map)?
                }
                Lifetime { bound, .. } => {
                    write!(f, "{}", bound.name.display(f.db.upcast(), f.edition()))?
                }
            }
        }
        f.write_str(",")?;
    }

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
            Some(name) => write!(f, "{}: ", name.display(f.db.upcast(), f.edition()))?,
            None => f.write_str("_: ")?,
        }
        data.type_ref.hir_fmt(f, &data.types_map)?;
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
        write!(f, "{}: ", data.name.display(f.db.upcast(), f.edition()))?;
        data.type_ref.hir_fmt(f, &data.types_map)?;
        Ok(())
    }
}

impl HirDisplay for Trait {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_trait_header(self, f)?;
        let def_id = GenericDefId::TraitId(self.id);
        let has_where_clause = write_where_clause(def_id, f)?;

        if let Some(limit) = f.entity_limit {
            let assoc_items = self.items(f.db);
            let count = assoc_items.len().min(limit);
            f.write_char(if !has_where_clause { ' ' } else { '\n' })?;
            if count == 0 {
                if assoc_items.is_empty() {
                    f.write_str("{}")?;
                } else {
                    f.write_str("{ /* … */ }")?;
                }
            } else {
                f.write_str("{\n")?;
                for item in &assoc_items[..count] {
                    f.write_str("    ")?;
                    match item {
                        AssocItem::Function(func) => func.hir_fmt(f),
                        AssocItem::Const(cst) => cst.hir_fmt(f),
                        AssocItem::TypeAlias(type_alias) => type_alias.hir_fmt(f),
                    }?;
                    f.write_str(";\n")?;
                }

                if assoc_items.len() > count {
                    f.write_str("    /* … */\n")?;
                }
                f.write_str("}")?;
            }
        }

        Ok(())
    }
}

fn write_trait_header(trait_: &Trait, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
    write_visibility(trait_.module(f.db).id, trait_.visibility(f.db), f)?;
    let data = f.db.trait_data(trait_.id);
    if data.is_unsafe {
        f.write_str("unsafe ")?;
    }
    if data.is_auto {
        f.write_str("auto ")?;
    }
    write!(f, "trait {}", data.name.display(f.db.upcast(), f.edition()))?;
    write_generic_params(GenericDefId::TraitId(trait_.id), f)?;
    Ok(())
}

impl HirDisplay for TraitAlias {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write_visibility(self.module(f.db).id, self.visibility(f.db), f)?;
        let data = f.db.trait_alias_data(self.id);
        write!(f, "trait {}", data.name.display(f.db.upcast(), f.edition()))?;
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
        write!(f, "type {}", data.name.display(f.db.upcast(), f.edition()))?;
        let def_id = GenericDefId::TypeAliasId(self.id);
        write_generic_params(def_id, f)?;
        if !data.bounds.is_empty() {
            f.write_str(": ")?;
            f.write_joined(
                data.bounds.iter().map(|bound| hir_display_with_types_map(bound, &data.types_map)),
                " + ",
            )?;
        }
        if let Some(ty) = data.type_ref {
            f.write_str(" = ")?;
            ty.hir_fmt(f, &data.types_map)?;
        }
        write_where_clause(def_id, f)?;
        Ok(())
    }
}

impl HirDisplay for Module {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        // FIXME: Module doesn't have visibility saved in data.
        match self.name(f.db) {
            Some(name) => write!(f, "mod {}", name.display(f.db.upcast(), f.edition())),
            None if self.is_crate_root() => match self.krate(f.db).display_name(f.db) {
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
        write!(f, " {}", self.name(f.db).display(f.db.upcast(), f.edition()))
    }
}
