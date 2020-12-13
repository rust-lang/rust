//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.
use std::sync::Arc;

use arena::{map::ArenaMap, Arena};
use base_db::FileId;
use either::Either;
use hir_expand::{
    name::{name, AsName, Name},
    InFile,
};
use syntax::ast::{self, GenericParamsOwner, NameOwner, TypeBoundsOwner};

use crate::{
    body::LowerCtx,
    child_by_source::ChildBySource,
    db::DefDatabase,
    dyn_map::DynMap,
    keys,
    src::HasSource,
    type_ref::{LifetimeRef, TypeBound, TypeRef},
    AdtId, GenericDefId, LifetimeParamId, LocalLifetimeParamId, LocalTypeParamId, Lookup,
    TypeParamId,
};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TypeParamData {
    pub name: Option<Name>,
    pub default: Option<TypeRef>,
    pub provenance: TypeParamProvenance,
}

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LifetimeParamData {
    pub name: Name,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TypeParamProvenance {
    TypeParamList,
    TraitSelf,
    ArgumentImplTrait,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GenericParams {
    pub types: Arena<TypeParamData>,
    pub lifetimes: Arena<LifetimeParamData>,
    pub where_predicates: Vec<WherePredicate>,
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WherePredicate {
    TypeBound { target: WherePredicateTypeTarget, bound: TypeBound },
    Lifetime { target: LifetimeRef, bound: LifetimeRef },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WherePredicateTypeTarget {
    TypeRef(TypeRef),
    /// For desugared where predicates that can directly refer to a type param.
    TypeParam(LocalTypeParamId),
    // FIXME: ForLifetime(Vec<LifetimeParamId>, TypeRef)
}

#[derive(Default)]
pub struct SourceMaps {
    pub type_params: ArenaMap<LocalTypeParamId, Either<ast::Trait, ast::TypeParam>>,
    pub lifetime_params: ArenaMap<LocalLifetimeParamId, ast::LifetimeParam>,
}

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> Arc<GenericParams> {
        let _p = profile::span("generic_params_query");

        let generics = match def {
            GenericDefId::FunctionId(id) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::AdtId(AdtId::StructId(id)) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::AdtId(AdtId::EnumId(id)) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::AdtId(AdtId::UnionId(id)) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::TraitId(id) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::TypeAliasId(id) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::ImplId(id) => {
                let id = id.lookup(db).id;
                let tree = db.item_tree(id.file_id);
                let item = &tree[id.value];
                tree[item.generic_params].clone()
            }
            GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => GenericParams::default(),
        };
        Arc::new(generics)
    }

    fn new(db: &dyn DefDatabase, def: GenericDefId) -> (GenericParams, InFile<SourceMaps>) {
        let mut generics = GenericParams::default();
        let mut sm = SourceMaps::default();

        // FIXME: add `: Sized` bound for everything except for `Self` in traits
        let file_id = match def {
            GenericDefId::FunctionId(it) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);
                generics.fill(&lower_ctx, &mut sm, &src.value);
                // lower `impl Trait` in arguments
                let data = db.function_data(it);
                for param in &data.params {
                    generics.fill_implicit_impl_trait_args(param);
                }
                src.file_id
            }
            GenericDefId::AdtId(AdtId::StructId(it)) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);
                generics.fill(&lower_ctx, &mut sm, &src.value);
                src.file_id
            }
            GenericDefId::AdtId(AdtId::UnionId(it)) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);
                generics.fill(&lower_ctx, &mut sm, &src.value);
                src.file_id
            }
            GenericDefId::AdtId(AdtId::EnumId(it)) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);
                generics.fill(&lower_ctx, &mut sm, &src.value);
                src.file_id
            }
            GenericDefId::TraitId(it) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);

                // traits get the Self type as an implicit first type parameter
                let self_param_id = generics.types.alloc(TypeParamData {
                    name: Some(name![Self]),
                    default: None,
                    provenance: TypeParamProvenance::TraitSelf,
                });
                sm.type_params.insert(self_param_id, Either::Left(src.value.clone()));
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name![Self].into());
                generics.fill_bounds(&lower_ctx, &src.value, Either::Left(self_param));

                generics.fill(&lower_ctx, &mut sm, &src.value);
                src.file_id
            }
            GenericDefId::TypeAliasId(it) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);

                generics.fill(&lower_ctx, &mut sm, &src.value);
                src.file_id
            }
            // Note that we don't add `Self` here: in `impl`s, `Self` is not a
            // type-parameter, but rather is a type-alias for impl's target
            // type, so this is handled by the resolver.
            GenericDefId::ImplId(it) => {
                let src = it.lookup(db).source(db);
                let lower_ctx = LowerCtx::new(db, src.file_id);

                generics.fill(&lower_ctx, &mut sm, &src.value);
                src.file_id
            }
            // We won't be using this ID anyway
            GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => FileId(!0).into(),
        };

        (generics, InFile::new(file_id, sm))
    }

    pub(crate) fn fill(
        &mut self,
        lower_ctx: &LowerCtx,
        sm: &mut SourceMaps,
        node: &dyn GenericParamsOwner,
    ) {
        if let Some(params) = node.generic_param_list() {
            self.fill_params(lower_ctx, sm, params)
        }
        if let Some(where_clause) = node.where_clause() {
            self.fill_where_predicates(lower_ctx, where_clause);
        }
    }

    pub(crate) fn fill_bounds(
        &mut self,
        lower_ctx: &LowerCtx,
        node: &dyn ast::TypeBoundsOwner,
        target: Either<TypeRef, LifetimeRef>,
    ) {
        for bound in
            node.type_bound_list().iter().flat_map(|type_bound_list| type_bound_list.bounds())
        {
            self.add_where_predicate_from_bound(lower_ctx, bound, target.clone());
        }
    }

    fn fill_params(
        &mut self,
        lower_ctx: &LowerCtx,
        sm: &mut SourceMaps,
        params: ast::GenericParamList,
    ) {
        for type_param in params.type_params() {
            let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
            // FIXME: Use `Path::from_src`
            let default = type_param.default_type().map(|it| TypeRef::from_ast(lower_ctx, it));
            let param = TypeParamData {
                name: Some(name.clone()),
                default,
                provenance: TypeParamProvenance::TypeParamList,
            };
            let param_id = self.types.alloc(param);
            sm.type_params.insert(param_id, Either::Right(type_param.clone()));

            let type_ref = TypeRef::Path(name.into());
            self.fill_bounds(&lower_ctx, &type_param, Either::Left(type_ref));
        }
        for lifetime_param in params.lifetime_params() {
            let name = lifetime_param
                .lifetime_token()
                .map_or_else(Name::missing, |tok| Name::new_lifetime(&tok));
            let param = LifetimeParamData { name: name.clone() };
            let param_id = self.lifetimes.alloc(param);
            sm.lifetime_params.insert(param_id, lifetime_param.clone());
            let lifetime_ref = LifetimeRef::new_name(name);
            self.fill_bounds(&lower_ctx, &lifetime_param, Either::Right(lifetime_ref));
        }
    }

    fn fill_where_predicates(&mut self, lower_ctx: &LowerCtx, where_clause: ast::WhereClause) {
        for pred in where_clause.predicates() {
            let target = if let Some(type_ref) = pred.ty() {
                Either::Left(TypeRef::from_ast(lower_ctx, type_ref))
            } else if let Some(lifetime_tok) = pred.lifetime_token() {
                Either::Right(LifetimeRef::from_token(lifetime_tok))
            } else {
                continue;
            };
            for bound in pred.type_bound_list().iter().flat_map(|l| l.bounds()) {
                self.add_where_predicate_from_bound(lower_ctx, bound, target.clone());
            }
        }
    }

    fn add_where_predicate_from_bound(
        &mut self,
        lower_ctx: &LowerCtx,
        bound: ast::TypeBound,
        target: Either<TypeRef, LifetimeRef>,
    ) {
        if bound.question_mark_token().is_some() {
            // FIXME: remove this bound
            return;
        }
        let bound = TypeBound::from_ast(lower_ctx, bound);
        let predicate = match (target, bound) {
            (Either::Left(type_ref), bound) => WherePredicate::TypeBound {
                target: WherePredicateTypeTarget::TypeRef(type_ref),
                bound,
            },
            (Either::Right(lifetime), TypeBound::Lifetime(bound)) => {
                WherePredicate::Lifetime { target: lifetime, bound }
            }
            _ => return,
        };
        self.where_predicates.push(predicate);
    }

    pub(crate) fn fill_implicit_impl_trait_args(&mut self, type_ref: &TypeRef) {
        type_ref.walk(&mut |type_ref| {
            if let TypeRef::ImplTrait(bounds) = type_ref {
                let param = TypeParamData {
                    name: None,
                    default: None,
                    provenance: TypeParamProvenance::ArgumentImplTrait,
                };
                let param_id = self.types.alloc(param);
                for bound in bounds {
                    self.where_predicates.push(WherePredicate::TypeBound {
                        target: WherePredicateTypeTarget::TypeParam(param_id),
                        bound: bound.clone(),
                    });
                }
            }
        });
    }

    pub fn find_by_name(&self, name: &Name) -> Option<LocalTypeParamId> {
        self.types
            .iter()
            .find_map(|(id, p)| if p.name.as_ref() == Some(name) { Some(id) } else { None })
    }

    pub fn find_trait_self_param(&self) -> Option<LocalTypeParamId> {
        self.types.iter().find_map(|(id, p)| {
            if p.provenance == TypeParamProvenance::TraitSelf {
                Some(id)
            } else {
                None
            }
        })
    }
}
impl GenericDefId {
    // FIXME: Change HasChildSource's ChildId AssocItem to be a generic parameter instead
    pub fn child_source(&self, db: &dyn DefDatabase) -> InFile<SourceMaps> {
        GenericParams::new(db, *self).1
    }
}

impl ChildBySource for GenericDefId {
    fn child_by_source(&self, db: &dyn DefDatabase) -> DynMap {
        let mut res = DynMap::default();
        let (_, sm) = GenericParams::new(db, *self);

        let sm = sm.as_ref();
        for (local_id, src) in sm.value.type_params.iter() {
            let id = TypeParamId { parent: *self, local_id };
            if let Either::Right(type_param) = src {
                res[keys::TYPE_PARAM].insert(sm.with_value(type_param.clone()), id)
            }
        }
        for (local_id, src) in sm.value.lifetime_params.iter() {
            let id = LifetimeParamId { parent: *self, local_id };
            res[keys::LIFETIME_PARAM].insert(sm.with_value(src.clone()), id);
        }
        res
    }
}
