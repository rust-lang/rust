//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.
use std::sync::Arc;

use either::Either;
use hir_expand::{
    name::{name, AsName, Name},
    InFile,
};
use ra_arena::{map::ArenaMap, Arena};
use ra_db::FileId;
use ra_prof::profile;
use ra_syntax::ast::{self, NameOwner, TypeBoundsOwner, TypeParamsOwner};

use crate::{
    child_by_source::ChildBySource,
    db::DefDatabase,
    dyn_map::DynMap,
    keys,
    src::HasChildSource,
    src::HasSource,
    type_ref::{TypeBound, TypeRef},
    AdtId, GenericDefId, LocalTypeParamId, Lookup, TypeParamId,
};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TypeParamData {
    pub name: Option<Name>,
    pub default: Option<TypeRef>,
    pub provenance: TypeParamProvenance,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TypeParamProvenance {
    TypeParamList,
    TraitSelf,
    ArgumentImplTrait,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParams {
    pub types: Arena<LocalTypeParamId, TypeParamData>,
    // lifetimes: Arena<LocalLifetimeParamId, LifetimeParamData>,
    pub where_predicates: Vec<WherePredicate>,
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WherePredicate {
    pub target: WherePredicateTarget,
    pub bound: TypeBound,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WherePredicateTarget {
    TypeRef(TypeRef),
    /// For desugared where predicates that can directly refer to a type param.
    TypeParam(LocalTypeParamId),
}

type SourceMap = ArenaMap<LocalTypeParamId, Either<ast::TraitDef, ast::TypeParam>>;

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> Arc<GenericParams> {
        let _p = profile("generic_params_query");
        let (params, _source_map) = GenericParams::new(db, def);
        Arc::new(params)
    }

    fn new(db: &dyn DefDatabase, def: GenericDefId) -> (GenericParams, InFile<SourceMap>) {
        let mut generics = GenericParams { types: Arena::default(), where_predicates: Vec::new() };
        let mut sm = ArenaMap::default();
        // FIXME: add `: Sized` bound for everything except for `Self` in traits
        let file_id = match def {
            GenericDefId::FunctionId(it) => {
                let src = it.lookup(db).source(db);
                generics.fill(&mut sm, &src.value);
                // lower `impl Trait` in arguments
                let data = db.function_data(it);
                for param in &data.params {
                    generics.fill_implicit_impl_trait_args(param);
                }
                src.file_id
            }
            GenericDefId::AdtId(AdtId::StructId(it)) => {
                let src = it.lookup(db).source(db);
                generics.fill(&mut sm, &src.value);
                src.file_id
            }
            GenericDefId::AdtId(AdtId::UnionId(it)) => {
                let src = it.lookup(db).source(db);
                generics.fill(&mut sm, &src.value);
                src.file_id
            }
            GenericDefId::AdtId(AdtId::EnumId(it)) => {
                let src = it.lookup(db).source(db);
                generics.fill(&mut sm, &src.value);
                src.file_id
            }
            GenericDefId::TraitId(it) => {
                let src = it.lookup(db).source(db);

                // traits get the Self type as an implicit first type parameter
                let self_param_id = generics.types.alloc(TypeParamData {
                    name: Some(name![Self]),
                    default: None,
                    provenance: TypeParamProvenance::TraitSelf,
                });
                sm.insert(self_param_id, Either::Left(src.value.clone()));
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name![Self].into());
                generics.fill_bounds(&src.value, self_param);

                generics.fill(&mut sm, &src.value);
                src.file_id
            }
            GenericDefId::TypeAliasId(it) => {
                let src = it.lookup(db).source(db);
                generics.fill(&mut sm, &src.value);
                src.file_id
            }
            // Note that we don't add `Self` here: in `impl`s, `Self` is not a
            // type-parameter, but rather is a type-alias for impl's target
            // type, so this is handled by the resolver.
            GenericDefId::ImplId(it) => {
                let src = it.lookup(db).source(db);
                generics.fill(&mut sm, &src.value);
                src.file_id
            }
            // We won't be using this ID anyway
            GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => FileId(!0).into(),
        };

        (generics, InFile::new(file_id, sm))
    }

    fn fill(&mut self, sm: &mut SourceMap, node: &dyn TypeParamsOwner) {
        if let Some(params) = node.type_param_list() {
            self.fill_params(sm, params)
        }
        if let Some(where_clause) = node.where_clause() {
            self.fill_where_predicates(where_clause);
        }
    }

    fn fill_bounds(&mut self, node: &dyn ast::TypeBoundsOwner, type_ref: TypeRef) {
        for bound in
            node.type_bound_list().iter().flat_map(|type_bound_list| type_bound_list.bounds())
        {
            self.add_where_predicate_from_bound(bound, type_ref.clone());
        }
    }

    fn fill_params(&mut self, sm: &mut SourceMap, params: ast::TypeParamList) {
        for type_param in params.type_params() {
            let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
            // FIXME: Use `Path::from_src`
            let default = type_param.default_type().map(TypeRef::from_ast);
            let param = TypeParamData {
                name: Some(name.clone()),
                default,
                provenance: TypeParamProvenance::TypeParamList,
            };
            let param_id = self.types.alloc(param);
            sm.insert(param_id, Either::Right(type_param.clone()));

            let type_ref = TypeRef::Path(name.into());
            self.fill_bounds(&type_param, type_ref);
        }
    }

    fn fill_where_predicates(&mut self, where_clause: ast::WhereClause) {
        for pred in where_clause.predicates() {
            let type_ref = match pred.type_ref() {
                Some(type_ref) => type_ref,
                None => continue,
            };
            let type_ref = TypeRef::from_ast(type_ref);
            for bound in pred.type_bound_list().iter().flat_map(|l| l.bounds()) {
                self.add_where_predicate_from_bound(bound, type_ref.clone());
            }
        }
    }

    fn add_where_predicate_from_bound(&mut self, bound: ast::TypeBound, type_ref: TypeRef) {
        if bound.has_question_mark() {
            // FIXME: remove this bound
            return;
        }
        let bound = TypeBound::from_ast(bound);
        self.where_predicates
            .push(WherePredicate { target: WherePredicateTarget::TypeRef(type_ref), bound });
    }

    fn fill_implicit_impl_trait_args(&mut self, type_ref: &TypeRef) {
        type_ref.walk(&mut |type_ref| {
            if let TypeRef::ImplTrait(bounds) = type_ref {
                let param = TypeParamData {
                    name: None,
                    default: None,
                    provenance: TypeParamProvenance::ArgumentImplTrait,
                };
                let param_id = self.types.alloc(param);
                for bound in bounds {
                    self.where_predicates.push(WherePredicate {
                        target: WherePredicateTarget::TypeParam(param_id),
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

impl HasChildSource for GenericDefId {
    type ChildId = LocalTypeParamId;
    type Value = Either<ast::TraitDef, ast::TypeParam>;
    fn child_source(&self, db: &dyn DefDatabase) -> InFile<SourceMap> {
        let (_, sm) = GenericParams::new(db, *self);
        sm
    }
}

impl ChildBySource for GenericDefId {
    fn child_by_source(&self, db: &dyn DefDatabase) -> DynMap {
        let mut res = DynMap::default();
        let arena_map = self.child_source(db);
        let arena_map = arena_map.as_ref();
        for (local_id, src) in arena_map.value.iter() {
            let id = TypeParamId { parent: *self, local_id };
            if let Either::Right(type_param) = src {
                res[keys::TYPE_PARAM].insert(arena_map.with_value(type_param.clone()), id)
            }
        }
        res
    }
}
