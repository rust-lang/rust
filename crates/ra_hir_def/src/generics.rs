//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.
use std::sync::Arc;

use hir_expand::name::{self, AsName, Name};
use ra_arena::Arena;
use ra_syntax::ast::{self, NameOwner, TypeBoundsOwner, TypeParamsOwner};

use crate::{
    db::DefDatabase,
    src::HasSource,
    type_ref::{TypeBound, TypeRef},
    AdtId, AstItemDef, GenericDefId, LocalGenericParamId, Lookup,
};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParamData {
    pub name: Name,
    pub default: Option<TypeRef>,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParams {
    pub params: Arena<LocalGenericParamId, GenericParamData>,
    pub where_predicates: Vec<WherePredicate>,
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WherePredicate {
    pub type_ref: TypeRef,
    pub bound: TypeBound,
}

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &impl DefDatabase,
        def: GenericDefId,
    ) -> Arc<GenericParams> {
        Arc::new(GenericParams::new(db, def.into()))
    }

    fn new(db: &impl DefDatabase, def: GenericDefId) -> GenericParams {
        let mut generics = GenericParams { params: Arena::default(), where_predicates: Vec::new() };
        // FIXME: add `: Sized` bound for everything except for `Self` in traits
        match def {
            GenericDefId::FunctionId(it) => generics.fill(&it.lookup(db).source(db).value),
            GenericDefId::AdtId(AdtId::StructId(it)) => generics.fill(&it.source(db).value),
            GenericDefId::AdtId(AdtId::UnionId(it)) => generics.fill(&it.source(db).value),
            GenericDefId::AdtId(AdtId::EnumId(it)) => generics.fill(&it.source(db).value),
            GenericDefId::TraitId(it) => {
                // traits get the Self type as an implicit first type parameter
                generics.params.alloc(GenericParamData { name: name::SELF_TYPE, default: None });
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name::SELF_TYPE.into());
                generics.fill_bounds(&it.source(db).value, self_param);

                generics.fill(&it.source(db).value);
            }
            GenericDefId::TypeAliasId(it) => generics.fill(&it.lookup(db).source(db).value),
            // Note that we don't add `Self` here: in `impl`s, `Self` is not a
            // type-parameter, but rather is a type-alias for impl's target
            // type, so this is handled by the resolver.
            GenericDefId::ImplId(it) => generics.fill(&it.source(db).value),
            GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => {}
        }

        generics
    }

    fn fill(&mut self, node: &dyn TypeParamsOwner) {
        if let Some(params) = node.type_param_list() {
            self.fill_params(params)
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

    fn fill_params(&mut self, params: ast::TypeParamList) {
        for type_param in params.type_params() {
            let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
            // FIXME: Use `Path::from_src`
            let default = type_param.default_type().map(TypeRef::from_ast);
            let param = GenericParamData { name: name.clone(), default };
            self.params.alloc(param);

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
        self.where_predicates.push(WherePredicate { type_ref, bound });
    }

    pub fn find_by_name(&self, name: &Name) -> Option<LocalGenericParamId> {
        self.params.iter().find_map(|(id, p)| if &p.name == name { Some(id) } else { None })
    }
}
