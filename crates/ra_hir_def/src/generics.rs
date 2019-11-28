//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.
use std::sync::Arc;

use hir_expand::name::{self, AsName, Name};
use ra_syntax::ast::{self, NameOwner, TypeBoundsOwner, TypeParamsOwner};

use crate::{
    db::DefDatabase,
    src::HasSource,
    type_ref::{TypeBound, TypeRef},
    AdtId, AstItemDef, ContainerId, GenericDefId, Lookup,
};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParam {
    // FIXME: give generic params proper IDs
    pub idx: u32,
    pub name: Name,
    pub default: Option<TypeRef>,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParams {
    pub parent_params: Option<Arc<GenericParams>>,
    pub params: Vec<GenericParam>,
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
        let parent_generics = parent_generic_def(db, def).map(|it| db.generic_params(it));
        Arc::new(GenericParams::new(db, def.into(), parent_generics))
    }

    fn new(
        db: &impl DefDatabase,
        def: GenericDefId,
        parent_params: Option<Arc<GenericParams>>,
    ) -> GenericParams {
        let mut generics =
            GenericParams { params: Vec::new(), parent_params, where_predicates: Vec::new() };
        let start = generics.parent_params.as_ref().map(|p| p.params.len()).unwrap_or(0) as u32;
        // FIXME: add `: Sized` bound for everything except for `Self` in traits
        match def {
            GenericDefId::FunctionId(it) => generics.fill(&it.lookup(db).source(db).value, start),
            GenericDefId::AdtId(AdtId::StructId(it)) => generics.fill(&it.source(db).value, start),
            GenericDefId::AdtId(AdtId::UnionId(it)) => generics.fill(&it.source(db).value, start),
            GenericDefId::AdtId(AdtId::EnumId(it)) => generics.fill(&it.source(db).value, start),
            GenericDefId::TraitId(it) => {
                // traits get the Self type as an implicit first type parameter
                generics.params.push(GenericParam {
                    idx: start,
                    name: name::SELF_TYPE,
                    default: None,
                });
                generics.fill(&it.source(db).value, start + 1);
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name::SELF_TYPE.into());
                generics.fill_bounds(&it.source(db).value, self_param);
            }
            GenericDefId::TypeAliasId(it) => generics.fill(&it.lookup(db).source(db).value, start),
            // Note that we don't add `Self` here: in `impl`s, `Self` is not a
            // type-parameter, but rather is a type-alias for impl's target
            // type, so this is handled by the resolver.
            GenericDefId::ImplId(it) => generics.fill(&it.source(db).value, start),
            GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => {}
        }

        generics
    }

    fn fill(&mut self, node: &impl TypeParamsOwner, start: u32) {
        if let Some(params) = node.type_param_list() {
            self.fill_params(params, start)
        }
        if let Some(where_clause) = node.where_clause() {
            self.fill_where_predicates(where_clause);
        }
    }

    fn fill_bounds(&mut self, node: &impl ast::TypeBoundsOwner, type_ref: TypeRef) {
        for bound in
            node.type_bound_list().iter().flat_map(|type_bound_list| type_bound_list.bounds())
        {
            self.add_where_predicate_from_bound(bound, type_ref.clone());
        }
    }

    fn fill_params(&mut self, params: ast::TypeParamList, start: u32) {
        for (idx, type_param) in params.type_params().enumerate() {
            let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
            // FIXME: Use `Path::from_src`
            let default = type_param.default_type().map(TypeRef::from_ast);
            let param = GenericParam { idx: idx as u32 + start, name: name.clone(), default };
            self.params.push(param);

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

    pub fn find_by_name(&self, name: &Name) -> Option<&GenericParam> {
        self.params.iter().find(|p| &p.name == name)
    }

    pub fn count_parent_params(&self) -> usize {
        self.parent_params.as_ref().map(|p| p.count_params_including_parent()).unwrap_or(0)
    }

    pub fn count_params_including_parent(&self) -> usize {
        let parent_count = self.count_parent_params();
        parent_count + self.params.len()
    }

    fn for_each_param<'a>(&'a self, f: &mut impl FnMut(&'a GenericParam)) {
        if let Some(parent) = &self.parent_params {
            parent.for_each_param(f);
        }
        self.params.iter().for_each(f);
    }

    pub fn params_including_parent(&self) -> Vec<&GenericParam> {
        let mut vec = Vec::with_capacity(self.count_params_including_parent());
        self.for_each_param(&mut |p| vec.push(p));
        vec
    }
}

fn parent_generic_def(db: &impl DefDatabase, def: GenericDefId) -> Option<GenericDefId> {
    let container = match def {
        GenericDefId::FunctionId(it) => it.lookup(db).container,
        GenericDefId::TypeAliasId(it) => it.lookup(db).container,
        GenericDefId::ConstId(it) => it.lookup(db).container,
        GenericDefId::EnumVariantId(it) => return Some(it.parent.into()),
        GenericDefId::AdtId(_) | GenericDefId::TraitId(_) | GenericDefId::ImplId(_) => return None,
    };

    match container {
        ContainerId::ImplId(it) => Some(it.into()),
        ContainerId::TraitId(it) => Some(it.into()),
        ContainerId::ModuleId(_) => None,
    }
}
