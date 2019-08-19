//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::sync::Arc;

use ra_syntax::ast::{self, DefaultTypeParamOwner, NameOwner, TypeBoundsOwner, TypeParamsOwner};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    name::SELF_TYPE,
    path::Path,
    type_ref::TypeRef,
    AdtDef, AsName, Container, Enum, EnumVariant, Function, HasSource, ImplBlock, Name, Struct,
    Trait, TypeAlias, Union,
};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParam {
    // FIXME: give generic params proper IDs
    pub(crate) idx: u32,
    pub(crate) name: Name,
    pub(crate) default: Option<Path>,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GenericParams {
    pub(crate) parent_params: Option<Arc<GenericParams>>,
    pub(crate) params: Vec<GenericParam>,
    pub(crate) where_predicates: Vec<WherePredicate>,
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WherePredicate {
    pub(crate) type_ref: TypeRef,
    pub(crate) trait_ref: Path,
}

// FIXME: consts can have type parameters from their parents (i.e. associated consts of traits)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Struct(Struct),
    Union(Union),
    Enum(Enum),
    Trait(Trait),
    TypeAlias(TypeAlias),
    ImplBlock(ImplBlock),
    // enum variants cannot have generics themselves, but their parent enums
    // can, and this makes some code easier to write
    EnumVariant(EnumVariant),
}
impl_froms!(GenericDef: Function, Struct, Union, Enum, Trait, TypeAlias, ImplBlock, EnumVariant);

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &(impl DefDatabase + AstDatabase),
        def: GenericDef,
    ) -> Arc<GenericParams> {
        let mut generics = GenericParams::default();
        let parent = match def {
            GenericDef::Function(it) => it.container(db).map(GenericDef::from),
            GenericDef::TypeAlias(it) => it.container(db).map(GenericDef::from),
            GenericDef::EnumVariant(it) => Some(it.parent_enum(db).into()),
            GenericDef::Struct(_)
            | GenericDef::Union(_)
            | GenericDef::Enum(_)
            | GenericDef::Trait(_) => None,
            GenericDef::ImplBlock(_) => None,
        };
        generics.parent_params = parent.map(|p| db.generic_params(p));
        let start = generics.parent_params.as_ref().map(|p| p.params.len()).unwrap_or(0) as u32;
        match def {
            GenericDef::Function(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::Struct(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::Union(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::Enum(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::Trait(it) => {
                // traits get the Self type as an implicit first type parameter
                generics.params.push(GenericParam { idx: start, name: SELF_TYPE, default: None });
                generics.fill(&it.source(db).ast, start + 1);
            }
            GenericDef::TypeAlias(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::ImplBlock(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::EnumVariant(_) => {}
        }

        Arc::new(generics)
    }

    fn fill(&mut self, node: &impl TypeParamsOwner, start: u32) {
        if let Some(params) = node.type_param_list() {
            self.fill_params(params, start)
        }
        if let Some(where_clause) = node.where_clause() {
            self.fill_where_predicates(where_clause);
        }
    }

    fn fill_params(&mut self, params: ast::TypeParamList, start: u32) {
        for (idx, type_param) in params.type_params().enumerate() {
            let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
            let default = type_param.default_type().and_then(|t| t.path()).and_then(Path::from_ast);

            let param = GenericParam { idx: idx as u32 + start, name: name.clone(), default };
            self.params.push(param);

            let type_ref = TypeRef::Path(name.into());
            for bound in type_param
                .type_bound_list()
                .iter()
                .flat_map(|type_bound_list| type_bound_list.bounds())
            {
                self.add_where_predicate_from_bound(bound, type_ref.clone());
            }
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
        let path = bound
            .type_ref()
            .and_then(|tr| match tr {
                ast::TypeRef::PathType(path) => path.path(),
                _ => None,
            })
            .and_then(Path::from_ast);
        let path = match path {
            Some(p) => p,
            None => return,
        };
        self.where_predicates.push(WherePredicate { type_ref, trait_ref: path });
    }

    pub(crate) fn find_by_name(&self, name: &Name) -> Option<&GenericParam> {
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

impl GenericDef {
    pub(crate) fn resolver(&self, db: &impl HirDatabase) -> crate::Resolver {
        match self {
            GenericDef::Function(inner) => inner.resolver(db),
            GenericDef::Struct(inner) => inner.resolver(db),
            GenericDef::Union(inner) => inner.resolver(db),
            GenericDef::Enum(inner) => inner.resolver(db),
            GenericDef::Trait(inner) => inner.resolver(db),
            GenericDef::TypeAlias(inner) => inner.resolver(db),
            GenericDef::ImplBlock(inner) => inner.resolver(db),
            GenericDef::EnumVariant(inner) => inner.parent_enum(db).resolver(db),
        }
    }
}

impl From<Container> for GenericDef {
    fn from(c: Container) -> Self {
        match c {
            Container::Trait(trait_) => trait_.into(),
            Container::ImplBlock(impl_block) => impl_block.into(),
        }
    }
}

impl From<crate::adt::AdtDef> for GenericDef {
    fn from(adt: crate::adt::AdtDef) -> Self {
        match adt {
            AdtDef::Struct(s) => s.into(),
            AdtDef::Union(u) => u.into(),
            AdtDef::Enum(e) => e.into(),
        }
    }
}

pub trait HasGenericParams: Copy {
    fn generic_params(self, db: &impl DefDatabase) -> Arc<GenericParams>;
}

impl<T> HasGenericParams for T
where
    T: Into<GenericDef> + Copy,
{
    fn generic_params(self, db: &impl DefDatabase) -> Arc<GenericParams> {
        db.generic_params(self.into())
    }
}
