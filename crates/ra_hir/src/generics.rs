//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::sync::Arc;

use hir_def::{
    path::Path,
    type_ref::{TypeBound, TypeRef},
};
use hir_expand::name::{self, AsName};
use ra_syntax::ast::{self, DefaultTypeParamOwner, NameOwner, TypeBoundsOwner, TypeParamsOwner};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    Adt, Const, Container, Enum, EnumVariant, Function, HasSource, ImplBlock, Name, Struct, Trait,
    TypeAlias, Union,
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParams {
    pub(crate) def: GenericDef,
    pub(crate) parent_params: Option<Arc<GenericParams>>,
    pub(crate) params: Vec<GenericParam>,
    pub(crate) where_predicates: Vec<WherePredicate>,
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WherePredicate {
    pub(crate) type_ref: TypeRef,
    pub(crate) bound: TypeBound,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Adt(Adt),
    Trait(Trait),
    TypeAlias(TypeAlias),
    ImplBlock(ImplBlock),
    // enum variants cannot have generics themselves, but their parent enums
    // can, and this makes some code easier to write
    EnumVariant(EnumVariant),
    // consts can have type parameters from their parents (i.e. associated consts of traits)
    Const(Const),
}
impl_froms!(
    GenericDef: Function,
    Adt(Struct, Enum, Union),
    Trait,
    TypeAlias,
    ImplBlock,
    EnumVariant,
    Const
);

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &(impl DefDatabase + AstDatabase),
        def: GenericDef,
    ) -> Arc<GenericParams> {
        let parent = match def {
            GenericDef::Function(it) => it.container(db).map(GenericDef::from),
            GenericDef::TypeAlias(it) => it.container(db).map(GenericDef::from),
            GenericDef::EnumVariant(it) => Some(it.parent_enum(db).into()),
            GenericDef::Adt(_) | GenericDef::Trait(_) => None,
            GenericDef::ImplBlock(_) | GenericDef::Const(_) => None,
        };
        let mut generics = GenericParams {
            def,
            params: Vec::new(),
            parent_params: parent.map(|p| db.generic_params(p)),
            where_predicates: Vec::new(),
        };
        let start = generics.parent_params.as_ref().map(|p| p.params.len()).unwrap_or(0) as u32;
        // FIXME: add `: Sized` bound for everything except for `Self` in traits
        match def {
            GenericDef::Function(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::Adt(Adt::Struct(it)) => generics.fill(&it.source(db).ast, start),
            GenericDef::Adt(Adt::Union(it)) => generics.fill(&it.source(db).ast, start),
            GenericDef::Adt(Adt::Enum(it)) => generics.fill(&it.source(db).ast, start),
            GenericDef::Trait(it) => {
                // traits get the Self type as an implicit first type parameter
                generics.params.push(GenericParam {
                    idx: start,
                    name: name::SELF_TYPE,
                    default: None,
                });
                generics.fill(&it.source(db).ast, start + 1);
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name::SELF_TYPE.into());
                generics.fill_bounds(&it.source(db).ast, self_param);
            }
            GenericDef::TypeAlias(it) => generics.fill(&it.source(db).ast, start),
            // Note that we don't add `Self` here: in `impl`s, `Self` is not a
            // type-parameter, but rather is a type-alias for impl's target
            // type, so this is handled by the resolver.
            GenericDef::ImplBlock(it) => generics.fill(&it.source(db).ast, start),
            GenericDef::EnumVariant(_) | GenericDef::Const(_) => {}
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
            let default = type_param.default_type().and_then(|t| t.path()).and_then(Path::from_ast);

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
            GenericDef::Adt(adt) => adt.resolver(db),
            GenericDef::Trait(inner) => inner.resolver(db),
            GenericDef::TypeAlias(inner) => inner.resolver(db),
            GenericDef::ImplBlock(inner) => inner.resolver(db),
            GenericDef::EnumVariant(inner) => inner.parent_enum(db).resolver(db),
            GenericDef::Const(inner) => inner.resolver(db),
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
