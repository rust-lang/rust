//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::sync::Arc;

use ra_syntax::ast::{self, NameOwner, TypeParamsOwner};

use crate::{db::PersistentHirDatabase, Name, AsName, Function, Struct, Enum, Trait, Type, ImplBlock};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParam {
    // TODO: give generic params proper IDs
    pub(crate) idx: u32,
    pub(crate) name: Name,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GenericParams {
    pub(crate) params: Vec<GenericParam>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Type(Type),
    ImplBlock(ImplBlock),
}
impl_froms!(GenericDef: Function, Struct, Enum, Trait, Type, ImplBlock);

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &impl PersistentHirDatabase,
        def: GenericDef,
    ) -> Arc<GenericParams> {
        let mut generics = GenericParams::default();
        match def {
            GenericDef::Function(it) => generics.fill(&*it.source(db).1),
            GenericDef::Struct(it) => generics.fill(&*it.source(db).1),
            GenericDef::Enum(it) => generics.fill(&*it.source(db).1),
            GenericDef::Trait(it) => generics.fill(&*it.source(db).1),
            GenericDef::Type(it) => generics.fill(&*it.source(db).1),
            GenericDef::ImplBlock(it) => generics.fill(&*it.source(db).1),
        }

        Arc::new(generics)
    }

    fn fill(&mut self, node: &impl TypeParamsOwner) {
        if let Some(params) = node.type_param_list() {
            self.fill_params(params)
        }
    }

    fn fill_params(&mut self, params: &ast::TypeParamList) {
        for (idx, type_param) in params.type_params().enumerate() {
            let name = type_param.name().map(AsName::as_name).unwrap_or_else(Name::missing);
            let param = GenericParam { idx: idx as u32, name };
            self.params.push(param);
        }
    }

    pub(crate) fn find_by_name(&self, name: &Name) -> Option<&GenericParam> {
        self.params.iter().find(|p| &p.name == name)
    }
}
