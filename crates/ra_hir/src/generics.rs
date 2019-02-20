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
    pub(crate) parent_params: Option<Arc<GenericParams>>,
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
        let parent = match def {
            GenericDef::Function(it) => it.impl_block(db),
            GenericDef::Type(it) => it.impl_block(db),
            GenericDef::Struct(_) | GenericDef::Enum(_) | GenericDef::Trait(_) => None,
            GenericDef::ImplBlock(_) => None,
        };
        generics.parent_params = parent.map(|p| p.generic_params(db));
        let start = generics.parent_params.as_ref().map(|p| p.params.len()).unwrap_or(0) as u32;
        match def {
            GenericDef::Function(it) => generics.fill(&*it.source(db).1, start),
            GenericDef::Struct(it) => generics.fill(&*it.source(db).1, start),
            GenericDef::Enum(it) => generics.fill(&*it.source(db).1, start),
            GenericDef::Trait(it) => generics.fill(&*it.source(db).1, start),
            GenericDef::Type(it) => generics.fill(&*it.source(db).1, start),
            GenericDef::ImplBlock(it) => generics.fill(&*it.source(db).1, start),
        }

        Arc::new(generics)
    }

    fn fill(&mut self, node: &impl TypeParamsOwner, start: u32) {
        if let Some(params) = node.type_param_list() {
            self.fill_params(params, start)
        }
    }

    fn fill_params(&mut self, params: &ast::TypeParamList, start: u32) {
        for (idx, type_param) in params.type_params().enumerate() {
            let name = type_param.name().map(AsName::as_name).unwrap_or_else(Name::missing);
            let param = GenericParam { idx: idx as u32 + start, name };
            self.params.push(param);
        }
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
