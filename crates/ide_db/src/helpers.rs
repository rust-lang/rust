//! A module with ide helpers for high-level ide features.
pub mod insert_use;
pub mod import_assets;

use std::collections::VecDeque;

use base_db::FileId;
use either::Either;
use hir::{Crate, Enum, ItemInNs, MacroDef, Module, ModuleDef, Name, ScopeDef, Semantics, Trait};
use syntax::ast::{self, make};

use crate::RootDatabase;

pub fn item_name(db: &RootDatabase, item: ItemInNs) -> Option<Name> {
    match item {
        ItemInNs::Types(module_def_id) => ModuleDef::from(module_def_id).name(db),
        ItemInNs::Values(module_def_id) => ModuleDef::from(module_def_id).name(db),
        ItemInNs::Macros(macro_def_id) => MacroDef::from(macro_def_id).name(db),
    }
}

/// Converts the mod path struct into its ast representation.
pub fn mod_path_to_ast(path: &hir::ModPath) -> ast::Path {
    let _p = profile::span("mod_path_to_ast");

    let mut segments = Vec::new();
    let mut is_abs = false;
    match path.kind {
        hir::PathKind::Plain => {}
        hir::PathKind::Super(0) => segments.push(make::path_segment_self()),
        hir::PathKind::Super(n) => segments.extend((0..n).map(|_| make::path_segment_super())),
        hir::PathKind::DollarCrate(_) | hir::PathKind::Crate => {
            segments.push(make::path_segment_crate())
        }
        hir::PathKind::Abs => is_abs = true,
    }

    segments.extend(
        path.segments()
            .iter()
            .map(|segment| make::path_segment(make::name_ref(&segment.to_string()))),
    );
    make::path_from_segments(segments, is_abs)
}

/// Iterates all `ModuleDef`s and `Impl` blocks of the given file.
pub fn visit_file_defs(
    sema: &Semantics<RootDatabase>,
    file_id: FileId,
    cb: &mut dyn FnMut(Either<hir::ModuleDef, hir::Impl>),
) {
    let db = sema.db;
    let module = match sema.to_module_def(file_id) {
        Some(it) => it,
        None => return,
    };
    let mut defs: VecDeque<_> = module.declarations(db).into();
    while let Some(def) = defs.pop_front() {
        if let ModuleDef::Module(submodule) = def {
            if let hir::ModuleSource::Module(_) = submodule.definition_source(db).value {
                defs.extend(submodule.declarations(db));
                submodule.impl_defs(db).into_iter().for_each(|impl_| cb(Either::Right(impl_)));
            }
        }
        cb(Either::Left(def));
    }
    module.impl_defs(db).into_iter().for_each(|impl_| cb(Either::Right(impl_)));
}

/// Helps with finding well-know things inside the standard library. This is
/// somewhat similar to the known paths infra inside hir, but it different; We
/// want to make sure that IDE specific paths don't become interesting inside
/// the compiler itself as well.
pub struct FamousDefs<'a, 'b>(pub &'a Semantics<'b, RootDatabase>, pub Option<Crate>);

#[allow(non_snake_case)]
impl FamousDefs<'_, '_> {
    pub const FIXTURE: &'static str = include_str!("helpers/famous_defs_fixture.rs");

    pub fn std(&self) -> Option<Crate> {
        self.find_crate("std")
    }

    pub fn core(&self) -> Option<Crate> {
        self.find_crate("core")
    }

    pub fn core_cmp_Ord(&self) -> Option<Trait> {
        self.find_trait("core:cmp:Ord")
    }

    pub fn core_convert_From(&self) -> Option<Trait> {
        self.find_trait("core:convert:From")
    }

    pub fn core_convert_Into(&self) -> Option<Trait> {
        self.find_trait("core:convert:Into")
    }

    pub fn core_option_Option(&self) -> Option<Enum> {
        self.find_enum("core:option:Option")
    }

    pub fn core_default_Default(&self) -> Option<Trait> {
        self.find_trait("core:default:Default")
    }

    pub fn core_iter_Iterator(&self) -> Option<Trait> {
        self.find_trait("core:iter:traits:iterator:Iterator")
    }

    pub fn core_iter(&self) -> Option<Module> {
        self.find_module("core:iter")
    }

    pub fn core_ops_Deref(&self) -> Option<Trait> {
        self.find_trait("core:ops:Deref")
    }

    fn find_trait(&self, path: &str) -> Option<Trait> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Trait(it)) => Some(it),
            _ => None,
        }
    }

    fn find_enum(&self, path: &str) -> Option<Enum> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Enum(it))) => Some(it),
            _ => None,
        }
    }

    fn find_module(&self, path: &str) -> Option<Module> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(it)) => Some(it),
            _ => None,
        }
    }

    fn find_crate(&self, name: &str) -> Option<Crate> {
        let krate = self.1?;
        let db = self.0.db;
        let res =
            krate.dependencies(db).into_iter().find(|dep| dep.name.to_string() == name)?.krate;
        Some(res)
    }

    fn find_def(&self, path: &str) -> Option<ScopeDef> {
        let db = self.0.db;
        let mut path = path.split(':');
        let trait_ = path.next_back()?;
        let std_crate = path.next()?;
        let std_crate = self.find_crate(std_crate)?;
        let mut module = std_crate.root_module(db);
        for segment in path {
            module = module.children(db).find_map(|child| {
                let name = child.name(db)?;
                if name.to_string() == segment {
                    Some(child)
                } else {
                    None
                }
            })?;
        }
        let def =
            module.scope(db, None).into_iter().find(|(name, _def)| name.to_string() == trait_)?.1;
        Some(def)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl SnippetCap {
    pub const fn new(allow_snippets: bool) -> Option<SnippetCap> {
        if allow_snippets {
            Some(SnippetCap { _private: () })
        } else {
            None
        }
    }
}
