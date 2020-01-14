//! `SourceBinder` should be the main entry point for getting info about source code.
//! It's main task is to map source syntax trees to hir-level IDs.
//!
//! It is intended to subsume `FromSource` and `SourceAnalyzer`.

use hir_def::{
    child_by_source::ChildBySource,
    dyn_map::DynMap,
    keys::{self, Key},
    resolver::{HasResolver, Resolver},
    ConstId, DefWithBodyId, EnumId, FunctionId, ImplId, ModuleId, StaticId, StructId, TraitId,
    UnionId, VariantId,
};
use hir_expand::InFile;
use ra_prof::profile;
use ra_syntax::{ast, match_ast, AstNode, SyntaxNode, TextUnit};
use rustc_hash::FxHashMap;

use crate::{db::HirDatabase, ModuleSource, SourceAnalyzer};

pub struct SourceBinder<'a, DB> {
    pub db: &'a DB,
    child_by_source_cache: FxHashMap<ChildContainer, DynMap>,
}

impl<DB: HirDatabase> SourceBinder<'_, DB> {
    pub fn new(db: &DB) -> SourceBinder<DB> {
        SourceBinder { db, child_by_source_cache: FxHashMap::default() }
    }

    pub fn analyze(
        &mut self,
        src: InFile<&SyntaxNode>,
        offset: Option<TextUnit>,
    ) -> SourceAnalyzer {
        let _p = profile("SourceBinder::analyzer");
        let container = match self.find_container(src) {
            Some(it) => it,
            None => return SourceAnalyzer::new_for_resolver(Resolver::default(), src),
        };

        let resolver = match container {
            ChildContainer::DefWithBodyId(def) => {
                return SourceAnalyzer::new_for_body(self.db, def, src, offset)
            }
            ChildContainer::TraitId(it) => it.resolver(self.db),
            ChildContainer::ImplId(it) => it.resolver(self.db),
            ChildContainer::ModuleId(it) => it.resolver(self.db),
            ChildContainer::EnumId(it) => it.resolver(self.db),
            ChildContainer::VariantId(it) => it.resolver(self.db),
        };
        SourceAnalyzer::new_for_resolver(resolver, src)
    }

    pub fn to_def<D, T>(&mut self, src: InFile<T>) -> Option<D>
    where
        D: From<T::ID>,
        T: ToId,
    {
        let id: T::ID = self.to_id(src)?;
        Some(id.into())
    }

    fn to_id<T: ToId>(&mut self, src: InFile<T>) -> Option<T::ID> {
        let container = self.find_container(src.as_ref().map(|it| it.syntax()))?;
        let db = self.db;
        let dyn_map =
            &*self.child_by_source_cache.entry(container).or_insert_with(|| match container {
                ChildContainer::DefWithBodyId(it) => it.child_by_source(db),
                ChildContainer::ModuleId(it) => it.child_by_source(db),
                ChildContainer::TraitId(it) => it.child_by_source(db),
                ChildContainer::ImplId(it) => it.child_by_source(db),
                ChildContainer::EnumId(it) => it.child_by_source(db),
                ChildContainer::VariantId(it) => it.child_by_source(db),
            });
        dyn_map[T::KEY].get(&src).copied()
    }

    fn find_container(&mut self, src: InFile<&SyntaxNode>) -> Option<ChildContainer> {
        for container in src.cloned().ancestors_with_macros(self.db).skip(1) {
            let res: ChildContainer = match_ast! {
                match (container.value) {
                    ast::TraitDef(it) => {
                        let def: TraitId = self.to_id(container.with_value(it))?;
                        def.into()
                    },
                    ast::ImplBlock(it) => {
                        let def: ImplId = self.to_id(container.with_value(it))?;
                        def.into()
                    },
                    ast::FnDef(it) => {
                        let def: FunctionId = self.to_id(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::StaticDef(it) => {
                        let def: StaticId = self.to_id(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::ConstDef(it) => {
                        let def: ConstId = self.to_id(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::EnumDef(it) => {
                        let def: EnumId = self.to_id(container.with_value(it))?;
                        def.into()
                    },
                    ast::StructDef(it) => {
                        let def: StructId = self.to_id(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    ast::UnionDef(it) => {
                        let def: UnionId = self.to_id(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    // FIXME: handle out-of-line modules here
                    _ => { continue },
                }
            };
            return Some(res);
        }

        let module_source = ModuleSource::from_child_node(self.db, src);
        let c = crate::Module::from_definition(self.db, src.with_value(module_source))?;
        Some(c.id.into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ChildContainer {
    DefWithBodyId(DefWithBodyId),
    ModuleId(ModuleId),
    TraitId(TraitId),
    ImplId(ImplId),
    EnumId(EnumId),
    VariantId(VariantId),
}
impl_froms! {
    ChildContainer:
    DefWithBodyId,
    ModuleId,
    TraitId,
    ImplId,
    EnumId,
    VariantId,
}

pub trait ToId: Sized + AstNode + 'static {
    type ID: Sized + Copy + 'static;
    const KEY: Key<Self, Self::ID>;
}

macro_rules! to_id_impls {
    ($(($id:ident, $ast:path, $key:path)),* ,) => {$(
        impl ToId for $ast {
            type ID = $id;
            const KEY: Key<Self, Self::ID> = $key;
        }
    )*}
}

to_id_impls![
    (StructId, ast::StructDef, keys::STRUCT),
    (UnionId, ast::UnionDef, keys::UNION),
    (EnumId, ast::EnumDef, keys::ENUM),
    (TraitId, ast::TraitDef, keys::TRAIT),
    (FunctionId, ast::FnDef, keys::FUNCTION),
    (StaticId, ast::StaticDef, keys::STATIC),
    (ConstId, ast::ConstDef, keys::CONST),
    // (TypeAlias, TypeAliasId, ast::TypeAliasDef, keys::TYPE_ALIAS),
    (ImplId, ast::ImplBlock, keys::IMPL),
];
