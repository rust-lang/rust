//! `SourceBinder` should be the main entry point for getting info about source code.
//! It's main task is to map source syntax trees to hir-level IDs.
//!
//! It is intended to subsume `FromSource` and `SourceAnalyzer`.

use hir_def::{
    child_by_source::ChildBySource,
    dyn_map::DynMap,
    keys::{self, Key},
    resolver::{HasResolver, Resolver},
    ConstId, DefWithBodyId, EnumId, EnumVariantId, FunctionId, GenericDefId, ImplId, ModuleId,
    StaticId, StructFieldId, StructId, TraitId, TypeAliasId, UnionId, VariantId,
};
use hir_expand::{name::AsName, AstId, InFile, MacroDefId, MacroDefKind};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, NameOwner},
    match_ast, AstNode, SyntaxNode, TextUnit,
};
use rustc_hash::FxHashMap;

use crate::{db::HirDatabase, Local, Module, ModuleSource, SourceAnalyzer, TypeParam};

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
            ChildContainer::GenericDefId(it) => it.resolver(self.db),
        };
        SourceAnalyzer::new_for_resolver(resolver, src)
    }

    pub fn to_def<T: ToDef>(&mut self, src: InFile<T>) -> Option<T::Def> {
        T::to_def(self, src)
    }

    fn to_id<T: ToId>(&mut self, src: InFile<T>) -> Option<T::ID> {
        T::to_id(self, src)
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

    fn child_by_source(&mut self, container: ChildContainer) -> &DynMap {
        let db = self.db;
        self.child_by_source_cache.entry(container).or_insert_with(|| match container {
            ChildContainer::DefWithBodyId(it) => it.child_by_source(db),
            ChildContainer::ModuleId(it) => it.child_by_source(db),
            ChildContainer::TraitId(it) => it.child_by_source(db),
            ChildContainer::ImplId(it) => it.child_by_source(db),
            ChildContainer::EnumId(it) => it.child_by_source(db),
            ChildContainer::VariantId(it) => it.child_by_source(db),
            ChildContainer::GenericDefId(it) => it.child_by_source(db),
        })
    }
}

pub trait ToId: Sized {
    type ID: Sized + Copy + 'static;
    fn to_id<DB: HirDatabase>(sb: &mut SourceBinder<'_, DB>, src: InFile<Self>)
        -> Option<Self::ID>;
}

pub trait ToDef: Sized + AstNode + 'static {
    type Def;
    fn to_def<DB: HirDatabase>(
        sb: &mut SourceBinder<'_, DB>,
        src: InFile<Self>,
    ) -> Option<Self::Def>;
}

macro_rules! to_def_impls {
    ($(($def:path, $ast:path)),* ,) => {$(
        impl ToDef for $ast {
            type Def = $def;
            fn to_def<DB: HirDatabase>(sb: &mut SourceBinder<'_, DB>, src: InFile<Self>)
                -> Option<Self::Def>
            { sb.to_id(src).map(Into::into) }
        }
    )*}
}

to_def_impls![
    (crate::Struct, ast::StructDef),
    (crate::Enum, ast::EnumDef),
    (crate::Union, ast::UnionDef),
    (crate::Trait, ast::TraitDef),
    (crate::ImplBlock, ast::ImplBlock),
    (crate::TypeAlias, ast::TypeAliasDef),
    (crate::Const, ast::ConstDef),
    (crate::Static, ast::StaticDef),
    (crate::Function, ast::FnDef),
    (crate::StructField, ast::RecordFieldDef),
    (crate::EnumVariant, ast::EnumVariant),
    (crate::MacroDef, ast::MacroCall), // this one is dubious, not all calls are macros
];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ChildContainer {
    DefWithBodyId(DefWithBodyId),
    ModuleId(ModuleId),
    TraitId(TraitId),
    ImplId(ImplId),
    EnumId(EnumId),
    VariantId(VariantId),
    /// XXX: this might be the same def as, for example an `EnumId`. However,
    /// here the children generic parameters, and not, eg enum variants.
    GenericDefId(GenericDefId),
}
impl_froms! {
    ChildContainer:
    DefWithBodyId,
    ModuleId,
    TraitId,
    ImplId,
    EnumId,
    VariantId,
    GenericDefId
}

pub trait ToIdByKey: Sized + AstNode + 'static {
    type ID: Sized + Copy + 'static;
    const KEY: Key<Self, Self::ID>;
}

impl<T: ToIdByKey> ToId for T {
    type ID = <T as ToIdByKey>::ID;
    fn to_id<DB: HirDatabase>(
        sb: &mut SourceBinder<'_, DB>,
        src: InFile<Self>,
    ) -> Option<Self::ID> {
        let container = sb.find_container(src.as_ref().map(|it| it.syntax()))?;
        let db = sb.db;
        let dyn_map =
            &*sb.child_by_source_cache.entry(container).or_insert_with(|| match container {
                ChildContainer::DefWithBodyId(it) => it.child_by_source(db),
                ChildContainer::ModuleId(it) => it.child_by_source(db),
                ChildContainer::TraitId(it) => it.child_by_source(db),
                ChildContainer::ImplId(it) => it.child_by_source(db),
                ChildContainer::EnumId(it) => it.child_by_source(db),
                ChildContainer::VariantId(it) => it.child_by_source(db),
                ChildContainer::GenericDefId(it) => it.child_by_source(db),
            });
        dyn_map[T::KEY].get(&src).copied()
    }
}

macro_rules! to_id_key_impls {
    ($(($id:ident, $ast:path, $key:path)),* ,) => {$(
        impl ToIdByKey for $ast {
            type ID = $id;
            const KEY: Key<Self, Self::ID> = $key;
        }
    )*}
}

to_id_key_impls![
    (StructId, ast::StructDef, keys::STRUCT),
    (UnionId, ast::UnionDef, keys::UNION),
    (EnumId, ast::EnumDef, keys::ENUM),
    (TraitId, ast::TraitDef, keys::TRAIT),
    (FunctionId, ast::FnDef, keys::FUNCTION),
    (StaticId, ast::StaticDef, keys::STATIC),
    (ConstId, ast::ConstDef, keys::CONST),
    (TypeAliasId, ast::TypeAliasDef, keys::TYPE_ALIAS),
    (ImplId, ast::ImplBlock, keys::IMPL),
    (StructFieldId, ast::RecordFieldDef, keys::RECORD_FIELD),
    (EnumVariantId, ast::EnumVariant, keys::ENUM_VARIANT),
];

// FIXME: use DynMap as well?
impl ToId for ast::MacroCall {
    type ID = MacroDefId;
    fn to_id<DB: HirDatabase>(
        sb: &mut SourceBinder<'_, DB>,
        src: InFile<Self>,
    ) -> Option<Self::ID> {
        let kind = MacroDefKind::Declarative;

        let module_src = ModuleSource::from_child_node(sb.db, src.as_ref().map(|it| it.syntax()));
        let module = crate::Module::from_definition(sb.db, InFile::new(src.file_id, module_src))?;
        let krate = Some(module.krate().id);

        let ast_id =
            Some(AstId::new(src.file_id, sb.db.ast_id_map(src.file_id).ast_id(&src.value)));

        Some(MacroDefId { krate, ast_id, kind })
    }
}

impl ToDef for ast::BindPat {
    type Def = Local;

    fn to_def<DB: HirDatabase>(sb: &mut SourceBinder<'_, DB>, src: InFile<Self>) -> Option<Local> {
        let file_id = src.file_id;
        let parent: DefWithBodyId = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::ConstDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::StaticDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::FnDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    _ => return None,
                }
            };
            Some(res)
        })?;
        let (_body, source_map) = sb.db.body_with_source_map(parent);
        let src = src.map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        Some(Local { parent: parent.into(), pat_id })
    }
}

impl ToDef for ast::TypeParam {
    type Def = TypeParam;

    fn to_def<DB: HirDatabase>(
        sb: &mut SourceBinder<'_, DB>,
        src: InFile<ast::TypeParam>,
    ) -> Option<TypeParam> {
        let mut sb = SourceBinder::new(sb.db);
        let file_id = src.file_id;
        let parent: GenericDefId = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::FnDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::StructDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::EnumDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::TraitDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::TypeAliasDef(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    ast::ImplBlock(value) => { sb.to_id(InFile { value, file_id})?.into() },
                    _ => return None,
                }
            };
            Some(res)
        })?;
        let &id = sb.child_by_source(parent.into())[keys::TYPE_PARAM].get(&src)?;
        Some(TypeParam { id })
    }
}

impl ToDef for ast::Module {
    type Def = Module;

    fn to_def<DB: HirDatabase>(
        sb: &mut SourceBinder<'_, DB>,
        src: InFile<ast::Module>,
    ) -> Option<Module> {
        {
            let _p = profile("ast::Module::to_def");
            let parent_declaration =
                src.value.syntax().ancestors().skip(1).find_map(ast::Module::cast);

            let parent_module = match parent_declaration {
                Some(parent_declaration) => {
                    let src_parent = InFile { file_id: src.file_id, value: parent_declaration };
                    sb.to_def(src_parent)
                }
                None => {
                    let source_file = sb.db.parse(src.file_id.original_file(sb.db)).tree();
                    let src_parent = InFile {
                        file_id: src.file_id,
                        value: ModuleSource::SourceFile(source_file),
                    };
                    Module::from_definition(sb.db, src_parent)
                }
            }?;

            let child_name = src.value.name()?.as_name();
            let def_map = sb.db.crate_def_map(parent_module.id.krate);
            let child_id = def_map[parent_module.id.local_id].children.get(&child_name)?;
            Some(parent_module.with_module_id(*child_id))
        }
    }
}
