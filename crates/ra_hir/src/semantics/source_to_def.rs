//! Maps *syntax* of various definitions to their semantic ids.

use hir_def::{
    child_by_source::ChildBySource,
    dyn_map::DynMap,
    expr::PatId,
    keys::{self, Key},
    ConstId, DefWithBodyId, EnumId, EnumVariantId, FieldId, FunctionId, GenericDefId, ImplId,
    ModuleId, StaticId, StructId, TraitId, TypeAliasId, TypeParamId, UnionId, VariantId,
};
use hir_expand::{name::AsName, AstId, MacroDefKind};
use ra_db::FileId;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, NameOwner},
    match_ast, AstNode, SyntaxNode,
};
use rustc_hash::FxHashMap;

use crate::{db::HirDatabase, InFile, MacroDefId};

pub(super) type SourceToDefCache = FxHashMap<ChildContainer, DynMap>;

pub(super) struct SourceToDefCtx<'a, 'b> {
    pub(super) db: &'b dyn HirDatabase,
    pub(super) cache: &'a mut SourceToDefCache,
}

impl SourceToDefCtx<'_, '_> {
    pub(super) fn file_to_def(&mut self, file: FileId) -> Option<ModuleId> {
        let _p = profile("SourceBinder::to_module_def");
        let (krate, local_id) = self.db.relevant_crates(file).iter().find_map(|&crate_id| {
            let crate_def_map = self.db.crate_def_map(crate_id);
            let local_id = crate_def_map.modules_for_file(file).next()?;
            Some((crate_id, local_id))
        })?;
        Some(ModuleId { krate, local_id })
    }

    pub(super) fn module_to_def(&mut self, src: InFile<ast::Module>) -> Option<ModuleId> {
        let _p = profile("module_to_def");
        let parent_declaration = src
            .as_ref()
            .map(|it| it.syntax())
            .cloned()
            .ancestors_with_macros(self.db.upcast())
            .skip(1)
            .find_map(|it| {
                let m = ast::Module::cast(it.value.clone())?;
                Some(it.with_value(m))
            });

        let parent_module = match parent_declaration {
            Some(parent_declaration) => self.module_to_def(parent_declaration),
            None => {
                let file_id = src.file_id.original_file(self.db.upcast());
                self.file_to_def(file_id)
            }
        }?;

        let child_name = src.value.name()?.as_name();
        let def_map = self.db.crate_def_map(parent_module.krate);
        let child_id = *def_map[parent_module.local_id].children.get(&child_name)?;
        Some(ModuleId { krate: parent_module.krate, local_id: child_id })
    }

    pub(super) fn trait_to_def(&mut self, src: InFile<ast::TraitDef>) -> Option<TraitId> {
        self.to_def(src, keys::TRAIT)
    }
    pub(super) fn impl_to_def(&mut self, src: InFile<ast::ImplDef>) -> Option<ImplId> {
        self.to_def(src, keys::IMPL)
    }
    pub(super) fn fn_to_def(&mut self, src: InFile<ast::FnDef>) -> Option<FunctionId> {
        self.to_def(src, keys::FUNCTION)
    }
    pub(super) fn struct_to_def(&mut self, src: InFile<ast::StructDef>) -> Option<StructId> {
        self.to_def(src, keys::STRUCT)
    }
    pub(super) fn enum_to_def(&mut self, src: InFile<ast::EnumDef>) -> Option<EnumId> {
        self.to_def(src, keys::ENUM)
    }
    pub(super) fn union_to_def(&mut self, src: InFile<ast::UnionDef>) -> Option<UnionId> {
        self.to_def(src, keys::UNION)
    }
    pub(super) fn static_to_def(&mut self, src: InFile<ast::StaticDef>) -> Option<StaticId> {
        self.to_def(src, keys::STATIC)
    }
    pub(super) fn const_to_def(&mut self, src: InFile<ast::ConstDef>) -> Option<ConstId> {
        self.to_def(src, keys::CONST)
    }
    pub(super) fn type_alias_to_def(
        &mut self,
        src: InFile<ast::TypeAliasDef>,
    ) -> Option<TypeAliasId> {
        self.to_def(src, keys::TYPE_ALIAS)
    }
    pub(super) fn record_field_to_def(
        &mut self,
        src: InFile<ast::RecordFieldDef>,
    ) -> Option<FieldId> {
        self.to_def(src, keys::RECORD_FIELD)
    }
    pub(super) fn tuple_field_to_def(
        &mut self,
        src: InFile<ast::TupleFieldDef>,
    ) -> Option<FieldId> {
        self.to_def(src, keys::TUPLE_FIELD)
    }
    pub(super) fn enum_variant_to_def(
        &mut self,
        src: InFile<ast::EnumVariant>,
    ) -> Option<EnumVariantId> {
        self.to_def(src, keys::ENUM_VARIANT)
    }
    pub(super) fn bind_pat_to_def(
        &mut self,
        src: InFile<ast::BindPat>,
    ) -> Option<(DefWithBodyId, PatId)> {
        let container = self.find_pat_container(src.as_ref().map(|it| it.syntax()))?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let src = src.map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        Some((container, pat_id))
    }

    fn to_def<Ast: AstNode + 'static, ID: Copy + 'static>(
        &mut self,
        src: InFile<Ast>,
        key: Key<Ast, ID>,
    ) -> Option<ID> {
        let container = self.find_container(src.as_ref().map(|it| it.syntax()))?;
        let db = self.db;
        let dyn_map =
            &*self.cache.entry(container).or_insert_with(|| container.child_by_source(db));
        dyn_map[key].get(&src).copied()
    }

    pub(super) fn type_param_to_def(&mut self, src: InFile<ast::TypeParam>) -> Option<TypeParamId> {
        let container: ChildContainer =
            self.find_type_param_container(src.as_ref().map(|it| it.syntax()))?.into();
        let db = self.db;
        let dyn_map =
            &*self.cache.entry(container).or_insert_with(|| container.child_by_source(db));
        dyn_map[keys::TYPE_PARAM].get(&src).copied()
    }

    // FIXME: use DynMap as well?
    pub(super) fn macro_call_to_def(&mut self, src: InFile<ast::MacroCall>) -> Option<MacroDefId> {
        let kind = MacroDefKind::Declarative;
        let file_id = src.file_id.original_file(self.db.upcast());
        let krate = self.file_to_def(file_id)?.krate;
        let file_ast_id = self.db.ast_id_map(src.file_id).ast_id(&src.value);
        let ast_id = Some(AstId::new(src.file_id, file_ast_id));
        Some(MacroDefId { krate: Some(krate), ast_id, kind, local_inner: false })
    }

    pub(super) fn find_container(&mut self, src: InFile<&SyntaxNode>) -> Option<ChildContainer> {
        for container in src.cloned().ancestors_with_macros(self.db.upcast()).skip(1) {
            let res: ChildContainer = match_ast! {
                match (container.value) {
                    ast::Module(it) => {
                        let def = self.module_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::TraitDef(it) => {
                        let def = self.trait_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::ImplDef(it) => {
                        let def = self.impl_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::FnDef(it) => {
                        let def = self.fn_to_def(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::StructDef(it) => {
                        let def = self.struct_to_def(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    ast::EnumDef(it) => {
                        let def = self.enum_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::UnionDef(it) => {
                        let def = self.union_to_def(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    ast::StaticDef(it) => {
                        let def = self.static_to_def(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::ConstDef(it) => {
                        let def = self.const_to_def(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    _ => continue,
                }
            };
            return Some(res);
        }

        let def = self.file_to_def(src.file_id.original_file(self.db.upcast()))?;
        Some(def.into())
    }

    fn find_type_param_container(&mut self, src: InFile<&SyntaxNode>) -> Option<GenericDefId> {
        for container in src.cloned().ancestors_with_macros(self.db.upcast()).skip(1) {
            let res: GenericDefId = match_ast! {
                match (container.value) {
                    ast::FnDef(it) => self.fn_to_def(container.with_value(it))?.into(),
                    ast::StructDef(it) => self.struct_to_def(container.with_value(it))?.into(),
                    ast::EnumDef(it) => self.enum_to_def(container.with_value(it))?.into(),
                    ast::TraitDef(it) => self.trait_to_def(container.with_value(it))?.into(),
                    ast::TypeAliasDef(it) => self.type_alias_to_def(container.with_value(it))?.into(),
                    ast::ImplDef(it) => self.impl_to_def(container.with_value(it))?.into(),
                    _ => continue,
                }
            };
            return Some(res);
        }
        None
    }

    fn find_pat_container(&mut self, src: InFile<&SyntaxNode>) -> Option<DefWithBodyId> {
        for container in src.cloned().ancestors_with_macros(self.db.upcast()).skip(1) {
            let res: DefWithBodyId = match_ast! {
                match (container.value) {
                    ast::ConstDef(it) => self.const_to_def(container.with_value(it))?.into(),
                    ast::StaticDef(it) => self.static_to_def(container.with_value(it))?.into(),
                    ast::FnDef(it) => self.fn_to_def(container.with_value(it))?.into(),
                    _ => continue,
                }
            };
            return Some(res);
        }
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ChildContainer {
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

impl ChildContainer {
    fn child_by_source(self, db: &dyn HirDatabase) -> DynMap {
        let db = db.upcast();
        match self {
            ChildContainer::DefWithBodyId(it) => it.child_by_source(db),
            ChildContainer::ModuleId(it) => it.child_by_source(db),
            ChildContainer::TraitId(it) => it.child_by_source(db),
            ChildContainer::ImplId(it) => it.child_by_source(db),
            ChildContainer::EnumId(it) => it.child_by_source(db),
            ChildContainer::VariantId(it) => it.child_by_source(db),
            ChildContainer::GenericDefId(it) => it.child_by_source(db),
        }
    }
}
