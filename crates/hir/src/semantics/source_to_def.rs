//! Maps *syntax* of various definitions to their semantic ids.

use base_db::FileId;
use hir_def::{
    child_by_source::ChildBySource,
    dyn_map::DynMap,
    expr::{LabelId, PatId},
    keys::{self, Key},
    ConstId, ConstParamId, DefWithBodyId, EnumId, EnumVariantId, FieldId, FunctionId, GenericDefId,
    ImplId, LifetimeParamId, ModuleId, StaticId, StructId, TraitId, TypeAliasId, TypeParamId,
    UnionId, VariantId,
};
use hir_expand::{name::AsName, AstId, MacroDefKind};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stdx::impl_from;
use syntax::{
    ast::{self, NameOwner},
    match_ast, AstNode, SyntaxNode,
};

use crate::{db::HirDatabase, InFile, MacroDefId};

pub(super) type SourceToDefCache = FxHashMap<ChildContainer, DynMap>;

pub(super) struct SourceToDefCtx<'a, 'b> {
    pub(super) db: &'b dyn HirDatabase,
    pub(super) cache: &'a mut SourceToDefCache,
}

impl SourceToDefCtx<'_, '_> {
    pub(super) fn file_to_def(&mut self, file: FileId) -> SmallVec<[ModuleId; 1]> {
        let _p = profile::span("SourceBinder::to_module_def");
        let mut mods = SmallVec::new();
        for &crate_id in self.db.relevant_crates(file).iter() {
            // FIXME: inner items
            let crate_def_map = self.db.crate_def_map(crate_id);
            mods.extend(
                crate_def_map
                    .modules_for_file(file)
                    .map(|local_id| crate_def_map.module_id(local_id)),
            )
        }
        mods
    }

    pub(super) fn module_to_def(&mut self, src: InFile<ast::Module>) -> Option<ModuleId> {
        let _p = profile::span("module_to_def");
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
                self.file_to_def(file_id).get(0).copied()
            }
        }?;

        let child_name = src.value.name()?.as_name();
        let def_map = parent_module.def_map(self.db.upcast());
        let child_id = *def_map[parent_module.local_id].children.get(&child_name)?;
        Some(def_map.module_id(child_id))
    }

    pub(super) fn trait_to_def(&mut self, src: InFile<ast::Trait>) -> Option<TraitId> {
        self.to_def(src, keys::TRAIT)
    }
    pub(super) fn impl_to_def(&mut self, src: InFile<ast::Impl>) -> Option<ImplId> {
        self.to_def(src, keys::IMPL)
    }
    pub(super) fn fn_to_def(&mut self, src: InFile<ast::Fn>) -> Option<FunctionId> {
        self.to_def(src, keys::FUNCTION)
    }
    pub(super) fn struct_to_def(&mut self, src: InFile<ast::Struct>) -> Option<StructId> {
        self.to_def(src, keys::STRUCT)
    }
    pub(super) fn enum_to_def(&mut self, src: InFile<ast::Enum>) -> Option<EnumId> {
        self.to_def(src, keys::ENUM)
    }
    pub(super) fn union_to_def(&mut self, src: InFile<ast::Union>) -> Option<UnionId> {
        self.to_def(src, keys::UNION)
    }
    pub(super) fn static_to_def(&mut self, src: InFile<ast::Static>) -> Option<StaticId> {
        self.to_def(src, keys::STATIC)
    }
    pub(super) fn const_to_def(&mut self, src: InFile<ast::Const>) -> Option<ConstId> {
        self.to_def(src, keys::CONST)
    }
    pub(super) fn type_alias_to_def(&mut self, src: InFile<ast::TypeAlias>) -> Option<TypeAliasId> {
        self.to_def(src, keys::TYPE_ALIAS)
    }
    pub(super) fn record_field_to_def(&mut self, src: InFile<ast::RecordField>) -> Option<FieldId> {
        self.to_def(src, keys::RECORD_FIELD)
    }
    pub(super) fn tuple_field_to_def(&mut self, src: InFile<ast::TupleField>) -> Option<FieldId> {
        self.to_def(src, keys::TUPLE_FIELD)
    }
    pub(super) fn enum_variant_to_def(
        &mut self,
        src: InFile<ast::Variant>,
    ) -> Option<EnumVariantId> {
        self.to_def(src, keys::VARIANT)
    }
    pub(super) fn bind_pat_to_def(
        &mut self,
        src: InFile<ast::IdentPat>,
    ) -> Option<(DefWithBodyId, PatId)> {
        let container = self.find_pat_or_label_container(src.as_ref().map(|it| it.syntax()))?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let src = src.map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        Some((container, pat_id))
    }
    pub(super) fn self_param_to_def(
        &mut self,
        src: InFile<ast::SelfParam>,
    ) -> Option<(DefWithBodyId, PatId)> {
        let container = self.find_pat_or_label_container(src.as_ref().map(|it| it.syntax()))?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let pat_id = source_map.node_self_param(src.as_ref())?;
        Some((container, pat_id))
    }
    pub(super) fn label_to_def(
        &mut self,
        src: InFile<ast::Label>,
    ) -> Option<(DefWithBodyId, LabelId)> {
        let container = self.find_pat_or_label_container(src.as_ref().map(|it| it.syntax()))?;
        let (_body, source_map) = self.db.body_with_source_map(container);
        let label_id = source_map.node_label(src.as_ref())?;
        Some((container, label_id))
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
            self.find_generic_param_container(src.as_ref().map(|it| it.syntax()))?.into();
        let db = self.db;
        let dyn_map =
            &*self.cache.entry(container).or_insert_with(|| container.child_by_source(db));
        dyn_map[keys::TYPE_PARAM].get(&src).copied()
    }

    pub(super) fn lifetime_param_to_def(
        &mut self,
        src: InFile<ast::LifetimeParam>,
    ) -> Option<LifetimeParamId> {
        let container: ChildContainer =
            self.find_generic_param_container(src.as_ref().map(|it| it.syntax()))?.into();
        let db = self.db;
        let dyn_map =
            &*self.cache.entry(container).or_insert_with(|| container.child_by_source(db));
        dyn_map[keys::LIFETIME_PARAM].get(&src).copied()
    }

    pub(super) fn const_param_to_def(
        &mut self,
        src: InFile<ast::ConstParam>,
    ) -> Option<ConstParamId> {
        let container: ChildContainer =
            self.find_generic_param_container(src.as_ref().map(|it| it.syntax()))?.into();
        let db = self.db;
        let dyn_map =
            &*self.cache.entry(container).or_insert_with(|| container.child_by_source(db));
        dyn_map[keys::CONST_PARAM].get(&src).copied()
    }

    // FIXME: use DynMap as well?
    pub(super) fn macro_rules_to_def(
        &mut self,
        src: InFile<ast::MacroRules>,
    ) -> Option<MacroDefId> {
        let kind = MacroDefKind::Declarative;
        let file_id = src.file_id.original_file(self.db.upcast());
        let krate = self.file_to_def(file_id).get(0).copied()?.krate();
        let file_ast_id = self.db.ast_id_map(src.file_id).ast_id(&src.value);
        let ast_id = Some(AstId::new(src.file_id, file_ast_id.upcast()));
        Some(MacroDefId { krate, ast_id, kind, local_inner: false })
    }

    pub(super) fn find_container(&mut self, src: InFile<&SyntaxNode>) -> Option<ChildContainer> {
        for container in src.cloned().ancestors_with_macros(self.db.upcast()).skip(1) {
            let res: ChildContainer = match_ast! {
                match (container.value) {
                    ast::Module(it) => {
                        let def = self.module_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::Trait(it) => {
                        let def = self.trait_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::Impl(it) => {
                        let def = self.impl_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::Fn(it) => {
                        let def = self.fn_to_def(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::Struct(it) => {
                        let def = self.struct_to_def(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    ast::Enum(it) => {
                        let def = self.enum_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::Union(it) => {
                        let def = self.union_to_def(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    ast::Static(it) => {
                        let def = self.static_to_def(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::Const(it) => {
                        let def = self.const_to_def(container.with_value(it))?;
                        DefWithBodyId::from(def).into()
                    },
                    ast::TypeAlias(it) => {
                        let def = self.type_alias_to_def(container.with_value(it))?;
                        def.into()
                    },
                    ast::Variant(it) => {
                        let def = self.enum_variant_to_def(container.with_value(it))?;
                        VariantId::from(def).into()
                    },
                    _ => continue,
                }
            };
            return Some(res);
        }

        let def = self.file_to_def(src.file_id.original_file(self.db.upcast())).get(0).copied()?;
        Some(def.into())
    }

    fn find_generic_param_container(&mut self, src: InFile<&SyntaxNode>) -> Option<GenericDefId> {
        for container in src.cloned().ancestors_with_macros(self.db.upcast()).skip(1) {
            let res: GenericDefId = match_ast! {
                match (container.value) {
                    ast::Fn(it) => self.fn_to_def(container.with_value(it))?.into(),
                    ast::Struct(it) => self.struct_to_def(container.with_value(it))?.into(),
                    ast::Enum(it) => self.enum_to_def(container.with_value(it))?.into(),
                    ast::Trait(it) => self.trait_to_def(container.with_value(it))?.into(),
                    ast::TypeAlias(it) => self.type_alias_to_def(container.with_value(it))?.into(),
                    ast::Impl(it) => self.impl_to_def(container.with_value(it))?.into(),
                    _ => continue,
                }
            };
            return Some(res);
        }
        None
    }

    fn find_pat_or_label_container(&mut self, src: InFile<&SyntaxNode>) -> Option<DefWithBodyId> {
        for container in src.cloned().ancestors_with_macros(self.db.upcast()).skip(1) {
            let res: DefWithBodyId = match_ast! {
                match (container.value) {
                    ast::Const(it) => self.const_to_def(container.with_value(it))?.into(),
                    ast::Static(it) => self.static_to_def(container.with_value(it))?.into(),
                    ast::Fn(it) => self.fn_to_def(container.with_value(it))?.into(),
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
    TypeAliasId(TypeAliasId),
    /// XXX: this might be the same def as, for example an `EnumId`. However,
    /// here the children are generic parameters, and not, eg enum variants.
    GenericDefId(GenericDefId),
}
impl_from! {
    DefWithBodyId,
    ModuleId,
    TraitId,
    ImplId,
    EnumId,
    VariantId,
    TypeAliasId,
    GenericDefId
    for ChildContainer
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
            ChildContainer::TypeAliasId(_) => DynMap::default(),
            ChildContainer::GenericDefId(it) => it.child_by_source(db),
        }
    }
}
