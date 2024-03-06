//! Defines database & queries for name resolution.
use base_db::{salsa, CrateId, FileId, SourceDatabase, Upcast};
use either::Either;
use hir_expand::{db::ExpandDatabase, HirFileId, MacroDefId};
use intern::Interned;
use la_arena::ArenaMap;
use span::MacroCallId;
use syntax::{ast, AstPtr};
use triomphe::Arc;

use crate::{
    attr::{Attrs, AttrsWithOwner},
    body::{scope::ExprScopes, Body, BodySourceMap},
    data::{
        adt::{EnumData, EnumVariantData, StructData},
        ConstData, ExternCrateDeclData, FunctionData, ImplData, Macro2Data, MacroRulesData,
        ProcMacroData, StaticData, TraitAliasData, TraitData, TypeAliasData,
    },
    generics::GenericParams,
    import_map::ImportMap,
    item_tree::{AttrOwner, ItemTree},
    lang_item::{self, LangItem, LangItemTarget, LangItems},
    nameres::{diagnostics::DefDiagnostics, DefMap},
    visibility::{self, Visibility},
    AttrDefId, BlockId, BlockLoc, ConstBlockId, ConstBlockLoc, ConstId, ConstLoc, DefWithBodyId,
    EnumId, EnumLoc, EnumVariantId, EnumVariantLoc, ExternBlockId, ExternBlockLoc, ExternCrateId,
    ExternCrateLoc, FunctionId, FunctionLoc, GenericDefId, ImplId, ImplLoc, InTypeConstId,
    InTypeConstLoc, LocalFieldId, Macro2Id, Macro2Loc, MacroId, MacroRulesId, MacroRulesLoc,
    MacroRulesLocFlags, ProcMacroId, ProcMacroLoc, StaticId, StaticLoc, StructId, StructLoc,
    TraitAliasId, TraitAliasLoc, TraitId, TraitLoc, TypeAliasId, TypeAliasLoc, UnionId, UnionLoc,
    UseId, UseLoc, VariantId,
};

#[salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    // region: items
    #[salsa::interned]
    fn intern_use(&self, loc: UseLoc) -> UseId;
    #[salsa::interned]
    fn intern_extern_crate(&self, loc: ExternCrateLoc) -> ExternCrateId;
    #[salsa::interned]
    fn intern_function(&self, loc: FunctionLoc) -> FunctionId;
    #[salsa::interned]
    fn intern_struct(&self, loc: StructLoc) -> StructId;
    #[salsa::interned]
    fn intern_union(&self, loc: UnionLoc) -> UnionId;
    #[salsa::interned]
    fn intern_enum(&self, loc: EnumLoc) -> EnumId;
    #[salsa::interned]
    fn intern_enum_variant(&self, loc: EnumVariantLoc) -> EnumVariantId;
    #[salsa::interned]
    fn intern_const(&self, loc: ConstLoc) -> ConstId;
    #[salsa::interned]
    fn intern_static(&self, loc: StaticLoc) -> StaticId;
    #[salsa::interned]
    fn intern_trait(&self, loc: TraitLoc) -> TraitId;
    #[salsa::interned]
    fn intern_trait_alias(&self, loc: TraitAliasLoc) -> TraitAliasId;
    #[salsa::interned]
    fn intern_type_alias(&self, loc: TypeAliasLoc) -> TypeAliasId;
    #[salsa::interned]
    fn intern_impl(&self, loc: ImplLoc) -> ImplId;
    #[salsa::interned]
    fn intern_extern_block(&self, loc: ExternBlockLoc) -> ExternBlockId;
    #[salsa::interned]
    fn intern_macro2(&self, loc: Macro2Loc) -> Macro2Id;
    #[salsa::interned]
    fn intern_proc_macro(&self, loc: ProcMacroLoc) -> ProcMacroId;
    #[salsa::interned]
    fn intern_macro_rules(&self, loc: MacroRulesLoc) -> MacroRulesId;
    // endregion: items

    #[salsa::interned]
    fn intern_block(&self, loc: BlockLoc) -> BlockId;
    #[salsa::interned]
    fn intern_anonymous_const(&self, id: ConstBlockLoc) -> ConstBlockId;
    #[salsa::interned]
    fn intern_in_type_const(&self, id: InTypeConstLoc) -> InTypeConstId;
}

#[salsa::query_group(DefDatabaseStorage)]
pub trait DefDatabase: InternDatabase + ExpandDatabase + Upcast<dyn ExpandDatabase> {
    #[salsa::input]
    fn expand_proc_attr_macros(&self) -> bool;

    #[salsa::invoke(ItemTree::file_item_tree_query)]
    fn file_item_tree(&self, file_id: HirFileId) -> Arc<ItemTree>;

    #[salsa::invoke(ItemTree::block_item_tree_query)]
    fn block_item_tree_query(&self, block_id: BlockId) -> Arc<ItemTree>;

    #[salsa::invoke(crate_def_map_wait)]
    #[salsa::transparent]
    fn crate_def_map(&self, krate: CrateId) -> Arc<DefMap>;

    #[salsa::invoke(DefMap::crate_def_map_query)]
    fn crate_def_map_query(&self, krate: CrateId) -> Arc<DefMap>;

    /// Computes the block-level `DefMap`.
    #[salsa::invoke(DefMap::block_def_map_query)]
    fn block_def_map(&self, block: BlockId) -> Arc<DefMap>;

    fn macro_def(&self, m: MacroId) -> MacroDefId;

    // region:data

    #[salsa::transparent]
    #[salsa::invoke(StructData::struct_data_query)]
    fn struct_data(&self, id: StructId) -> Arc<StructData>;

    #[salsa::invoke(StructData::struct_data_with_diagnostics_query)]
    fn struct_data_with_diagnostics(&self, id: StructId) -> (Arc<StructData>, DefDiagnostics);

    #[salsa::transparent]
    #[salsa::invoke(StructData::union_data_query)]
    fn union_data(&self, id: UnionId) -> Arc<StructData>;

    #[salsa::invoke(StructData::union_data_with_diagnostics_query)]
    fn union_data_with_diagnostics(&self, id: UnionId) -> (Arc<StructData>, DefDiagnostics);

    #[salsa::invoke(EnumData::enum_data_query)]
    fn enum_data(&self, e: EnumId) -> Arc<EnumData>;

    #[salsa::transparent]
    #[salsa::invoke(EnumVariantData::enum_variant_data_query)]
    fn enum_variant_data(&self, id: EnumVariantId) -> Arc<EnumVariantData>;

    #[salsa::invoke(EnumVariantData::enum_variant_data_with_diagnostics_query)]
    fn enum_variant_data_with_diagnostics(
        &self,
        id: EnumVariantId,
    ) -> (Arc<EnumVariantData>, DefDiagnostics);

    #[salsa::transparent]
    #[salsa::invoke(ImplData::impl_data_query)]
    fn impl_data(&self, e: ImplId) -> Arc<ImplData>;

    #[salsa::invoke(ImplData::impl_data_with_diagnostics_query)]
    fn impl_data_with_diagnostics(&self, e: ImplId) -> (Arc<ImplData>, DefDiagnostics);

    #[salsa::transparent]
    #[salsa::invoke(TraitData::trait_data_query)]
    fn trait_data(&self, e: TraitId) -> Arc<TraitData>;

    #[salsa::invoke(TraitData::trait_data_with_diagnostics_query)]
    fn trait_data_with_diagnostics(&self, tr: TraitId) -> (Arc<TraitData>, DefDiagnostics);

    #[salsa::invoke(TraitAliasData::trait_alias_query)]
    fn trait_alias_data(&self, e: TraitAliasId) -> Arc<TraitAliasData>;

    #[salsa::invoke(TypeAliasData::type_alias_data_query)]
    fn type_alias_data(&self, e: TypeAliasId) -> Arc<TypeAliasData>;

    #[salsa::invoke(FunctionData::fn_data_query)]
    fn function_data(&self, func: FunctionId) -> Arc<FunctionData>;

    #[salsa::invoke(ConstData::const_data_query)]
    fn const_data(&self, konst: ConstId) -> Arc<ConstData>;

    #[salsa::invoke(StaticData::static_data_query)]
    fn static_data(&self, konst: StaticId) -> Arc<StaticData>;

    #[salsa::invoke(Macro2Data::macro2_data_query)]
    fn macro2_data(&self, makro: Macro2Id) -> Arc<Macro2Data>;

    #[salsa::invoke(MacroRulesData::macro_rules_data_query)]
    fn macro_rules_data(&self, makro: MacroRulesId) -> Arc<MacroRulesData>;

    #[salsa::invoke(ProcMacroData::proc_macro_data_query)]
    fn proc_macro_data(&self, makro: ProcMacroId) -> Arc<ProcMacroData>;

    #[salsa::invoke(ExternCrateDeclData::extern_crate_decl_data_query)]
    fn extern_crate_decl_data(&self, extern_crate: ExternCrateId) -> Arc<ExternCrateDeclData>;

    // endregion:data

    #[salsa::invoke(Body::body_with_source_map_query)]
    fn body_with_source_map(&self, def: DefWithBodyId) -> (Arc<Body>, Arc<BodySourceMap>);

    #[salsa::invoke(Body::body_query)]
    fn body(&self, def: DefWithBodyId) -> Arc<Body>;

    #[salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, def: DefWithBodyId) -> Arc<ExprScopes>;

    #[salsa::invoke(GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDefId) -> Interned<GenericParams>;

    // region:attrs

    #[salsa::invoke(Attrs::fields_attrs_query)]
    fn fields_attrs(&self, def: VariantId) -> Arc<ArenaMap<LocalFieldId, Attrs>>;

    #[salsa::invoke(crate::attr::fields_attrs_source_map)]
    fn fields_attrs_source_map(
        &self,
        def: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, AstPtr<Either<ast::TupleField, ast::RecordField>>>>;

    #[salsa::invoke(AttrsWithOwner::attrs_query)]
    fn attrs(&self, def: AttrDefId) -> Attrs;

    #[salsa::transparent]
    #[salsa::invoke(lang_item::lang_attr)]
    fn lang_attr(&self, def: AttrDefId) -> Option<LangItem>;

    // endregion:attrs

    #[salsa::invoke(LangItems::lang_item_query)]
    fn lang_item(&self, start_crate: CrateId, item: LangItem) -> Option<LangItemTarget>;

    #[salsa::invoke(ImportMap::import_map_query)]
    fn import_map(&self, krate: CrateId) -> Arc<ImportMap>;

    // region:visibilities

    #[salsa::invoke(visibility::field_visibilities_query)]
    fn field_visibilities(&self, var: VariantId) -> Arc<ArenaMap<LocalFieldId, Visibility>>;

    // FIXME: unify function_visibility and const_visibility?
    #[salsa::invoke(visibility::function_visibility_query)]
    fn function_visibility(&self, def: FunctionId) -> Visibility;

    #[salsa::invoke(visibility::const_visibility_query)]
    fn const_visibility(&self, def: ConstId) -> Visibility;

    // endregion:visibilities

    #[salsa::invoke(LangItems::crate_lang_items_query)]
    fn crate_lang_items(&self, krate: CrateId) -> Option<Arc<LangItems>>;

    #[salsa::invoke(crate::lang_item::notable_traits_in_deps)]
    fn notable_traits_in_deps(&self, krate: CrateId) -> Arc<[Arc<[TraitId]>]>;
    #[salsa::invoke(crate::lang_item::crate_notable_traits)]
    fn crate_notable_traits(&self, krate: CrateId) -> Option<Arc<[TraitId]>>;

    fn crate_supports_no_std(&self, crate_id: CrateId) -> bool;

    fn include_macro_invoc(&self, crate_id: CrateId) -> Vec<(MacroCallId, FileId)>;
}

// return: macro call id and include file id
fn include_macro_invoc(db: &dyn DefDatabase, krate: CrateId) -> Vec<(MacroCallId, FileId)> {
    db.crate_def_map(krate)
        .modules
        .values()
        .flat_map(|m| m.scope.iter_macro_invoc())
        .filter_map(|invoc| {
            db.lookup_intern_macro_call(*invoc.1)
                .include_file_id(db.upcast(), *invoc.1)
                .map(|x| (*invoc.1, x))
        })
        .collect()
}

fn crate_def_map_wait(db: &dyn DefDatabase, krate: CrateId) -> Arc<DefMap> {
    let _p = tracing::span!(tracing::Level::INFO, "crate_def_map:wait").entered();
    db.crate_def_map_query(krate)
}

fn crate_supports_no_std(db: &dyn DefDatabase, crate_id: CrateId) -> bool {
    let file = db.crate_graph()[crate_id].root_file_id;
    let item_tree = db.file_item_tree(file.into());
    let attrs = item_tree.raw_attrs(AttrOwner::TopLevel);
    for attr in &**attrs {
        match attr.path().as_ident().and_then(|id| id.as_text()) {
            Some(ident) if ident == "no_std" => return true,
            Some(ident) if ident == "cfg_attr" => {}
            _ => continue,
        }

        // This is a `cfg_attr`; check if it could possibly expand to `no_std`.
        // Syntax is: `#[cfg_attr(condition(cfg, style), attr0, attr1, <...>)]`
        let tt = match attr.token_tree_value() {
            Some(tt) => &tt.token_trees,
            None => continue,
        };

        let segments =
            tt.split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(p)) if p.char == ','));
        for output in segments.skip(1) {
            match output {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.text == "no_std" => {
                    return true
                }
                _ => {}
            }
        }
    }

    false
}

fn macro_def(db: &dyn DefDatabase, id: MacroId) -> MacroDefId {
    use hir_expand::InFile;

    use crate::{Lookup, MacroDefKind, MacroExpander};

    let kind = |expander, file_id, m| {
        let in_file = InFile::new(file_id, m);
        match expander {
            MacroExpander::Declarative => MacroDefKind::Declarative(in_file),
            MacroExpander::BuiltIn(it) => MacroDefKind::BuiltIn(it, in_file),
            MacroExpander::BuiltInAttr(it) => MacroDefKind::BuiltInAttr(it, in_file),
            MacroExpander::BuiltInDerive(it) => MacroDefKind::BuiltInDerive(it, in_file),
            MacroExpander::BuiltInEager(it) => MacroDefKind::BuiltInEager(it, in_file),
        }
    };

    match id {
        MacroId::Macro2Id(it) => {
            let loc: Macro2Loc = it.lookup(db);

            let item_tree = loc.id.item_tree(db);
            let makro = &item_tree[loc.id.value];
            MacroDefId {
                krate: loc.container.krate,
                kind: kind(loc.expander, loc.id.file_id(), makro.ast_id.upcast()),
                local_inner: false,
                allow_internal_unsafe: loc.allow_internal_unsafe,
                span: db
                    .span_map(loc.id.file_id())
                    .span_for_range(db.ast_id_map(loc.id.file_id()).get(makro.ast_id).text_range()),
                edition: loc.edition,
            }
        }

        MacroId::MacroRulesId(it) => {
            let loc: MacroRulesLoc = it.lookup(db);

            let item_tree = loc.id.item_tree(db);
            let makro = &item_tree[loc.id.value];
            MacroDefId {
                krate: loc.container.krate,
                kind: kind(loc.expander, loc.id.file_id(), makro.ast_id.upcast()),
                local_inner: loc.flags.contains(MacroRulesLocFlags::LOCAL_INNER),
                allow_internal_unsafe: loc
                    .flags
                    .contains(MacroRulesLocFlags::ALLOW_INTERNAL_UNSAFE),
                span: db
                    .span_map(loc.id.file_id())
                    .span_for_range(db.ast_id_map(loc.id.file_id()).get(makro.ast_id).text_range()),
                edition: loc.edition,
            }
        }
        MacroId::ProcMacroId(it) => {
            let loc = it.lookup(db);

            let item_tree = loc.id.item_tree(db);
            let makro = &item_tree[loc.id.value];
            MacroDefId {
                krate: loc.container.krate,
                kind: MacroDefKind::ProcMacro(
                    loc.expander,
                    loc.kind,
                    InFile::new(loc.id.file_id(), makro.ast_id),
                ),
                local_inner: false,
                allow_internal_unsafe: false,
                span: db
                    .span_map(loc.id.file_id())
                    .span_for_range(db.ast_id_map(loc.id.file_id()).get(makro.ast_id).text_range()),
                edition: loc.edition,
            }
        }
    }
}
