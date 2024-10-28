//! Defines database & queries for name resolution.
use base_db::{ra_salsa, CrateId, SourceDatabase, Upcast};
use either::Either;
use hir_expand::{db::ExpandDatabase, HirFileId, MacroDefId};
use intern::{sym, Interned};
use la_arena::ArenaMap;
use span::{EditionedFileId, MacroCallId};
use syntax::{ast, AstPtr};
use triomphe::Arc;

use crate::{
    attr::{Attrs, AttrsWithOwner},
    body::{scope::ExprScopes, Body, BodySourceMap},
    data::{
        adt::{EnumData, EnumVariantData, StructData, VariantData},
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

#[ra_salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    // region: items
    #[ra_salsa::interned]
    fn intern_use(&self, loc: UseLoc) -> UseId;
    #[ra_salsa::interned]
    fn intern_extern_crate(&self, loc: ExternCrateLoc) -> ExternCrateId;
    #[ra_salsa::interned]
    fn intern_function(&self, loc: FunctionLoc) -> FunctionId;
    #[ra_salsa::interned]
    fn intern_struct(&self, loc: StructLoc) -> StructId;
    #[ra_salsa::interned]
    fn intern_union(&self, loc: UnionLoc) -> UnionId;
    #[ra_salsa::interned]
    fn intern_enum(&self, loc: EnumLoc) -> EnumId;
    #[ra_salsa::interned]
    fn intern_enum_variant(&self, loc: EnumVariantLoc) -> EnumVariantId;
    #[ra_salsa::interned]
    fn intern_const(&self, loc: ConstLoc) -> ConstId;
    #[ra_salsa::interned]
    fn intern_static(&self, loc: StaticLoc) -> StaticId;
    #[ra_salsa::interned]
    fn intern_trait(&self, loc: TraitLoc) -> TraitId;
    #[ra_salsa::interned]
    fn intern_trait_alias(&self, loc: TraitAliasLoc) -> TraitAliasId;
    #[ra_salsa::interned]
    fn intern_type_alias(&self, loc: TypeAliasLoc) -> TypeAliasId;
    #[ra_salsa::interned]
    fn intern_impl(&self, loc: ImplLoc) -> ImplId;
    #[ra_salsa::interned]
    fn intern_extern_block(&self, loc: ExternBlockLoc) -> ExternBlockId;
    #[ra_salsa::interned]
    fn intern_macro2(&self, loc: Macro2Loc) -> Macro2Id;
    #[ra_salsa::interned]
    fn intern_proc_macro(&self, loc: ProcMacroLoc) -> ProcMacroId;
    #[ra_salsa::interned]
    fn intern_macro_rules(&self, loc: MacroRulesLoc) -> MacroRulesId;
    // endregion: items

    #[ra_salsa::interned]
    fn intern_block(&self, loc: BlockLoc) -> BlockId;
    #[ra_salsa::interned]
    fn intern_anonymous_const(&self, id: ConstBlockLoc) -> ConstBlockId;
    #[ra_salsa::interned]
    fn intern_in_type_const(&self, id: InTypeConstLoc) -> InTypeConstId;
}

#[ra_salsa::query_group(DefDatabaseStorage)]
pub trait DefDatabase: InternDatabase + ExpandDatabase + Upcast<dyn ExpandDatabase> {
    /// Whether to expand procedural macros during name resolution.
    #[ra_salsa::input]
    fn expand_proc_attr_macros(&self) -> bool;

    /// Computes an [`ItemTree`] for the given file or macro expansion.
    #[ra_salsa::invoke(ItemTree::file_item_tree_query)]
    fn file_item_tree(&self, file_id: HirFileId) -> Arc<ItemTree>;

    #[ra_salsa::invoke(ItemTree::block_item_tree_query)]
    fn block_item_tree(&self, block_id: BlockId) -> Arc<ItemTree>;

    #[ra_salsa::invoke(DefMap::crate_def_map_query)]
    fn crate_def_map(&self, krate: CrateId) -> Arc<DefMap>;

    /// Computes the block-level `DefMap`.
    #[ra_salsa::invoke(DefMap::block_def_map_query)]
    fn block_def_map(&self, block: BlockId) -> Arc<DefMap>;

    /// Turns a MacroId into a MacroDefId, describing the macro's definition post name resolution.
    fn macro_def(&self, m: MacroId) -> MacroDefId;

    // region:data

    #[ra_salsa::transparent]
    #[ra_salsa::invoke(StructData::struct_data_query)]
    fn struct_data(&self, id: StructId) -> Arc<StructData>;

    #[ra_salsa::invoke(StructData::struct_data_with_diagnostics_query)]
    fn struct_data_with_diagnostics(&self, id: StructId) -> (Arc<StructData>, DefDiagnostics);

    #[ra_salsa::transparent]
    #[ra_salsa::invoke(StructData::union_data_query)]
    fn union_data(&self, id: UnionId) -> Arc<StructData>;

    #[ra_salsa::invoke(StructData::union_data_with_diagnostics_query)]
    fn union_data_with_diagnostics(&self, id: UnionId) -> (Arc<StructData>, DefDiagnostics);

    #[ra_salsa::invoke(EnumData::enum_data_query)]
    fn enum_data(&self, e: EnumId) -> Arc<EnumData>;

    #[ra_salsa::transparent]
    #[ra_salsa::invoke(EnumVariantData::enum_variant_data_query)]
    fn enum_variant_data(&self, id: EnumVariantId) -> Arc<EnumVariantData>;

    #[ra_salsa::invoke(EnumVariantData::enum_variant_data_with_diagnostics_query)]
    fn enum_variant_data_with_diagnostics(
        &self,
        id: EnumVariantId,
    ) -> (Arc<EnumVariantData>, DefDiagnostics);

    #[ra_salsa::transparent]
    #[ra_salsa::invoke(VariantData::variant_data)]
    fn variant_data(&self, id: VariantId) -> Arc<VariantData>;
    #[ra_salsa::transparent]
    #[ra_salsa::invoke(ImplData::impl_data_query)]
    fn impl_data(&self, e: ImplId) -> Arc<ImplData>;

    #[ra_salsa::invoke(ImplData::impl_data_with_diagnostics_query)]
    fn impl_data_with_diagnostics(&self, e: ImplId) -> (Arc<ImplData>, DefDiagnostics);

    #[ra_salsa::transparent]
    #[ra_salsa::invoke(TraitData::trait_data_query)]
    fn trait_data(&self, e: TraitId) -> Arc<TraitData>;

    #[ra_salsa::invoke(TraitData::trait_data_with_diagnostics_query)]
    fn trait_data_with_diagnostics(&self, tr: TraitId) -> (Arc<TraitData>, DefDiagnostics);

    #[ra_salsa::invoke(TraitAliasData::trait_alias_query)]
    fn trait_alias_data(&self, e: TraitAliasId) -> Arc<TraitAliasData>;

    #[ra_salsa::invoke(TypeAliasData::type_alias_data_query)]
    fn type_alias_data(&self, e: TypeAliasId) -> Arc<TypeAliasData>;

    #[ra_salsa::invoke(FunctionData::fn_data_query)]
    fn function_data(&self, func: FunctionId) -> Arc<FunctionData>;

    #[ra_salsa::invoke(ConstData::const_data_query)]
    fn const_data(&self, konst: ConstId) -> Arc<ConstData>;

    #[ra_salsa::invoke(StaticData::static_data_query)]
    fn static_data(&self, statik: StaticId) -> Arc<StaticData>;

    #[ra_salsa::invoke(Macro2Data::macro2_data_query)]
    fn macro2_data(&self, makro: Macro2Id) -> Arc<Macro2Data>;

    #[ra_salsa::invoke(MacroRulesData::macro_rules_data_query)]
    fn macro_rules_data(&self, makro: MacroRulesId) -> Arc<MacroRulesData>;

    #[ra_salsa::invoke(ProcMacroData::proc_macro_data_query)]
    fn proc_macro_data(&self, makro: ProcMacroId) -> Arc<ProcMacroData>;

    #[ra_salsa::invoke(ExternCrateDeclData::extern_crate_decl_data_query)]
    fn extern_crate_decl_data(&self, extern_crate: ExternCrateId) -> Arc<ExternCrateDeclData>;

    // endregion:data

    #[ra_salsa::invoke(Body::body_with_source_map_query)]
    #[ra_salsa::lru]
    fn body_with_source_map(&self, def: DefWithBodyId) -> (Arc<Body>, Arc<BodySourceMap>);

    #[ra_salsa::invoke(Body::body_query)]
    fn body(&self, def: DefWithBodyId) -> Arc<Body>;

    #[ra_salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, def: DefWithBodyId) -> Arc<ExprScopes>;

    #[ra_salsa::invoke(GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDefId) -> Interned<GenericParams>;

    // region:attrs

    #[ra_salsa::invoke(Attrs::fields_attrs_query)]
    fn fields_attrs(&self, def: VariantId) -> Arc<ArenaMap<LocalFieldId, Attrs>>;

    // should this really be a query?
    #[ra_salsa::invoke(crate::attr::fields_attrs_source_map)]
    fn fields_attrs_source_map(
        &self,
        def: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, AstPtr<Either<ast::TupleField, ast::RecordField>>>>;

    #[ra_salsa::invoke(AttrsWithOwner::attrs_query)]
    fn attrs(&self, def: AttrDefId) -> Attrs;

    #[ra_salsa::transparent]
    #[ra_salsa::invoke(lang_item::lang_attr)]
    fn lang_attr(&self, def: AttrDefId) -> Option<LangItem>;

    // endregion:attrs

    #[ra_salsa::invoke(LangItems::lang_item_query)]
    fn lang_item(&self, start_crate: CrateId, item: LangItem) -> Option<LangItemTarget>;

    #[ra_salsa::invoke(ImportMap::import_map_query)]
    fn import_map(&self, krate: CrateId) -> Arc<ImportMap>;

    // region:visibilities

    #[ra_salsa::invoke(visibility::field_visibilities_query)]
    fn field_visibilities(&self, var: VariantId) -> Arc<ArenaMap<LocalFieldId, Visibility>>;

    // FIXME: unify function_visibility and const_visibility?
    #[ra_salsa::invoke(visibility::function_visibility_query)]
    fn function_visibility(&self, def: FunctionId) -> Visibility;

    #[ra_salsa::invoke(visibility::const_visibility_query)]
    fn const_visibility(&self, def: ConstId) -> Visibility;

    // endregion:visibilities

    #[ra_salsa::invoke(LangItems::crate_lang_items_query)]
    fn crate_lang_items(&self, krate: CrateId) -> Option<Arc<LangItems>>;

    #[ra_salsa::invoke(crate::lang_item::notable_traits_in_deps)]
    fn notable_traits_in_deps(&self, krate: CrateId) -> Arc<[Arc<[TraitId]>]>;
    #[ra_salsa::invoke(crate::lang_item::crate_notable_traits)]
    fn crate_notable_traits(&self, krate: CrateId) -> Option<Arc<[TraitId]>>;

    fn crate_supports_no_std(&self, crate_id: CrateId) -> bool;

    fn include_macro_invoc(&self, crate_id: CrateId) -> Arc<[(MacroCallId, EditionedFileId)]>;
}

// return: macro call id and include file id
fn include_macro_invoc(
    db: &dyn DefDatabase,
    krate: CrateId,
) -> Arc<[(MacroCallId, EditionedFileId)]> {
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

fn crate_supports_no_std(db: &dyn DefDatabase, crate_id: CrateId) -> bool {
    let file = db.crate_graph()[crate_id].root_file_id();
    let item_tree = db.file_item_tree(file.into());
    let attrs = item_tree.raw_attrs(AttrOwner::TopLevel);
    for attr in &**attrs {
        match attr.path().as_ident() {
            Some(ident) if *ident == sym::no_std.clone() => return true,
            Some(ident) if *ident == sym::cfg_attr.clone() => {}
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
                [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.sym == sym::no_std => {
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
            MacroExpander::BuiltIn(it) => MacroDefKind::BuiltIn(in_file, it),
            MacroExpander::BuiltInAttr(it) => MacroDefKind::BuiltInAttr(in_file, it),
            MacroExpander::BuiltInDerive(it) => MacroDefKind::BuiltInDerive(in_file, it),
            MacroExpander::BuiltInEager(it) => MacroDefKind::BuiltInEager(in_file, it),
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
                    InFile::new(loc.id.file_id(), makro.ast_id),
                    loc.expander,
                    loc.kind,
                ),
                local_inner: false,
                allow_internal_unsafe: false,
                edition: loc.edition,
            }
        }
    }
}
