//! Defines database & queries for name resolution.
use base_db::{salsa, CrateId, SourceDatabase, Upcast};
use either::Either;
use hir_expand::{db::ExpandDatabase, HirFileId};
use intern::Interned;
use la_arena::ArenaMap;
use syntax::{ast, AstPtr};
use triomphe::Arc;

use crate::{
    attr::{Attrs, AttrsWithOwner},
    body::{scope::ExprScopes, Body, BodySourceMap},
    data::{
        adt::{EnumData, StructData},
        ConstData, FunctionData, ImplData, Macro2Data, MacroRulesData, ProcMacroData, StaticData,
        TraitAliasData, TraitData, TypeAliasData,
    },
    generics::GenericParams,
    import_map::ImportMap,
    item_tree::{AttrOwner, ItemTree},
    lang_item::{LangItem, LangItemTarget, LangItems},
    nameres::{diagnostics::DefDiagnostic, DefMap},
    visibility::{self, Visibility},
    AttrDefId, BlockId, BlockLoc, ConstBlockId, ConstBlockLoc, ConstId, ConstLoc, DefWithBodyId,
    EnumId, EnumLoc, ExternBlockId, ExternBlockLoc, FunctionId, FunctionLoc, GenericDefId, ImplId,
    ImplLoc, InTypeConstId, InTypeConstLoc, LocalEnumVariantId, LocalFieldId, Macro2Id, Macro2Loc,
    MacroRulesId, MacroRulesLoc, ProcMacroId, ProcMacroLoc, StaticId, StaticLoc, StructId,
    StructLoc, TraitAliasId, TraitAliasLoc, TraitId, TraitLoc, TypeAliasId, TypeAliasLoc, UnionId,
    UnionLoc, VariantId,
};

#[salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    // region: items
    #[salsa::interned]
    fn intern_function(&self, loc: FunctionLoc) -> FunctionId;
    #[salsa::interned]
    fn intern_struct(&self, loc: StructLoc) -> StructId;
    #[salsa::interned]
    fn intern_union(&self, loc: UnionLoc) -> UnionId;
    #[salsa::interned]
    fn intern_enum(&self, loc: EnumLoc) -> EnumId;
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

    #[salsa::invoke(crate_def_map_wait)]
    #[salsa::transparent]
    fn crate_def_map(&self, krate: CrateId) -> Arc<DefMap>;

    #[salsa::invoke(DefMap::crate_def_map_query)]
    fn crate_def_map_query(&self, krate: CrateId) -> Arc<DefMap>;

    /// Computes the block-level `DefMap`, returning `None` when `block` doesn't contain any inner
    /// items directly.
    ///
    /// For example:
    ///
    /// ```
    /// fn f() { // (0)
    ///     { // (1)
    ///         fn inner() {}
    ///     }
    /// }
    /// ```
    ///
    /// The `block_def_map` for block 0 would return `None`, while `block_def_map` of block 1 would
    /// return a `DefMap` containing `inner`.
    #[salsa::invoke(DefMap::block_def_map_query)]
    fn block_def_map(&self, block: BlockId) -> Arc<DefMap>;

    // region:data

    #[salsa::invoke(StructData::struct_data_query)]
    fn struct_data(&self, id: StructId) -> Arc<StructData>;

    #[salsa::invoke(StructData::struct_data_with_diagnostics_query)]
    fn struct_data_with_diagnostics(&self, id: StructId)
        -> (Arc<StructData>, Arc<[DefDiagnostic]>);

    #[salsa::invoke(StructData::union_data_query)]
    fn union_data(&self, id: UnionId) -> Arc<StructData>;

    #[salsa::invoke(StructData::union_data_with_diagnostics_query)]
    fn union_data_with_diagnostics(&self, id: UnionId) -> (Arc<StructData>, Arc<[DefDiagnostic]>);

    #[salsa::invoke(EnumData::enum_data_query)]
    fn enum_data(&self, e: EnumId) -> Arc<EnumData>;

    #[salsa::invoke(EnumData::enum_data_with_diagnostics_query)]
    fn enum_data_with_diagnostics(&self, e: EnumId) -> (Arc<EnumData>, Arc<[DefDiagnostic]>);

    #[salsa::invoke(ImplData::impl_data_query)]
    fn impl_data(&self, e: ImplId) -> Arc<ImplData>;

    #[salsa::invoke(ImplData::impl_data_with_diagnostics_query)]
    fn impl_data_with_diagnostics(&self, e: ImplId) -> (Arc<ImplData>, Arc<[DefDiagnostic]>);

    #[salsa::invoke(TraitData::trait_data_query)]
    fn trait_data(&self, e: TraitId) -> Arc<TraitData>;

    #[salsa::invoke(TraitData::trait_data_with_diagnostics_query)]
    fn trait_data_with_diagnostics(&self, tr: TraitId) -> (Arc<TraitData>, Arc<[DefDiagnostic]>);

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

    #[salsa::invoke(Attrs::variants_attrs_query)]
    fn variants_attrs(&self, def: EnumId) -> Arc<ArenaMap<LocalEnumVariantId, Attrs>>;

    #[salsa::invoke(Attrs::fields_attrs_query)]
    fn fields_attrs(&self, def: VariantId) -> Arc<ArenaMap<LocalFieldId, Attrs>>;

    #[salsa::invoke(crate::attr::variants_attrs_source_map)]
    fn variants_attrs_source_map(
        &self,
        def: EnumId,
    ) -> Arc<ArenaMap<LocalEnumVariantId, AstPtr<ast::Variant>>>;

    #[salsa::invoke(crate::attr::fields_attrs_source_map)]
    fn fields_attrs_source_map(
        &self,
        def: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, Either<AstPtr<ast::TupleField>, AstPtr<ast::RecordField>>>>;

    #[salsa::invoke(AttrsWithOwner::attrs_query)]
    fn attrs(&self, def: AttrDefId) -> Attrs;

    #[salsa::transparent]
    #[salsa::invoke(AttrsWithOwner::attrs_with_owner)]
    fn attrs_with_owner(&self, def: AttrDefId) -> AttrsWithOwner;

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
    fn crate_lang_items(&self, krate: CrateId) -> Arc<LangItems>;

    #[salsa::transparent]
    fn crate_limits(&self, crate_id: CrateId) -> CrateLimits;

    #[salsa::transparent]
    fn recursion_limit(&self, crate_id: CrateId) -> u32;

    fn crate_supports_no_std(&self, crate_id: CrateId) -> bool;
}

fn crate_def_map_wait(db: &dyn DefDatabase, krate: CrateId) -> Arc<DefMap> {
    let _p = profile::span("crate_def_map:wait");
    db.crate_def_map_query(krate)
}

pub struct CrateLimits {
    /// The maximum depth for potentially infinitely-recursive compile-time operations like macro expansion or auto-dereference.
    pub recursion_limit: u32,
}

fn crate_limits(db: &dyn DefDatabase, crate_id: CrateId) -> CrateLimits {
    let def_map = db.crate_def_map(crate_id);

    CrateLimits {
        // 128 is the default in rustc.
        recursion_limit: def_map.recursion_limit().unwrap_or(128),
    }
}

fn recursion_limit(db: &dyn DefDatabase, crate_id: CrateId) -> u32 {
    db.crate_limits(crate_id).recursion_limit
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

        let segments = tt.split(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Punct(p)) if p.char == ',' => true,
            _ => false,
        });
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
