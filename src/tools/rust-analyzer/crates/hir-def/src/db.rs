//! Defines database & queries for name resolution.
use base_db::{Crate, RootQueryDb, SourceDatabase};
use either::Either;
use hir_expand::{
    EditionedFileId, HirFileId, InFile, Lookup, MacroCallId, MacroDefId, MacroDefKind,
    db::ExpandDatabase,
};
use intern::sym;
use la_arena::ArenaMap;
use syntax::{AstPtr, ast};
use triomphe::Arc;

use crate::{
    AssocItemId, AttrDefId, ConstId, ConstLoc, DefWithBodyId, EnumId, EnumLoc, EnumVariantId,
    EnumVariantLoc, ExternBlockId, ExternBlockLoc, ExternCrateId, ExternCrateLoc, FunctionId,
    FunctionLoc, GenericDefId, ImplId, ImplLoc, LocalFieldId, Macro2Id, Macro2Loc, MacroExpander,
    MacroId, MacroRulesId, MacroRulesLoc, MacroRulesLocFlags, ProcMacroId, ProcMacroLoc, StaticId,
    StaticLoc, StructId, StructLoc, TraitAliasId, TraitAliasLoc, TraitId, TraitLoc, TypeAliasId,
    TypeAliasLoc, UnionId, UnionLoc, UseId, UseLoc, VariantId,
    attr::{Attrs, AttrsWithOwner},
    expr_store::{
        Body, BodySourceMap, ExpressionStore, ExpressionStoreSourceMap, scope::ExprScopes,
    },
    hir::generics::GenericParams,
    import_map::ImportMap,
    item_tree::{ItemTree, file_item_tree_query},
    lang_item::{self, LangItem},
    nameres::{assoc::TraitItems, crate_def_map, diagnostics::DefDiagnostics},
    signatures::{
        ConstSignature, EnumSignature, FunctionSignature, ImplSignature, StaticSignature,
        StructSignature, TraitAliasSignature, TraitSignature, TypeAliasSignature, UnionSignature,
        VariantFields,
    },
    tt,
    visibility::{self, Visibility},
};

use salsa::plumbing::AsId;

#[query_group::query_group(InternDatabaseStorage)]
pub trait InternDatabase: RootQueryDb {
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
    // // endregion: items
}

#[query_group::query_group]
pub trait DefDatabase: InternDatabase + ExpandDatabase + SourceDatabase {
    /// Whether to expand procedural macros during name resolution.
    #[salsa::input]
    fn expand_proc_attr_macros(&self) -> bool;

    /// Computes an [`ItemTree`] for the given file or macro expansion.
    #[salsa::invoke(file_item_tree_query)]
    #[salsa::transparent]
    fn file_item_tree(&self, file_id: HirFileId) -> &ItemTree;

    /// Turns a MacroId into a MacroDefId, describing the macro's definition post name resolution.
    #[salsa::invoke(macro_def)]
    fn macro_def(&self, m: MacroId) -> MacroDefId;

    // region:data

    #[salsa::invoke(VariantFields::query)]
    fn variant_fields_with_source_map(
        &self,
        id: VariantId,
    ) -> (Arc<VariantFields>, Arc<ExpressionStoreSourceMap>);

    #[salsa::transparent]
    #[salsa::invoke(TraitItems::trait_items_query)]
    fn trait_items(&self, e: TraitId) -> Arc<TraitItems>;

    #[salsa::invoke(TraitItems::trait_items_with_diagnostics_query)]
    fn trait_items_with_diagnostics(&self, tr: TraitId) -> (Arc<TraitItems>, DefDiagnostics);

    #[salsa::tracked]
    fn variant_fields(&self, id: VariantId) -> Arc<VariantFields> {
        self.variant_fields_with_source_map(id).0
    }

    #[salsa::tracked]
    fn trait_signature(&self, trait_: TraitId) -> Arc<TraitSignature> {
        self.trait_signature_with_source_map(trait_).0
    }

    #[salsa::tracked]
    fn impl_signature(&self, impl_: ImplId) -> Arc<ImplSignature> {
        self.impl_signature_with_source_map(impl_).0
    }

    #[salsa::tracked]
    fn struct_signature(&self, struct_: StructId) -> Arc<StructSignature> {
        self.struct_signature_with_source_map(struct_).0
    }

    #[salsa::tracked]
    fn union_signature(&self, union_: UnionId) -> Arc<UnionSignature> {
        self.union_signature_with_source_map(union_).0
    }

    #[salsa::tracked]
    fn enum_signature(&self, e: EnumId) -> Arc<EnumSignature> {
        self.enum_signature_with_source_map(e).0
    }

    #[salsa::tracked]
    fn const_signature(&self, e: ConstId) -> Arc<ConstSignature> {
        self.const_signature_with_source_map(e).0
    }

    #[salsa::tracked]
    fn static_signature(&self, e: StaticId) -> Arc<StaticSignature> {
        self.static_signature_with_source_map(e).0
    }

    #[salsa::tracked]
    fn function_signature(&self, e: FunctionId) -> Arc<FunctionSignature> {
        self.function_signature_with_source_map(e).0
    }

    #[salsa::tracked]
    fn trait_alias_signature(&self, e: TraitAliasId) -> Arc<TraitAliasSignature> {
        self.trait_alias_signature_with_source_map(e).0
    }

    #[salsa::tracked]
    fn type_alias_signature(&self, e: TypeAliasId) -> Arc<TypeAliasSignature> {
        self.type_alias_signature_with_source_map(e).0
    }

    #[salsa::invoke(TraitSignature::query)]
    fn trait_signature_with_source_map(
        &self,
        trait_: TraitId,
    ) -> (Arc<TraitSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(ImplSignature::query)]
    fn impl_signature_with_source_map(
        &self,
        impl_: ImplId,
    ) -> (Arc<ImplSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(StructSignature::query)]
    fn struct_signature_with_source_map(
        &self,
        struct_: StructId,
    ) -> (Arc<StructSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(UnionSignature::query)]
    fn union_signature_with_source_map(
        &self,
        union_: UnionId,
    ) -> (Arc<UnionSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(EnumSignature::query)]
    fn enum_signature_with_source_map(
        &self,
        e: EnumId,
    ) -> (Arc<EnumSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(ConstSignature::query)]
    fn const_signature_with_source_map(
        &self,
        e: ConstId,
    ) -> (Arc<ConstSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(StaticSignature::query)]
    fn static_signature_with_source_map(
        &self,
        e: StaticId,
    ) -> (Arc<StaticSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(FunctionSignature::query)]
    fn function_signature_with_source_map(
        &self,
        e: FunctionId,
    ) -> (Arc<FunctionSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(TraitAliasSignature::query)]
    fn trait_alias_signature_with_source_map(
        &self,
        e: TraitAliasId,
    ) -> (Arc<TraitAliasSignature>, Arc<ExpressionStoreSourceMap>);

    #[salsa::invoke(TypeAliasSignature::query)]
    fn type_alias_signature_with_source_map(
        &self,
        e: TypeAliasId,
    ) -> (Arc<TypeAliasSignature>, Arc<ExpressionStoreSourceMap>);

    // endregion:data

    #[salsa::invoke(Body::body_with_source_map_query)]
    #[salsa::lru(512)]
    fn body_with_source_map(&self, def: DefWithBodyId) -> (Arc<Body>, Arc<BodySourceMap>);

    #[salsa::invoke(Body::body_query)]
    fn body(&self, def: DefWithBodyId) -> Arc<Body>;

    #[salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, def: DefWithBodyId) -> Arc<ExprScopes>;

    #[salsa::transparent]
    #[salsa::invoke(GenericParams::new)]
    fn generic_params(&self, def: GenericDefId) -> Arc<GenericParams>;

    #[salsa::transparent]
    #[salsa::invoke(GenericParams::generic_params_and_store)]
    fn generic_params_and_store(
        &self,
        def: GenericDefId,
    ) -> (Arc<GenericParams>, Arc<ExpressionStore>);

    #[salsa::transparent]
    #[salsa::invoke(GenericParams::generic_params_and_store_and_source_map)]
    fn generic_params_and_store_and_source_map(
        &self,
        def: GenericDefId,
    ) -> (Arc<GenericParams>, Arc<ExpressionStore>, Arc<ExpressionStoreSourceMap>);

    // region:attrs

    #[salsa::invoke(Attrs::fields_attrs_query)]
    fn fields_attrs(&self, def: VariantId) -> Arc<ArenaMap<LocalFieldId, Attrs>>;

    // should this really be a query?
    #[salsa::invoke(crate::attr::fields_attrs_source_map)]
    fn fields_attrs_source_map(
        &self,
        def: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, AstPtr<Either<ast::TupleField, ast::RecordField>>>>;

    // FIXME: Make this a non-interned query.
    #[salsa::invoke_interned(AttrsWithOwner::attrs_query)]
    fn attrs(&self, def: AttrDefId) -> Attrs;

    #[salsa::transparent]
    #[salsa::invoke(lang_item::lang_attr)]
    fn lang_attr(&self, def: AttrDefId) -> Option<LangItem>;

    // endregion:attrs

    #[salsa::invoke(ImportMap::import_map_query)]
    fn import_map(&self, krate: Crate) -> Arc<ImportMap>;

    // region:visibilities

    #[salsa::invoke(visibility::field_visibilities_query)]
    fn field_visibilities(&self, var: VariantId) -> Arc<ArenaMap<LocalFieldId, Visibility>>;

    #[salsa::invoke(visibility::assoc_visibility_query)]
    fn assoc_visibility(&self, def: AssocItemId) -> Visibility;

    // endregion:visibilities

    #[salsa::invoke(crate::lang_item::notable_traits_in_deps)]
    fn notable_traits_in_deps(&self, krate: Crate) -> Arc<[Arc<[TraitId]>]>;
    #[salsa::invoke(crate::lang_item::crate_notable_traits)]
    fn crate_notable_traits(&self, krate: Crate) -> Option<Arc<[TraitId]>>;

    #[salsa::invoke(crate_supports_no_std)]
    fn crate_supports_no_std(&self, crate_id: Crate) -> bool;

    #[salsa::invoke(include_macro_invoc)]
    fn include_macro_invoc(&self, crate_id: Crate) -> Arc<[(MacroCallId, EditionedFileId)]>;
}

// return: macro call id and include file id
fn include_macro_invoc(
    db: &dyn DefDatabase,
    krate: Crate,
) -> Arc<[(MacroCallId, EditionedFileId)]> {
    crate_def_map(db, krate)
        .modules
        .values()
        .flat_map(|m| m.scope.iter_macro_invoc())
        .filter_map(|invoc| {
            db.lookup_intern_macro_call(*invoc.1)
                .include_file_id(db, *invoc.1)
                .map(|x| (*invoc.1, x))
        })
        .collect()
}

fn crate_supports_no_std(db: &dyn DefDatabase, crate_id: Crate) -> bool {
    let file = crate_id.data(db).root_file_id(db);
    let item_tree = db.file_item_tree(file.into());
    let attrs = item_tree.top_level_raw_attrs();
    for attr in &**attrs {
        match attr.path().as_ident() {
            Some(ident) if *ident == sym::no_std => return true,
            Some(ident) if *ident == sym::cfg_attr => {}
            _ => continue,
        }

        // This is a `cfg_attr`; check if it could possibly expand to `no_std`.
        // Syntax is: `#[cfg_attr(condition(cfg, style), attr0, attr1, <...>)]`
        let tt = match attr.token_tree_value() {
            Some(tt) => tt.token_trees(),
            None => continue,
        };

        let segments =
            tt.split(|tt| matches!(tt, tt::TtElement::Leaf(tt::Leaf::Punct(p)) if p.char == ','));
        for output in segments.skip(1) {
            match output.flat_tokens() {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.sym == sym::no_std => {
                    return true;
                }
                _ => {}
            }
        }
    }

    false
}

fn macro_def(db: &dyn DefDatabase, id: MacroId) -> MacroDefId {
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

            MacroDefId {
                krate: loc.container.krate,
                kind: kind(loc.expander, loc.id.file_id, loc.id.value.upcast()),
                local_inner: false,
                allow_internal_unsafe: loc.allow_internal_unsafe,
                edition: loc.edition,
            }
        }
        MacroId::MacroRulesId(it) => {
            let loc: MacroRulesLoc = it.lookup(db);

            MacroDefId {
                krate: loc.container.krate,
                kind: kind(loc.expander, loc.id.file_id, loc.id.value.upcast()),
                local_inner: loc.flags.contains(MacroRulesLocFlags::LOCAL_INNER),
                allow_internal_unsafe: loc
                    .flags
                    .contains(MacroRulesLocFlags::ALLOW_INTERNAL_UNSAFE),
                edition: loc.edition,
            }
        }
        MacroId::ProcMacroId(it) => {
            let loc = it.lookup(db);

            MacroDefId {
                krate: loc.container.krate,
                kind: MacroDefKind::ProcMacro(loc.id, loc.expander, loc.kind),
                local_inner: false,
                allow_internal_unsafe: false,
                edition: loc.edition,
            }
        }
    }
}
