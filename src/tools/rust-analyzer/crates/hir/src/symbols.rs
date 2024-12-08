//! File symbol extraction.

use hir_def::{
    db::DefDatabase,
    item_scope::ItemInNs,
    src::{HasChildSource, HasSource},
    AdtId, AssocItemId, DefWithBodyId, HasModule, ImplId, Lookup, MacroId, ModuleDefId, ModuleId,
    TraitId,
};
use hir_expand::HirFileId;
use hir_ty::{
    db::HirDatabase,
    display::{hir_display_with_types_map, HirDisplay},
};
use span::Edition;
use syntax::{ast::HasName, AstNode, AstPtr, SmolStr, SyntaxNode, SyntaxNodePtr, ToSmolStr};

use crate::{Module, ModuleDef, Semantics};

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileSymbol {
    pub name: SmolStr,
    pub def: ModuleDef,
    pub loc: DeclarationLocation,
    pub container_name: Option<SmolStr>,
    /// Whether this symbol is a doc alias for the original symbol.
    pub is_alias: bool,
    pub is_assoc: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclarationLocation {
    /// The file id for both the `ptr` and `name_ptr`.
    pub hir_file_id: HirFileId,
    /// This points to the whole syntax node of the declaration.
    pub ptr: SyntaxNodePtr,
    /// This points to the [`syntax::ast::Name`] identifier of the declaration.
    pub name_ptr: AstPtr<syntax::ast::Name>,
}

impl DeclarationLocation {
    pub fn syntax<DB: HirDatabase>(&self, sema: &Semantics<'_, DB>) -> SyntaxNode {
        let root = sema.parse_or_expand(self.hir_file_id);
        self.ptr.to_node(&root)
    }
}

/// Represents an outstanding module that the symbol collector must collect symbols from.
struct SymbolCollectorWork {
    module_id: ModuleId,
    parent: Option<DefWithBodyId>,
}

pub struct SymbolCollector<'a> {
    db: &'a dyn HirDatabase,
    symbols: Vec<FileSymbol>,
    work: Vec<SymbolCollectorWork>,
    current_container_name: Option<SmolStr>,
    edition: Edition,
}

/// Given a [`ModuleId`] and a [`HirDatabase`], use the DefMap for the module's crate to collect
/// all symbols that should be indexed for the given module.
impl<'a> SymbolCollector<'a> {
    pub fn new(db: &'a dyn HirDatabase) -> Self {
        SymbolCollector {
            db,
            symbols: Default::default(),
            work: Default::default(),
            current_container_name: None,
            edition: Edition::Edition2015,
        }
    }

    pub fn collect(&mut self, module: Module) {
        self.edition = module.krate().edition(self.db);

        // The initial work is the root module we're collecting, additional work will
        // be populated as we traverse the module's definitions.
        self.work.push(SymbolCollectorWork { module_id: module.into(), parent: None });

        while let Some(work) = self.work.pop() {
            self.do_work(work);
        }
    }

    pub fn finish(self) -> Vec<FileSymbol> {
        self.symbols
    }

    pub fn collect_module(db: &dyn HirDatabase, module: Module) -> Vec<FileSymbol> {
        let mut symbol_collector = SymbolCollector::new(db);
        symbol_collector.collect(module);
        symbol_collector.finish()
    }

    fn do_work(&mut self, work: SymbolCollectorWork) {
        self.db.unwind_if_cancelled();

        let parent_name = work.parent.and_then(|id| self.def_with_body_id_name(id));
        self.with_container_name(parent_name, |s| s.collect_from_module(work.module_id));
    }

    fn collect_from_module(&mut self, module_id: ModuleId) {
        let def_map = module_id.def_map(self.db.upcast());
        let scope = &def_map[module_id.local_id].scope;

        for module_def_id in scope.declarations() {
            match module_def_id {
                ModuleDefId::ModuleId(id) => self.push_module(id),
                ModuleDefId::FunctionId(id) => {
                    self.push_decl(id, false);
                    self.collect_from_body(id);
                }
                ModuleDefId::AdtId(AdtId::StructId(id)) => self.push_decl(id, false),
                ModuleDefId::AdtId(AdtId::EnumId(id)) => self.push_decl(id, false),
                ModuleDefId::AdtId(AdtId::UnionId(id)) => self.push_decl(id, false),
                ModuleDefId::ConstId(id) => {
                    self.push_decl(id, false);
                    self.collect_from_body(id);
                }
                ModuleDefId::StaticId(id) => {
                    self.push_decl(id, false);
                    self.collect_from_body(id);
                }
                ModuleDefId::TraitId(id) => {
                    self.push_decl(id, false);
                    self.collect_from_trait(id);
                }
                ModuleDefId::TraitAliasId(id) => {
                    self.push_decl(id, false);
                }
                ModuleDefId::TypeAliasId(id) => {
                    self.push_decl(id, false);
                }
                ModuleDefId::MacroId(id) => match id {
                    MacroId::Macro2Id(id) => self.push_decl(id, false),
                    MacroId::MacroRulesId(id) => self.push_decl(id, false),
                    MacroId::ProcMacroId(id) => self.push_decl(id, false),
                },
                // Don't index these.
                ModuleDefId::BuiltinType(_) => {}
                ModuleDefId::EnumVariantId(_) => {}
            }
        }

        for impl_id in scope.impls() {
            self.collect_from_impl(impl_id);
        }

        // Record renamed imports.
        // FIXME: In case it imports multiple items under different namespaces we just pick one arbitrarily
        // for now.
        for id in scope.imports() {
            let source = id.import.child_source(self.db.upcast());
            let Some(use_tree_src) = source.value.get(id.idx) else { continue };
            let Some(rename) = use_tree_src.rename() else { continue };
            let Some(name) = rename.name() else { continue };

            let res = scope.fully_resolve_import(self.db.upcast(), id);
            res.iter_items().for_each(|(item, _)| {
                let def = match item {
                    ItemInNs::Types(def) | ItemInNs::Values(def) => def,
                    ItemInNs::Macros(def) => ModuleDefId::from(def),
                }
                .into();
                let dec_loc = DeclarationLocation {
                    hir_file_id: source.file_id,
                    ptr: SyntaxNodePtr::new(use_tree_src.syntax()),
                    name_ptr: AstPtr::new(&name),
                };

                self.symbols.push(FileSymbol {
                    name: name.text().into(),
                    def,
                    container_name: self.current_container_name.clone(),
                    loc: dec_loc,
                    is_alias: false,
                    is_assoc: false,
                });
            });
        }

        for const_id in scope.unnamed_consts() {
            self.collect_from_body(const_id);
        }

        for (_, id) in scope.legacy_macros() {
            for &id in id {
                if id.module(self.db.upcast()) == module_id {
                    match id {
                        MacroId::Macro2Id(id) => self.push_decl(id, false),
                        MacroId::MacroRulesId(id) => self.push_decl(id, false),
                        MacroId::ProcMacroId(id) => self.push_decl(id, false),
                    }
                }
            }
        }
    }

    fn collect_from_body(&mut self, body_id: impl Into<DefWithBodyId>) {
        let body_id = body_id.into();
        let body = self.db.body(body_id);

        // Descend into the blocks and enqueue collection of all modules within.
        for (_, def_map) in body.blocks(self.db.upcast()) {
            for (id, _) in def_map.modules() {
                self.work.push(SymbolCollectorWork {
                    module_id: def_map.module_id(id),
                    parent: Some(body_id),
                });
            }
        }
    }

    fn collect_from_impl(&mut self, impl_id: ImplId) {
        let impl_data = self.db.impl_data(impl_id);
        let impl_name = Some(
            hir_display_with_types_map(impl_data.self_ty, &impl_data.types_map)
                .display(self.db, self.edition)
                .to_smolstr(),
        );
        self.with_container_name(impl_name, |s| {
            for &assoc_item_id in impl_data.items.iter() {
                s.push_assoc_item(assoc_item_id)
            }
        })
    }

    fn collect_from_trait(&mut self, trait_id: TraitId) {
        let trait_data = self.db.trait_data(trait_id);
        self.with_container_name(Some(trait_data.name.as_str().into()), |s| {
            for &(_, assoc_item_id) in &trait_data.items {
                s.push_assoc_item(assoc_item_id);
            }
        });
    }

    fn with_container_name(&mut self, container_name: Option<SmolStr>, f: impl FnOnce(&mut Self)) {
        if let Some(container_name) = container_name {
            let prev = self.current_container_name.replace(container_name);
            f(self);
            self.current_container_name = prev;
        } else {
            f(self);
        }
    }

    fn def_with_body_id_name(&self, body_id: DefWithBodyId) -> Option<SmolStr> {
        match body_id {
            DefWithBodyId::FunctionId(id) => {
                Some(self.db.function_data(id).name.display_no_db(self.edition).to_smolstr())
            }
            DefWithBodyId::StaticId(id) => {
                Some(self.db.static_data(id).name.display_no_db(self.edition).to_smolstr())
            }
            DefWithBodyId::ConstId(id) => {
                Some(self.db.const_data(id).name.as_ref()?.display_no_db(self.edition).to_smolstr())
            }
            DefWithBodyId::VariantId(id) => {
                Some(self.db.enum_variant_data(id).name.display_no_db(self.edition).to_smolstr())
            }
            DefWithBodyId::InTypeConstId(_) => Some("in type const".into()),
        }
    }

    fn push_assoc_item(&mut self, assoc_item_id: AssocItemId) {
        match assoc_item_id {
            AssocItemId::FunctionId(id) => self.push_decl(id, true),
            AssocItemId::ConstId(id) => self.push_decl(id, true),
            AssocItemId::TypeAliasId(id) => self.push_decl(id, true),
        }
    }

    fn push_decl<'db, L>(&mut self, id: L, is_assoc: bool)
    where
        L: Lookup<Database<'db> = dyn DefDatabase + 'db> + Into<ModuleDefId>,
        <L as Lookup>::Data: HasSource,
        <<L as Lookup>::Data as HasSource>::Value: HasName,
    {
        let loc = id.lookup(self.db.upcast());
        let source = loc.source(self.db.upcast());
        let Some(name_node) = source.value.name() else { return };
        let def = ModuleDef::from(id.into());
        let dec_loc = DeclarationLocation {
            hir_file_id: source.file_id,
            ptr: SyntaxNodePtr::new(source.value.syntax()),
            name_ptr: AstPtr::new(&name_node),
        };

        if let Some(attrs) = def.attrs(self.db) {
            for alias in attrs.doc_aliases() {
                self.symbols.push(FileSymbol {
                    name: alias.as_str().into(),
                    def,
                    loc: dec_loc.clone(),
                    container_name: self.current_container_name.clone(),
                    is_alias: true,
                    is_assoc,
                });
            }
        }

        self.symbols.push(FileSymbol {
            name: name_node.text().into(),
            def,
            container_name: self.current_container_name.clone(),
            loc: dec_loc,
            is_alias: false,
            is_assoc,
        });
    }

    fn push_module(&mut self, module_id: ModuleId) {
        let def_map = module_id.def_map(self.db.upcast());
        let module_data = &def_map[module_id.local_id];
        let Some(declaration) = module_data.origin.declaration() else { return };
        let module = declaration.to_node(self.db.upcast());
        let Some(name_node) = module.name() else { return };
        let dec_loc = DeclarationLocation {
            hir_file_id: declaration.file_id,
            ptr: SyntaxNodePtr::new(module.syntax()),
            name_ptr: AstPtr::new(&name_node),
        };

        let def = ModuleDef::Module(module_id.into());

        if let Some(attrs) = def.attrs(self.db) {
            for alias in attrs.doc_aliases() {
                self.symbols.push(FileSymbol {
                    name: alias.as_str().into(),
                    def,
                    loc: dec_loc.clone(),
                    container_name: self.current_container_name.clone(),
                    is_alias: true,
                    is_assoc: false,
                });
            }
        }

        self.symbols.push(FileSymbol {
            name: name_node.text().into(),
            def: ModuleDef::Module(module_id.into()),
            container_name: self.current_container_name.clone(),
            loc: dec_loc,
            is_alias: false,
            is_assoc: false,
        });
    }
}
