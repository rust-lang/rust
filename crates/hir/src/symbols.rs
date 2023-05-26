//! File symbol extraction.

use base_db::FileRange;
use hir_def::{
    src::HasSource, AdtId, AssocItemId, DefWithBodyId, HasModule, ImplId, Lookup, MacroId,
    ModuleDefId, ModuleId, TraitId,
};
use hir_expand::{HirFileId, InFile};
use hir_ty::db::HirDatabase;
use syntax::{ast::HasName, AstNode, SmolStr, SyntaxNode, SyntaxNodePtr};

use crate::{Module, ModuleDef, Semantics};

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileSymbol {
    // even though name can be derived from the def, we store it for efficiency
    pub name: SmolStr,
    pub def: ModuleDef,
    pub loc: DeclarationLocation,
    pub container_name: Option<SmolStr>,
    pub is_alias: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclarationLocation {
    /// The file id for both the `ptr` and `name_ptr`.
    pub hir_file_id: HirFileId,
    /// This points to the whole syntax node of the declaration.
    pub ptr: SyntaxNodePtr,
    /// This points to the [`syntax::ast::Name`] identifier of the declaration.
    pub name_ptr: SyntaxNodePtr,
}

impl DeclarationLocation {
    pub fn syntax<DB: HirDatabase>(&self, sema: &Semantics<'_, DB>) -> SyntaxNode {
        let root = sema.parse_or_expand(self.hir_file_id);
        self.ptr.to_node(&root)
    }

    pub fn original_range(&self, db: &dyn HirDatabase) -> FileRange {
        if let Some(file_id) = self.hir_file_id.file_id() {
            // fast path to prevent parsing
            return FileRange { file_id, range: self.ptr.text_range() };
        }
        let node = resolve_node(db, self.hir_file_id, &self.ptr);
        node.as_ref().original_file_range(db.upcast())
    }

    pub fn original_name_range(&self, db: &dyn HirDatabase) -> Option<FileRange> {
        if let Some(file_id) = self.hir_file_id.file_id() {
            // fast path to prevent parsing
            return Some(FileRange { file_id, range: self.name_ptr.text_range() });
        }
        let node = resolve_node(db, self.hir_file_id, &self.name_ptr);
        node.as_ref().original_file_range_opt(db.upcast())
    }
}

fn resolve_node(
    db: &dyn HirDatabase,
    file_id: HirFileId,
    ptr: &SyntaxNodePtr,
) -> InFile<SyntaxNode> {
    let root = db.parse_or_expand(file_id);
    let node = ptr.to_node(&root);
    InFile::new(file_id, node)
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
        }
    }

    pub fn collect(&mut self, module: Module) {
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
                    self.push_decl(id);
                    self.collect_from_body(id);
                }
                ModuleDefId::AdtId(AdtId::StructId(id)) => self.push_decl(id),
                ModuleDefId::AdtId(AdtId::EnumId(id)) => self.push_decl(id),
                ModuleDefId::AdtId(AdtId::UnionId(id)) => self.push_decl(id),
                ModuleDefId::ConstId(id) => {
                    self.push_decl(id);
                    self.collect_from_body(id);
                }
                ModuleDefId::StaticId(id) => {
                    self.push_decl(id);
                    self.collect_from_body(id);
                }
                ModuleDefId::TraitId(id) => {
                    self.push_decl(id);
                    self.collect_from_trait(id);
                }
                ModuleDefId::TraitAliasId(id) => {
                    self.push_decl(id);
                }
                ModuleDefId::TypeAliasId(id) => {
                    self.push_decl(id);
                }
                ModuleDefId::MacroId(id) => match id {
                    MacroId::Macro2Id(id) => self.push_decl(id),
                    MacroId::MacroRulesId(id) => self.push_decl(id),
                    MacroId::ProcMacroId(id) => self.push_decl(id),
                },
                // Don't index these.
                ModuleDefId::BuiltinType(_) => {}
                ModuleDefId::EnumVariantId(_) => {}
            }
        }

        for impl_id in scope.impls() {
            self.collect_from_impl(impl_id);
        }

        for const_id in scope.unnamed_consts() {
            self.collect_from_body(const_id);
        }

        for (_, id) in scope.legacy_macros() {
            for &id in id {
                if id.module(self.db.upcast()) == module_id {
                    match id {
                        MacroId::Macro2Id(id) => self.push_decl(id),
                        MacroId::MacroRulesId(id) => self.push_decl(id),
                        MacroId::ProcMacroId(id) => self.push_decl(id),
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
        for &assoc_item_id in &impl_data.items {
            self.push_assoc_item(assoc_item_id)
        }
    }

    fn collect_from_trait(&mut self, trait_id: TraitId) {
        let trait_data = self.db.trait_data(trait_id);
        self.with_container_name(trait_data.name.as_text(), |s| {
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
            DefWithBodyId::FunctionId(id) => Some(self.db.function_data(id).name.to_smol_str()),
            DefWithBodyId::StaticId(id) => Some(self.db.static_data(id).name.to_smol_str()),
            DefWithBodyId::ConstId(id) => Some(self.db.const_data(id).name.as_ref()?.to_smol_str()),
            DefWithBodyId::VariantId(id) => {
                Some(self.db.enum_data(id.parent).variants[id.local_id].name.to_smol_str())
            }
        }
    }

    fn push_assoc_item(&mut self, assoc_item_id: AssocItemId) {
        match assoc_item_id {
            AssocItemId::FunctionId(id) => self.push_decl(id),
            AssocItemId::ConstId(id) => self.push_decl(id),
            AssocItemId::TypeAliasId(id) => self.push_decl(id),
        }
    }

    fn push_decl<L>(&mut self, id: L)
    where
        L: Lookup + Into<ModuleDefId>,
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
            name_ptr: SyntaxNodePtr::new(name_node.syntax()),
        };

        if let Some(attrs) = def.attrs(self.db) {
            for alias in attrs.doc_aliases() {
                self.symbols.push(FileSymbol {
                    name: alias,
                    def,
                    loc: dec_loc.clone(),
                    container_name: self.current_container_name.clone(),
                    is_alias: true,
                });
            }
        }

        self.symbols.push(FileSymbol {
            name: name_node.text().into(),
            def,
            container_name: self.current_container_name.clone(),
            loc: dec_loc,
            is_alias: false,
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
            name_ptr: SyntaxNodePtr::new(name_node.syntax()),
        };

        let def = ModuleDef::Module(module_id.into());

        if let Some(attrs) = def.attrs(self.db) {
            for alias in attrs.doc_aliases() {
                self.symbols.push(FileSymbol {
                    name: alias,
                    def,
                    loc: dec_loc.clone(),
                    container_name: self.current_container_name.clone(),
                    is_alias: true,
                });
            }
        }

        self.symbols.push(FileSymbol {
            name: name_node.text().into(),
            def: ModuleDef::Module(module_id.into()),
            container_name: self.current_container_name.clone(),
            loc: dec_loc,
            is_alias: false,
        });
    }
}
