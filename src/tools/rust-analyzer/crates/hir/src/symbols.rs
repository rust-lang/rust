//! File symbol extraction.

use base_db::FileRange;
use hir_def::{
    item_tree::ItemTreeNode, src::HasSource, AdtId, AssocItemId, AssocItemLoc, DefWithBodyId,
    HasModule, ImplId, ItemContainerId, Lookup, MacroId, ModuleDefId, ModuleId, TraitId,
};
use hir_expand::{HirFileId, InFile};
use hir_ty::db::HirDatabase;
use syntax::{ast::HasName, AstNode, SmolStr, SyntaxNode, SyntaxNodePtr};

use crate::{Module, Semantics};

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileSymbol {
    pub name: SmolStr,
    pub loc: DeclarationLocation,
    pub kind: FileSymbolKind,
    pub container_name: Option<SmolStr>,
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
    pub fn syntax<DB: HirDatabase>(&self, sema: &Semantics<'_, DB>) -> Option<SyntaxNode> {
        let root = sema.parse_or_expand(self.hir_file_id)?;
        Some(self.ptr.to_node(&root))
    }

    pub fn original_range(&self, db: &dyn HirDatabase) -> Option<FileRange> {
        let node = resolve_node(db, self.hir_file_id, &self.ptr)?;
        Some(node.as_ref().original_file_range(db.upcast()))
    }

    pub fn original_name_range(&self, db: &dyn HirDatabase) -> Option<FileRange> {
        let node = resolve_node(db, self.hir_file_id, &self.name_ptr)?;
        node.as_ref().original_file_range_opt(db.upcast())
    }
}

fn resolve_node(
    db: &dyn HirDatabase,
    file_id: HirFileId,
    ptr: &SyntaxNodePtr,
) -> Option<InFile<SyntaxNode>> {
    let root = db.parse_or_expand(file_id)?;
    let node = ptr.to_node(&root);
    Some(InFile::new(file_id, node))
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum FileSymbolKind {
    Const,
    Enum,
    Function,
    Macro,
    Module,
    Static,
    Struct,
    Trait,
    TraitAlias,
    TypeAlias,
    Union,
}

impl FileSymbolKind {
    pub fn is_type(self: FileSymbolKind) -> bool {
        matches!(
            self,
            FileSymbolKind::Struct
                | FileSymbolKind::Enum
                | FileSymbolKind::Trait
                | FileSymbolKind::TypeAlias
                | FileSymbolKind::Union
        )
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
}

/// Given a [`ModuleId`] and a [`HirDatabase`], use the DefMap for the module's crate to collect
/// all symbols that should be indexed for the given module.
impl<'a> SymbolCollector<'a> {
    pub fn collect(db: &dyn HirDatabase, module: Module) -> Vec<FileSymbol> {
        let mut symbol_collector = SymbolCollector {
            db,
            symbols: Default::default(),
            current_container_name: None,
            // The initial work is the root module we're collecting, additional work will
            // be populated as we traverse the module's definitions.
            work: vec![SymbolCollectorWork { module_id: module.into(), parent: None }],
        };

        while let Some(work) = symbol_collector.work.pop() {
            symbol_collector.do_work(work);
        }

        symbol_collector.symbols
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
                    self.push_decl_assoc(id, FileSymbolKind::Function);
                    self.collect_from_body(id);
                }
                ModuleDefId::AdtId(AdtId::StructId(id)) => {
                    self.push_decl(id, FileSymbolKind::Struct)
                }
                ModuleDefId::AdtId(AdtId::EnumId(id)) => self.push_decl(id, FileSymbolKind::Enum),
                ModuleDefId::AdtId(AdtId::UnionId(id)) => self.push_decl(id, FileSymbolKind::Union),
                ModuleDefId::ConstId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::Const);
                    self.collect_from_body(id);
                }
                ModuleDefId::StaticId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::Static);
                    self.collect_from_body(id);
                }
                ModuleDefId::TraitId(id) => {
                    self.push_decl(id, FileSymbolKind::Trait);
                    self.collect_from_trait(id);
                }
                ModuleDefId::TraitAliasId(id) => {
                    self.push_decl(id, FileSymbolKind::TraitAlias);
                }
                ModuleDefId::TypeAliasId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::TypeAlias);
                }
                ModuleDefId::MacroId(id) => match id {
                    MacroId::Macro2Id(id) => self.push_decl(id, FileSymbolKind::Macro),
                    MacroId::MacroRulesId(id) => self.push_decl(id, FileSymbolKind::Macro),
                    MacroId::ProcMacroId(id) => self.push_decl(id, FileSymbolKind::Macro),
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
                        MacroId::Macro2Id(id) => self.push_decl(id, FileSymbolKind::Macro),
                        MacroId::MacroRulesId(id) => self.push_decl(id, FileSymbolKind::Macro),
                        MacroId::ProcMacroId(id) => self.push_decl(id, FileSymbolKind::Macro),
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

    fn current_container_name(&self) -> Option<SmolStr> {
        self.current_container_name.clone()
    }

    fn def_with_body_id_name(&self, body_id: DefWithBodyId) -> Option<SmolStr> {
        match body_id {
            DefWithBodyId::FunctionId(id) => Some(
                id.lookup(self.db.upcast()).source(self.db.upcast()).value.name()?.text().into(),
            ),
            DefWithBodyId::StaticId(id) => Some(
                id.lookup(self.db.upcast()).source(self.db.upcast()).value.name()?.text().into(),
            ),
            DefWithBodyId::ConstId(id) => Some(
                id.lookup(self.db.upcast()).source(self.db.upcast()).value.name()?.text().into(),
            ),
            DefWithBodyId::VariantId(id) => Some({
                let db = self.db.upcast();
                id.parent.lookup(db).source(db).value.name()?.text().into()
            }),
        }
    }

    fn push_assoc_item(&mut self, assoc_item_id: AssocItemId) {
        match assoc_item_id {
            AssocItemId::FunctionId(id) => self.push_decl_assoc(id, FileSymbolKind::Function),
            AssocItemId::ConstId(id) => self.push_decl_assoc(id, FileSymbolKind::Const),
            AssocItemId::TypeAliasId(id) => self.push_decl_assoc(id, FileSymbolKind::TypeAlias),
        }
    }

    fn push_decl_assoc<L, T>(&mut self, id: L, kind: FileSymbolKind)
    where
        L: Lookup<Data = AssocItemLoc<T>>,
        T: ItemTreeNode,
        <T as ItemTreeNode>::Source: HasName,
    {
        fn container_name(db: &dyn HirDatabase, container: ItemContainerId) -> Option<SmolStr> {
            match container {
                ItemContainerId::ModuleId(module_id) => {
                    let module = Module::from(module_id);
                    module.name(db).and_then(|name| name.as_text())
                }
                ItemContainerId::TraitId(trait_id) => {
                    let trait_data = db.trait_data(trait_id);
                    trait_data.name.as_text()
                }
                ItemContainerId::ImplId(_) | ItemContainerId::ExternBlockId(_) => None,
            }
        }

        self.push_file_symbol(|s| {
            let loc = id.lookup(s.db.upcast());
            let source = loc.source(s.db.upcast());
            let name_node = source.value.name()?;
            let container_name =
                container_name(s.db, loc.container).or_else(|| s.current_container_name());

            Some(FileSymbol {
                name: name_node.text().into(),
                kind,
                container_name,
                loc: DeclarationLocation {
                    hir_file_id: source.file_id,
                    ptr: SyntaxNodePtr::new(source.value.syntax()),
                    name_ptr: SyntaxNodePtr::new(name_node.syntax()),
                },
            })
        })
    }

    fn push_decl<L>(&mut self, id: L, kind: FileSymbolKind)
    where
        L: Lookup,
        <L as Lookup>::Data: HasSource,
        <<L as Lookup>::Data as HasSource>::Value: HasName,
    {
        self.push_file_symbol(|s| {
            let loc = id.lookup(s.db.upcast());
            let source = loc.source(s.db.upcast());
            let name_node = source.value.name()?;

            Some(FileSymbol {
                name: name_node.text().into(),
                kind,
                container_name: s.current_container_name(),
                loc: DeclarationLocation {
                    hir_file_id: source.file_id,
                    ptr: SyntaxNodePtr::new(source.value.syntax()),
                    name_ptr: SyntaxNodePtr::new(name_node.syntax()),
                },
            })
        })
    }

    fn push_module(&mut self, module_id: ModuleId) {
        self.push_file_symbol(|s| {
            let def_map = module_id.def_map(s.db.upcast());
            let module_data = &def_map[module_id.local_id];
            let declaration = module_data.origin.declaration()?;
            let module = declaration.to_node(s.db.upcast());
            let name_node = module.name()?;

            Some(FileSymbol {
                name: name_node.text().into(),
                kind: FileSymbolKind::Module,
                container_name: s.current_container_name(),
                loc: DeclarationLocation {
                    hir_file_id: declaration.file_id,
                    ptr: SyntaxNodePtr::new(module.syntax()),
                    name_ptr: SyntaxNodePtr::new(name_node.syntax()),
                },
            })
        })
    }

    fn push_file_symbol(&mut self, f: impl FnOnce(&Self) -> Option<FileSymbol>) {
        if let Some(file_symbol) = f(self) {
            self.symbols.push(file_symbol);
        }
    }
}
