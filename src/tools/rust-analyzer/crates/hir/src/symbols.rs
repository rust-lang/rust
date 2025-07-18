//! File symbol extraction.

use base_db::FxIndexSet;
use either::Either;
use hir_def::{
    AdtId, AssocItemId, Complete, DefWithBodyId, ExternCrateId, HasModule, ImplId, Lookup, MacroId,
    ModuleDefId, ModuleId, TraitId,
    db::DefDatabase,
    item_scope::{ImportId, ImportOrExternCrate, ImportOrGlob},
    per_ns::Item,
    src::{HasChildSource, HasSource},
    visibility::{Visibility, VisibilityExplicitness},
};
use hir_expand::{HirFileId, name::Name};
use hir_ty::{
    db::HirDatabase,
    display::{HirDisplay, hir_display_with_store},
};
use intern::Symbol;
use rustc_hash::FxHashMap;
use syntax::{AstNode, AstPtr, SmolStr, SyntaxNode, SyntaxNodePtr, ToSmolStr, ast::HasName};

use crate::{HasCrate, Module, ModuleDef, Semantics};

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileSymbol {
    pub name: Symbol,
    pub def: ModuleDef,
    pub loc: DeclarationLocation,
    pub container_name: Option<SmolStr>,
    /// Whether this symbol is a doc alias for the original symbol.
    pub is_alias: bool,
    pub is_assoc: bool,
    pub is_import: bool,
    pub do_not_complete: Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclarationLocation {
    /// The file id for both the `ptr` and `name_ptr`.
    pub hir_file_id: HirFileId,
    /// This points to the whole syntax node of the declaration.
    pub ptr: SyntaxNodePtr,
    /// This points to the [`syntax::ast::Name`] identifier of the declaration.
    pub name_ptr: AstPtr<Either<syntax::ast::Name, syntax::ast::NameRef>>,
}

impl DeclarationLocation {
    pub fn syntax<DB: HirDatabase>(&self, sema: &Semantics<'_, DB>) -> SyntaxNode {
        let root = sema.parse_or_expand(self.hir_file_id);
        self.ptr.to_node(&root)
    }
}

/// Represents an outstanding module that the symbol collector must collect symbols from.
#[derive(Debug)]
struct SymbolCollectorWork {
    module_id: ModuleId,
    parent: Option<Name>,
}

pub struct SymbolCollector<'a> {
    db: &'a dyn HirDatabase,
    symbols: FxIndexSet<FileSymbol>,
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

    pub fn new_module(db: &dyn HirDatabase, module: Module) -> Box<[FileSymbol]> {
        let mut symbol_collector = SymbolCollector::new(db);
        symbol_collector.collect(module);
        symbol_collector.finish()
    }

    pub fn collect(&mut self, module: Module) {
        let _p = tracing::info_span!("SymbolCollector::collect", ?module).entered();
        tracing::info!(?module, "SymbolCollector::collect");

        // The initial work is the root module we're collecting, additional work will
        // be populated as we traverse the module's definitions.
        self.work.push(SymbolCollectorWork { module_id: module.into(), parent: None });

        while let Some(work) = self.work.pop() {
            self.do_work(work);
        }
    }

    pub fn finish(self) -> Box<[FileSymbol]> {
        self.symbols.into_iter().collect()
    }

    fn do_work(&mut self, work: SymbolCollectorWork) {
        let _p = tracing::info_span!("SymbolCollector::do_work", ?work).entered();
        tracing::info!(?work, "SymbolCollector::do_work");
        self.db.unwind_if_revision_cancelled();

        let parent_name = work.parent.map(|name| name.as_str().to_smolstr());
        self.with_container_name(parent_name, |s| s.collect_from_module(work.module_id));
    }

    fn collect_from_module(&mut self, module_id: ModuleId) {
        let push_decl = |this: &mut Self, def, name| {
            match def {
                ModuleDefId::ModuleId(id) => this.push_module(id, name),
                ModuleDefId::FunctionId(id) => {
                    this.push_decl(id, name, false, None);
                    this.collect_from_body(id, Some(name.clone()));
                }
                ModuleDefId::AdtId(AdtId::StructId(id)) => {
                    this.push_decl(id, name, false, None);
                }
                ModuleDefId::AdtId(AdtId::EnumId(id)) => {
                    this.push_decl(id, name, false, None);
                    let enum_name = this.db.enum_signature(id).name.as_str().to_smolstr();
                    this.with_container_name(Some(enum_name), |this| {
                        let variants = id.enum_variants(this.db);
                        for (variant_id, variant_name, _) in &variants.variants {
                            this.push_decl(*variant_id, variant_name, true, None);
                        }
                    });
                }
                ModuleDefId::AdtId(AdtId::UnionId(id)) => {
                    this.push_decl(id, name, false, None);
                }
                ModuleDefId::ConstId(id) => {
                    this.push_decl(id, name, false, None);
                    this.collect_from_body(id, Some(name.clone()));
                }
                ModuleDefId::StaticId(id) => {
                    this.push_decl(id, name, false, None);
                    this.collect_from_body(id, Some(name.clone()));
                }
                ModuleDefId::TraitId(id) => {
                    let trait_do_not_complete = this.push_decl(id, name, false, None);
                    this.collect_from_trait(id, trait_do_not_complete);
                }
                ModuleDefId::TraitAliasId(id) => {
                    this.push_decl(id, name, false, None);
                }
                ModuleDefId::TypeAliasId(id) => {
                    this.push_decl(id, name, false, None);
                }
                ModuleDefId::MacroId(id) => {
                    match id {
                        MacroId::Macro2Id(id) => this.push_decl(id, name, false, None),
                        MacroId::MacroRulesId(id) => this.push_decl(id, name, false, None),
                        MacroId::ProcMacroId(id) => this.push_decl(id, name, false, None),
                    };
                }
                // Don't index these.
                ModuleDefId::BuiltinType(_) => {}
                ModuleDefId::EnumVariantId(_) => {}
            }
        };

        // Nested trees are very common, so a cache here will hit a lot.
        let import_child_source_cache = &mut FxHashMap::default();

        let is_explicit_import = |vis| match vis {
            Visibility::Public => true,
            Visibility::PubCrate(_) => true,
            Visibility::Module(_, VisibilityExplicitness::Explicit) => true,
            Visibility::Module(_, VisibilityExplicitness::Implicit) => false,
        };

        let mut push_import = |this: &mut Self, i: ImportId, name: &Name, def: ModuleDefId, vis| {
            let source = import_child_source_cache
                .entry(i.use_)
                .or_insert_with(|| i.use_.child_source(this.db));
            let Some(use_tree_src) = source.value.get(i.idx) else { return };
            let rename = use_tree_src.rename().and_then(|rename| rename.name());
            let name_syntax = match rename {
                Some(name) => Some(Either::Left(name)),
                None if is_explicit_import(vis) => {
                    (|| use_tree_src.path()?.segment()?.name_ref().map(Either::Right))()
                }
                None => None,
            };
            let Some(name_syntax) = name_syntax else {
                return;
            };
            let dec_loc = DeclarationLocation {
                hir_file_id: source.file_id,
                ptr: SyntaxNodePtr::new(use_tree_src.syntax()),
                name_ptr: AstPtr::new(&name_syntax),
            };
            this.symbols.insert(FileSymbol {
                name: name.symbol().clone(),
                def: def.into(),
                container_name: this.current_container_name.clone(),
                loc: dec_loc,
                is_alias: false,
                is_assoc: false,
                is_import: true,
                do_not_complete: Complete::Yes,
            });
        };

        let push_extern_crate =
            |this: &mut Self, i: ExternCrateId, name: &Name, def: ModuleDefId, vis| {
                let loc = i.lookup(this.db);
                let source = loc.source(this.db);
                let rename = source.value.rename().and_then(|rename| rename.name());

                let name_syntax = match rename {
                    Some(name) => Some(Either::Left(name)),
                    None if is_explicit_import(vis) => None,
                    None => source.value.name_ref().map(Either::Right),
                };
                let Some(name_syntax) = name_syntax else {
                    return;
                };
                let dec_loc = DeclarationLocation {
                    hir_file_id: source.file_id,
                    ptr: SyntaxNodePtr::new(source.value.syntax()),
                    name_ptr: AstPtr::new(&name_syntax),
                };
                this.symbols.insert(FileSymbol {
                    name: name.symbol().clone(),
                    def: def.into(),
                    container_name: this.current_container_name.clone(),
                    loc: dec_loc,
                    is_alias: false,
                    is_assoc: false,
                    is_import: false,
                    do_not_complete: Complete::Yes,
                });
            };

        let def_map = module_id.def_map(self.db);
        let scope = &def_map[module_id.local_id].scope;

        for impl_id in scope.impls() {
            self.collect_from_impl(impl_id);
        }

        for (name, Item { def, vis, import }) in scope.types() {
            if let Some(i) = import {
                match i {
                    ImportOrExternCrate::Import(i) => push_import(self, i, name, def, vis),
                    ImportOrExternCrate::Glob(_) => (),
                    ImportOrExternCrate::ExternCrate(i) => {
                        push_extern_crate(self, i, name, def, vis)
                    }
                }

                continue;
            }
            // self is a declaration
            push_decl(self, def, name)
        }

        for (name, Item { def, vis, import }) in scope.macros() {
            if let Some(i) = import {
                match i {
                    ImportOrExternCrate::Import(i) => push_import(self, i, name, def.into(), vis),
                    ImportOrExternCrate::Glob(_) => (),
                    ImportOrExternCrate::ExternCrate(_) => (),
                }
                continue;
            }
            // self is a declaration
            push_decl(self, def.into(), name)
        }

        for (name, Item { def, vis, import }) in scope.values() {
            if let Some(i) = import {
                match i {
                    ImportOrGlob::Import(i) => push_import(self, i, name, def, vis),
                    ImportOrGlob::Glob(_) => (),
                }
                continue;
            }
            // self is a declaration
            push_decl(self, def, name)
        }

        for const_id in scope.unnamed_consts() {
            self.collect_from_body(const_id, None);
        }

        for (name, id) in scope.legacy_macros() {
            for &id in id {
                if id.module(self.db) == module_id {
                    match id {
                        MacroId::Macro2Id(id) => self.push_decl(id, name, false, None),
                        MacroId::MacroRulesId(id) => self.push_decl(id, name, false, None),
                        MacroId::ProcMacroId(id) => self.push_decl(id, name, false, None),
                    };
                }
            }
        }
    }

    fn collect_from_body(&mut self, body_id: impl Into<DefWithBodyId>, name: Option<Name>) {
        let body_id = body_id.into();
        let body = self.db.body(body_id);

        // Descend into the blocks and enqueue collection of all modules within.
        for (_, def_map) in body.blocks(self.db) {
            for (id, _) in def_map.modules() {
                self.work.push(SymbolCollectorWork {
                    module_id: def_map.module_id(id),
                    parent: name.clone(),
                });
            }
        }
    }

    fn collect_from_impl(&mut self, impl_id: ImplId) {
        let impl_data = self.db.impl_signature(impl_id);
        let impl_name = Some(
            hir_display_with_store(impl_data.self_ty, &impl_data.store)
                .display(
                    self.db,
                    crate::Impl::from(impl_id).krate(self.db).to_display_target(self.db),
                )
                .to_smolstr(),
        );
        self.with_container_name(impl_name, |s| {
            for &(ref name, assoc_item_id) in &impl_id.impl_items(self.db).items {
                s.push_assoc_item(assoc_item_id, name, None)
            }
        })
    }

    fn collect_from_trait(&mut self, trait_id: TraitId, trait_do_not_complete: Complete) {
        let trait_data = self.db.trait_signature(trait_id);
        self.with_container_name(Some(trait_data.name.as_str().into()), |s| {
            for &(ref name, assoc_item_id) in &trait_id.trait_items(self.db).items {
                s.push_assoc_item(assoc_item_id, name, Some(trait_do_not_complete));
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

    fn push_assoc_item(
        &mut self,
        assoc_item_id: AssocItemId,
        name: &Name,
        trait_do_not_complete: Option<Complete>,
    ) {
        match assoc_item_id {
            AssocItemId::FunctionId(id) => self.push_decl(id, name, true, trait_do_not_complete),
            AssocItemId::ConstId(id) => self.push_decl(id, name, true, trait_do_not_complete),
            AssocItemId::TypeAliasId(id) => self.push_decl(id, name, true, trait_do_not_complete),
        };
    }

    fn push_decl<L>(
        &mut self,
        id: L,
        name: &Name,
        is_assoc: bool,
        trait_do_not_complete: Option<Complete>,
    ) -> Complete
    where
        L: Lookup<Database = dyn DefDatabase> + Into<ModuleDefId>,
        <L as Lookup>::Data: HasSource,
        <<L as Lookup>::Data as HasSource>::Value: HasName,
    {
        let loc = id.lookup(self.db);
        let source = loc.source(self.db);
        let Some(name_node) = source.value.name() else { return Complete::Yes };
        let def = ModuleDef::from(id.into());
        let dec_loc = DeclarationLocation {
            hir_file_id: source.file_id,
            ptr: SyntaxNodePtr::new(source.value.syntax()),
            name_ptr: AstPtr::new(&name_node).wrap_left(),
        };

        let mut do_not_complete = Complete::Yes;

        if let Some(attrs) = def.attrs(self.db) {
            do_not_complete = Complete::extract(matches!(def, ModuleDef::Trait(_)), &attrs);
            if let Some(trait_do_not_complete) = trait_do_not_complete {
                do_not_complete = Complete::for_trait_item(trait_do_not_complete, do_not_complete);
            }

            for alias in attrs.doc_aliases() {
                self.symbols.insert(FileSymbol {
                    name: alias.clone(),
                    def,
                    loc: dec_loc.clone(),
                    container_name: self.current_container_name.clone(),
                    is_alias: true,
                    is_assoc,
                    is_import: false,
                    do_not_complete,
                });
            }
        }

        self.symbols.insert(FileSymbol {
            name: name.symbol().clone(),
            def,
            container_name: self.current_container_name.clone(),
            loc: dec_loc,
            is_alias: false,
            is_assoc,
            is_import: false,
            do_not_complete,
        });

        do_not_complete
    }

    fn push_module(&mut self, module_id: ModuleId, name: &Name) {
        let def_map = module_id.def_map(self.db);
        let module_data = &def_map[module_id.local_id];
        let Some(declaration) = module_data.origin.declaration() else { return };
        let module = declaration.to_node(self.db);
        let Some(name_node) = module.name() else { return };
        let dec_loc = DeclarationLocation {
            hir_file_id: declaration.file_id,
            ptr: SyntaxNodePtr::new(module.syntax()),
            name_ptr: AstPtr::new(&name_node).wrap_left(),
        };

        let def = ModuleDef::Module(module_id.into());

        let mut do_not_complete = Complete::Yes;
        if let Some(attrs) = def.attrs(self.db) {
            do_not_complete = Complete::extract(matches!(def, ModuleDef::Trait(_)), &attrs);

            for alias in attrs.doc_aliases() {
                self.symbols.insert(FileSymbol {
                    name: alias.clone(),
                    def,
                    loc: dec_loc.clone(),
                    container_name: self.current_container_name.clone(),
                    is_alias: true,
                    is_assoc: false,
                    is_import: false,
                    do_not_complete,
                });
            }
        }

        self.symbols.insert(FileSymbol {
            name: name.symbol().clone(),
            def: ModuleDef::Module(module_id.into()),
            container_name: self.current_container_name.clone(),
            loc: dec_loc,
            is_alias: false,
            is_assoc: false,
            is_import: false,
            do_not_complete,
        });
    }
}
