// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Reduced graph building
//!
//! Here we build the "reduced graph": the graph of the module tree without
//! any imports resolved.

use resolve_imports::ImportDirectiveSubclass::{self, GlobImport};
use Module;
use Namespace::{self, TypeNS, ValueNS};
use {NameBinding, NameBindingKind};
use ParentLink::{ModuleParentLink, BlockParentLink};
use Resolver;
use {resolve_error, resolve_struct_error, ResolutionError};

use rustc::middle::cstore::{CrateStore, ChildItem, DlDef};
use rustc::lint;
use rustc::hir::def::*;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::ty::{self, VariantKind};

use syntax::ast::{Name, NodeId};
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::keywords;
use syntax::codemap::{Span, DUMMY_SP};

use rustc::hir;
use rustc::hir::{Block, DeclItem};
use rustc::hir::{ForeignItem, ForeignItemFn, ForeignItemStatic};
use rustc::hir::{Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn};
use rustc::hir::{ItemForeignMod, ItemImpl, ItemMod, ItemStatic, ItemDefaultImpl};
use rustc::hir::{ItemStruct, ItemTrait, ItemTy, ItemUse};
use rustc::hir::{PathListIdent, PathListMod, StmtDecl};
use rustc::hir::{Variant, ViewPathGlob, ViewPathList, ViewPathSimple};
use rustc::hir::intravisit::{self, Visitor};

trait ToNameBinding<'a> {
    fn to_name_binding(self) -> NameBinding<'a>;
}

impl<'a> ToNameBinding<'a> for (Module<'a>, Span, ty::Visibility) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding { kind: NameBindingKind::Module(self.0), span: Some(self.1), vis: self.2 }
    }
}

impl<'a> ToNameBinding<'a> for (Def, Span, ty::Visibility) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding { kind: NameBindingKind::Def(self.0), span: Some(self.1), vis: self.2 }
    }
}

impl<'b, 'tcx:'b> Resolver<'b, 'tcx> {
    /// Constructs the reduced graph for the entire crate.
    pub fn build_reduced_graph(&mut self, krate: &hir::Crate) {
        let mut visitor = BuildReducedGraphVisitor {
            parent: self.graph_root,
            resolver: self,
        };
        intravisit::walk_crate(&mut visitor, krate);
    }

    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined.
    fn try_define<T>(&self, parent: Module<'b>, name: Name, ns: Namespace, def: T)
        where T: ToNameBinding<'b>
    {
        let _ = parent.try_define_child(name, ns, def.to_name_binding());
    }

    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined;
    /// otherwise, reports an error.
    fn define<T: ToNameBinding<'b>>(&self, parent: Module<'b>, name: Name, ns: Namespace, def: T) {
        let binding = def.to_name_binding();
        if let Err(old_binding) = parent.try_define_child(name, ns, binding.clone()) {
            self.report_conflict(parent, name, ns, old_binding, &binding);
        }
    }

    fn block_needs_anonymous_module(&mut self, block: &Block) -> bool {
        fn is_item(statement: &hir::Stmt) -> bool {
            if let StmtDecl(ref declaration, _) = statement.node {
                if let DeclItem(_) = declaration.node {
                    return true;
                }
            }
            false
        }

        // If any statements are items, we need to create an anonymous module
        block.stmts.iter().any(is_item)
    }

    fn sanity_check_import(&self, view_path: &hir::ViewPath, id: NodeId) {
        let path = match view_path.node {
            ViewPathSimple(_, ref path) |
            ViewPathGlob (ref path) |
            ViewPathList(ref path, _) => path
        };

        // Check for type parameters
        let found_param = path.segments.iter().any(|segment| {
            !segment.parameters.types().is_empty() ||
            !segment.parameters.lifetimes().is_empty() ||
            !segment.parameters.bindings().is_empty()
        });
        if found_param {
            self.session.span_err(path.span, "type or lifetime parameters in import path");
        }

        // Checking for special identifiers in path
        // prevent `self` or `super` at beginning of global path
        if path.global && path.segments.len() > 0 {
            let first = path.segments[0].identifier.name;
            if first == keywords::Super.name() || first == keywords::SelfValue.name() {
                self.session.add_lint(
                    lint::builtin::SUPER_OR_SELF_IN_GLOBAL_PATH, id, path.span,
                    format!("expected identifier, found keyword `{}`", first)
                );
            }
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &Item, parent_ref: &mut Module<'b>) {
        let parent = *parent_ref;
        let name = item.name;
        let sp = item.span;
        self.current_module = parent;
        let vis = self.resolve_visibility(&item.vis);

        match item.node {
            ItemUse(ref view_path) => {
                // Extract and intern the module part of the path. For
                // globs and lists, the path is found directly in the AST;
                // for simple paths we have to munge the path a little.
                let module_path: Vec<Name> = match view_path.node {
                    ViewPathSimple(_, ref full_path) => {
                        full_path.segments
                                 .split_last()
                                 .unwrap()
                                 .1
                                 .iter()
                                 .map(|seg| seg.identifier.name)
                                 .collect()
                    }

                    ViewPathGlob(ref module_ident_path) |
                    ViewPathList(ref module_ident_path, _) => {
                        module_ident_path.segments
                                         .iter()
                                         .map(|seg| seg.identifier.name)
                                         .collect()
                    }
                };

                self.sanity_check_import(view_path, item.id);

                // Build up the import directives.
                let is_prelude = item.attrs.iter().any(|attr| attr.name() == "prelude_import");

                match view_path.node {
                    ViewPathSimple(binding, ref full_path) => {
                        let source_name = full_path.segments.last().unwrap().identifier.name;
                        if source_name.as_str() == "mod" || source_name.as_str() == "self" {
                            resolve_error(self,
                                          view_path.span,
                                          ResolutionError::SelfImportsOnlyAllowedWithin);
                        }

                        let subclass = ImportDirectiveSubclass::single(binding, source_name);
                        let span = view_path.span;
                        parent.add_import_directive(module_path, subclass, span, item.id, vis);
                        self.unresolved_imports += 1;
                    }
                    ViewPathList(_, ref source_items) => {
                        // Make sure there's at most one `mod` import in the list.
                        let mod_spans = source_items.iter()
                                                    .filter_map(|item| {
                                                        match item.node {
                                                            PathListMod { .. } => Some(item.span),
                                                            _ => None,
                                                        }
                                                    })
                                                    .collect::<Vec<Span>>();
                        if mod_spans.len() > 1 {
                            let mut e = resolve_struct_error(self,
                                          mod_spans[0],
                                          ResolutionError::SelfImportCanOnlyAppearOnceInTheList);
                            for other_span in mod_spans.iter().skip(1) {
                                e.span_note(*other_span, "another `self` import appears here");
                            }
                            e.emit();
                        }

                        for source_item in source_items {
                            let (module_path, name, rename) = match source_item.node {
                                PathListIdent { name, rename, .. } =>
                                    (module_path.clone(), name, rename.unwrap_or(name)),
                                PathListMod { rename, .. } => {
                                    let name = match module_path.last() {
                                        Some(name) => *name,
                                        None => {
                                            resolve_error(
                                                self,
                                                source_item.span,
                                                ResolutionError::
                                                SelfImportOnlyInImportListWithNonEmptyPrefix
                                            );
                                            continue;
                                        }
                                    };
                                    let module_path = module_path.split_last().unwrap().1;
                                    let rename = rename.unwrap_or(name);
                                    (module_path.to_vec(), name, rename)
                                }
                            };
                            let subclass = ImportDirectiveSubclass::single(rename, name);
                            let (span, id) = (source_item.span, source_item.node.id());
                            parent.add_import_directive(module_path, subclass, span, id, vis);
                            self.unresolved_imports += 1;
                        }
                    }
                    ViewPathGlob(_) => {
                        let subclass = GlobImport { is_prelude: is_prelude };
                        let span = view_path.span;
                        parent.add_import_directive(module_path, subclass, span, item.id, vis);
                        self.unresolved_imports += 1;
                    }
                }
            }

            ItemExternCrate(_) => {
                // n.b. we don't need to look at the path option here, because cstore already
                // did
                if let Some(crate_id) = self.session.cstore.extern_mod_stmt_cnum(item.id) {
                    let def_id = DefId {
                        krate: crate_id,
                        index: CRATE_DEF_INDEX,
                    };
                    let parent_link = ModuleParentLink(parent, name);
                    let def = Def::Mod(def_id);
                    let module = self.new_extern_crate_module(parent_link, def, item.id);
                    self.define(parent, name, TypeNS, (module, sp, vis));

                    self.build_reduced_graph_for_external_crate(module);
                }
            }

            ItemMod(..) => {
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::Mod(self.ast_map.local_def_id(item.id));
                let module = self.new_module(parent_link, Some(def), false);
                self.define(parent, name, TypeNS, (module, sp, vis));
                self.module_map.insert(item.id, module);
                *parent_ref = module;
            }

            ItemForeignMod(..) => {}

            // These items live in the value namespace.
            ItemStatic(_, m, _) => {
                let mutbl = m == hir::MutMutable;
                let def = Def::Static(self.ast_map.local_def_id(item.id), mutbl);
                self.define(parent, name, ValueNS, (def, sp, vis));
            }
            ItemConst(_, _) => {
                let def = Def::Const(self.ast_map.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, sp, vis));
            }
            ItemFn(_, _, _, _, _, _) => {
                let def = Def::Fn(self.ast_map.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, sp, vis));
            }

            // These items live in the type namespace.
            ItemTy(..) => {
                let def = Def::TyAlias(self.ast_map.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, vis));
            }

            ItemEnum(ref enum_definition, _) => {
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::Enum(self.ast_map.local_def_id(item.id));
                let module = self.new_module(parent_link, Some(def), false);
                self.define(parent, name, TypeNS, (module, sp, vis));

                for variant in &(*enum_definition).variants {
                    let item_def_id = self.ast_map.local_def_id(item.id);
                    self.build_reduced_graph_for_variant(variant, item_def_id, module, vis);
                }
            }

            // These items live in both the type and value namespaces.
            ItemStruct(ref struct_def, _) => {
                // Define a name in the type namespace.
                let def = Def::Struct(self.ast_map.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, vis));

                // If this is a newtype or unit-like struct, define a name
                // in the value namespace as well
                if !struct_def.is_struct() {
                    let def = Def::Struct(self.ast_map.local_def_id(struct_def.id()));
                    self.define(parent, name, ValueNS, (def, sp, vis));
                }

                // Record the def ID and fields of this struct.
                let field_names = struct_def.fields().iter().map(|field| {
                    self.resolve_visibility(&field.vis);
                    field.name
                }).collect();
                let item_def_id = self.ast_map.local_def_id(item.id);
                self.structs.insert(item_def_id, field_names);
            }

            ItemDefaultImpl(_, _) | ItemImpl(..) => {}

            ItemTrait(_, _, _, ref items) => {
                let def_id = self.ast_map.local_def_id(item.id);

                // Add all the items within to a new module.
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::Trait(def_id);
                let module_parent = self.new_module(parent_link, Some(def), false);
                self.define(parent, name, TypeNS, (module_parent, sp, vis));

                // Add the names of all the items to the trait info.
                for item in items {
                    let item_def_id = self.ast_map.local_def_id(item.id);
                    let (def, ns) = match item.node {
                        hir::ConstTraitItem(..) => (Def::AssociatedConst(item_def_id), ValueNS),
                        hir::MethodTraitItem(..) => (Def::Method(item_def_id), ValueNS),
                        hir::TypeTraitItem(..) => (Def::AssociatedTy(def_id, item_def_id), TypeNS),
                    };

                    self.define(module_parent, item.name, ns, (def, item.span, vis));

                    self.trait_item_map.insert((item.name, def_id), item_def_id);
                }
            }
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn build_reduced_graph_for_variant(&mut self,
                                       variant: &Variant,
                                       item_id: DefId,
                                       parent: Module<'b>,
                                       vis: ty::Visibility) {
        let name = variant.node.name;
        if variant.node.data.is_struct() {
            // Not adding fields for variants as they are not accessed with a self receiver
            let variant_def_id = self.ast_map.local_def_id(variant.node.data.id());
            self.structs.insert(variant_def_id, Vec::new());
        }

        // Variants are always treated as importable to allow them to be glob used.
        // All variants are defined in both type and value namespaces as future-proofing.
        let def = Def::Variant(item_id, self.ast_map.local_def_id(variant.node.data.id()));
        self.define(parent, name, ValueNS, (def, variant.span, vis));
        self.define(parent, name, TypeNS, (def, variant.span, vis));
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self,
                                            foreign_item: &ForeignItem,
                                            parent: Module<'b>) {
        let name = foreign_item.name;

        let def = match foreign_item.node {
            ForeignItemFn(..) => {
                Def::Fn(self.ast_map.local_def_id(foreign_item.id))
            }
            ForeignItemStatic(_, m) => {
                Def::Static(self.ast_map.local_def_id(foreign_item.id), m)
            }
        };
        self.current_module = parent;
        let vis = self.resolve_visibility(&foreign_item.vis);
        self.define(parent, name, ValueNS, (def, foreign_item.span, vis));
    }

    fn build_reduced_graph_for_block(&mut self, block: &Block, parent: &mut Module<'b>) {
        if self.block_needs_anonymous_module(block) {
            let block_id = block.id;

            debug!("(building reduced graph for block) creating a new anonymous module for block \
                    {}",
                   block_id);

            let parent_link = BlockParentLink(parent, block_id);
            let new_module = self.new_module(parent_link, None, false);
            self.module_map.insert(block_id, new_module);
            *parent = new_module;
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_def(&mut self, parent: Module<'b>, xcdef: ChildItem) {
        let def = match xcdef.def {
            DlDef(def) => def,
            _ => return,
        };

        if let Def::ForeignMod(def_id) = def {
            // Foreign modules have no names. Recur and populate eagerly.
            for child in self.session.cstore.item_children(def_id) {
                self.build_reduced_graph_for_external_crate_def(parent, child);
            }
            return;
        }

        let name = xcdef.name;
        let vis = if parent.is_trait() { ty::Visibility::Public } else { xcdef.vis };

        match def {
            Def::Mod(_) | Def::ForeignMod(_) | Def::Enum(..) => {
                debug!("(building reduced graph for external crate) building module {} {:?}",
                       name, vis);
                let parent_link = ModuleParentLink(parent, name);
                let module = self.new_module(parent_link, Some(def), true);
                self.try_define(parent, name, TypeNS, (module, DUMMY_SP, vis));
            }
            Def::Variant(_, variant_id) => {
                debug!("(building reduced graph for external crate) building variant {}", name);
                // Variants are always treated as importable to allow them to be glob used.
                // All variants are defined in both type and value namespaces as future-proofing.
                self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));
                self.try_define(parent, name, ValueNS, (def, DUMMY_SP, vis));
                if self.session.cstore.variant_kind(variant_id) == Some(VariantKind::Struct) {
                    // Not adding fields for variants as they are not accessed with a self receiver
                    self.structs.insert(variant_id, Vec::new());
                }
            }
            Def::Fn(..) |
            Def::Static(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::Method(..) => {
                debug!("(building reduced graph for external crate) building value (fn/static) {}",
                       name);
                self.try_define(parent, name, ValueNS, (def, DUMMY_SP, vis));
            }
            Def::Trait(def_id) => {
                debug!("(building reduced graph for external crate) building type {}", name);

                // If this is a trait, add all the trait item names to the trait
                // info.

                let trait_item_def_ids = self.session.cstore.trait_item_def_ids(def_id);
                for trait_item_def in &trait_item_def_ids {
                    let trait_item_name =
                        self.session.cstore.item_name(trait_item_def.def_id());

                    debug!("(building reduced graph for external crate) ... adding trait item \
                            '{}'",
                           trait_item_name);

                    self.trait_item_map.insert((trait_item_name, def_id), trait_item_def.def_id());
                }

                let parent_link = ModuleParentLink(parent, name);
                let module = self.new_module(parent_link, Some(def), true);
                self.try_define(parent, name, TypeNS, (module, DUMMY_SP, vis));
            }
            Def::TyAlias(..) | Def::AssociatedTy(..) => {
                debug!("(building reduced graph for external crate) building type {}", name);
                self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));
            }
            Def::Struct(def_id)
                if self.session.cstore.tuple_struct_definition_if_ctor(def_id).is_none() => {
                debug!("(building reduced graph for external crate) building type and value for {}",
                       name);
                self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));
                if let Some(ctor_def_id) = self.session.cstore.struct_ctor_def_id(def_id) {
                    let def = Def::Struct(ctor_def_id);
                    self.try_define(parent, name, ValueNS, (def, DUMMY_SP, vis));
                }

                // Record the def ID and fields of this struct.
                let fields = self.session.cstore.struct_field_names(def_id);
                self.structs.insert(def_id, fields);
            }
            Def::Struct(..) => {}
            Def::Local(..) |
            Def::PrimTy(..) |
            Def::TyParam(..) |
            Def::Upvar(..) |
            Def::Label(..) |
            Def::SelfTy(..) |
            Def::Err => {
                bug!("didn't expect `{:?}`", def);
            }
        }
    }

    /// Builds the reduced graph rooted at the 'use' directive for an external
    /// crate.
    fn build_reduced_graph_for_external_crate(&mut self, root: Module<'b>) {
        let root_cnum = root.def_id().unwrap().krate;
        for child in self.session.cstore.crate_top_level_items(root_cnum) {
            self.build_reduced_graph_for_external_crate_def(root, child);
        }
    }

    /// Ensures that the reduced graph rooted at the given external module
    /// is built, building it if it is not.
    pub fn populate_module_if_necessary(&mut self, module: Module<'b>) {
        if module.populated.get() { return }
        for child in self.session.cstore.item_children(module.def_id().unwrap()) {
            self.build_reduced_graph_for_external_crate_def(module, child);
        }
        module.populated.set(true)
    }
}

struct BuildReducedGraphVisitor<'a, 'b: 'a, 'tcx: 'b> {
    resolver: &'a mut Resolver<'b, 'tcx>,
    parent: Module<'b>,
}

impl<'a, 'b, 'v, 'tcx> Visitor<'v> for BuildReducedGraphVisitor<'a, 'b, 'tcx> {
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.resolver.ast_map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &Item) {
        let old_parent = self.parent;
        self.resolver.build_reduced_graph_for_item(item, &mut self.parent);
        intravisit::walk_item(self, item);
        self.parent = old_parent;
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.resolver.build_reduced_graph_for_foreign_item(foreign_item, &self.parent);
    }

    fn visit_block(&mut self, block: &Block) {
        let old_parent = self.parent;
        self.resolver.build_reduced_graph_for_block(block, &mut self.parent);
        intravisit::walk_block(self, block);
        self.parent = old_parent;
    }
}
