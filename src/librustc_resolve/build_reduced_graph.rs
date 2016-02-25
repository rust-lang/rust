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

use DefModifiers;
use resolve_imports::ImportDirective;
use resolve_imports::ImportDirectiveSubclass::{self, SingleImport, GlobImport};
use Module;
use Namespace::{self, TypeNS, ValueNS};
use {NameBinding, NameBindingKind};
use module_to_string;
use ParentLink::{ModuleParentLink, BlockParentLink};
use Resolver;
use resolve_imports::Shadowable;
use {resolve_error, resolve_struct_error, ResolutionError};

use rustc::middle::cstore::{CrateStore, ChildItem, DlDef, DlField, DlImpl};
use rustc::middle::def::*;
use rustc::middle::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::middle::ty::VariantKind;

use syntax::ast::{Name, NodeId};
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::special_idents;
use syntax::codemap::{Span, DUMMY_SP};

use rustc_front::hir;
use rustc_front::hir::{Block, DeclItem};
use rustc_front::hir::{ForeignItem, ForeignItemFn, ForeignItemStatic};
use rustc_front::hir::{Item, ItemConst, ItemEnum, ItemExternCrate, ItemFn};
use rustc_front::hir::{ItemForeignMod, ItemImpl, ItemMod, ItemStatic, ItemDefaultImpl};
use rustc_front::hir::{ItemStruct, ItemTrait, ItemTy, ItemUse};
use rustc_front::hir::{NamedField, PathListIdent, PathListMod};
use rustc_front::hir::StmtDecl;
use rustc_front::hir::UnnamedField;
use rustc_front::hir::{Variant, ViewPathGlob, ViewPathList, ViewPathSimple};
use rustc_front::hir::Visibility;
use rustc_front::intravisit::{self, Visitor};

use std::mem::replace;
use std::ops::{Deref, DerefMut};

struct GraphBuilder<'a, 'b: 'a, 'tcx: 'b> {
    resolver: &'a mut Resolver<'b, 'tcx>,
}

impl<'a, 'b:'a, 'tcx:'b> Deref for GraphBuilder<'a, 'b, 'tcx> {
    type Target = Resolver<'b, 'tcx>;

    fn deref(&self) -> &Resolver<'b, 'tcx> {
        &*self.resolver
    }
}

impl<'a, 'b:'a, 'tcx:'b> DerefMut for GraphBuilder<'a, 'b, 'tcx> {
    fn deref_mut(&mut self) -> &mut Resolver<'b, 'tcx> {
        &mut *self.resolver
    }
}

trait ToNameBinding<'a> {
    fn to_name_binding(self) -> NameBinding<'a>;
}

impl<'a> ToNameBinding<'a> for (Module<'a>, Span) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding::create_from_module(self.0, Some(self.1))
    }
}

impl<'a> ToNameBinding<'a> for (Def, Span, DefModifiers) {
    fn to_name_binding(self) -> NameBinding<'a> {
        let kind = NameBindingKind::Def(self.0);
        NameBinding { modifiers: self.2, kind: kind, span: Some(self.1) }
    }
}

impl<'a, 'b:'a, 'tcx:'b> GraphBuilder<'a, 'b, 'tcx> {
    /// Constructs the reduced graph for the entire crate.
    fn build_reduced_graph(self, krate: &hir::Crate) {
        let mut visitor = BuildReducedGraphVisitor {
            parent: self.graph_root,
            builder: self,
        };
        intravisit::walk_crate(&mut visitor, krate);
    }

    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined.
    fn try_define<T>(&self, parent: Module<'b>, name: Name, ns: Namespace, def: T)
        where T: ToNameBinding<'b>
    {
        let _ = parent.try_define_child(name, ns, self.new_name_binding(def.to_name_binding()));
    }

    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined;
    /// otherwise, reports an error.
    fn define<T: ToNameBinding<'b>>(&self, parent: Module<'b>, name: Name, ns: Namespace, def: T) {
        let binding = self.new_name_binding(def.to_name_binding());
        let old_binding = match parent.try_define_child(name, ns, binding) {
            Ok(()) => return,
            Err(old_binding) => old_binding,
        };

        let span = binding.span.unwrap_or(DUMMY_SP);
        if !old_binding.is_extern_crate() && !binding.is_extern_crate() {
            // Record an error here by looking up the namespace that had the duplicate
            let ns_str = match ns { TypeNS => "type or module", ValueNS => "value" };
            let resolution_error = ResolutionError::DuplicateDefinition(ns_str, name);
            let mut err = resolve_struct_error(self, span, resolution_error);

            if let Some(sp) = old_binding.span {
                let note = format!("first definition of {} `{}` here", ns_str, name);
                err.span_note(sp, &note);
            }
            err.emit();
        } else if old_binding.is_extern_crate() && binding.is_extern_crate() {
            span_err!(self.session,
                      span,
                      E0259,
                      "an external crate named `{}` has already been imported into this module",
                      name);
        } else {
            span_err!(self.session,
                      span,
                      E0260,
                      "the name `{}` conflicts with an external crate \
                      that has been imported into this module",
                      name);
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

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &Item, parent: Module<'b>) -> Module<'b> {
        let name = item.name;
        let sp = item.span;
        let is_public = item.vis == hir::Public;
        let modifiers = if is_public {
            DefModifiers::PUBLIC
        } else {
            DefModifiers::empty()
        } | DefModifiers::IMPORTABLE;

        match item.node {
            ItemUse(ref view_path) => {
                // Extract and intern the module part of the path. For
                // globs and lists, the path is found directly in the AST;
                // for simple paths we have to munge the path a little.
                let module_path = match view_path.node {
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

                // Build up the import directives.
                let shadowable = item.attrs.iter().any(|attr| {
                    attr.name() == special_idents::prelude_import.name.as_str()
                });
                let shadowable = if shadowable {
                    Shadowable::Always
                } else {
                    Shadowable::Never
                };

                match view_path.node {
                    ViewPathSimple(binding, ref full_path) => {
                        let source_name = full_path.segments.last().unwrap().identifier.name;
                        if source_name.as_str() == "mod" || source_name.as_str() == "self" {
                            resolve_error(self,
                                          view_path.span,
                                          ResolutionError::SelfImportsOnlyAllowedWithin);
                        }

                        let subclass = SingleImport(binding, source_name);
                        self.build_import_directive(parent,
                                                    module_path,
                                                    subclass,
                                                    view_path.span,
                                                    item.id,
                                                    is_public,
                                                    shadowable);
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
                            self.build_import_directive(parent,
                                                        module_path,
                                                        SingleImport(rename, name),
                                                        source_item.span,
                                                        source_item.node.id(),
                                                        is_public,
                                                        shadowable);
                        }
                    }
                    ViewPathGlob(_) => {
                        self.build_import_directive(parent,
                                                    module_path,
                                                    GlobImport,
                                                    view_path.span,
                                                    item.id,
                                                    is_public,
                                                    shadowable);
                    }
                }
                parent
            }

            ItemExternCrate(_) => {
                // n.b. we don't need to look at the path option here, because cstore already
                // did
                if let Some(crate_id) = self.session.cstore.extern_mod_stmt_cnum(item.id) {
                    let def_id = DefId {
                        krate: crate_id,
                        index: CRATE_DEF_INDEX,
                    };
                    self.external_exports.insert(def_id);
                    let parent_link = ModuleParentLink(parent, name);
                    let def = Def::Mod(def_id);
                    let module = self.new_extern_crate_module(parent_link, def, is_public, item.id);
                    self.define(parent, name, TypeNS, (module, sp));

                    if is_public {
                        let export = Export { name: name, def_id: def_id };
                        if let Some(def_id) = parent.def_id() {
                            let node_id = self.resolver.ast_map.as_local_node_id(def_id).unwrap();
                            self.export_map.entry(node_id).or_insert(Vec::new()).push(export);
                        }
                    }

                    self.build_reduced_graph_for_external_crate(module);
                }
                parent
            }

            ItemMod(..) => {
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::Mod(self.ast_map.local_def_id(item.id));
                let module = self.new_module(parent_link, Some(def), false, is_public);
                self.define(parent, name, TypeNS, (module, sp));
                parent.module_children.borrow_mut().insert(item.id, module);
                module
            }

            ItemForeignMod(..) => parent,

            // These items live in the value namespace.
            ItemStatic(_, m, _) => {
                let mutbl = m == hir::MutMutable;
                let def = Def::Static(self.ast_map.local_def_id(item.id), mutbl);
                self.define(parent, name, ValueNS, (def, sp, modifiers));
                parent
            }
            ItemConst(_, _) => {
                let def = Def::Const(self.ast_map.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, sp, modifiers));
                parent
            }
            ItemFn(_, _, _, _, _, _) => {
                let def = Def::Fn(self.ast_map.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, sp, modifiers));
                parent
            }

            // These items live in the type namespace.
            ItemTy(..) => {
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::TyAlias(self.ast_map.local_def_id(item.id));
                let module = self.new_module(parent_link, Some(def), false, is_public);
                self.define(parent, name, TypeNS, (module, sp));
                parent
            }

            ItemEnum(ref enum_definition, _) => {
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::Enum(self.ast_map.local_def_id(item.id));
                let module = self.new_module(parent_link, Some(def), false, is_public);
                self.define(parent, name, TypeNS, (module, sp));

                let variant_modifiers = if is_public {
                    DefModifiers::empty()
                } else {
                    DefModifiers::PRIVATE_VARIANT
                };
                for variant in &(*enum_definition).variants {
                    let item_def_id = self.ast_map.local_def_id(item.id);
                    self.build_reduced_graph_for_variant(variant, item_def_id,
                                                         module, variant_modifiers);
                }
                parent
            }

            // These items live in both the type and value namespaces.
            ItemStruct(ref struct_def, _) => {
                // Define a name in the type namespace.
                let def = Def::Struct(self.ast_map.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, modifiers));

                // If this is a newtype or unit-like struct, define a name
                // in the value namespace as well
                if !struct_def.is_struct() {
                    let def = Def::Struct(self.ast_map.local_def_id(struct_def.id()));
                    self.define(parent, name, ValueNS, (def, sp, modifiers));
                }

                // Record the def ID and fields of this struct.
                let named_fields = struct_def.fields()
                                             .iter()
                                             .filter_map(|f| {
                                                 match f.node.kind {
                                                     NamedField(name, _) => Some(name),
                                                     UnnamedField(_) => None,
                                                 }
                                             })
                                             .collect();
                let item_def_id = self.ast_map.local_def_id(item.id);
                self.structs.insert(item_def_id, named_fields);

                parent
            }

            ItemDefaultImpl(_, _) |
            ItemImpl(..) => parent,

            ItemTrait(_, _, _, ref items) => {
                let def_id = self.ast_map.local_def_id(item.id);

                // Add all the items within to a new module.
                let parent_link = ModuleParentLink(parent, name);
                let def = Def::Trait(def_id);
                let module_parent = self.new_module(parent_link, Some(def), false, is_public);
                self.define(parent, name, TypeNS, (module_parent, sp));

                // Add the names of all the items to the trait info.
                for item in items {
                    let item_def_id = self.ast_map.local_def_id(item.id);
                    let (def, ns) = match item.node {
                        hir::ConstTraitItem(..) => (Def::AssociatedConst(item_def_id), ValueNS),
                        hir::MethodTraitItem(..) => (Def::Method(item_def_id), ValueNS),
                        hir::TypeTraitItem(..) => (Def::AssociatedTy(def_id, item_def_id), TypeNS),
                    };

                    let modifiers = DefModifiers::PUBLIC; // NB: not DefModifiers::IMPORTABLE
                    self.define(module_parent, item.name, ns, (def, item.span, modifiers));

                    self.trait_item_map.insert((item.name, def_id), item_def_id);
                }

                parent
            }
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn build_reduced_graph_for_variant(&mut self,
                                       variant: &Variant,
                                       item_id: DefId,
                                       parent: Module<'b>,
                                       variant_modifiers: DefModifiers) {
        let name = variant.node.name;
        if variant.node.data.is_struct() {
            // Not adding fields for variants as they are not accessed with a self receiver
            let variant_def_id = self.ast_map.local_def_id(variant.node.data.id());
            self.structs.insert(variant_def_id, Vec::new());
        }

        // Variants are always treated as importable to allow them to be glob used.
        // All variants are defined in both type and value namespaces as future-proofing.
        let modifiers = DefModifiers::PUBLIC | DefModifiers::IMPORTABLE | variant_modifiers;
        let def = Def::Variant(item_id, self.ast_map.local_def_id(variant.node.data.id()));

        self.define(parent, name, ValueNS, (def, variant.span, modifiers));
        self.define(parent, name, TypeNS, (def, variant.span, modifiers));
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self,
                                            foreign_item: &ForeignItem,
                                            parent: Module<'b>) {
        let name = foreign_item.name;
        let is_public = foreign_item.vis == hir::Public;
        let modifiers = if is_public {
            DefModifiers::PUBLIC
        } else {
            DefModifiers::empty()
        } | DefModifiers::IMPORTABLE;

        let def = match foreign_item.node {
            ForeignItemFn(..) => {
                Def::Fn(self.ast_map.local_def_id(foreign_item.id))
            }
            ForeignItemStatic(_, m) => {
                Def::Static(self.ast_map.local_def_id(foreign_item.id), m)
            }
        };
        self.define(parent, name, ValueNS, (def, foreign_item.span, modifiers));
    }

    fn build_reduced_graph_for_block(&mut self, block: &Block, parent: Module<'b>) -> Module<'b> {
        if self.block_needs_anonymous_module(block) {
            let block_id = block.id;

            debug!("(building reduced graph for block) creating a new anonymous module for block \
                    {}",
                   block_id);

            let parent_link = BlockParentLink(parent, block_id);
            let new_module = self.new_module(parent_link, None, false, false);
            parent.module_children.borrow_mut().insert(block_id, new_module);
            new_module
        } else {
            parent
        }
    }

    fn handle_external_def(&mut self,
                           def: Def,
                           vis: Visibility,
                           final_ident: &str,
                           name: Name,
                           new_parent: Module<'b>) {
        debug!("(building reduced graph for external crate) building external def {}, priv {:?}",
               final_ident,
               vis);
        let is_public = vis == hir::Public || new_parent.is_trait();

        let mut modifiers = DefModifiers::empty();
        if is_public {
            modifiers = modifiers | DefModifiers::PUBLIC;
        }
        if new_parent.is_normal() {
            modifiers = modifiers | DefModifiers::IMPORTABLE;
        }

        let is_exported = is_public &&
                          match new_parent.def_id() {
            None => true,
            Some(did) => self.external_exports.contains(&did),
        };
        if is_exported {
            self.external_exports.insert(def.def_id());
        }

        match def {
            Def::Mod(_) | Def::ForeignMod(_) | Def::Enum(..) | Def::TyAlias(..) => {
                debug!("(building reduced graph for external crate) building module {} {}",
                       final_ident,
                       is_public);
                let parent_link = ModuleParentLink(new_parent, name);
                let module = self.new_module(parent_link, Some(def), true, is_public);
                self.try_define(new_parent, name, TypeNS, (module, DUMMY_SP));
            }
            Def::Variant(_, variant_id) => {
                debug!("(building reduced graph for external crate) building variant {}",
                       final_ident);
                // Variants are always treated as importable to allow them to be glob used.
                // All variants are defined in both type and value namespaces as future-proofing.
                let modifiers = DefModifiers::PUBLIC | DefModifiers::IMPORTABLE;
                self.try_define(new_parent, name, TypeNS, (def, DUMMY_SP, modifiers));
                self.try_define(new_parent, name, ValueNS, (def, DUMMY_SP, modifiers));
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
                       final_ident);
                self.try_define(new_parent, name, ValueNS, (def, DUMMY_SP, modifiers));
            }
            Def::Trait(def_id) => {
                debug!("(building reduced graph for external crate) building type {}",
                       final_ident);

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

                    if is_exported {
                        self.external_exports.insert(trait_item_def.def_id());
                    }
                }

                let parent_link = ModuleParentLink(new_parent, name);
                let module = self.new_module(parent_link, Some(def), true, is_public);
                self.try_define(new_parent, name, TypeNS, (module, DUMMY_SP));
            }
            Def::AssociatedTy(..) => {
                debug!("(building reduced graph for external crate) building type {}",
                       final_ident);
                self.try_define(new_parent, name, TypeNS, (def, DUMMY_SP, modifiers));
            }
            Def::Struct(def_id)
                if self.session.cstore.tuple_struct_definition_if_ctor(def_id).is_none() => {
                debug!("(building reduced graph for external crate) building type and value for \
                        {}",
                       final_ident);
                self.try_define(new_parent, name, TypeNS, (def, DUMMY_SP, modifiers));
                if let Some(ctor_def_id) = self.session.cstore.struct_ctor_def_id(def_id) {
                    let def = Def::Struct(ctor_def_id);
                    self.try_define(new_parent, name, ValueNS, (def, DUMMY_SP, modifiers));
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
                panic!("didn't expect `{:?}`", def);
            }
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_def(&mut self,
                                                  root: Module<'b>,
                                                  xcdef: ChildItem) {
        match xcdef.def {
            DlDef(def) => {
                // Add the new child item, if necessary.
                match def {
                    Def::ForeignMod(def_id) => {
                        // Foreign modules have no names. Recur and populate
                        // eagerly.
                        for child in self.session.cstore.item_children(def_id) {
                            self.build_reduced_graph_for_external_crate_def(root, child)
                        }
                    }
                    _ => {
                        self.handle_external_def(def,
                                                 xcdef.vis,
                                                 &xcdef.name.as_str(),
                                                 xcdef.name,
                                                 root);
                    }
                }
            }
            DlImpl(_) => {
                debug!("(building reduced graph for external crate) ignoring impl");
            }
            DlField => {
                debug!("(building reduced graph for external crate) ignoring field");
            }
        }
    }

    /// Builds the reduced graph rooted at the given external module.
    fn populate_external_module(&mut self, module: Module<'b>) {
        debug!("(populating external module) attempting to populate {}",
               module_to_string(module));

        let def_id = match module.def_id() {
            None => {
                debug!("(populating external module) ... no def ID!");
                return;
            }
            Some(def_id) => def_id,
        };

        for child in self.session.cstore.item_children(def_id) {
            debug!("(populating external module) ... found ident: {}",
                   child.name);
            self.build_reduced_graph_for_external_crate_def(module, child);
        }
        module.populated.set(true)
    }

    /// Ensures that the reduced graph rooted at the given external module
    /// is built, building it if it is not.
    fn populate_module_if_necessary(&mut self, module: Module<'b>) {
        if !module.populated.get() {
            self.populate_external_module(module)
        }
        assert!(module.populated.get())
    }

    /// Builds the reduced graph rooted at the 'use' directive for an external
    /// crate.
    fn build_reduced_graph_for_external_crate(&mut self, root: Module<'b>) {
        let root_cnum = root.def_id().unwrap().krate;
        for child in self.session.cstore.crate_top_level_items(root_cnum) {
            self.build_reduced_graph_for_external_crate_def(root, child);
        }
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(&mut self,
                              module_: Module<'b>,
                              module_path: Vec<Name>,
                              subclass: ImportDirectiveSubclass,
                              span: Span,
                              id: NodeId,
                              is_public: bool,
                              shadowable: Shadowable) {
        module_.unresolved_imports
               .borrow_mut()
               .push(ImportDirective::new(module_path, subclass, span, id, is_public, shadowable));
        self.unresolved_imports += 1;

        if is_public {
            module_.inc_pub_count();
        }

        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match subclass {
            SingleImport(target, _) => {
                module_.increment_outstanding_references_for(target, ValueNS);
                module_.increment_outstanding_references_for(target, TypeNS);
            }
            GlobImport => {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module_.inc_glob_count();
                if is_public {
                    module_.inc_pub_glob_count();
                }
            }
        }
    }
}

struct BuildReducedGraphVisitor<'a, 'b: 'a, 'tcx: 'b> {
    builder: GraphBuilder<'a, 'b, 'tcx>,
    parent: Module<'b>,
}

impl<'a, 'b, 'v, 'tcx> Visitor<'v> for BuildReducedGraphVisitor<'a, 'b, 'tcx> {
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.builder.resolver.ast_map.expect_item(item.id))
    }

    fn visit_item(&mut self, item: &Item) {
        let p = self.builder.build_reduced_graph_for_item(item, &self.parent);
        let old_parent = replace(&mut self.parent, p);
        intravisit::walk_item(self, item);
        self.parent = old_parent;
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.builder.build_reduced_graph_for_foreign_item(foreign_item, &self.parent);
    }

    fn visit_block(&mut self, block: &Block) {
        let np = self.builder.build_reduced_graph_for_block(block, &self.parent);
        let old_parent = replace(&mut self.parent, np);
        intravisit::walk_block(self, block);
        self.parent = old_parent;
    }
}

pub fn build_reduced_graph(resolver: &mut Resolver, krate: &hir::Crate) {
    GraphBuilder { resolver: resolver }.build_reduced_graph(krate);
}

pub fn populate_module_if_necessary<'a, 'tcx>(resolver: &mut Resolver<'a, 'tcx>,
                                              module: Module<'a>) {
    GraphBuilder { resolver: resolver }.populate_module_if_necessary(module);
}
