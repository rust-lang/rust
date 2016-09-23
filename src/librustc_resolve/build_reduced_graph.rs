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
use {Module, ModuleS, ModuleKind};
use Namespace::{self, TypeNS, ValueNS};
use {NameBinding, NameBindingKind, ToNameBinding};
use Resolver;
use {resolve_error, resolve_struct_error, ResolutionError};

use rustc::hir::def::*;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::hir::map::DefPathData;
use rustc::ty;

use std::cell::Cell;

use syntax::ast::Name;
use syntax::attr;
use syntax::parse::token;

use syntax::ast::{Block, Crate};
use syntax::ast::{ForeignItem, ForeignItemKind, Item, ItemKind};
use syntax::ast::{Mutability, StmtKind, TraitItem, TraitItemKind};
use syntax::ast::{Variant, ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::parse::token::keywords;
use syntax::visit::{self, Visitor};

use syntax_pos::{Span, DUMMY_SP};

impl<'a> ToNameBinding<'a> for (Module<'a>, Span, ty::Visibility) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding { kind: NameBindingKind::Module(self.0), span: self.1, vis: self.2 }
    }
}

impl<'a> ToNameBinding<'a> for (Def, Span, ty::Visibility) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding { kind: NameBindingKind::Def(self.0), span: self.1, vis: self.2 }
    }
}

impl<'b> Resolver<'b> {
    /// Constructs the reduced graph for the entire crate.
    pub fn build_reduced_graph(&mut self, krate: &Crate) {
        visit::walk_crate(&mut BuildReducedGraphVisitor { resolver: self }, krate);
    }

    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined;
    /// otherwise, reports an error.
    fn define<T>(&mut self, parent: Module<'b>, name: Name, ns: Namespace, def: T)
        where T: ToNameBinding<'b>,
    {
        let binding = def.to_name_binding();
        if let Err(old_binding) = self.try_define(parent, name, ns, binding.clone()) {
            self.report_conflict(parent, name, ns, old_binding, &binding);
        }
    }

    fn block_needs_anonymous_module(&mut self, block: &Block) -> bool {
        // If any statements are items, we need to create an anonymous module
        block.stmts.iter().any(|statement| match statement.node {
            StmtKind::Item(_) => true,
            _ => false,
        })
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &Item) {
        self.crate_loader.process_item(item, &self.definitions);

        let parent = self.current_module;
        let name = item.ident.name;
        let sp = item.span;
        let vis = self.resolve_visibility(&item.vis);

        match item.node {
            ItemKind::Use(ref view_path) => {
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

                // Build up the import directives.
                let is_prelude = attr::contains_name(&item.attrs, "prelude_import");

                match view_path.node {
                    ViewPathSimple(binding, ref full_path) => {
                        let source_name = full_path.segments.last().unwrap().identifier.name;
                        if source_name.as_str() == "mod" || source_name.as_str() == "self" {
                            resolve_error(self,
                                          view_path.span,
                                          ResolutionError::SelfImportsOnlyAllowedWithin);
                        }

                        let subclass = ImportDirectiveSubclass::single(binding.name, source_name);
                        let span = view_path.span;
                        self.add_import_directive(module_path, subclass, span, item.id, vis);
                    }
                    ViewPathList(_, ref source_items) => {
                        // Make sure there's at most one `mod` import in the list.
                        let mod_spans = source_items.iter().filter_map(|item| {
                            if item.node.name.name == keywords::SelfValue.name() {
                                Some(item.span)
                            } else {
                                None
                            }
                        }).collect::<Vec<Span>>();

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
                            let node = source_item.node;
                            let (module_path, name, rename) = {
                                if node.name.name != keywords::SelfValue.name() {
                                    let rename = node.rename.unwrap_or(node.name).name;
                                    (module_path.clone(), node.name.name, rename)
                                } else {
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
                                    let rename = node.rename.map(|i| i.name).unwrap_or(name);
                                    (module_path.to_vec(), name, rename)
                                }
                            };
                            let subclass = ImportDirectiveSubclass::single(rename, name);
                            let (span, id) = (source_item.span, source_item.node.id);
                            self.add_import_directive(module_path, subclass, span, id, vis);
                        }
                    }
                    ViewPathGlob(_) => {
                        let subclass = GlobImport {
                            is_prelude: is_prelude,
                            max_vis: Cell::new(ty::Visibility::PrivateExternal),
                        };
                        let span = view_path.span;
                        self.add_import_directive(module_path, subclass, span, item.id, vis);
                    }
                }
            }

            ItemKind::ExternCrate(_) => {
                // n.b. we don't need to look at the path option here, because cstore already
                // did
                if let Some(crate_id) = self.session.cstore.extern_mod_stmt_cnum(item.id) {
                    let def_id = DefId {
                        krate: crate_id,
                        index: CRATE_DEF_INDEX,
                    };
                    let module = self.arenas.alloc_module(ModuleS {
                        extern_crate_id: Some(item.id),
                        populated: Cell::new(false),
                        ..ModuleS::new(Some(parent), ModuleKind::Def(Def::Mod(def_id), name))
                    });
                    self.define(parent, name, TypeNS, (module, sp, vis));

                    self.populate_module_if_necessary(module);
                }
            }

            ItemKind::Mod(..) => {
                let def = Def::Mod(self.definitions.local_def_id(item.id));
                let module = self.arenas.alloc_module(ModuleS {
                    no_implicit_prelude: parent.no_implicit_prelude || {
                        attr::contains_name(&item.attrs, "no_implicit_prelude")
                    },
                    normal_ancestor_id: Some(item.id),
                    ..ModuleS::new(Some(parent), ModuleKind::Def(def, name))
                });
                self.define(parent, name, TypeNS, (module, sp, vis));
                self.module_map.insert(item.id, module);

                // Descend into the module.
                self.current_module = module;
            }

            ItemKind::ForeignMod(..) => {}

            // These items live in the value namespace.
            ItemKind::Static(_, m, _) => {
                let mutbl = m == Mutability::Mutable;
                let def = Def::Static(self.definitions.local_def_id(item.id), mutbl);
                self.define(parent, name, ValueNS, (def, sp, vis));
            }
            ItemKind::Const(..) => {
                let def = Def::Const(self.definitions.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, sp, vis));
            }
            ItemKind::Fn(..) => {
                let def = Def::Fn(self.definitions.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, sp, vis));
            }

            // These items live in the type namespace.
            ItemKind::Ty(..) => {
                let def = Def::TyAlias(self.definitions.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, vis));
            }

            ItemKind::Enum(ref enum_definition, _) => {
                let def = Def::Enum(self.definitions.local_def_id(item.id));
                let module = self.new_module(parent, ModuleKind::Def(def, name), true);
                self.define(parent, name, TypeNS, (module, sp, vis));

                for variant in &(*enum_definition).variants {
                    self.build_reduced_graph_for_variant(variant, module, vis);
                }
            }

            // These items live in both the type and value namespaces.
            ItemKind::Struct(ref struct_def, _) => {
                // Define a name in the type namespace.
                let def = Def::Struct(self.definitions.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, vis));

                // If this is a tuple or unit struct, define a name
                // in the value namespace as well.
                if !struct_def.is_struct() {
                    let def = Def::Struct(self.definitions.local_def_id(struct_def.id()));
                    self.define(parent, name, ValueNS, (def, sp, vis));
                }

                // Record the def ID and fields of this struct.
                let field_names = struct_def.fields().iter().enumerate().map(|(index, field)| {
                    self.resolve_visibility(&field.vis);
                    field.ident.map(|ident| ident.name)
                               .unwrap_or_else(|| token::intern(&index.to_string()))
                }).collect();
                let item_def_id = self.definitions.local_def_id(item.id);
                self.structs.insert(item_def_id, field_names);
            }

            ItemKind::Union(ref vdata, _) => {
                let def = Def::Union(self.definitions.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, vis));

                // Record the def ID and fields of this union.
                let field_names = vdata.fields().iter().enumerate().map(|(index, field)| {
                    self.resolve_visibility(&field.vis);
                    field.ident.map(|ident| ident.name)
                               .unwrap_or_else(|| token::intern(&index.to_string()))
                }).collect();
                let item_def_id = self.definitions.local_def_id(item.id);
                self.structs.insert(item_def_id, field_names);
            }

            ItemKind::DefaultImpl(..) | ItemKind::Impl(..) => {}

            ItemKind::Trait(..) => {
                let def_id = self.definitions.local_def_id(item.id);

                // Add all the items within to a new module.
                let module =
                    self.new_module(parent, ModuleKind::Def(Def::Trait(def_id), name), true);
                self.define(parent, name, TypeNS, (module, sp, vis));
                self.current_module = module;
            }
            ItemKind::Mac(_) => panic!("unexpanded macro in resolve!"),
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn build_reduced_graph_for_variant(&mut self,
                                       variant: &Variant,
                                       parent: Module<'b>,
                                       vis: ty::Visibility) {
        let name = variant.node.name.name;
        if variant.node.data.is_struct() {
            // Not adding fields for variants as they are not accessed with a self receiver
            let variant_def_id = self.definitions.local_def_id(variant.node.data.id());
            self.structs.insert(variant_def_id, Vec::new());
        }

        // Variants are always treated as importable to allow them to be glob used.
        // All variants are defined in both type and value namespaces as future-proofing.
        let def = Def::Variant(self.definitions.local_def_id(variant.node.data.id()));
        self.define(parent, name, ValueNS, (def, variant.span, vis));
        self.define(parent, name, TypeNS, (def, variant.span, vis));
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self, foreign_item: &ForeignItem) {
        let parent = self.current_module;
        let name = foreign_item.ident.name;

        let def = match foreign_item.node {
            ForeignItemKind::Fn(..) => {
                Def::Fn(self.definitions.local_def_id(foreign_item.id))
            }
            ForeignItemKind::Static(_, m) => {
                Def::Static(self.definitions.local_def_id(foreign_item.id), m)
            }
        };
        let vis = self.resolve_visibility(&foreign_item.vis);
        self.define(parent, name, ValueNS, (def, foreign_item.span, vis));
    }

    fn build_reduced_graph_for_block(&mut self, block: &Block) {
        let parent = self.current_module;
        if self.block_needs_anonymous_module(block) {
            let block_id = block.id;

            debug!("(building reduced graph for block) creating a new anonymous module for block \
                    {}",
                   block_id);

            let new_module = self.new_module(parent, ModuleKind::Block(block_id), true);
            self.module_map.insert(block_id, new_module);
            self.current_module = new_module; // Descend into the block.
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_def(&mut self, parent: Module<'b>,
                                                  child: Export) {
        let def_id = child.def_id;
        let name = child.name;

        let def = if let Some(def) = self.session.cstore.describe_def(def_id) {
            def
        } else {
            return;
        };

        let vis = if parent.is_trait() {
            ty::Visibility::Public
        } else {
            self.session.cstore.visibility(def_id)
        };

        match def {
            Def::Mod(_) | Def::Enum(..) => {
                debug!("(building reduced graph for external crate) building module {} {:?}",
                       name, vis);
                let module = self.new_module(parent, ModuleKind::Def(def, name), false);
                let _ = self.try_define(parent, name, TypeNS, (module, DUMMY_SP, vis));
            }
            Def::Variant(variant_id) => {
                debug!("(building reduced graph for external crate) building variant {}", name);
                // Variants are always treated as importable to allow them to be glob used.
                // All variants are defined in both type and value namespaces as future-proofing.
                let _ = self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));
                let _ = self.try_define(parent, name, ValueNS, (def, DUMMY_SP, vis));
                if self.session.cstore.variant_kind(variant_id) == Some(ty::VariantKind::Struct) {
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
                let _ = self.try_define(parent, name, ValueNS, (def, DUMMY_SP, vis));
            }
            Def::Trait(_) => {
                debug!("(building reduced graph for external crate) building type {}", name);

                // If this is a trait, add all the trait item names to the trait
                // info.

                let trait_item_def_ids = self.session.cstore.impl_or_trait_items(def_id);
                for &trait_item_def in &trait_item_def_ids {
                    let trait_item_name =
                        self.session.cstore.def_key(trait_item_def)
                            .disambiguated_data.data.get_opt_name()
                            .expect("opt_item_name returned None for trait");

                    debug!("(building reduced graph for external crate) ... adding trait item \
                            '{}'",
                           trait_item_name);

                    self.trait_item_map.insert((trait_item_name, def_id), false);
                }

                let module = self.new_module(parent, ModuleKind::Def(def, name), false);
                let _ = self.try_define(parent, name, TypeNS, (module, DUMMY_SP, vis));
            }
            Def::TyAlias(..) | Def::AssociatedTy(..) => {
                debug!("(building reduced graph for external crate) building type {}", name);
                let _ = self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));
            }
            Def::Struct(_)
                if self.session.cstore.def_key(def_id).disambiguated_data.data !=
                   DefPathData::StructCtor
                => {
                debug!("(building reduced graph for external crate) building type and value for {}",
                       name);
                let _ = self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));
                if let Some(ctor_def_id) = self.session.cstore.struct_ctor_def_id(def_id) {
                    let def = Def::Struct(ctor_def_id);
                    let _ = self.try_define(parent, name, ValueNS, (def, DUMMY_SP, vis));
                }

                // Record the def ID and fields of this struct.
                let fields = self.session.cstore.struct_field_names(def_id);
                self.structs.insert(def_id, fields);
            }
            Def::Union(_) => {
                let _ = self.try_define(parent, name, TypeNS, (def, DUMMY_SP, vis));

                // Record the def ID and fields of this union.
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

struct BuildReducedGraphVisitor<'a, 'b: 'a> {
    resolver: &'a mut Resolver<'b>,
}

impl<'a, 'b> Visitor for BuildReducedGraphVisitor<'a, 'b> {
    fn visit_item(&mut self, item: &Item) {
        let parent = self.resolver.current_module;
        self.resolver.build_reduced_graph_for_item(item);
        visit::walk_item(self, item);
        self.resolver.current_module = parent;
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.resolver.build_reduced_graph_for_foreign_item(foreign_item);
    }

    fn visit_block(&mut self, block: &Block) {
        let parent = self.resolver.current_module;
        self.resolver.build_reduced_graph_for_block(block);
        visit::walk_block(self, block);
        self.resolver.current_module = parent;
    }

    fn visit_trait_item(&mut self, item: &TraitItem) {
        let parent = self.resolver.current_module;
        let def_id = parent.def_id().unwrap();

        // Add the item to the trait info.
        let item_def_id = self.resolver.definitions.local_def_id(item.id);
        let mut is_static_method = false;
        let (def, ns) = match item.node {
            TraitItemKind::Const(..) => (Def::AssociatedConst(item_def_id), ValueNS),
            TraitItemKind::Method(ref sig, _) => {
                is_static_method = !sig.decl.has_self();
                (Def::Method(item_def_id), ValueNS)
            }
            TraitItemKind::Type(..) => (Def::AssociatedTy(item_def_id), TypeNS),
            TraitItemKind::Macro(_) => panic!("unexpanded macro in resolve!"),
        };

        self.resolver.trait_item_map.insert((item.ident.name, def_id), is_static_method);

        let vis = ty::Visibility::Public;
        self.resolver.define(parent, item.ident.name, ns, (def, item.span, vis));

        self.resolver.current_module = parent.parent.unwrap(); // nearest normal ancestor
        visit::walk_trait_item(self, item);
        self.resolver.current_module = parent;
    }
}
