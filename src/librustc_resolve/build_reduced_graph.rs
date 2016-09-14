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

use macros;
use resolve_imports::ImportDirectiveSubclass::{self, GlobImport};
use {Module, ModuleS, ModuleKind};
use Namespace::{self, TypeNS, ValueNS};
use {NameBinding, NameBindingKind, ToNameBinding};
use Resolver;
use {resolve_error, resolve_struct_error, ResolutionError};

use rustc::middle::cstore::LoadedMacroKind;
use rustc::hir::def::*;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::ty;

use std::cell::Cell;
use std::rc::Rc;

use syntax::ast::Name;
use syntax::attr;
use syntax::parse::token;

use syntax::ast::{self, Block, ForeignItem, ForeignItemKind, Item, ItemKind};
use syntax::ast::{Mutability, StmtKind, TraitItem, TraitItemKind};
use syntax::ast::{Variant, ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::ext::base::{MultiItemModifier, Resolver as SyntaxResolver};
use syntax::ext::hygiene::Mark;
use syntax::feature_gate::{self, emit_feature_err};
use syntax::ext::tt::macro_rules;
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
            StmtKind::Item(_) | StmtKind::Mac(_) => true,
            _ => false,
        })
    }

    fn insert_field_names(&mut self, def_id: DefId, field_names: Vec<Name>) {
        if !field_names.is_empty() {
            self.field_names.insert(def_id, field_names);
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &Item, expansion: Mark) {
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
                // We need to error on `#[macro_use] extern crate` when it isn't at the
                // crate root, because `$crate` won't work properly.
                let is_crate_root = self.current_module.parent.is_none();
                for loaded_macro in self.crate_loader.load_macros(item, is_crate_root) {
                    match loaded_macro.kind {
                        LoadedMacroKind::Def(mut def) => {
                            let name = def.ident.name;
                            if def.use_locally {
                                let ext = macro_rules::compile(&self.session.parse_sess, &def);
                                let shadowing =
                                    self.resolve_macro_name(Mark::root(), name, false).is_some();
                                self.expansion_data[&Mark::root()].module.macros.borrow_mut()
                                    .insert(name, macros::NameBinding {
                                        ext: Rc::new(ext),
                                        expansion: expansion,
                                        shadowing: shadowing,
                                        span: loaded_macro.import_site,
                                    });
                                self.macro_names.insert(name);
                            }
                            if def.export {
                                def.id = self.next_node_id();
                                self.exported_macros.push(def);
                            }
                        }
                        LoadedMacroKind::CustomDerive(name, ext) => {
                            self.insert_custom_derive(&name, ext, item.span);
                        }
                    }
                }
                self.crate_loader.process_item(item, &self.definitions);

                // n.b. we don't need to look at the path option here, because cstore already did
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

            ItemKind::Mod(..) if item.ident == keywords::Invalid.ident() => {} // Crate root

            ItemKind::Mod(..) => {
                let def = Def::Mod(self.definitions.local_def_id(item.id));
                let module = self.arenas.alloc_module(ModuleS {
                    no_implicit_prelude: parent.no_implicit_prelude || {
                        attr::contains_name(&item.attrs, "no_implicit_prelude")
                    },
                    normal_ancestor_id: Some(item.id),
                    macros_escape: self.contains_macro_use(&item.attrs),
                    ..ModuleS::new(Some(parent), ModuleKind::Def(def, name))
                });
                self.define(parent, name, TypeNS, (module, sp, vis));
                self.module_map.insert(item.id, module);

                // Descend into the module.
                self.current_module = module;
            }

            ItemKind::ForeignMod(..) => self.crate_loader.process_item(item, &self.definitions),

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
                    let ctor_def = Def::StructCtor(self.definitions.local_def_id(struct_def.id()),
                                                   CtorKind::from_ast(struct_def));
                    self.define(parent, name, ValueNS, (ctor_def, sp, vis));
                }

                // Record field names for error reporting.
                let field_names = struct_def.fields().iter().filter_map(|field| {
                    self.resolve_visibility(&field.vis);
                    field.ident.map(|ident| ident.name)
                }).collect();
                let item_def_id = self.definitions.local_def_id(item.id);
                self.insert_field_names(item_def_id, field_names);
            }

            ItemKind::Union(ref vdata, _) => {
                let def = Def::Union(self.definitions.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, sp, vis));

                // Record field names for error reporting.
                let field_names = vdata.fields().iter().filter_map(|field| {
                    self.resolve_visibility(&field.vis);
                    field.ident.map(|ident| ident.name)
                }).collect();
                let item_def_id = self.definitions.local_def_id(item.id);
                self.insert_field_names(item_def_id, field_names);
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
        let def_id = self.definitions.local_def_id(variant.node.data.id());

        // Define a name in the type namespace.
        let def = Def::Variant(def_id);
        self.define(parent, name, TypeNS, (def, variant.span, vis));

        // Define a constructor name in the value namespace.
        // Braced variants, unlike structs, generate unusable names in
        // value namespace, they are reserved for possible future use.
        let ctor_kind = CtorKind::from_ast(&variant.node.data);
        let ctor_def = Def::VariantCtor(def_id, ctor_kind);
        self.define(parent, name, ValueNS, (ctor_def, variant.span, vis));
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
        let name = child.name;
        let def = child.def;
        let def_id = def.def_id();
        let vis = if parent.is_trait() {
            ty::Visibility::Public
        } else {
            self.session.cstore.visibility(def_id)
        };

        match def {
            Def::Mod(..) | Def::Enum(..) => {
                let module = self.new_module(parent, ModuleKind::Def(def, name), false);
                self.define(parent, name, TypeNS, (module, DUMMY_SP, vis));
            }
            Def::Variant(..) => {
                self.define(parent, name, TypeNS, (def, DUMMY_SP, vis));
            }
            Def::VariantCtor(..) => {
                self.define(parent, name, ValueNS, (def, DUMMY_SP, vis));
            }
            Def::Fn(..) |
            Def::Static(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::Method(..) => {
                self.define(parent, name, ValueNS, (def, DUMMY_SP, vis));
            }
            Def::Trait(..) => {
                let module = self.new_module(parent, ModuleKind::Def(def, name), false);
                self.define(parent, name, TypeNS, (module, DUMMY_SP, vis));

                // If this is a trait, add all the trait item names to the trait info.
                let trait_item_def_ids = self.session.cstore.impl_or_trait_items(def_id);
                for trait_item_def_id in trait_item_def_ids {
                    let trait_item_name = self.session.cstore.def_key(trait_item_def_id)
                                              .disambiguated_data.data.get_opt_name()
                                              .expect("opt_item_name returned None for trait");
                    self.trait_item_map.insert((trait_item_name, def_id), false);
                }
            }
            Def::TyAlias(..) | Def::AssociatedTy(..) => {
                self.define(parent, name, TypeNS, (def, DUMMY_SP, vis));
            }
            Def::Struct(..) => {
                self.define(parent, name, TypeNS, (def, DUMMY_SP, vis));

                // Record field names for error reporting.
                let field_names = self.session.cstore.struct_field_names(def_id);
                self.insert_field_names(def_id, field_names);
            }
            Def::StructCtor(..) => {
                self.define(parent, name, ValueNS, (def, DUMMY_SP, vis));
            }
            Def::Union(..) => {
                self.define(parent, name, TypeNS, (def, DUMMY_SP, vis));

                // Record field names for error reporting.
                let field_names = self.session.cstore.struct_field_names(def_id);
                self.insert_field_names(def_id, field_names);
            }
            Def::Local(..) |
            Def::PrimTy(..) |
            Def::TyParam(..) |
            Def::Upvar(..) |
            Def::Label(..) |
            Def::SelfTy(..) |
            Def::Err => {
                bug!("unexpected definition: {:?}", def);
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

    // does this attribute list contain "macro_use"?
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            if attr.check_name("macro_escape") {
                let msg = "macro_escape is a deprecated synonym for macro_use";
                let mut err = self.session.struct_span_warn(attr.span, msg);
                if let ast::AttrStyle::Inner = attr.node.style {
                    err.help("consider an outer attribute, #[macro_use] mod ...").emit();
                } else {
                    err.emit();
                }
            } else if !attr.check_name("macro_use") {
                continue;
            }

            if !attr.is_word() {
                self.session.span_err(attr.span, "arguments to macro_use are not allowed here");
            }
            return true;
        }

        false
    }

    fn insert_custom_derive(&mut self, name: &str, ext: Rc<MultiItemModifier>, sp: Span) {
        if !self.session.features.borrow().rustc_macro {
            let sess = &self.session.parse_sess;
            let msg = "loading custom derive macro crates is experimentally supported";
            emit_feature_err(sess, "rustc_macro", sp, feature_gate::GateIssue::Language, msg);
        }
        if self.derive_modes.insert(token::intern(name), ext).is_some() {
            self.session.span_err(sp, &format!("cannot shadow existing derive mode `{}`", name));
        }
    }
}

pub struct BuildReducedGraphVisitor<'a, 'b: 'a> {
    pub resolver: &'a mut Resolver<'b>,
    pub expansion: Mark,
}

impl<'a, 'b> BuildReducedGraphVisitor<'a, 'b> {
    fn visit_invoc(&mut self, id: ast::NodeId) {
        self.resolver.expansion_data.get_mut(&Mark::from_placeholder_id(id)).unwrap().module =
            self.resolver.current_module;
    }
}

macro_rules! method {
    ($visit:ident: $ty:ty, $invoc:path, $walk:ident) => {
        fn $visit(&mut self, node: &$ty) {
            match node.node {
                $invoc(..) => self.visit_invoc(node.id),
                _ => visit::$walk(self, node),
            }
        }
    }
}

impl<'a, 'b> Visitor for BuildReducedGraphVisitor<'a, 'b> {
    method!(visit_impl_item: ast::ImplItem, ast::ImplItemKind::Macro, walk_impl_item);
    method!(visit_stmt:      ast::Stmt,     ast::StmtKind::Mac,       walk_stmt);
    method!(visit_expr:      ast::Expr,     ast::ExprKind::Mac,       walk_expr);
    method!(visit_pat:       ast::Pat,      ast::PatKind::Mac,        walk_pat);
    method!(visit_ty:        ast::Ty,       ast::TyKind::Mac,         walk_ty);

    fn visit_item(&mut self, item: &Item) {
        match item.node {
            ItemKind::Mac(..) if item.id == ast::DUMMY_NODE_ID => return, // Scope placeholder
            ItemKind::Mac(..) => return self.visit_invoc(item.id),
            _ => {}
        }

        let parent = self.resolver.current_module;
        self.resolver.build_reduced_graph_for_item(item, self.expansion);
        visit::walk_item(self, item);
        self.resolver.current_module = parent;
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.resolver.build_reduced_graph_for_foreign_item(foreign_item);
        visit::walk_foreign_item(self, foreign_item);
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

        if let TraitItemKind::Macro(_) = item.node {
            return self.visit_invoc(item.id);
        }

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
            TraitItemKind::Macro(_) => bug!(),  // handled above
        };

        self.resolver.trait_item_map.insert((item.ident.name, def_id), is_static_method);

        let vis = ty::Visibility::Public;
        self.resolver.define(parent, item.ident.name, ns, (def, item.span, vis));

        self.resolver.current_module = parent.parent.unwrap(); // nearest normal ancestor
        visit::walk_trait_item(self, item);
        self.resolver.current_module = parent;
    }
}
