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

use macros::{InvocationData, LegacyScope};
use resolve_imports::ImportDirective;
use resolve_imports::ImportDirectiveSubclass::{self, GlobImport, SingleImport};
use {Resolver, Module, ModuleS, ModuleKind, NameBinding, NameBindingKind, ToNameBinding};
use Namespace::{self, TypeNS, ValueNS, MacroNS};
use ResolveResult::Success;
use {resolve_error, resolve_struct_error, ResolutionError};

use rustc::middle::cstore::{DepKind, LoadedMacro};
use rustc::hir::def::*;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId};
use rustc::ty;

use std::cell::Cell;
use std::rc::Rc;

use syntax::ast::Name;
use syntax::attr;

use syntax::ast::{self, Block, ForeignItem, ForeignItemKind, Item, ItemKind};
use syntax::ast::{Mutability, StmtKind, TraitItem, TraitItemKind};
use syntax::ast::{Variant, ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::ext::base::SyntaxExtension;
use syntax::ext::base::Determinacy::Undetermined;
use syntax::ext::expand::mark_tts;
use syntax::ext::hygiene::Mark;
use syntax::ext::tt::macro_rules;
use syntax::symbol::keywords;
use syntax::visit::{self, Visitor};

use syntax_pos::{Span, DUMMY_SP};

impl<'a> ToNameBinding<'a> for (Module<'a>, ty::Visibility, Span, Mark) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding {
            kind: NameBindingKind::Module(self.0),
            vis: self.1,
            span: self.2,
            expansion: self.3,
        }
    }
}

impl<'a> ToNameBinding<'a> for (Def, ty::Visibility, Span, Mark) {
    fn to_name_binding(self) -> NameBinding<'a> {
        NameBinding {
            kind: NameBindingKind::Def(self.0),
            vis: self.1,
            span: self.2,
            expansion: self.3,
        }
    }
}

#[derive(Default, PartialEq, Eq)]
struct LegacyMacroImports {
    import_all: Option<Span>,
    imports: Vec<(Name, Span)>,
    reexports: Vec<(Name, Span)>,
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
                let module_path: Vec<_> = match view_path.node {
                    ViewPathSimple(_, ref full_path) => {
                        full_path.segments
                                 .split_last()
                                 .unwrap()
                                 .1
                                 .iter()
                                 .map(|seg| seg.identifier)
                                 .collect()
                    }

                    ViewPathGlob(ref module_ident_path) |
                    ViewPathList(ref module_ident_path, _) => {
                        module_ident_path.segments
                                         .iter()
                                         .map(|seg| seg.identifier)
                                         .collect()
                    }
                };

                // Build up the import directives.
                let is_prelude = attr::contains_name(&item.attrs, "prelude_import");

                match view_path.node {
                    ViewPathSimple(binding, ref full_path) => {
                        let mut source = full_path.segments.last().unwrap().identifier;
                        let source_name = source.name;
                        if source_name == "mod" || source_name == "self" {
                            resolve_error(self,
                                          view_path.span,
                                          ResolutionError::SelfImportsOnlyAllowedWithin);
                        } else if source_name == "$crate" && full_path.segments.len() == 1 {
                            let crate_root = self.resolve_crate_var(source.ctxt);
                            let crate_name = match crate_root.kind {
                                ModuleKind::Def(_, name) => name,
                                ModuleKind::Block(..) => unreachable!(),
                            };
                            source.name = crate_name;

                            self.session.struct_span_warn(item.span, "`$crate` may not be imported")
                                .note("`use $crate;` was erroneously allowed and \
                                       will become a hard error in a future release")
                                .emit();
                        }

                        let subclass = SingleImport {
                            target: binding.name,
                            source: source.name,
                            result: self.per_ns(|_, _| Cell::new(Err(Undetermined))),
                        };
                        self.add_import_directive(
                            module_path, subclass, view_path.span, item.id, vis, expansion,
                        );
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
                                        Some(ident) => ident.name,
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
                            let subclass = SingleImport {
                                target: rename,
                                source: name,
                                result: self.per_ns(|_, _| Cell::new(Err(Undetermined))),
                            };
                            let id = source_item.node.id;
                            self.add_import_directive(
                                module_path, subclass, source_item.span, id, vis, expansion,
                            );
                        }
                    }
                    ViewPathGlob(_) => {
                        let subclass = GlobImport {
                            is_prelude: is_prelude,
                            max_vis: Cell::new(ty::Visibility::PrivateExternal),
                        };
                        self.add_import_directive(
                            module_path, subclass, view_path.span, item.id, vis, expansion,
                        );
                    }
                }
            }

            ItemKind::ExternCrate(_) => {
                self.crate_loader.process_item(item, &self.definitions);

                // n.b. we don't need to look at the path option here, because cstore already did
                let crate_id = self.session.cstore.extern_mod_stmt_cnum(item.id).unwrap();
                let module = self.get_extern_crate_root(crate_id);
                let binding = (module, ty::Visibility::Public, sp, expansion).to_name_binding();
                let binding = self.arenas.alloc_name_binding(binding);
                let directive = self.arenas.alloc_import_directive(ImportDirective {
                    id: item.id,
                    parent: parent,
                    imported_module: Cell::new(Some(module)),
                    subclass: ImportDirectiveSubclass::ExternCrate,
                    span: item.span,
                    module_path: Vec::new(),
                    vis: Cell::new(vis),
                    expansion: expansion,
                });
                let imported_binding = self.import(binding, directive);
                self.define(parent, name, TypeNS, imported_binding);
                self.populate_module_if_necessary(module);
                self.process_legacy_macro_imports(item, module, expansion);
            }

            ItemKind::Mod(..) if item.ident == keywords::Invalid.ident() => {} // Crate root

            ItemKind::Mod(..) => {
                let def = Def::Mod(self.definitions.local_def_id(item.id));
                let module = self.arenas.alloc_module(ModuleS {
                    no_implicit_prelude: parent.no_implicit_prelude || {
                        attr::contains_name(&item.attrs, "no_implicit_prelude")
                    },
                    normal_ancestor_id: Some(item.id),
                    ..ModuleS::new(Some(parent), ModuleKind::Def(def, name))
                });
                self.define(parent, name, TypeNS, (module, vis, sp, expansion));
                self.module_map.insert(item.id, module);

                // Descend into the module.
                self.current_module = module;
            }

            ItemKind::ForeignMod(..) => self.crate_loader.process_item(item, &self.definitions),

            // These items live in the value namespace.
            ItemKind::Static(_, m, _) => {
                let mutbl = m == Mutability::Mutable;
                let def = Def::Static(self.definitions.local_def_id(item.id), mutbl);
                self.define(parent, name, ValueNS, (def, vis, sp, expansion));
            }
            ItemKind::Const(..) => {
                let def = Def::Const(self.definitions.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, vis, sp, expansion));
            }
            ItemKind::Fn(..) => {
                let def = Def::Fn(self.definitions.local_def_id(item.id));
                self.define(parent, name, ValueNS, (def, vis, sp, expansion));
            }

            // These items live in the type namespace.
            ItemKind::Ty(..) => {
                let def = Def::TyAlias(self.definitions.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, vis, sp, expansion));
            }

            ItemKind::Enum(ref enum_definition, _) => {
                let def = Def::Enum(self.definitions.local_def_id(item.id));
                let module = self.new_module(parent, ModuleKind::Def(def, name), true);
                self.define(parent, name, TypeNS, (module, vis, sp, expansion));

                for variant in &(*enum_definition).variants {
                    self.build_reduced_graph_for_variant(variant, module, vis, expansion);
                }
            }

            // These items live in both the type and value namespaces.
            ItemKind::Struct(ref struct_def, _) => {
                // Define a name in the type namespace.
                let def = Def::Struct(self.definitions.local_def_id(item.id));
                self.define(parent, name, TypeNS, (def, vis, sp, expansion));

                // If this is a tuple or unit struct, define a name
                // in the value namespace as well.
                if !struct_def.is_struct() {
                    let ctor_def = Def::StructCtor(self.definitions.local_def_id(struct_def.id()),
                                                   CtorKind::from_ast(struct_def));
                    self.define(parent, name, ValueNS, (ctor_def, vis, sp, expansion));
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
                self.define(parent, name, TypeNS, (def, vis, sp, expansion));

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
                self.define(parent, name, TypeNS, (module, vis, sp, expansion));
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
                                       vis: ty::Visibility,
                                       expansion: Mark) {
        let name = variant.node.name.name;
        let def_id = self.definitions.local_def_id(variant.node.data.id());

        // Define a name in the type namespace.
        let def = Def::Variant(def_id);
        self.define(parent, name, TypeNS, (def, vis, variant.span, expansion));

        // Define a constructor name in the value namespace.
        // Braced variants, unlike structs, generate unusable names in
        // value namespace, they are reserved for possible future use.
        let ctor_kind = CtorKind::from_ast(&variant.node.data);
        let ctor_def = Def::VariantCtor(def_id, ctor_kind);
        self.define(parent, name, ValueNS, (ctor_def, vis, variant.span, expansion));
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self, item: &ForeignItem, expansion: Mark) {
        let parent = self.current_module;
        let name = item.ident.name;

        let def = match item.node {
            ForeignItemKind::Fn(..) => {
                Def::Fn(self.definitions.local_def_id(item.id))
            }
            ForeignItemKind::Static(_, m) => {
                Def::Static(self.definitions.local_def_id(item.id), m)
            }
        };
        let vis = self.resolve_visibility(&item.vis);
        self.define(parent, name, ValueNS, (def, vis, item.span, expansion));
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
    fn build_reduced_graph_for_external_crate_def(&mut self, parent: Module<'b>, child: Export) {
        let name = child.name;
        let def = child.def;
        let def_id = def.def_id();
        let vis = match def {
            Def::Macro(..) => ty::Visibility::Public,
            _ if parent.is_trait() => ty::Visibility::Public,
            _ => self.session.cstore.visibility(def_id),
        };

        match def {
            Def::Mod(..) | Def::Enum(..) => {
                let module = self.new_module(parent, ModuleKind::Def(def, name), false);
                self.define(parent, name, TypeNS, (module, vis, DUMMY_SP, Mark::root()));
            }
            Def::Variant(..) => {
                self.define(parent, name, TypeNS, (def, vis, DUMMY_SP, Mark::root()));
            }
            Def::VariantCtor(..) => {
                self.define(parent, name, ValueNS, (def, vis, DUMMY_SP, Mark::root()));
            }
            Def::Fn(..) |
            Def::Static(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::Method(..) => {
                self.define(parent, name, ValueNS, (def, vis, DUMMY_SP, Mark::root()));
            }
            Def::Trait(..) => {
                let module = self.new_module(parent, ModuleKind::Def(def, name), false);
                self.define(parent, name, TypeNS, (module, vis, DUMMY_SP, Mark::root()));

                // If this is a trait, add all the trait item names to the trait info.
                let trait_item_def_ids = self.session.cstore.associated_item_def_ids(def_id);
                for trait_item_def_id in trait_item_def_ids {
                    let trait_item_name = self.session.cstore.def_key(trait_item_def_id)
                                              .disambiguated_data.data.get_opt_name()
                                              .expect("opt_item_name returned None for trait");
                    self.trait_item_map.insert((trait_item_name, def_id), false);
                }
            }
            Def::TyAlias(..) | Def::AssociatedTy(..) => {
                self.define(parent, name, TypeNS, (def, vis, DUMMY_SP, Mark::root()));
            }
            Def::Struct(..) => {
                self.define(parent, name, TypeNS, (def, vis, DUMMY_SP, Mark::root()));

                // Record field names for error reporting.
                let field_names = self.session.cstore.struct_field_names(def_id);
                self.insert_field_names(def_id, field_names);
            }
            Def::StructCtor(..) => {
                self.define(parent, name, ValueNS, (def, vis, DUMMY_SP, Mark::root()));
            }
            Def::Union(..) => {
                self.define(parent, name, TypeNS, (def, vis, DUMMY_SP, Mark::root()));

                // Record field names for error reporting.
                let field_names = self.session.cstore.struct_field_names(def_id);
                self.insert_field_names(def_id, field_names);
            }
            Def::Macro(..) => {
                self.define(parent, name, MacroNS, (def, vis, DUMMY_SP, Mark::root()));
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

    fn get_extern_crate_root(&mut self, cnum: CrateNum) -> Module<'b> {
        let def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
        let macros_only = self.session.cstore.dep_kind(cnum) == DepKind::MacrosOnly;
        let arenas = self.arenas;
        *self.extern_crate_roots.entry((cnum, macros_only)).or_insert_with(|| {
            arenas.alloc_module(ModuleS {
                populated: Cell::new(false),
                ..ModuleS::new(None, ModuleKind::Def(Def::Mod(def_id), keywords::Invalid.name()))
            })
        })
    }

    pub fn get_macro(&mut self, binding: &'b NameBinding<'b>) -> Rc<SyntaxExtension> {
        let def_id = match binding.kind {
            NameBindingKind::Def(Def::Macro(def_id)) => def_id,
            NameBindingKind::Import { binding, .. } => return self.get_macro(binding),
            NameBindingKind::Ambiguity { b1, .. } => return self.get_macro(b1),
            _ => panic!("Expected Def::Macro(..)"),
        };
        if let Some(ext) = self.macro_map.get(&def_id) {
            return ext.clone();
        }

        let mut macro_rules = match self.session.cstore.load_macro(def_id, &self.session) {
            LoadedMacro::MacroRules(macro_rules) => macro_rules,
            LoadedMacro::ProcMacro(ext) => return ext,
        };

        let mark = Mark::fresh();
        let invocation = self.arenas.alloc_invocation_data(InvocationData {
            module: Cell::new(self.get_extern_crate_root(def_id.krate)),
            def_index: CRATE_DEF_INDEX,
            const_integer: false,
            legacy_scope: Cell::new(LegacyScope::Empty),
            expansion: Cell::new(LegacyScope::Empty),
        });
        self.invocations.insert(mark, invocation);
        macro_rules.body = mark_tts(&macro_rules.body, mark);
        let ext = Rc::new(macro_rules::compile(&self.session.parse_sess, &macro_rules));
        self.macro_map.insert(def_id, ext.clone());
        ext
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

    fn legacy_import_macro(&mut self,
                           name: Name,
                           binding: &'b NameBinding<'b>,
                           span: Span,
                           allow_shadowing: bool) {
        self.used_crates.insert(binding.def().def_id().krate);
        self.macro_names.insert(name);
        if self.builtin_macros.insert(name, binding).is_some() && !allow_shadowing {
            let msg = format!("`{}` is already in scope", name);
            let note =
                "macro-expanded `#[macro_use]`s may not shadow existing macros (see RFC 1560)";
            self.session.struct_span_err(span, &msg).note(note).emit();
        }
    }

    fn process_legacy_macro_imports(&mut self, item: &Item, module: Module<'b>, expansion: Mark) {
        let allow_shadowing = expansion == Mark::root();
        let legacy_imports = self.legacy_macro_imports(&item.attrs);
        let cnum = module.def_id().unwrap().krate;

        // `#[macro_use]` and `#[macro_reexport]` are only allowed at the crate root.
        if self.current_module.parent.is_some() && legacy_imports != LegacyMacroImports::default() {
            span_err!(self.session, item.span, E0468,
                      "an `extern crate` loading macros must be at the crate root");
        } else if self.session.cstore.dep_kind(cnum) == DepKind::MacrosOnly &&
                  legacy_imports == LegacyMacroImports::default() {
            let msg = "custom derive crates and `#[no_link]` crates have no effect without \
                       `#[macro_use]`";
            self.session.span_warn(item.span, msg);
            self.used_crates.insert(cnum); // Avoid the normal unused extern crate warning
        }

        if let Some(span) = legacy_imports.import_all {
            module.for_each_child(|name, ns, binding| if ns == MacroNS {
                self.legacy_import_macro(name, binding, span, allow_shadowing);
            });
        } else {
            for (name, span) in legacy_imports.imports {
                let result = self.resolve_name_in_module(module, name, MacroNS, false, None);
                if let Success(binding) = result {
                    self.legacy_import_macro(name, binding, span, allow_shadowing);
                } else {
                    span_err!(self.session, span, E0469, "imported macro not found");
                }
            }
        }
        for (name, span) in legacy_imports.reexports {
            self.used_crates.insert(module.def_id().unwrap().krate);
            let result = self.resolve_name_in_module(module, name, MacroNS, false, None);
            if let Success(binding) = result {
                self.macro_exports.push(Export { name: name, def: binding.def() });
            } else {
                span_err!(self.session, span, E0470, "reexported macro not found");
            }
        }
    }

    // does this attribute list contain "macro_use"?
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            if attr.check_name("macro_escape") {
                let msg = "macro_escape is a deprecated synonym for macro_use";
                let mut err = self.session.struct_span_warn(attr.span, msg);
                if let ast::AttrStyle::Inner = attr.style {
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

    fn legacy_macro_imports(&mut self, attrs: &[ast::Attribute]) -> LegacyMacroImports {
        let mut imports = LegacyMacroImports::default();
        for attr in attrs {
            if attr.check_name("macro_use") {
                match attr.meta_item_list() {
                    Some(names) => for attr in names {
                        if let Some(word) = attr.word() {
                            imports.imports.push((word.name(), attr.span()));
                        } else {
                            span_err!(self.session, attr.span(), E0466, "bad macro import");
                        }
                    },
                    None => imports.import_all = Some(attr.span),
                }
            } else if attr.check_name("macro_reexport") {
                let bad_macro_reexport = |this: &mut Self, span| {
                    span_err!(this.session, span, E0467, "bad macro reexport");
                };
                if let Some(names) = attr.meta_item_list() {
                    for attr in names {
                        if let Some(word) = attr.word() {
                            imports.reexports.push((word.name(), attr.span()));
                        } else {
                            bad_macro_reexport(self, attr.span());
                        }
                    }
                } else {
                    bad_macro_reexport(self, attr.span());
                }
            }
        }
        imports
    }
}

pub struct BuildReducedGraphVisitor<'a, 'b: 'a> {
    pub resolver: &'a mut Resolver<'b>,
    pub legacy_scope: LegacyScope<'b>,
    pub expansion: Mark,
}

impl<'a, 'b> BuildReducedGraphVisitor<'a, 'b> {
    fn visit_invoc(&mut self, id: ast::NodeId) -> &'b InvocationData<'b> {
        let mark = Mark::from_placeholder_id(id);
        self.resolver.current_module.unresolved_invocations.borrow_mut().insert(mark);
        let invocation = self.resolver.invocations[&mark];
        invocation.module.set(self.resolver.current_module);
        invocation.legacy_scope.set(self.legacy_scope);
        invocation
    }
}

macro_rules! method {
    ($visit:ident: $ty:ty, $invoc:path, $walk:ident) => {
        fn $visit(&mut self, node: &$ty) {
            if let $invoc(..) = node.node {
                self.visit_invoc(node.id);
            } else {
                visit::$walk(self, node);
            }
        }
    }
}

impl<'a, 'b> Visitor for BuildReducedGraphVisitor<'a, 'b> {
    method!(visit_impl_item: ast::ImplItem, ast::ImplItemKind::Macro, walk_impl_item);
    method!(visit_expr:      ast::Expr,     ast::ExprKind::Mac,       walk_expr);
    method!(visit_pat:       ast::Pat,      ast::PatKind::Mac,        walk_pat);
    method!(visit_ty:        ast::Ty,       ast::TyKind::Mac,         walk_ty);

    fn visit_item(&mut self, item: &Item) {
        let macro_use = match item.node {
            ItemKind::Mac(..) if item.id == ast::DUMMY_NODE_ID => return, // Scope placeholder
            ItemKind::Mac(..) => {
                return self.legacy_scope = LegacyScope::Expansion(self.visit_invoc(item.id));
            }
            ItemKind::Mod(..) => self.resolver.contains_macro_use(&item.attrs),
            _ => false,
        };

        let (parent, legacy_scope) = (self.resolver.current_module, self.legacy_scope);
        self.resolver.build_reduced_graph_for_item(item, self.expansion);
        visit::walk_item(self, item);
        self.resolver.current_module = parent;
        if !macro_use {
            self.legacy_scope = legacy_scope;
        }
    }

    fn visit_stmt(&mut self, stmt: &ast::Stmt) {
        if let ast::StmtKind::Mac(..) = stmt.node {
            self.legacy_scope = LegacyScope::Expansion(self.visit_invoc(stmt.id));
        } else {
            visit::walk_stmt(self, stmt);
        }
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.resolver.build_reduced_graph_for_foreign_item(foreign_item, self.expansion);
        visit::walk_foreign_item(self, foreign_item);
    }

    fn visit_block(&mut self, block: &Block) {
        let (parent, legacy_scope) = (self.resolver.current_module, self.legacy_scope);
        self.resolver.build_reduced_graph_for_block(block);
        visit::walk_block(self, block);
        self.resolver.current_module = parent;
        self.legacy_scope = legacy_scope;
    }

    fn visit_trait_item(&mut self, item: &TraitItem) {
        let parent = self.resolver.current_module;
        let def_id = parent.def_id().unwrap();

        if let TraitItemKind::Macro(_) = item.node {
            self.visit_invoc(item.id);
            return
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
        self.resolver.define(parent, item.ident.name, ns, (def, vis, item.span, self.expansion));

        self.resolver.current_module = parent.parent.unwrap(); // nearest normal ancestor
        visit::walk_trait_item(self, item);
        self.resolver.current_module = parent;
    }
}
