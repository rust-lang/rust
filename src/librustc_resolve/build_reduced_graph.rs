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

use {DefModifiers, PUBLIC, IMPORTABLE};
use ImportDirective;
use ImportDirectiveSubclass::{self, SingleImport, GlobImport};
use ImportResolution;
use Module;
use ModuleKind::*;
use Namespace::{TypeNS, ValueNS};
use NameBindings;
use ParentLink::{self, ModuleParentLink, BlockParentLink};
use Resolver;
use RibKind::*;
use Shadowable;
use TypeNsDef;
use TypeParameters::HasTypeParameters;

use self::DuplicateCheckingMode::*;
use self::NamespaceError::*;

use rustc::metadata::csearch;
use rustc::metadata::decoder::{DefLike, DlDef, DlField, DlImpl};
use rustc::middle::def::*;
use rustc::middle::subst::FnSpace;

use syntax::ast::{Block, Crate};
use syntax::ast::{DeclItem, DefId};
use syntax::ast::{ForeignItem, ForeignItemFn, ForeignItemStatic};
use syntax::ast::{Item, ItemConst, ItemEnum, ItemFn};
use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic};
use syntax::ast::{ItemStruct, ItemTrait, ItemTy};
use syntax::ast::{MethodImplItem, Name, NamedField, NodeId};
use syntax::ast::{PathListIdent, PathListMod};
use syntax::ast::{Public, SelfStatic};
use syntax::ast::StmtDecl;
use syntax::ast::StructVariantKind;
use syntax::ast::TupleVariantKind;
use syntax::ast::TyObjectSum;
use syntax::ast::{TypeImplItem, UnnamedField};
use syntax::ast::{Variant, ViewItem, ViewItemExternCrate};
use syntax::ast::{ViewItemUse, ViewPathGlob, ViewPathList, ViewPathSimple};
use syntax::ast::{Visibility};
use syntax::ast::TyPath;
use syntax::ast;
use syntax::ast_util::{self, PostExpansionMethod, local_def};
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::{self, special_idents};
use syntax::codemap::{Span, DUMMY_SP};
use syntax::visit::{self, Visitor};

use std::mem::replace;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

// Specifies how duplicates should be handled when adding a child item if
// another item exists with the same name in some namespace.
#[derive(Copy, PartialEq)]
enum DuplicateCheckingMode {
    ForbidDuplicateModules,
    ForbidDuplicateTypesAndModules,
    ForbidDuplicateValues,
    ForbidDuplicateTypesAndValues,
    OverwriteDuplicates
}

#[derive(Copy, PartialEq)]
enum NamespaceError {
    NoError,
    ModuleError,
    TypeError,
    ValueError
}

fn namespace_error_to_string(ns: NamespaceError) -> &'static str {
    match ns {
        NoError                 => "",
        ModuleError | TypeError => "type or module",
        ValueError              => "value",
    }
}

struct GraphBuilder<'a, 'b:'a, 'tcx:'b> {
    resolver: &'a mut Resolver<'b, 'tcx>
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

impl<'a, 'b:'a, 'tcx:'b> GraphBuilder<'a, 'b, 'tcx> {
    /// Constructs the reduced graph for the entire crate.
    fn build_reduced_graph(self, krate: &ast::Crate) {
        let parent = self.graph_root.get_module();
        let mut visitor = BuildReducedGraphVisitor {
            builder: self,
            parent: parent
        };
        visit::walk_crate(&mut visitor, krate);
    }

    /// Adds a new child item to the module definition of the parent node and
    /// returns its corresponding name bindings as well as the current parent.
    /// Or, if we're inside a block, creates (or reuses) an anonymous module
    /// corresponding to the innermost block ID and returns the name bindings
    /// as well as the newly-created parent.
    ///
    /// # Panics
    ///
    /// Panics if this node does not have a module definition and we are not inside
    /// a block.
    fn add_child(&self,
                 name: Name,
                 parent: &Rc<Module>,
                 duplicate_checking_mode: DuplicateCheckingMode,
                 // For printing errors
                 sp: Span)
                 -> Rc<NameBindings> {
        // If this is the immediate descendant of a module, then we add the
        // child name directly. Otherwise, we create or reuse an anonymous
        // module and add the child to that.

        self.check_for_conflicts_between_external_crates_and_items(&**parent,
                                                                   name,
                                                                   sp);

        // Add or reuse the child.
        let child = parent.children.borrow().get(&name).cloned();
        match child {
            None => {
                let child = Rc::new(NameBindings::new());
                parent.children.borrow_mut().insert(name, child.clone());
                child
            }
            Some(child) => {
                // Enforce the duplicate checking mode:
                //
                // * If we're requesting duplicate module checking, check that
                //   there isn't a module in the module with the same name.
                //
                // * If we're requesting duplicate type checking, check that
                //   there isn't a type in the module with the same name.
                //
                // * If we're requesting duplicate value checking, check that
                //   there isn't a value in the module with the same name.
                //
                // * If we're requesting duplicate type checking and duplicate
                //   value checking, check that there isn't a duplicate type
                //   and a duplicate value with the same name.
                //
                // * If no duplicate checking was requested at all, do
                //   nothing.

                let mut duplicate_type = NoError;
                let ns = match duplicate_checking_mode {
                    ForbidDuplicateModules => {
                        if child.get_module_if_available().is_some() {
                            duplicate_type = ModuleError;
                        }
                        Some(TypeNS)
                    }
                    ForbidDuplicateTypesAndModules => {
                        match child.def_for_namespace(TypeNS) {
                            None => {}
                            Some(_) if child.get_module_if_available()
                                            .map(|m| m.kind.get()) ==
                                       Some(ImplModuleKind) => {}
                            Some(_) => duplicate_type = TypeError
                        }
                        Some(TypeNS)
                    }
                    ForbidDuplicateValues => {
                        if child.defined_in_namespace(ValueNS) {
                            duplicate_type = ValueError;
                        }
                        Some(ValueNS)
                    }
                    ForbidDuplicateTypesAndValues => {
                        let mut n = None;
                        match child.def_for_namespace(TypeNS) {
                            Some(DefMod(_)) | None => {}
                            Some(_) => {
                                n = Some(TypeNS);
                                duplicate_type = TypeError;
                            }
                        };
                        if child.defined_in_namespace(ValueNS) {
                            duplicate_type = ValueError;
                            n = Some(ValueNS);
                        }
                        n
                    }
                    OverwriteDuplicates => None
                };
                if duplicate_type != NoError {
                    // Return an error here by looking up the namespace that
                    // had the duplicate.
                    let ns = ns.unwrap();
                    self.resolve_error(sp,
                        &format!("duplicate definition of {} `{}`",
                             namespace_error_to_string(duplicate_type),
                             token::get_name(name))[]);
                    {
                        let r = child.span_for_namespace(ns);
                        for sp in r.iter() {
                            self.session.span_note(*sp,
                                 &format!("first definition of {} `{}` here",
                                      namespace_error_to_string(duplicate_type),
                                      token::get_name(name))[]);
                        }
                    }
                }
                child
            }
        }
    }

    fn block_needs_anonymous_module(&mut self, block: &Block) -> bool {
        // If the block has view items, we need an anonymous module.
        if block.view_items.len() > 0 {
            return true;
        }

        // Check each statement.
        for statement in block.stmts.iter() {
            match statement.node {
                StmtDecl(ref declaration, _) => {
                    match declaration.node {
                        DeclItem(_) => {
                            return true;
                        }
                        _ => {
                            // Keep searching.
                        }
                    }
                }
                _ => {
                    // Keep searching.
                }
            }
        }

        // If we found neither view items nor items, we don't need to create
        // an anonymous module.

        return false;
    }

    fn get_parent_link(&mut self, parent: &Rc<Module>, name: Name) -> ParentLink {
        ModuleParentLink(parent.downgrade(), name)
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &Item, parent: &Rc<Module>) -> Rc<Module> {
        let name = item.ident.name;
        let sp = item.span;
        let is_public = item.vis == ast::Public;
        let modifiers = if is_public { PUBLIC } else { DefModifiers::empty() } | IMPORTABLE;

        match item.node {
            ItemMod(..) => {
                let name_bindings = self.add_child(name, parent, ForbidDuplicateModules, sp);

                let parent_link = self.get_parent_link(parent, name);
                let def_id = DefId { krate: 0, node: item.id };
                name_bindings.define_module(parent_link,
                                            Some(def_id),
                                            NormalModuleKind,
                                            false,
                                            item.vis == ast::Public,
                                            sp);

                name_bindings.get_module()
            }

            ItemForeignMod(..) => parent.clone(),

            // These items live in the value namespace.
            ItemStatic(_, m, _) => {
                let name_bindings = self.add_child(name, parent, ForbidDuplicateValues, sp);
                let mutbl = m == ast::MutMutable;

                name_bindings.define_value(DefStatic(local_def(item.id), mutbl), sp, modifiers);
                parent.clone()
            }
            ItemConst(_, _) => {
                self.add_child(name, parent, ForbidDuplicateValues, sp)
                    .define_value(DefConst(local_def(item.id)), sp, modifiers);
                parent.clone()
            }
            ItemFn(_, _, _, _, _) => {
                let name_bindings = self.add_child(name, parent, ForbidDuplicateValues, sp);

                let def = DefFn(local_def(item.id), false);
                name_bindings.define_value(def, sp, modifiers);
                parent.clone()
            }

            // These items live in the type namespace.
            ItemTy(..) => {
                let name_bindings =
                    self.add_child(name, parent, ForbidDuplicateTypesAndModules, sp);

                name_bindings.define_type(DefTy(local_def(item.id), false), sp, modifiers);
                parent.clone()
            }

            ItemEnum(ref enum_definition, _) => {
                let name_bindings =
                    self.add_child(name, parent, ForbidDuplicateTypesAndModules, sp);

                name_bindings.define_type(DefTy(local_def(item.id), true), sp, modifiers);

                let parent_link = self.get_parent_link(parent, name);
                // We want to make sure the module type is EnumModuleKind
                // even if there's already an ImplModuleKind module defined,
                // since that's how we prevent duplicate enum definitions
                name_bindings.set_module_kind(parent_link,
                                              Some(local_def(item.id)),
                                              EnumModuleKind,
                                              false,
                                              is_public,
                                              sp);

                let module = name_bindings.get_module();

                for variant in (*enum_definition).variants.iter() {
                    self.build_reduced_graph_for_variant(
                        &**variant,
                        local_def(item.id),
                        &module);
                }
                parent.clone()
            }

            // These items live in both the type and value namespaces.
            ItemStruct(ref struct_def, _) => {
                // Adding to both Type and Value namespaces or just Type?
                let (forbid, ctor_id) = match struct_def.ctor_id {
                    Some(ctor_id)   => (ForbidDuplicateTypesAndValues, Some(ctor_id)),
                    None            => (ForbidDuplicateTypesAndModules, None)
                };

                let name_bindings = self.add_child(name, parent, forbid, sp);

                // Define a name in the type namespace.
                name_bindings.define_type(DefTy(local_def(item.id), false), sp, modifiers);

                // If this is a newtype or unit-like struct, define a name
                // in the value namespace as well
                if let Some(cid) = ctor_id {
                    name_bindings.define_value(DefStruct(local_def(cid)), sp, modifiers);
                }

                // Record the def ID and fields of this struct.
                let named_fields = struct_def.fields.iter().filter_map(|f| {
                    match f.node.kind {
                        NamedField(ident, _) => Some(ident.name),
                        UnnamedField(_) => None
                    }
                }).collect();
                self.structs.insert(local_def(item.id), named_fields);

                parent.clone()
            }

            ItemImpl(_, _, _, None, ref ty, ref impl_items) => {
                // If this implements an anonymous trait, then add all the
                // methods within to a new module, if the type was defined
                // within this module.

                let mod_name = match ty.node {
                    TyPath(ref path, _) if path.segments.len() == 1 => {
                        // FIXME(18446) we should distinguish between the name of
                        // a trait and the name of an impl of that trait.
                        Some(path.segments.last().unwrap().identifier.name)
                    }
                    TyObjectSum(ref lhs_ty, _) => {
                        match lhs_ty.node {
                            TyPath(ref path, _) if path.segments.len() == 1 => {
                                Some(path.segments.last().unwrap().identifier.name)
                            }
                            _ => {
                                None
                            }
                        }
                    }
                    _ => {
                        None
                    }
                };

                match mod_name {
                    None => {
                        self.resolve_error(ty.span,
                                           "inherent implementations may \
                                            only be implemented in the same \
                                            module as the type they are \
                                            implemented for")
                    }
                    Some(mod_name) => {
                        // Create the module and add all methods.
                        let parent_opt = parent.children.borrow().get(&mod_name).cloned();
                        let new_parent = match parent_opt {
                            // It already exists
                            Some(ref child) if child.get_module_if_available()
                                .is_some() &&
                                (child.get_module().kind.get() == ImplModuleKind ||
                                 child.get_module().kind.get() == TraitModuleKind) => {
                                    child.get_module()
                                }
                            Some(ref child) if child.get_module_if_available()
                                .is_some() &&
                                child.get_module().kind.get() ==
                                EnumModuleKind => child.get_module(),
                            // Create the module
                            _ => {
                                let name_bindings =
                                    self.add_child(mod_name, parent, ForbidDuplicateModules, sp);

                                let parent_link = self.get_parent_link(parent, name);
                                let def_id = local_def(item.id);
                                let ns = TypeNS;
                                let is_public =
                                    !name_bindings.defined_in_namespace(ns) ||
                                    name_bindings.defined_in_public_namespace(ns);

                                name_bindings.define_module(parent_link,
                                                            Some(def_id),
                                                            ImplModuleKind,
                                                            false,
                                                            is_public,
                                                            sp);

                                name_bindings.get_module()
                            }
                        };

                        // For each implementation item...
                        for impl_item in impl_items.iter() {
                            match *impl_item {
                                MethodImplItem(ref method) => {
                                    // Add the method to the module.
                                    let name = method.pe_ident().name;
                                    let method_name_bindings =
                                        self.add_child(name,
                                                       &new_parent,
                                                       ForbidDuplicateValues,
                                                       method.span);
                                    let def = match method.pe_explicit_self()
                                        .node {
                                            SelfStatic => {
                                                // Static methods become
                                                // `DefStaticMethod`s.
                                                DefStaticMethod(local_def(method.id),
                                                                FromImpl(local_def(item.id)))
                                            }
                                            _ => {
                                                // Non-static methods become
                                                // `DefMethod`s.
                                                DefMethod(local_def(method.id),
                                                          None,
                                                          FromImpl(local_def(item.id)))
                                            }
                                        };

                                    // NB: not IMPORTABLE
                                    let modifiers = if method.pe_vis() == ast::Public {
                                        PUBLIC
                                    } else {
                                        DefModifiers::empty()
                                    };
                                    method_name_bindings.define_value(
                                        def,
                                        method.span,
                                        modifiers);
                                }
                                TypeImplItem(ref typedef) => {
                                    // Add the typedef to the module.
                                    let name = typedef.ident.name;
                                    let typedef_name_bindings =
                                        self.add_child(
                                            name,
                                            &new_parent,
                                            ForbidDuplicateTypesAndModules,
                                            typedef.span);
                                    let def = DefAssociatedTy(local_def(
                                        typedef.id));
                                    // NB: not IMPORTABLE
                                    let modifiers = if typedef.vis == ast::Public {
                                        PUBLIC
                                    } else {
                                        DefModifiers::empty()
                                    };
                                    typedef_name_bindings.define_type(
                                        def,
                                        typedef.span,
                                        modifiers);
                                }
                            }
                        }
                    }
                }

                parent.clone()
            }

            ItemImpl(_, _, _, Some(_), _, _) => parent.clone(),

            ItemTrait(_, _, _, ref items) => {
                let name_bindings =
                    self.add_child(name, parent, ForbidDuplicateTypesAndModules, sp);

                // Add all the items within to a new module.
                let parent_link = self.get_parent_link(parent, name);
                name_bindings.define_module(parent_link,
                                            Some(local_def(item.id)),
                                            TraitModuleKind,
                                            false,
                                            item.vis == ast::Public,
                                            sp);
                let module_parent = name_bindings.get_module();

                let def_id = local_def(item.id);

                // Add the names of all the items to the trait info.
                for trait_item in items.iter() {
                    let (name, kind) = match *trait_item {
                        ast::RequiredMethod(_) |
                        ast::ProvidedMethod(_) => {
                            let ty_m = ast_util::trait_item_to_ty_method(trait_item);

                            let name = ty_m.ident.name;

                            // Add it as a name in the trait module.
                            let (def, static_flag) = match ty_m.explicit_self
                                                               .node {
                                SelfStatic => {
                                    // Static methods become `DefStaticMethod`s.
                                    (DefStaticMethod(
                                            local_def(ty_m.id),
                                            FromTrait(local_def(item.id))),
                                     StaticMethodTraitItemKind)
                                }
                                _ => {
                                    // Non-static methods become `DefMethod`s.
                                    (DefMethod(local_def(ty_m.id),
                                               Some(local_def(item.id)),
                                               FromTrait(local_def(item.id))),
                                     NonstaticMethodTraitItemKind)
                                }
                            };

                            let method_name_bindings =
                                self.add_child(name,
                                               &module_parent,
                                               ForbidDuplicateTypesAndValues,
                                               ty_m.span);
                            // NB: not IMPORTABLE
                            method_name_bindings.define_value(def,
                                                              ty_m.span,
                                                              PUBLIC);

                            (name, static_flag)
                        }
                        ast::TypeTraitItem(ref associated_type) => {
                            let def = DefAssociatedTy(local_def(
                                    associated_type.ty_param.id));

                            let name_bindings =
                                self.add_child(associated_type.ty_param.ident.name,
                                               &module_parent,
                                               ForbidDuplicateTypesAndValues,
                                               associated_type.ty_param.span);
                            // NB: not IMPORTABLE
                            name_bindings.define_type(def,
                                                      associated_type.ty_param.span,
                                                      PUBLIC);

                            (associated_type.ty_param.ident.name, TypeTraitItemKind)
                        }
                    };

                    self.trait_item_map.insert((name, def_id), kind);
                }

                name_bindings.define_type(DefTrait(def_id), sp, modifiers);
                parent.clone()
            }
            ItemMac(..) => parent.clone()
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn build_reduced_graph_for_variant(&mut self,
                                       variant: &Variant,
                                       item_id: DefId,
                                       parent: &Rc<Module>) {
        let name = variant.node.name.name;
        let is_exported = match variant.node.kind {
            TupleVariantKind(_) => false,
            StructVariantKind(_) => {
                // Not adding fields for variants as they are not accessed with a self receiver
                self.structs.insert(local_def(variant.node.id), Vec::new());
                true
            }
        };

        let child = self.add_child(name, parent,
                                   ForbidDuplicateTypesAndValues,
                                   variant.span);
        // variants are always treated as importable to allow them to be glob
        // used
        child.define_value(DefVariant(item_id,
                                      local_def(variant.node.id), is_exported),
                           variant.span, PUBLIC | IMPORTABLE);
        child.define_type(DefVariant(item_id,
                                     local_def(variant.node.id), is_exported),
                          variant.span, PUBLIC | IMPORTABLE);
    }

    /// Constructs the reduced graph for one 'view item'. View items consist
    /// of imports and use directives.
    fn build_reduced_graph_for_view_item(&mut self, view_item: &ViewItem, parent: &Rc<Module>) {
        match view_item.node {
            ViewItemUse(ref view_path) => {
                // Extract and intern the module part of the path. For
                // globs and lists, the path is found directly in the AST;
                // for simple paths we have to munge the path a little.
                let module_path = match view_path.node {
                    ViewPathSimple(_, ref full_path, _) => {
                        full_path.segments
                            .init()
                            .iter().map(|ident| ident.identifier.name)
                            .collect()
                    }

                    ViewPathGlob(ref module_ident_path, _) |
                    ViewPathList(ref module_ident_path, _, _) => {
                        module_ident_path.segments
                            .iter().map(|ident| ident.identifier.name).collect()
                    }
                };

                // Build up the import directives.
                let is_public = view_item.vis == ast::Public;
                let shadowable =
                    view_item.attrs
                             .iter()
                             .any(|attr| {
                                 attr.name() == token::get_name(
                                    special_idents::prelude_import.name)
                             });
                let shadowable = if shadowable {
                    Shadowable::Always
                } else {
                    Shadowable::Never
                };

                match view_path.node {
                    ViewPathSimple(binding, ref full_path, id) => {
                        let source_name =
                            full_path.segments.last().unwrap().identifier.name;
                        if token::get_name(source_name).get() == "mod" ||
                           token::get_name(source_name).get() == "self" {
                            self.resolve_error(view_path.span,
                                "`self` imports are only allowed within a { } list");
                        }

                        let subclass = SingleImport(binding.name,
                                                    source_name);
                        self.build_import_directive(&**parent,
                                                    module_path,
                                                    subclass,
                                                    view_path.span,
                                                    id,
                                                    is_public,
                                                    shadowable);
                    }
                    ViewPathList(_, ref source_items, _) => {
                        // Make sure there's at most one `mod` import in the list.
                        let mod_spans = source_items.iter().filter_map(|item| match item.node {
                            PathListMod { .. } => Some(item.span),
                            _ => None
                        }).collect::<Vec<Span>>();
                        if mod_spans.len() > 1 {
                            self.resolve_error(mod_spans[0],
                                "`self` import can only appear once in the list");
                            for other_span in mod_spans.iter().skip(1) {
                                self.session.span_note(*other_span,
                                    "another `self` import appears here");
                            }
                        }

                        for source_item in source_items.iter() {
                            let (module_path, name) = match source_item.node {
                                PathListIdent { name, .. } =>
                                    (module_path.clone(), name.name),
                                PathListMod { .. } => {
                                    let name = match module_path.last() {
                                        Some(name) => *name,
                                        None => {
                                            self.resolve_error(source_item.span,
                                                "`self` import can only appear in an import list \
                                                 with a non-empty prefix");
                                            continue;
                                        }
                                    };
                                    let module_path = module_path.init();
                                    (module_path.to_vec(), name)
                                }
                            };
                            self.build_import_directive(
                                &**parent,
                                module_path,
                                SingleImport(name, name),
                                source_item.span,
                                source_item.node.id(),
                                is_public,
                                shadowable);
                        }
                    }
                    ViewPathGlob(_, id) => {
                        self.build_import_directive(&**parent,
                                                    module_path,
                                                    GlobImport,
                                                    view_path.span,
                                                    id,
                                                    is_public,
                                                    shadowable);
                    }
                }
            }

            ViewItemExternCrate(name, _, node_id) => {
                // n.b. we don't need to look at the path option here, because cstore already did
                for &crate_id in self.session.cstore
                                     .find_extern_mod_stmt_cnum(node_id).iter() {
                    let def_id = DefId { krate: crate_id, node: 0 };
                    self.external_exports.insert(def_id);
                    let parent_link = ModuleParentLink(parent.downgrade(), name.name);
                    let external_module = Rc::new(Module::new(parent_link,
                                                              Some(def_id),
                                                              NormalModuleKind,
                                                              false,
                                                              true));
                    debug!("(build reduced graph for item) found extern `{}`",
                            self.module_to_string(&*external_module));
                    self.check_for_conflicts_between_external_crates(
                        &**parent,
                        name.name,
                        view_item.span);
                    parent.external_module_children.borrow_mut()
                          .insert(name.name, external_module.clone());
                    self.build_reduced_graph_for_external_crate(&external_module);
                }
            }
        }
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item<F>(&mut self,
                                               foreign_item: &ForeignItem,
                                               parent: &Rc<Module>,
                                               f: F) where
        F: FnOnce(&mut Resolver),
    {
        let name = foreign_item.ident.name;
        let is_public = foreign_item.vis == ast::Public;
        let modifiers = if is_public { PUBLIC } else { DefModifiers::empty() } | IMPORTABLE;
        let name_bindings =
            self.add_child(name, parent, ForbidDuplicateValues,
                           foreign_item.span);

        match foreign_item.node {
            ForeignItemFn(_, ref generics) => {
                let def = DefFn(local_def(foreign_item.id), false);
                name_bindings.define_value(def, foreign_item.span, modifiers);

                self.with_type_parameter_rib(
                    HasTypeParameters(generics,
                                      FnSpace,
                                      foreign_item.id,
                                      NormalRibKind),
                    f);
            }
            ForeignItemStatic(_, m) => {
                let def = DefStatic(local_def(foreign_item.id), m);
                name_bindings.define_value(def, foreign_item.span, modifiers);

                f(self.resolver)
            }
        }
    }

    fn build_reduced_graph_for_block(&mut self, block: &Block, parent: &Rc<Module>) -> Rc<Module> {
        if self.block_needs_anonymous_module(block) {
            let block_id = block.id;

            debug!("(building reduced graph for block) creating a new \
                    anonymous module for block {}",
                   block_id);

            let new_module = Rc::new(Module::new(
                BlockParentLink(parent.downgrade(), block_id),
                None,
                AnonymousModuleKind,
                false,
                false));
            parent.anonymous_children.borrow_mut().insert(block_id, new_module.clone());
            new_module
        } else {
            parent.clone()
        }
    }

    fn handle_external_def(&mut self,
                           def: Def,
                           vis: Visibility,
                           child_name_bindings: &NameBindings,
                           final_ident: &str,
                           name: Name,
                           new_parent: &Rc<Module>) {
        debug!("(building reduced graph for \
                external crate) building external def, priv {:?}",
               vis);
        let is_public = vis == ast::Public;
        let modifiers = if is_public { PUBLIC } else { DefModifiers::empty() } | IMPORTABLE;
        let is_exported = is_public && match new_parent.def_id.get() {
            None => true,
            Some(did) => self.external_exports.contains(&did)
        };
        if is_exported {
            self.external_exports.insert(def.def_id());
        }

        let kind = match def {
            DefTy(_, true) => EnumModuleKind,
            DefStruct(..) | DefTy(..) => ImplModuleKind,
            _ => NormalModuleKind
        };

        match def {
          DefMod(def_id) | DefForeignMod(def_id) | DefStruct(def_id) |
          DefTy(def_id, _) => {
            let type_def = child_name_bindings.type_def.borrow().clone();
            match type_def {
              Some(TypeNsDef { module_def: Some(module_def), .. }) => {
                debug!("(building reduced graph for external crate) \
                        already created module");
                module_def.def_id.set(Some(def_id));
              }
              Some(_) | None => {
                debug!("(building reduced graph for \
                        external crate) building module \
                        {}", final_ident);
                let parent_link = self.get_parent_link(new_parent, name);

                child_name_bindings.define_module(parent_link,
                                                  Some(def_id),
                                                  kind,
                                                  true,
                                                  is_public,
                                                  DUMMY_SP);
              }
            }
          }
          _ => {}
        }

        match def {
          DefMod(_) | DefForeignMod(_) => {}
          DefVariant(_, variant_id, is_struct) => {
              debug!("(building reduced graph for external crate) building \
                      variant {}",
                      final_ident);
              // variants are always treated as importable to allow them to be
              // glob used
              let modifiers = PUBLIC | IMPORTABLE;
              if is_struct {
                  child_name_bindings.define_type(def, DUMMY_SP, modifiers);
                  // Not adding fields for variants as they are not accessed with a self receiver
                  self.structs.insert(variant_id, Vec::new());
              } else {
                  child_name_bindings.define_value(def, DUMMY_SP, modifiers);
              }
          }
          DefFn(ctor_id, true) => {
            child_name_bindings.define_value(
                csearch::get_tuple_struct_definition_if_ctor(&self.session.cstore, ctor_id)
                    .map_or(def, |_| DefStruct(ctor_id)), DUMMY_SP, modifiers);
          }
          DefFn(..) | DefStaticMethod(..) | DefStatic(..) | DefConst(..) | DefMethod(..) => {
            debug!("(building reduced graph for external \
                    crate) building value (fn/static) {}", final_ident);
            // impl methods have already been defined with the correct importability modifier
            let mut modifiers = match *child_name_bindings.value_def.borrow() {
                Some(ref def) => (modifiers & !IMPORTABLE) | (def.modifiers & IMPORTABLE),
                None => modifiers
            };
            if new_parent.kind.get() != NormalModuleKind {
                modifiers = modifiers & !IMPORTABLE;
            }
            child_name_bindings.define_value(def, DUMMY_SP, modifiers);
          }
          DefTrait(def_id) => {
              debug!("(building reduced graph for external \
                      crate) building type {}", final_ident);

              // If this is a trait, add all the trait item names to the trait
              // info.

              let trait_item_def_ids =
                csearch::get_trait_item_def_ids(&self.session.cstore, def_id);
              for trait_item_def_id in trait_item_def_ids.iter() {
                  let (trait_item_name, trait_item_kind) =
                      csearch::get_trait_item_name_and_kind(
                          &self.session.cstore,
                          trait_item_def_id.def_id());

                  debug!("(building reduced graph for external crate) ... \
                          adding trait item '{}'",
                         token::get_name(trait_item_name));

                  self.trait_item_map.insert((trait_item_name, def_id), trait_item_kind);

                  if is_exported {
                      self.external_exports
                          .insert(trait_item_def_id.def_id());
                  }
              }

              child_name_bindings.define_type(def, DUMMY_SP, modifiers);

              // Define a module if necessary.
              let parent_link = self.get_parent_link(new_parent, name);
              child_name_bindings.set_module_kind(parent_link,
                                                  Some(def_id),
                                                  TraitModuleKind,
                                                  true,
                                                  is_public,
                                                  DUMMY_SP)
          }
          DefTy(..) | DefAssociatedTy(..) | DefAssociatedPath(..) => {
              debug!("(building reduced graph for external \
                      crate) building type {}", final_ident);

              child_name_bindings.define_type(def, DUMMY_SP, modifiers);
          }
          DefStruct(def_id) => {
            debug!("(building reduced graph for external \
                    crate) building type and value for {}",
                   final_ident);
            child_name_bindings.define_type(def, DUMMY_SP, modifiers);
            let fields = csearch::get_struct_fields(&self.session.cstore, def_id).iter().map(|f| {
                f.name
            }).collect::<Vec<_>>();

            if fields.len() == 0 {
                child_name_bindings.define_value(def, DUMMY_SP, modifiers);
            }

            // Record the def ID and fields of this struct.
            self.structs.insert(def_id, fields);
          }
          DefLocal(..) | DefPrimTy(..) | DefTyParam(..) |
          DefUse(..) | DefUpvar(..) | DefRegion(..) |
          DefTyParamBinder(..) | DefLabel(..) | DefSelfTy(..) => {
            panic!("didn't expect `{:?}`", def);
          }
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_def(&mut self,
                                                  root: &Rc<Module>,
                                                  def_like: DefLike,
                                                  name: Name,
                                                  visibility: Visibility) {
        match def_like {
            DlDef(def) => {
                // Add the new child item, if necessary.
                match def {
                    DefForeignMod(def_id) => {
                        // Foreign modules have no names. Recur and populate
                        // eagerly.
                        csearch::each_child_of_item(&self.session.cstore,
                                                    def_id,
                                                    |def_like,
                                                     child_name,
                                                     vis| {
                            self.build_reduced_graph_for_external_crate_def(
                                root,
                                def_like,
                                child_name,
                                vis)
                        });
                    }
                    _ => {
                        let child_name_bindings =
                            self.add_child(name,
                                           root,
                                           OverwriteDuplicates,
                                           DUMMY_SP);

                        self.handle_external_def(def,
                                                 visibility,
                                                 &*child_name_bindings,
                                                 token::get_name(name).get(),
                                                 name,
                                                 root);
                    }
                }
            }
            DlImpl(def) => {
                match csearch::get_type_name_if_impl(&self.session.cstore, def) {
                    None => {}
                    Some(final_name) => {
                        let methods_opt =
                            csearch::get_methods_if_impl(&self.session.cstore, def);
                        match methods_opt {
                            Some(ref methods) if
                                methods.len() >= 1 => {
                                debug!("(building reduced graph for \
                                        external crate) processing \
                                        static methods for type name {}",
                                        token::get_name(final_name));

                                let child_name_bindings =
                                    self.add_child(
                                        final_name,
                                        root,
                                        OverwriteDuplicates,
                                        DUMMY_SP);

                                // Process the static methods. First,
                                // create the module.
                                let type_module;
                                let type_def = child_name_bindings.type_def.borrow().clone();
                                match type_def {
                                    Some(TypeNsDef {
                                        module_def: Some(module_def),
                                        ..
                                    }) => {
                                        // We already have a module. This
                                        // is OK.
                                        type_module = module_def;

                                        // Mark it as an impl module if
                                        // necessary.
                                        type_module.kind.set(ImplModuleKind);
                                    }
                                    Some(_) | None => {
                                        let parent_link =
                                            self.get_parent_link(root, final_name);
                                        child_name_bindings.define_module(
                                            parent_link,
                                            Some(def),
                                            ImplModuleKind,
                                            true,
                                            true,
                                            DUMMY_SP);
                                        type_module =
                                            child_name_bindings.
                                                get_module();
                                    }
                                }

                                // Add each static method to the module.
                                let new_parent = type_module;
                                for method_info in methods.iter() {
                                    let name = method_info.name;
                                    debug!("(building reduced graph for \
                                             external crate) creating \
                                             static method '{}'",
                                           token::get_name(name));

                                    let method_name_bindings =
                                        self.add_child(name,
                                                       &new_parent,
                                                       OverwriteDuplicates,
                                                       DUMMY_SP);
                                    let def = DefFn(method_info.def_id, false);

                                    // NB: not IMPORTABLE
                                    let modifiers = if visibility == ast::Public {
                                        PUBLIC
                                    } else {
                                        DefModifiers::empty()
                                    };
                                    method_name_bindings.define_value(
                                        def, DUMMY_SP, modifiers);
                                }
                            }

                            // Otherwise, do nothing.
                            Some(_) | None => {}
                        }
                    }
                }
            }
            DlField => {
                debug!("(building reduced graph for external crate) \
                        ignoring field");
            }
        }
    }

    /// Builds the reduced graph rooted at the given external module.
    fn populate_external_module(&mut self, module: &Rc<Module>) {
        debug!("(populating external module) attempting to populate {}",
               self.module_to_string(&**module));

        let def_id = match module.def_id.get() {
            None => {
                debug!("(populating external module) ... no def ID!");
                return
            }
            Some(def_id) => def_id,
        };

        csearch::each_child_of_item(&self.session.cstore,
                                    def_id,
                                    |def_like, child_name, visibility| {
            debug!("(populating external module) ... found ident: {}",
                   token::get_name(child_name));
            self.build_reduced_graph_for_external_crate_def(module,
                                                            def_like,
                                                            child_name,
                                                            visibility)
        });
        module.populated.set(true)
    }

    /// Ensures that the reduced graph rooted at the given external module
    /// is built, building it if it is not.
    fn populate_module_if_necessary(&mut self, module: &Rc<Module>) {
        if !module.populated.get() {
            self.populate_external_module(module)
        }
        assert!(module.populated.get())
    }

    /// Builds the reduced graph rooted at the 'use' directive for an external
    /// crate.
    fn build_reduced_graph_for_external_crate(&mut self, root: &Rc<Module>) {
        csearch::each_top_level_item_of_crate(&self.session.cstore,
                                              root.def_id
                                                  .get()
                                                  .unwrap()
                                                  .krate,
                                              |def_like, name, visibility| {
            self.build_reduced_graph_for_external_crate_def(root, def_like, name, visibility)
        });
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(&mut self,
                              module_: &Module,
                              module_path: Vec<Name>,
                              subclass: ImportDirectiveSubclass,
                              span: Span,
                              id: NodeId,
                              is_public: bool,
                              shadowable: Shadowable) {
        module_.imports.borrow_mut().push(ImportDirective::new(module_path,
                                                               subclass,
                                                               span,
                                                               id,
                                                               is_public,
                                                               shadowable));
        self.unresolved_imports += 1;
        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match subclass {
            SingleImport(target, _) => {
                debug!("(building import directive) building import \
                        directive: {}::{}",
                       self.names_to_string(&module_.imports.borrow().last().unwrap().
                                                             module_path[]),
                       token::get_name(target));

                let mut import_resolutions = module_.import_resolutions
                                                    .borrow_mut();
                match import_resolutions.get_mut(&target) {
                    Some(resolution) => {
                        debug!("(building import directive) bumping \
                                reference");
                        resolution.outstanding_references += 1;

                        // the source of this name is different now
                        resolution.type_id = id;
                        resolution.value_id = id;
                        resolution.is_public = is_public;
                        return;
                    }
                    None => {}
                }
                debug!("(building import directive) creating new");
                let mut resolution = ImportResolution::new(id, is_public);
                resolution.outstanding_references = 1;
                import_resolutions.insert(target, resolution);
            }
            GlobImport => {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module_.glob_count.set(module_.glob_count.get() + 1);
            }
        }
    }
}

struct BuildReducedGraphVisitor<'a, 'b:'a, 'tcx:'b> {
    builder: GraphBuilder<'a, 'b, 'tcx>,
    parent: Rc<Module>
}

impl<'a, 'b, 'v, 'tcx> Visitor<'v> for BuildReducedGraphVisitor<'a, 'b, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        let p = self.builder.build_reduced_graph_for_item(item, &self.parent);
        let old_parent = replace(&mut self.parent, p);
        visit::walk_item(self, item);
        self.parent = old_parent;
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        let parent = &self.parent;
        self.builder.build_reduced_graph_for_foreign_item(foreign_item,
                                                          parent,
                                                          |r| {
            let mut v = BuildReducedGraphVisitor {
                builder: GraphBuilder { resolver: r },
                parent: parent.clone()
            };
            visit::walk_foreign_item(&mut v, foreign_item);
        })
    }

    fn visit_view_item(&mut self, view_item: &ViewItem) {
        self.builder.build_reduced_graph_for_view_item(view_item, &self.parent);
    }

    fn visit_block(&mut self, block: &Block) {
        let np = self.builder.build_reduced_graph_for_block(block, &self.parent);
        let old_parent = replace(&mut self.parent, np);
        visit::walk_block(self, block);
        self.parent = old_parent;
    }
}

pub fn build_reduced_graph(resolver: &mut Resolver, krate: &ast::Crate) {
    GraphBuilder {
        resolver: resolver
    }.build_reduced_graph(krate);
}

pub fn populate_module_if_necessary(resolver: &mut Resolver, module: &Rc<Module>) {
    GraphBuilder {
        resolver: resolver
    }.populate_module_if_necessary(module);
}
