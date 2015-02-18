// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_privacy"]
#![unstable(feature = "rustc_private")]
#![staged_api]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/nightly/")]

#![feature(core)]
#![feature(int_uint)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;

extern crate rustc;

use self::PrivacyResult::*;
use self::FieldName::*;

use std::mem::replace;

use rustc::metadata::csearch;
use rustc::middle::def;
use rustc::middle::privacy::ImportUse::*;
use rustc::middle::privacy::LastPrivate::*;
use rustc::middle::privacy::PrivateDep::*;
use rustc::middle::privacy::{ExportedItems, PublicItems, LastPrivateMap};
use rustc::middle::privacy::{ExternalExports};
use rustc::middle::ty::{MethodTypeParam, MethodStatic};
use rustc::middle::ty::{MethodCall, MethodMap, MethodOrigin, MethodParam};
use rustc::middle::ty::{MethodStaticClosure, MethodObject};
use rustc::middle::ty::{MethodTraitObject};
use rustc::middle::ty::{self, Ty};
use rustc::util::nodemap::{NodeMap, NodeSet};

use syntax::{ast, ast_map};
use syntax::ast_util::{is_local, local_def, PostExpansionMethod};
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit::{self, Visitor};

type Context<'a, 'tcx> = (&'a MethodMap<'tcx>, &'a def::ExportMap);

/// Result of a checking operation - None => no errors were found. Some => an
/// error and contains the span and message for reporting that error and
/// optionally the same for a note about the error.
type CheckResult = Option<(Span, String, Option<(Span, String)>)>;

////////////////////////////////////////////////////////////////////////////////
/// The parent visitor, used to determine what's the parent of what (node-wise)
////////////////////////////////////////////////////////////////////////////////

struct ParentVisitor {
    parents: NodeMap<ast::NodeId>,
    curparent: ast::NodeId,
}

impl<'v> Visitor<'v> for ParentVisitor {
    fn visit_item(&mut self, item: &ast::Item) {
        self.parents.insert(item.id, self.curparent);

        let prev = self.curparent;
        match item.node {
            ast::ItemMod(..) => { self.curparent = item.id; }
            // Enum variants are parented to the enum definition itself because
            // they inherit privacy
            ast::ItemEnum(ref def, _) => {
                for variant in &def.variants {
                    // The parent is considered the enclosing enum because the
                    // enum will dictate the privacy visibility of this variant
                    // instead.
                    self.parents.insert(variant.node.id, item.id);
                }
            }

            // Trait methods are always considered "public", but if the trait is
            // private then we need some private item in the chain from the
            // method to the root. In this case, if the trait is private, then
            // parent all the methods to the trait to indicate that they're
            // private.
            ast::ItemTrait(_, _, _, ref methods) if item.vis != ast::Public => {
                for m in methods {
                    match *m {
                        ast::ProvidedMethod(ref m) => {
                            self.parents.insert(m.id, item.id);
                        }
                        ast::RequiredMethod(ref m) => {
                            self.parents.insert(m.id, item.id);
                        }
                        ast::TypeTraitItem(_) => {}
                    };
                }
            }

            _ => {}
        }
        visit::walk_item(self, item);
        self.curparent = prev;
    }

    fn visit_foreign_item(&mut self, a: &ast::ForeignItem) {
        self.parents.insert(a.id, self.curparent);
        visit::walk_foreign_item(self, a);
    }

    fn visit_fn(&mut self, a: visit::FnKind<'v>, b: &'v ast::FnDecl,
                c: &'v ast::Block, d: Span, id: ast::NodeId) {
        // We already took care of some trait methods above, otherwise things
        // like impl methods and pub trait methods are parented to the
        // containing module, not the containing trait.
        if !self.parents.contains_key(&id) {
            self.parents.insert(id, self.curparent);
        }
        visit::walk_fn(self, a, b, c, d);
    }

    fn visit_struct_def(&mut self, s: &ast::StructDef, _: ast::Ident,
                        _: &'v ast::Generics, n: ast::NodeId) {
        // Struct constructors are parented to their struct definitions because
        // they essentially are the struct definitions.
        match s.ctor_id {
            Some(id) => { self.parents.insert(id, n); }
            None => {}
        }

        // While we have the id of the struct definition, go ahead and parent
        // all the fields.
        for field in &s.fields {
            self.parents.insert(field.node.id, self.curparent);
        }
        visit::walk_struct_def(self, s)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the ast
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    export_map: &'a def::ExportMap,

    // This flag is an indicator of whether the previous item in the
    // hierarchical chain was exported or not. This is the indicator of whether
    // children should be exported as well. Note that this can flip from false
    // to true if a reexported module is entered (or an action similar).
    prev_exported: bool,

    // This is a list of all exported items in the AST. An exported item is any
    // function/method/item which is usable by external crates. This essentially
    // means that the result is "public all the way down", but the "path down"
    // may jump across private boundaries through reexport statements.
    exported_items: ExportedItems,

    // This sets contains all the destination nodes which are publicly
    // re-exported. This is *not* a set of all reexported nodes, only a set of
    // all nodes which are reexported *and* reachable from external crates. This
    // means that the destination of the reexport is exported, and hence the
    // destination must also be exported.
    reexports: NodeSet,

    // These two fields are closely related to one another in that they are only
    // used for generation of the 'PublicItems' set, not for privacy checking at
    // all
    public_items: PublicItems,
    prev_public: bool,
}

impl<'a, 'tcx> EmbargoVisitor<'a, 'tcx> {
    // There are checks inside of privacy which depend on knowing whether a
    // trait should be exported or not. The two current consumers of this are:
    //
    //  1. Should default methods of a trait be exported?
    //  2. Should the methods of an implementation of a trait be exported?
    //
    // The answer to both of these questions partly rely on whether the trait
    // itself is exported or not. If the trait is somehow exported, then the
    // answers to both questions must be yes. Right now this question involves
    // more analysis than is currently done in rustc, so we conservatively
    // answer "yes" so that all traits need to be exported.
    fn exported_trait(&self, _id: ast::NodeId) -> bool {
        true
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for EmbargoVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        let orig_all_pub = self.prev_public;
        self.prev_public = orig_all_pub && item.vis == ast::Public;
        if self.prev_public {
            self.public_items.insert(item.id);
        }

        let orig_all_exported = self.prev_exported;
        match item.node {
            // impls/extern blocks do not break the "public chain" because they
            // cannot have visibility qualifiers on them anyway
            ast::ItemImpl(..) | ast::ItemForeignMod(..) => {}

            // Traits are a little special in that even if they themselves are
            // not public they may still be exported.
            ast::ItemTrait(..) => {
                self.prev_exported = self.exported_trait(item.id);
            }

            // Private by default, hence we only retain the "public chain" if
            // `pub` is explicitly listed.
            _ => {
                self.prev_exported =
                    (orig_all_exported && item.vis == ast::Public) ||
                     self.reexports.contains(&item.id);
            }
        }

        let public_first = self.prev_exported &&
                           self.exported_items.insert(item.id);

        match item.node {
            // Enum variants inherit from their parent, so if the enum is
            // public all variants are public unless they're explicitly priv
            ast::ItemEnum(ref def, _) if public_first => {
                for variant in &def.variants {
                    self.exported_items.insert(variant.node.id);
                }
            }

            // Implementations are a little tricky to determine what's exported
            // out of them. Here's a few cases which are currently defined:
            //
            // * Impls for private types do not need to export their methods
            //   (either public or private methods)
            //
            // * Impls for public types only have public methods exported
            //
            // * Public trait impls for public types must have all methods
            //   exported.
            //
            // * Private trait impls for public types can be ignored
            //
            // * Public trait impls for private types have their methods
            //   exported. I'm not entirely certain that this is the correct
            //   thing to do, but I have seen use cases of where this will cause
            //   undefined symbols at linkage time if this case is not handled.
            //
            // * Private trait impls for private types can be completely ignored
            ast::ItemImpl(_, _, _, _, ref ty, ref impl_items) => {
                let public_ty = match ty.node {
                    ast::TyPath(_, id) => {
                        match self.tcx.def_map.borrow()[id].clone() {
                            def::DefPrimTy(..) => true,
                            def => {
                                let did = def.def_id();
                                !is_local(did) ||
                                 self.exported_items.contains(&did.node)
                            }
                        }
                    }
                    _ => true,
                };
                let tr = ty::impl_trait_ref(self.tcx, local_def(item.id));
                let public_trait = tr.clone().map_or(false, |tr| {
                    !is_local(tr.def_id) ||
                     self.exported_items.contains(&tr.def_id.node)
                });

                if public_ty || public_trait {
                    for impl_item in impl_items {
                        match *impl_item {
                            ast::MethodImplItem(ref method) => {
                                let meth_public =
                                    match method.pe_explicit_self().node {
                                        ast::SelfStatic => public_ty,
                                        _ => true,
                                    } && method.pe_vis() == ast::Public;
                                if meth_public || tr.is_some() {
                                    self.exported_items.insert(method.id);
                                }
                            }
                            ast::TypeImplItem(_) => {}
                        }
                    }
                }
            }

            // Default methods on traits are all public so long as the trait
            // is public
            ast::ItemTrait(_, _, _, ref methods) if public_first => {
                for method in methods {
                    match *method {
                        ast::ProvidedMethod(ref m) => {
                            debug!("provided {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                        ast::RequiredMethod(ref m) => {
                            debug!("required {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                        ast::TypeTraitItem(ref t) => {
                            debug!("typedef {}", t.ty_param.id);
                            self.exported_items.insert(t.ty_param.id);
                        }
                    }
                }
            }

            // Struct constructors are public if the struct is all public.
            ast::ItemStruct(ref def, _) if public_first => {
                match def.ctor_id {
                    Some(id) => { self.exported_items.insert(id); }
                    None => {}
                }
            }

            ast::ItemTy(ref ty, _) if public_first => {
                if let ast::TyPath(_, id) = ty.node {
                    match self.tcx.def_map.borrow()[id].clone() {
                        def::DefPrimTy(..) | def::DefTyParam(..) => {},
                        def => {
                            let did = def.def_id();
                            if is_local(did) {
                                self.exported_items.insert(did.node);
                            }
                        }
                    }
                }
            }

            _ => {}
        }

        visit::walk_item(self, item);

        self.prev_exported = orig_all_exported;
        self.prev_public = orig_all_pub;
    }

    fn visit_foreign_item(&mut self, a: &ast::ForeignItem) {
        if (self.prev_exported && a.vis == ast::Public) || self.reexports.contains(&a.id) {
            self.exported_items.insert(a.id);
        }
    }

    fn visit_mod(&mut self, m: &ast::Mod, _sp: Span, id: ast::NodeId) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_exported {
            assert!(self.export_map.contains_key(&id), "wut {}", id);
            for export in &self.export_map[id] {
                if is_local(export.def_id) {
                    self.reexports.insert(export.def_id.node);
                }
            }
        }
        visit::walk_mod(self, m)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy visitor, where privacy checks take place (violations reported)
////////////////////////////////////////////////////////////////////////////////

struct PrivacyVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    curitem: ast::NodeId,
    in_foreign: bool,
    parents: NodeMap<ast::NodeId>,
    external_exports: ExternalExports,
    last_private_map: LastPrivateMap,
}

enum PrivacyResult {
    Allowable,
    ExternallyDenied,
    DisallowedBy(ast::NodeId),
}

enum FieldName {
    UnnamedField(uint), // index
    // (Name, not Ident, because struct fields are not macro-hygienic)
    NamedField(ast::Name),
}

impl<'a, 'tcx> PrivacyVisitor<'a, 'tcx> {
    // used when debugging
    fn nodestr(&self, id: ast::NodeId) -> String {
        self.tcx.map.node_to_string(id).to_string()
    }

    // Determines whether the given definition is public from the point of view
    // of the current item.
    fn def_privacy(&self, did: ast::DefId) -> PrivacyResult {
        if !is_local(did) {
            if self.external_exports.contains(&did) {
                debug!("privacy - {:?} was externally exported", did);
                return Allowable;
            }
            debug!("privacy - is {:?} a public method", did);

            return match self.tcx.impl_or_trait_items.borrow().get(&did) {
                Some(&ty::MethodTraitItem(ref meth)) => {
                    debug!("privacy - well at least it's a method: {:?}",
                           *meth);
                    match meth.container {
                        ty::TraitContainer(id) => {
                            debug!("privacy - recursing on trait {:?}", id);
                            self.def_privacy(id)
                        }
                        ty::ImplContainer(id) => {
                            match ty::impl_trait_ref(self.tcx, id) {
                                Some(t) => {
                                    debug!("privacy - impl of trait {:?}", id);
                                    self.def_privacy(t.def_id)
                                }
                                None => {
                                    debug!("privacy - found a method {:?}",
                                            meth.vis);
                                    if meth.vis == ast::Public {
                                        Allowable
                                    } else {
                                        ExternallyDenied
                                    }
                                }
                            }
                        }
                    }
                }
                Some(&ty::TypeTraitItem(ref typedef)) => {
                    match typedef.container {
                        ty::TraitContainer(id) => {
                            debug!("privacy - recursing on trait {:?}", id);
                            self.def_privacy(id)
                        }
                        ty::ImplContainer(id) => {
                            match ty::impl_trait_ref(self.tcx, id) {
                                Some(t) => {
                                    debug!("privacy - impl of trait {:?}", id);
                                    self.def_privacy(t.def_id)
                                }
                                None => {
                                    debug!("privacy - found a typedef {:?}",
                                            typedef.vis);
                                    if typedef.vis == ast::Public {
                                        Allowable
                                    } else {
                                        ExternallyDenied
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    debug!("privacy - nope, not even a method");
                    ExternallyDenied
                }
            };
        }

        debug!("privacy - local {} not public all the way down",
               self.tcx.map.node_to_string(did.node));
        // return quickly for things in the same module
        if self.parents.get(&did.node) == self.parents.get(&self.curitem) {
            debug!("privacy - same parent, we're done here");
            return Allowable;
        }

        // We now know that there is at least one private member between the
        // destination and the root.
        let mut closest_private_id = did.node;
        loop {
            debug!("privacy - examining {}", self.nodestr(closest_private_id));
            let vis = match self.tcx.map.find(closest_private_id) {
                // If this item is a method, then we know for sure that it's an
                // actual method and not a static method. The reason for this is
                // that these cases are only hit in the ExprMethodCall
                // expression, and ExprCall will have its path checked later
                // (the path of the trait/impl) if it's a static method.
                //
                // With this information, then we can completely ignore all
                // trait methods. The privacy violation would be if the trait
                // couldn't get imported, not if the method couldn't be used
                // (all trait methods are public).
                //
                // However, if this is an impl method, then we dictate this
                // decision solely based on the privacy of the method
                // invocation.
                // FIXME(#10573) is this the right behavior? Why not consider
                //               where the method was defined?
                Some(ast_map::NodeImplItem(ii)) => {
                    match *ii {
                        ast::MethodImplItem(ref m) => {
                            let imp = self.tcx.map
                                          .get_parent_did(closest_private_id);
                            match ty::impl_trait_ref(self.tcx, imp) {
                                Some(..) => return Allowable,
                                _ if m.pe_vis() == ast::Public => {
                                    return Allowable
                                }
                                _ => m.pe_vis()
                            }
                        }
                        ast::TypeImplItem(_) => return Allowable,
                    }
                }
                Some(ast_map::NodeTraitItem(_)) => {
                    return Allowable;
                }

                // This is not a method call, extract the visibility as one
                // would normally look at it
                Some(ast_map::NodeItem(it)) => it.vis,
                Some(ast_map::NodeForeignItem(_)) => {
                    self.tcx.map.get_foreign_vis(closest_private_id)
                }
                Some(ast_map::NodeVariant(..)) => {
                    ast::Public // need to move up a level (to the enum)
                }
                _ => ast::Public,
            };
            if vis != ast::Public { break }
            // if we've reached the root, then everything was allowable and this
            // access is public.
            if closest_private_id == ast::CRATE_NODE_ID { return Allowable }
            closest_private_id = self.parents[closest_private_id];

            // If we reached the top, then we were public all the way down and
            // we can allow this access.
            if closest_private_id == ast::DUMMY_NODE_ID { return Allowable }
        }
        debug!("privacy - closest priv {}", self.nodestr(closest_private_id));
        if self.private_accessible(closest_private_id) {
            Allowable
        } else {
            DisallowedBy(closest_private_id)
        }
    }

    /// For a local private node in the AST, this function will determine
    /// whether the node is accessible by the current module that iteration is
    /// inside.
    fn private_accessible(&self, id: ast::NodeId) -> bool {
        let parent = self.parents[id];
        debug!("privacy - accessible parent {}", self.nodestr(parent));

        // After finding `did`'s closest private member, we roll ourselves back
        // to see if this private member's parent is anywhere in our ancestry.
        // By the privacy rules, we can access all of our ancestor's private
        // members, so that's why we test the parent, and not the did itself.
        let mut cur = self.curitem;
        loop {
            debug!("privacy - questioning {}, {}", self.nodestr(cur), cur);
            match cur {
                // If the relevant parent is in our history, then we're allowed
                // to look inside any of our ancestor's immediate private items,
                // so this access is valid.
                x if x == parent => return true,

                // If we've reached the root, then we couldn't access this item
                // in the first place
                ast::DUMMY_NODE_ID => return false,

                // Keep going up
                _ => {}
            }

            cur = self.parents[cur];
        }
    }

    fn report_error(&self, result: CheckResult) -> bool {
        match result {
            None => true,
            Some((span, msg, note)) => {
                self.tcx.sess.span_err(span, &msg[..]);
                match note {
                    Some((span, msg)) => {
                        self.tcx.sess.span_note(span, &msg[..])
                    }
                    None => {},
                }
                false
            },
        }
    }

    /// Guarantee that a particular definition is public. Returns a CheckResult
    /// which contains any errors found. These can be reported using `report_error`.
    /// If the result is `None`, no errors were found.
    fn ensure_public(&self, span: Span, to_check: ast::DefId,
                     source_did: Option<ast::DefId>, msg: &str) -> CheckResult {
        let id = match self.def_privacy(to_check) {
            ExternallyDenied => {
                return Some((span, format!("{} is private", msg), None))
            }
            Allowable => return None,
            DisallowedBy(id) => id,
        };

        // If we're disallowed by a particular id, then we attempt to give a
        // nice error message to say why it was disallowed. It was either
        // because the item itself is private or because its parent is private
        // and its parent isn't in our ancestry.
        let (err_span, err_msg) = if id == source_did.unwrap_or(to_check).node {
            return Some((span, format!("{} is private", msg), None));
        } else {
            (span, format!("{} is inaccessible", msg))
        };
        let item = match self.tcx.map.find(id) {
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    // If an impl disallowed this item, then this is resolve's
                    // way of saying that a struct/enum's static method was
                    // invoked, and the struct/enum itself is private. Crawl
                    // back up the chains to find the relevant struct/enum that
                    // was private.
                    ast::ItemImpl(_, _, _, _, ref ty, _) => {
                        let id = match ty.node {
                            ast::TyPath(_, id) => id,
                            _ => return Some((err_span, err_msg, None)),
                        };
                        let def = self.tcx.def_map.borrow()[id].clone();
                        let did = def.def_id();
                        assert!(is_local(did));
                        match self.tcx.map.get(did.node) {
                            ast_map::NodeItem(item) => item,
                            _ => self.tcx.sess.span_bug(item.span,
                                                        "path is not an item")
                        }
                    }
                    _ => item
                }
            }
            Some(..) | None => return Some((err_span, err_msg, None)),
        };
        let desc = match item.node {
            ast::ItemMod(..) => "module",
            ast::ItemTrait(..) => "trait",
            ast::ItemStruct(..) => "struct",
            ast::ItemEnum(..) => "enum",
            _ => return Some((err_span, err_msg, None))
        };
        let msg = format!("{} `{}` is private", desc,
                          token::get_ident(item.ident));
        Some((err_span, err_msg, Some((span, msg))))
    }

    // Checks that a field is in scope.
    fn check_field(&mut self,
                   span: Span,
                   id: ast::DefId,
                   name: FieldName) {
        let fields = ty::lookup_struct_fields(self.tcx, id);
        let field = match name {
            NamedField(f_name) => {
                debug!("privacy - check named field {} in struct {:?}", f_name, id);
                fields.iter().find(|f| f.name == f_name).unwrap()
            }
            UnnamedField(idx) => &fields[idx]
        };
        if field.vis == ast::Public ||
            (is_local(field.id) && self.private_accessible(field.id.node)) {
            return
        }

        let struct_type = ty::lookup_item_type(self.tcx, id).ty;
        let struct_desc = match struct_type.sty {
            ty::ty_struct(_, _) =>
                format!("struct `{}`", ty::item_path_str(self.tcx, id)),
            // struct variant fields have inherited visibility
            ty::ty_enum(..) => return,
            _ => self.tcx.sess.span_bug(span, "can't find struct for field")
        };
        let msg = match name {
            NamedField(name) => format!("field `{}` of {} is private",
                                        token::get_name(name), struct_desc),
            UnnamedField(idx) => format!("field #{} of {} is private",
                                         idx + 1, struct_desc),
        };
        self.tcx.sess.span_err(span, &msg[..]);
    }

    // Given the ID of a method, checks to ensure it's in scope.
    fn check_static_method(&mut self,
                           span: Span,
                           method_id: ast::DefId,
                           name: ast::Ident) {
        // If the method is a default method, we need to use the def_id of
        // the default implementation.
        let method_id = match ty::impl_or_trait_item(self.tcx, method_id) {
            ty::MethodTraitItem(method_type) => {
                method_type.provided_source.unwrap_or(method_id)
            }
            ty::TypeTraitItem(_) => method_id,
        };

        let string = token::get_ident(name);
        self.report_error(self.ensure_public(span,
                                             method_id,
                                             None,
                                             &format!("method `{}`",
                                                     string)[]));
    }

    // Checks that a path is in scope.
    fn check_path(&mut self, span: Span, path_id: ast::NodeId, path: &ast::Path) {
        debug!("privacy - path {}", self.nodestr(path_id));
        let orig_def = self.tcx.def_map.borrow()[path_id].clone();
        let ck = |tyname: &str| {
            let ck_public = |def: ast::DefId| {
                debug!("privacy - ck_public {:?}", def);
                let name = token::get_ident(path.segments.last().unwrap().identifier);
                let origdid = orig_def.def_id();
                self.ensure_public(span,
                                   def,
                                   Some(origdid),
                                   &format!("{} `{}`", tyname, name)[])
            };

            match self.last_private_map[path_id] {
                LastMod(AllPublic) => {},
                LastMod(DependsOn(def)) => {
                    self.report_error(ck_public(def));
                },
                LastImport { value_priv,
                             value_used: check_value,
                             type_priv,
                             type_used: check_type } => {
                    // This dance with found_error is because we don't want to
                    // report a privacy error twice for the same directive.
                    let found_error = match (type_priv, check_type) {
                        (Some(DependsOn(def)), Used) => {
                            !self.report_error(ck_public(def))
                        },
                        _ => false,
                    };
                    if !found_error {
                        match (value_priv, check_value) {
                            (Some(DependsOn(def)), Used) => {
                                self.report_error(ck_public(def));
                            },
                            _ => {},
                        }
                    }
                    // If an import is not used in either namespace, we still
                    // want to check that it could be legal. Therefore we check
                    // in both namespaces and only report an error if both would
                    // be illegal. We only report one error, even if it is
                    // illegal to import from both namespaces.
                    match (value_priv, check_value, type_priv, check_type) {
                        (Some(p), Unused, None, _) |
                        (None, _, Some(p), Unused) => {
                            let p = match p {
                                AllPublic => None,
                                DependsOn(def) => ck_public(def),
                            };
                            if p.is_some() {
                                self.report_error(p);
                            }
                        },
                        (Some(v), Unused, Some(t), Unused) => {
                            let v = match v {
                                AllPublic => None,
                                DependsOn(def) => ck_public(def),
                            };
                            let t = match t {
                                AllPublic => None,
                                DependsOn(def) => ck_public(def),
                            };
                            if let (Some(_), Some(t)) = (v, t) {
                                self.report_error(Some(t));
                            }
                        },
                        _ => {},
                    }
                },
            }
        };
        // FIXME(#12334) Imports can refer to definitions in both the type and
        // value namespaces. The privacy information is aware of this, but the
        // def map is not. Therefore the names we work out below will not always
        // be accurate and we can get slightly wonky error messages (but type
        // checking is always correct).
        match self.tcx.def_map.borrow()[path_id].clone() {
            def::DefStaticMethod(..) => ck("static method"),
            def::DefFn(..) => ck("function"),
            def::DefStatic(..) => ck("static"),
            def::DefConst(..) => ck("const"),
            def::DefVariant(..) => ck("variant"),
            def::DefTy(_, false) => ck("type"),
            def::DefTy(_, true) => ck("enum"),
            def::DefTrait(..) => ck("trait"),
            def::DefStruct(..) => ck("struct"),
            def::DefMethod(_, Some(..), _) => ck("trait method"),
            def::DefMethod(..) => ck("method"),
            def::DefMod(..) => ck("module"),
            _ => {}
        }
    }

    // Checks that a method is in scope.
    fn check_method(&mut self, span: Span, origin: &MethodOrigin,
                    ident: ast::Ident) {
        match *origin {
            MethodStatic(method_id) => {
                self.check_static_method(span, method_id, ident)
            }
            MethodStaticClosure(_) => {}
            // Trait methods are always all public. The only controlling factor
            // is whether the trait itself is accessible or not.
            MethodTypeParam(MethodParam { ref trait_ref, .. }) |
            MethodTraitObject(MethodObject { ref trait_ref, .. }) => {
                self.report_error(self.ensure_public(span, trait_ref.def_id,
                                                     None, "source trait"));
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for PrivacyVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        match item.node {
            ast::ItemUse(ref vpath) => {
                match vpath.node {
                    ast::ViewPathSimple(..) | ast::ViewPathGlob(..) => {}
                    ast::ViewPathList(ref prefix, ref list) => {
                        for pid in list {
                            match pid.node {
                                ast::PathListIdent { id, name } => {
                                    debug!("privacy - ident item {}", id);
                                    let seg = ast::PathSegment {
                                        identifier: name,
                                        parameters: ast::PathParameters::none(),
                                    };
                                    let segs = vec![seg];
                                    let path = ast::Path {
                                        global: false,
                                        span: pid.span,
                                        segments: segs,
                                    };
                                    self.check_path(pid.span, id, &path);
                                }
                                ast::PathListMod { id } => {
                                    debug!("privacy - mod item {}", id);
                                    self.check_path(pid.span, id, prefix);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        let orig_curitem = replace(&mut self.curitem, item.id);
        visit::walk_item(self, item);
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: &ast::Expr) {
        match expr.node {
            ast::ExprField(ref base, ident) => {
                if let ty::ty_struct(id, _) = ty::expr_ty_adjusted(self.tcx, &**base).sty {
                    self.check_field(expr.span, id, NamedField(ident.node.name));
                }
            }
            ast::ExprTupField(ref base, idx) => {
                if let ty::ty_struct(id, _) = ty::expr_ty_adjusted(self.tcx, &**base).sty {
                    self.check_field(expr.span, id, UnnamedField(idx.node));
                }
            }
            ast::ExprMethodCall(ident, _, _) => {
                let method_call = MethodCall::expr(expr.id);
                match self.tcx.method_map.borrow().get(&method_call) {
                    None => {
                        self.tcx.sess.span_bug(expr.span,
                                                "method call not in \
                                                method map");
                    }
                    Some(method) => {
                        debug!("(privacy checking) checking impl method");
                        self.check_method(expr.span, &method.origin, ident.node);
                    }
                }
            }
            ast::ExprStruct(_, ref fields, _) => {
                match ty::expr_ty(self.tcx, expr).sty {
                    ty::ty_struct(ctor_id, _) => {
                        // RFC 736: ensure all unmentioned fields are visible.
                        // Rather than computing the set of unmentioned fields
                        // (i.e. `all_fields - fields`), just check them all.
                        let all_fields = ty::lookup_struct_fields(self.tcx, ctor_id);
                        for field in all_fields {
                            self.check_field(expr.span, ctor_id,
                                             NamedField(field.name));
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.borrow()[expr.id].clone() {
                            def::DefVariant(_, variant_id, _) => {
                                for field in fields {
                                    self.check_field(expr.span, variant_id,
                                                     NamedField(field.ident.node.name));
                                }
                            }
                            _ => self.tcx.sess.span_bug(expr.span,
                                                        "resolve didn't \
                                                         map enum struct \
                                                         constructor to a \
                                                         variant def"),
                        }
                    }
                    _ => self.tcx.sess.span_bug(expr.span, "struct expr \
                                                            didn't have \
                                                            struct type?!"),
                }
            }
            ast::ExprPath(_) | ast::ExprQPath(_) => {
                let guard = |did: ast::DefId| {
                    let fields = ty::lookup_struct_fields(self.tcx, did);
                    let any_priv = fields.iter().any(|f| {
                        f.vis != ast::Public && (
                            !is_local(f.id) ||
                            !self.private_accessible(f.id.node))
                    });
                    if any_priv {
                        self.tcx.sess.span_err(expr.span,
                            "cannot invoke tuple struct constructor \
                             with private fields");
                    }
                };
                match self.tcx.def_map.borrow().get(&expr.id) {
                    Some(&def::DefStruct(did)) => {
                        guard(if is_local(did) {
                            local_def(self.tcx.map.get_parent(did.node))
                        } else {
                            // "tuple structs" with zero fields (such as
                            // `pub struct Foo;`) don't have a ctor_id, hence
                            // the unwrap_or to the same struct id.
                            let maybe_did =
                                csearch::get_tuple_struct_definition_if_ctor(
                                    &self.tcx.sess.cstore, did);
                            maybe_did.unwrap_or(did)
                        })
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pattern: &ast::Pat) {
        // Foreign functions do not have their patterns mapped in the def_map,
        // and there's nothing really relevant there anyway, so don't bother
        // checking privacy. If you can name the type then you can pass it to an
        // external C function anyway.
        if self.in_foreign { return }

        match pattern.node {
            ast::PatStruct(_, ref fields, _) => {
                match ty::pat_ty(self.tcx, pattern).sty {
                    ty::ty_struct(id, _) => {
                        for field in fields {
                            self.check_field(pattern.span, id,
                                             NamedField(field.node.ident.name));
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.borrow().get(&pattern.id) {
                            Some(&def::DefVariant(_, variant_id, _)) => {
                                for field in fields {
                                    self.check_field(pattern.span, variant_id,
                                                     NamedField(field.node.ident.name));
                                }
                            }
                            _ => self.tcx.sess.span_bug(pattern.span,
                                                        "resolve didn't \
                                                         map enum struct \
                                                         pattern to a \
                                                         variant def"),
                        }
                    }
                    _ => self.tcx.sess.span_bug(pattern.span,
                                                "struct pattern didn't have \
                                                 struct type?!"),
                }
            }

            // Patterns which bind no fields are allowable (the path is check
            // elsewhere).
            ast::PatEnum(_, Some(ref fields)) => {
                match ty::pat_ty(self.tcx, pattern).sty {
                    ty::ty_struct(id, _) => {
                        for (i, field) in fields.iter().enumerate() {
                            if let ast::PatWild(..) = field.node {
                                continue
                            }
                            self.check_field(field.span, id, UnnamedField(i));
                        }
                    }
                    ty::ty_enum(..) => {
                        // enum fields have no privacy at this time
                    }
                    _ => {}
                }

            }
            _ => {}
        }

        visit::walk_pat(self, pattern);
    }

    fn visit_foreign_item(&mut self, fi: &ast::ForeignItem) {
        self.in_foreign = true;
        visit::walk_foreign_item(self, fi);
        self.in_foreign = false;
    }

    fn visit_path(&mut self, path: &ast::Path, id: ast::NodeId) {
        self.check_path(path.span, id, path);
        visit::walk_path(self, path);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy sanity check visitor, ensures unnecessary visibility isn't here
////////////////////////////////////////////////////////////////////////////////

struct SanePrivacyVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    in_fn: bool,
}

impl<'a, 'tcx, 'v> Visitor<'v> for SanePrivacyVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        if self.in_fn {
            self.check_all_inherited(item);
        } else {
            self.check_sane_privacy(item);
        }

        let in_fn = self.in_fn;
        let orig_in_fn = replace(&mut self.in_fn, match item.node {
            ast::ItemMod(..) => false, // modules turn privacy back on
            _ => in_fn,           // otherwise we inherit
        });
        visit::walk_item(self, item);
        self.in_fn = orig_in_fn;
    }

    fn visit_fn(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, s: Span, _: ast::NodeId) {
        // This catches both functions and methods
        let orig_in_fn = replace(&mut self.in_fn, true);
        visit::walk_fn(self, fk, fd, b, s);
        self.in_fn = orig_in_fn;
    }
}

impl<'a, 'tcx> SanePrivacyVisitor<'a, 'tcx> {
    /// Validates all of the visibility qualifiers placed on the item given. This
    /// ensures that there are no extraneous qualifiers that don't actually do
    /// anything. In theory these qualifiers wouldn't parse, but that may happen
    /// later on down the road...
    fn check_sane_privacy(&self, item: &ast::Item) {
        let tcx = self.tcx;
        let check_inherited = |sp: Span, vis: ast::Visibility, note: &str| {
            if vis != ast::Inherited {
                tcx.sess.span_err(sp, "unnecessary visibility qualifier");
                if note.len() > 0 {
                    tcx.sess.span_note(sp, note);
                }
            }
        };
        match item.node {
            // implementations of traits don't need visibility qualifiers because
            // that's controlled by having the trait in scope.
            ast::ItemImpl(_, _, _, Some(..), _, ref impl_items) => {
                check_inherited(item.span, item.vis,
                                "visibility qualifiers have no effect on trait \
                                 impls");
                for impl_item in impl_items {
                    match *impl_item {
                        ast::MethodImplItem(ref m) => {
                            check_inherited(m.span, m.pe_vis(), "");
                        }
                        ast::TypeImplItem(_) => {}
                    }
                }
            }

            ast::ItemImpl(..) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual methods instead");
            }
            ast::ItemForeignMod(..) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual functions \
                                 instead");
            }

            ast::ItemEnum(ref def, _) => {
                for v in &def.variants {
                    match v.node.vis {
                        ast::Public => {
                            if item.vis == ast::Public {
                                tcx.sess.span_err(v.span, "unnecessary `pub` \
                                                           visibility");
                            }
                        }
                        ast::Inherited => {}
                    }
                }
            }

            ast::ItemTrait(_, _, _, ref methods) => {
                for m in methods {
                    match *m {
                        ast::ProvidedMethod(ref m) => {
                            check_inherited(m.span, m.pe_vis(),
                                            "unnecessary visibility");
                        }
                        ast::RequiredMethod(ref m) => {
                            check_inherited(m.span, m.vis,
                                            "unnecessary visibility");
                        }
                        ast::TypeTraitItem(_) => {}
                    }
                }
            }

            ast::ItemConst(..) | ast::ItemStatic(..) | ast::ItemStruct(..) |
            ast::ItemFn(..) | ast::ItemMod(..) | ast::ItemTy(..) |
            ast::ItemExternCrate(_) | ast::ItemUse(_) | ast::ItemMac(..) => {}
        }
    }

    /// When inside of something like a function or a method, visibility has no
    /// control over anything so this forbids any mention of any visibility
    fn check_all_inherited(&self, item: &ast::Item) {
        let tcx = self.tcx;
        fn check_inherited(tcx: &ty::ctxt, sp: Span, vis: ast::Visibility) {
            if vis != ast::Inherited {
                tcx.sess.span_err(sp, "visibility has no effect inside functions");
            }
        }
        let check_struct = |def: &ast::StructDef| {
            for f in &def.fields {
               match f.node.kind {
                    ast::NamedField(_, p) => check_inherited(tcx, f.span, p),
                    ast::UnnamedField(..) => {}
                }
            }
        };
        check_inherited(tcx, item.span, item.vis);
        match item.node {
            ast::ItemImpl(_, _, _, _, _, ref impl_items) => {
                for impl_item in impl_items {
                    match *impl_item {
                        ast::MethodImplItem(ref m) => {
                            check_inherited(tcx, m.span, m.pe_vis());
                        }
                        ast::TypeImplItem(_) => {}
                    }
                }
            }
            ast::ItemForeignMod(ref fm) => {
                for i in &fm.items {
                    check_inherited(tcx, i.span, i.vis);
                }
            }
            ast::ItemEnum(ref def, _) => {
                for v in &def.variants {
                    check_inherited(tcx, v.span, v.node.vis);
                }
            }

            ast::ItemStruct(ref def, _) => check_struct(&**def),

            ast::ItemTrait(_, _, _, ref methods) => {
                for m in methods {
                    match *m {
                        ast::RequiredMethod(..) => {}
                        ast::ProvidedMethod(ref m) => check_inherited(tcx, m.span,
                                                                m.pe_vis()),
                        ast::TypeTraitItem(_) => {}
                    }
                }
            }

            ast::ItemExternCrate(_) | ast::ItemUse(_) |
            ast::ItemStatic(..) | ast::ItemConst(..) |
            ast::ItemFn(..) | ast::ItemMod(..) | ast::ItemTy(..) |
            ast::ItemMac(..) => {}
        }
    }
}

struct VisiblePrivateTypesVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    exported_items: &'a ExportedItems,
    public_items: &'a PublicItems,
    in_variant: bool,
}

struct CheckTypeForPrivatenessVisitor<'a, 'b: 'a, 'tcx: 'b> {
    inner: &'a VisiblePrivateTypesVisitor<'b, 'tcx>,
    /// whether the type refers to private types.
    contains_private: bool,
    /// whether we've recurred at all (i.e. if we're pointing at the
    /// first type on which visit_ty was called).
    at_outer_type: bool,
    // whether that first type is a public path.
    outer_type_is_public_path: bool,
}

impl<'a, 'tcx> VisiblePrivateTypesVisitor<'a, 'tcx> {
    fn path_is_private_type(&self, path_id: ast::NodeId) -> bool {
        let did = match self.tcx.def_map.borrow().get(&path_id).cloned() {
            // `int` etc. (None doesn't seem to occur.)
            None | Some(def::DefPrimTy(..)) => return false,
            Some(def) => def.def_id()
        };
        // A path can only be private if:
        // it's in this crate...
        if !is_local(did) {
            return false
        }
        // .. and it corresponds to a private type in the AST (this returns
        // None for type parameters)
        match self.tcx.map.find(did.node) {
            Some(ast_map::NodeItem(ref item)) => item.vis != ast::Public,
            Some(_) | None => false,
        }
    }

    fn trait_is_public(&self, trait_id: ast::NodeId) -> bool {
        // FIXME: this would preferably be using `exported_items`, but all
        // traits are exported currently (see `EmbargoVisitor.exported_trait`)
        self.public_items.contains(&trait_id)
    }

    fn check_ty_param_bound(&self,
                            ty_param_bound: &ast::TyParamBound) {
        if let ast::TraitTyParamBound(ref trait_ref, _) = *ty_param_bound {
            if !self.tcx.sess.features.borrow().visible_private_types &&
                self.path_is_private_type(trait_ref.trait_ref.ref_id) {
                    let span = trait_ref.trait_ref.path.span;
                    self.tcx.sess.span_err(span,
                                           "private trait in exported type \
                                            parameter bound");
            }
        }
    }
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for CheckTypeForPrivatenessVisitor<'a, 'b, 'tcx> {
    fn visit_ty(&mut self, ty: &ast::Ty) {
        if let ast::TyPath(_, path_id) = ty.node {
            if self.inner.path_is_private_type(path_id) {
                self.contains_private = true;
                // found what we're looking for so let's stop
                // working.
                return
            } else if self.at_outer_type {
                self.outer_type_is_public_path = true;
            }
        }
        self.at_outer_type = false;
        visit::walk_ty(self, ty)
    }

    // don't want to recurse into [, .. expr]
    fn visit_expr(&mut self, _: &ast::Expr) {}
}

impl<'a, 'tcx, 'v> Visitor<'v> for VisiblePrivateTypesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        match item.node {
            // contents of a private mod can be reexported, so we need
            // to check internals.
            ast::ItemMod(_) => {}

            // An `extern {}` doesn't introduce a new privacy
            // namespace (the contents have their own privacies).
            ast::ItemForeignMod(_) => {}

            ast::ItemTrait(_, _, ref bounds, _) => {
                if !self.trait_is_public(item.id) {
                    return
                }

                for bound in &**bounds {
                    self.check_ty_param_bound(bound)
                }
            }

            // impls need some special handling to try to offer useful
            // error messages without (too many) false positives
            // (i.e. we could just return here to not check them at
            // all, or some worse estimation of whether an impl is
            // publicly visible.
            ast::ItemImpl(_, _, ref g, ref trait_ref, ref self_, ref impl_items) => {
                // `impl [... for] Private` is never visible.
                let self_contains_private;
                // impl [... for] Public<...>, but not `impl [... for]
                // ~[Public]` or `(Public,)` etc.
                let self_is_public_path;

                // check the properties of the Self type:
                {
                    let mut visitor = CheckTypeForPrivatenessVisitor {
                        inner: self,
                        contains_private: false,
                        at_outer_type: true,
                        outer_type_is_public_path: false,
                    };
                    visitor.visit_ty(&**self_);
                    self_contains_private = visitor.contains_private;
                    self_is_public_path = visitor.outer_type_is_public_path;
                }

                // miscellaneous info about the impl

                // `true` iff this is `impl Private for ...`.
                let not_private_trait =
                    trait_ref.as_ref().map_or(true, // no trait counts as public trait
                                              |tr| {
                        let did = ty::trait_ref_to_def_id(self.tcx, tr);

                        !is_local(did) || self.trait_is_public(did.node)
                    });

                // `true` iff this is a trait impl or at least one method is public.
                //
                // `impl Public { $( fn ...() {} )* }` is not visible.
                //
                // This is required over just using the methods' privacy
                // directly because we might have `impl<T: Foo<Private>> ...`,
                // and we shouldn't warn about the generics if all the methods
                // are private (because `T` won't be visible externally).
                let trait_or_some_public_method =
                    trait_ref.is_some() ||
                    impl_items.iter()
                              .any(|impl_item| {
                                  match *impl_item {
                                      ast::MethodImplItem(ref m) => {
                                          self.exported_items.contains(&m.id)
                                      }
                                      ast::TypeImplItem(_) => false,
                                  }
                              });

                if !self_contains_private &&
                        not_private_trait &&
                        trait_or_some_public_method {

                    visit::walk_generics(self, g);

                    match *trait_ref {
                        None => {
                            for impl_item in impl_items {
                                match *impl_item {
                                    ast::MethodImplItem(ref method) => {
                                        visit::walk_method_helper(self, &**method)
                                    }
                                    ast::TypeImplItem(_) => {}
                                }
                            }
                        }
                        Some(ref tr) => {
                            // Any private types in a trait impl fall into two
                            // categories.
                            // 1. mentioned in the trait definition
                            // 2. mentioned in the type params/generics
                            //
                            // Those in 1. can only occur if the trait is in
                            // this crate and will've been warned about on the
                            // trait definition (there's no need to warn twice
                            // so we don't check the methods).
                            //
                            // Those in 2. are warned via walk_generics and this
                            // call here.
                            self.visit_trait_ref(tr)
                        }
                    }
                } else if trait_ref.is_none() && self_is_public_path {
                    // impl Public<Private> { ... }. Any public static
                    // methods will be visible as `Public::foo`.
                    let mut found_pub_static = false;
                    for impl_item in impl_items {
                        match *impl_item {
                            ast::MethodImplItem(ref method) => {
                                if method.pe_explicit_self().node ==
                                        ast::SelfStatic &&
                                        self.exported_items
                                            .contains(&method.id) {
                                    found_pub_static = true;
                                    visit::walk_method_helper(self, &**method);
                                }
                            }
                            ast::TypeImplItem(_) => {}
                        }
                    }
                    if found_pub_static {
                        visit::walk_generics(self, g)
                    }
                }
                return
            }

            // `type ... = ...;` can contain private types, because
            // we're introducing a new name.
            ast::ItemTy(..) => return,

            // not at all public, so we don't care
            _ if !self.exported_items.contains(&item.id) => return,

            _ => {}
        }

        // we've carefully constructed it so that if we're here, then
        // any `visit_ty`'s will be called on things that are in
        // public signatures, i.e. things that we're interested in for
        // this visitor.
        visit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &ast::Generics) {
        for ty_param in &*generics.ty_params {
            for bound in &*ty_param.bounds {
                self.check_ty_param_bound(bound)
            }
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &ast::WherePredicate::BoundPredicate(ref bound_pred) => {
                    for bound in &*bound_pred.bounds {
                        self.check_ty_param_bound(bound)
                    }
                }
                &ast::WherePredicate::RegionPredicate(_) => {}
                &ast::WherePredicate::EqPredicate(ref eq_pred) => {
                    self.visit_ty(&*eq_pred.ty);
                }
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &ast::ForeignItem) {
        if self.exported_items.contains(&item.id) {
            visit::walk_foreign_item(self, item)
        }
    }

    fn visit_fn(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, s: Span, id: ast::NodeId) {
        // needs special handling for methods.
        if self.exported_items.contains(&id) {
            visit::walk_fn(self, fk, fd, b, s);
        }
    }

    fn visit_ty(&mut self, t: &ast::Ty) {
        if let ast::TyPath(ref p, path_id) = t.node {
            if !self.tcx.sess.features.borrow().visible_private_types &&
                self.path_is_private_type(path_id) {
                self.tcx.sess.span_err(p.span,
                                       "private type in exported type signature");
            }
        }
        visit::walk_ty(self, t)
    }

    fn visit_variant(&mut self, v: &ast::Variant, g: &ast::Generics) {
        if self.exported_items.contains(&v.node.id) {
            self.in_variant = true;
            visit::walk_variant(self, v, g);
            self.in_variant = false;
        }
    }

    fn visit_struct_field(&mut self, s: &ast::StructField) {
        match s.node.kind {
            ast::NamedField(_, vis) if vis == ast::Public || self.in_variant => {
                visit::walk_struct_field(self, s);
            }
            _ => {}
        }
    }


    // we don't need to introspect into these at all: an
    // expression/block context can't possibly contain exported things.
    // (Making them no-ops stops us from traversing the whole AST without
    // having to be super careful about our `walk_...` calls above.)
    fn visit_block(&mut self, _: &ast::Block) {}
    fn visit_expr(&mut self, _: &ast::Expr) {}
}

pub fn check_crate(tcx: &ty::ctxt,
                   export_map: &def::ExportMap,
                   external_exports: ExternalExports,
                   last_private_map: LastPrivateMap)
                   -> (ExportedItems, PublicItems) {
    let krate = tcx.map.krate();

    // Figure out who everyone's parent is
    let mut visitor = ParentVisitor {
        parents: NodeMap(),
        curparent: ast::DUMMY_NODE_ID,
    };
    visit::walk_crate(&mut visitor, krate);

    // Use the parent map to check the privacy of everything
    let mut visitor = PrivacyVisitor {
        curitem: ast::DUMMY_NODE_ID,
        in_foreign: false,
        tcx: tcx,
        parents: visitor.parents,
        external_exports: external_exports,
        last_private_map: last_private_map,
    };
    visit::walk_crate(&mut visitor, krate);

    // Sanity check to make sure that all privacy usage and controls are
    // reasonable.
    let mut visitor = SanePrivacyVisitor {
        in_fn: false,
        tcx: tcx,
    };
    visit::walk_crate(&mut visitor, krate);

    tcx.sess.abort_if_errors();

    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx: tcx,
        exported_items: NodeSet(),
        public_items: NodeSet(),
        reexports: NodeSet(),
        export_map: export_map,
        prev_exported: true,
        prev_public: true,
    };
    loop {
        let before = visitor.exported_items.len();
        visit::walk_crate(&mut visitor, krate);
        if before == visitor.exported_items.len() {
            break
        }
    }

    let EmbargoVisitor { exported_items, public_items, .. } = visitor;

    {
        let mut visitor = VisiblePrivateTypesVisitor {
            tcx: tcx,
            exported_items: &exported_items,
            public_items: &public_items,
            in_variant: false,
        };
        visit::walk_crate(&mut visitor, krate);
    }
    return (exported_items, public_items);
}
