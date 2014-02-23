// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.

use std::mem::replace;
use collections::{HashSet, HashMap};

use metadata::csearch;
use middle::resolve;
use middle::ty;
use middle::typeck::{method_map, method_origin, method_param};
use middle::typeck::{method_static, method_object};

use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{is_local, def_id_of_def, local_def};
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::opt_vec;
use syntax::visit;
use syntax::visit::Visitor;

type Context<'a> = (&'a method_map, &'a resolve::ExportMap2);

/// A set of AST nodes exported by the crate.
pub type ExportedItems = HashSet<ast::NodeId>;

/// A set of AST nodes that are fully public in the crate. This map is used for
/// documentation purposes (reexporting a private struct inlines the doc,
/// reexporting a public struct doesn't inline the doc).
pub type PublicItems = HashSet<ast::NodeId>;

/// Result of a checking operation - None => no errors were found. Some => an
/// error and contains the span and message for reporting that error and
/// optionally the same for a note about the error.
type CheckResult = Option<(Span, ~str, Option<(Span, ~str)>)>;

////////////////////////////////////////////////////////////////////////////////
/// The parent visitor, used to determine what's the parent of what (node-wise)
////////////////////////////////////////////////////////////////////////////////

struct ParentVisitor {
    parents: HashMap<ast::NodeId, ast::NodeId>,
    curparent: ast::NodeId,
}

impl Visitor<()> for ParentVisitor {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        self.parents.insert(item.id, self.curparent);

        let prev = self.curparent;
        match item.node {
            ast::ItemMod(..) => { self.curparent = item.id; }
            // Enum variants are parented to the enum definition itself beacuse
            // they inherit privacy
            ast::ItemEnum(ref def, _) => {
                for variant in def.variants.iter() {
                    // If variants are private, then their logical "parent" is
                    // the enclosing module because everyone in the enclosing
                    // module can still use the private variant
                    if variant.node.vis == ast::Private {
                        self.parents.insert(variant.node.id, self.curparent);

                    // Otherwise, if the variant is public, then the parent is
                    // considered the enclosing enum because the enum will
                    // dictate the privacy visibility of this variant instead.
                    } else {
                        self.parents.insert(variant.node.id, item.id);
                    }
                }
            }

            // Trait methods are always considered "public", but if the trait is
            // private then we need some private item in the chain from the
            // method to the root. In this case, if the trait is private, then
            // parent all the methods to the trait to indicate that they're
            // private.
            ast::ItemTrait(_, _, ref methods) if item.vis != ast::Public => {
                for m in methods.iter() {
                    match *m {
                        ast::Provided(ref m) => self.parents.insert(m.id, item.id),
                        ast::Required(ref m) => self.parents.insert(m.id, item.id),
                    };
                }
            }

            _ => {}
        }
        visit::walk_item(self, item, ());
        self.curparent = prev;
    }

    fn visit_foreign_item(&mut self, a: &ast::ForeignItem, _: ()) {
        self.parents.insert(a.id, self.curparent);
        visit::walk_foreign_item(self, a, ());
    }

    fn visit_fn(&mut self, a: &visit::FnKind, b: &ast::FnDecl,
                c: &ast::Block, d: Span, id: ast::NodeId, _: ()) {
        // We already took care of some trait methods above, otherwise things
        // like impl methods and pub trait methods are parented to the
        // containing module, not the containing trait.
        if !self.parents.contains_key(&id) {
            self.parents.insert(id, self.curparent);
        }
        visit::walk_fn(self, a, b, c, d, id, ());
    }

    fn visit_struct_def(&mut self, s: &ast::StructDef, i: ast::Ident,
                        g: &ast::Generics, n: ast::NodeId, _: ()) {
        // Struct constructors are parented to their struct definitions because
        // they essentially are the struct definitions.
        match s.ctor_id {
            Some(id) => { self.parents.insert(id, n); }
            None => {}
        }

        // While we have the id of the struct definition, go ahead and parent
        // all the fields.
        for field in s.fields.iter() {
            self.parents.insert(field.node.id, self.curparent);
        }
        visit::walk_struct_def(self, s, i, g, n, ())
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the ast
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'a> {
    tcx: ty::ctxt,
    exp_map2: &'a resolve::ExportMap2,

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
    reexports: HashSet<ast::NodeId>,

    // These two fields are closely related to one another in that they are only
    // used for generation of the 'PublicItems' set, not for privacy checking at
    // all
    public_items: PublicItems,
    prev_public: bool,
}

impl<'a> EmbargoVisitor<'a> {
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

impl<'a> Visitor<()> for EmbargoVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
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
                for variant in def.variants.iter() {
                    if variant.node.vis != ast::Private {
                        self.exported_items.insert(variant.node.id);
                    }
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
            ast::ItemImpl(_, _, ref ty, ref methods) => {
                let public_ty = match ty.node {
                    ast::TyPath(_, _, id) => {
                        let def_map = self.tcx.def_map.borrow();
                        match def_map.get().get_copy(&id) {
                            ast::DefPrimTy(..) => true,
                            def => {
                                let did = def_id_of_def(def);
                                !is_local(did) ||
                                 self.exported_items.contains(&did.node)
                            }
                        }
                    }
                    _ => true,
                };
                let tr = ty::impl_trait_ref(self.tcx, local_def(item.id));
                let public_trait = tr.map_or(false, |tr| {
                    !is_local(tr.def_id) ||
                     self.exported_items.contains(&tr.def_id.node)
                });

                if public_ty || public_trait {
                    for method in methods.iter() {
                        let meth_public = match method.explicit_self.node {
                            ast::SelfStatic => public_ty,
                            _ => true,
                        } && method.vis == ast::Public;
                        if meth_public || tr.is_some() {
                            self.exported_items.insert(method.id);
                        }
                    }
                }
            }

            // Default methods on traits are all public so long as the trait
            // is public
            ast::ItemTrait(_, _, ref methods) if public_first => {
                for method in methods.iter() {
                    match *method {
                        ast::Provided(ref m) => {
                            debug!("provided {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                        ast::Required(ref m) => {
                            debug!("required {}", m.id);
                            self.exported_items.insert(m.id);
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

            _ => {}
        }

        visit::walk_item(self, item, ());

        self.prev_exported = orig_all_exported;
        self.prev_public = orig_all_pub;
    }

    fn visit_foreign_item(&mut self, a: &ast::ForeignItem, _: ()) {
        if self.prev_exported && a.vis == ast::Public {
            self.exported_items.insert(a.id);
        }
    }

    fn visit_mod(&mut self, m: &ast::Mod, _sp: Span, id: ast::NodeId, _: ()) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_exported {
            let exp_map2 = self.exp_map2.borrow();
            assert!(exp_map2.get().contains_key(&id), "wut {:?}", id);
            for export in exp_map2.get().get(&id).iter() {
                if is_local(export.def_id) {
                    self.reexports.insert(export.def_id.node);
                }
            }
        }
        visit::walk_mod(self, m, ())
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy visitor, where privacy checks take place (violations reported)
////////////////////////////////////////////////////////////////////////////////

struct PrivacyVisitor<'a> {
    tcx: ty::ctxt,
    curitem: ast::NodeId,
    in_fn: bool,
    in_foreign: bool,
    method_map: &'a method_map,
    parents: HashMap<ast::NodeId, ast::NodeId>,
    external_exports: resolve::ExternalExports,
    last_private_map: resolve::LastPrivateMap,
}

enum PrivacyResult {
    Allowable,
    ExternallyDenied,
    DisallowedBy(ast::NodeId),
}

impl<'a> PrivacyVisitor<'a> {
    // used when debugging
    fn nodestr(&self, id: ast::NodeId) -> ~str {
        self.tcx.map.node_to_str(id)
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

            let methods = self.tcx.methods.borrow();
            return match methods.get().find(&did) {
                Some(meth) => {
                    debug!("privacy - well at least it's a method: {:?}", meth);
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
                None => {
                    debug!("privacy - nope, not even a method");
                    ExternallyDenied
                }
            };
        }

        debug!("privacy - local {:?} not public all the way down", did);
        // return quickly for things in the same module
        if self.parents.find(&did.node) == self.parents.find(&self.curitem) {
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
                Some(ast_map::NodeMethod(ref m)) => {
                    let imp = self.tcx.map.get_parent_did(closest_private_id);
                    match ty::impl_trait_ref(self.tcx, imp) {
                        Some(..) => return Allowable,
                        _ if m.vis == ast::Public => return Allowable,
                        _ => m.vis
                    }
                }
                Some(ast_map::NodeTraitMethod(_)) => {
                    return Allowable;
                }

                // This is not a method call, extract the visibility as one
                // would normally look at it
                Some(ast_map::NodeItem(it)) => it.vis,
                Some(ast_map::NodeForeignItem(_)) => {
                    self.tcx.map.get_foreign_vis(closest_private_id)
                }
                Some(ast_map::NodeVariant(ref v)) => {
                    // sadly enum variants still inherit visibility, so only
                    // break out of this is explicitly private
                    if v.node.vis == ast::Private { break }
                    ast::Public // need to move up a level (to the enum)
                }
                _ => ast::Public,
            };
            if vis != ast::Public { break }
            // if we've reached the root, then everything was allowable and this
            // access is public.
            if closest_private_id == ast::CRATE_NODE_ID { return Allowable }
            closest_private_id = *self.parents.get(&closest_private_id);

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
        let parent = *self.parents.get(&id);
        debug!("privacy - accessible parent {}", self.nodestr(parent));

        // After finding `did`'s closest private member, we roll ourselves back
        // to see if this private member's parent is anywhere in our ancestry.
        // By the privacy rules, we can access all of our ancestor's private
        // members, so that's why we test the parent, and not the did itself.
        let mut cur = self.curitem;
        loop {
            debug!("privacy - questioning {}", self.nodestr(cur));
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

            cur = *self.parents.get(&cur);
        }
    }

    fn report_error(&self, result: CheckResult) -> bool {
        match result {
            None => true,
            Some((span, msg, note)) => {
                self.tcx.sess.span_err(span, msg);
                match note {
                    Some((span, msg)) => self.tcx.sess.span_note(span, msg),
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
        match self.def_privacy(to_check) {
            ExternallyDenied => Some((span, format!("{} is private", msg), None)),
            DisallowedBy(id) => {
                let (err_span, err_msg) = if id == source_did.unwrap_or(to_check).node {
                    return Some((span, format!("{} is private", msg), None));
                } else {
                    (span, format!("{} is inaccessible", msg))
                };
                match self.tcx.map.find(id) {
                    Some(ast_map::NodeItem(item)) => {
                        let desc = match item.node {
                            ast::ItemMod(..) => "module",
                            ast::ItemTrait(..) => "trait",
                            _ => return Some((err_span, err_msg, None)),
                        };
                        let msg = format!("{} `{}` is private",
                                          desc,
                                          token::get_ident(item.ident));
                        Some((err_span, err_msg, Some((span, msg))))
                    },
                    _ => Some((err_span, err_msg, None)),
                }
            },
            Allowable => None,
        }
    }

    // Checks that a field is in scope.
    // FIXME #6993: change type (and name) from Ident to Name
    fn check_field(&mut self, span: Span, id: ast::DefId, ident: ast::Ident,
                   enum_id: Option<ast::DefId>) {
        let fields = ty::lookup_struct_fields(self.tcx, id);
        let struct_vis = if is_local(id) {
            match self.tcx.map.get(id.node) {
                ast_map::NodeItem(ref it) => it.vis,
                ast_map::NodeVariant(ref v) => {
                    if v.node.vis == ast::Inherited {
                        let parent = self.tcx.map.get_parent(id.node);
                        self.tcx.map.expect_item(parent).vis
                    } else {
                        v.node.vis
                    }
                }
                _ => {
                    self.tcx.sess.span_bug(span,
                                           format!("not an item or variant def"));
                }
            }
        } else {
            let cstore = self.tcx.sess.cstore;
            match enum_id {
                Some(enum_id) => {
                    let v = csearch::get_enum_variants(self.tcx, enum_id);
                    match v.iter().find(|v| v.id == id) {
                        Some(variant) => {
                            if variant.vis == ast::Inherited {
                                csearch::get_item_visibility(cstore, enum_id)
                            } else {
                                variant.vis
                            }
                        }
                        None => {
                            self.tcx.sess.span_bug(span, "no xcrate variant");
                        }
                    }
                }
                None => csearch::get_item_visibility(cstore, id)
            }
        };

        for field in fields.iter() {
            if field.name != ident.name { continue; }
            // public structs have public fields by default, and private structs
            // have private fields by default.
            if struct_vis == ast::Public && field.vis != ast::Private { break }
            if struct_vis != ast::Public && field.vis == ast::Public { break }
            if !is_local(field.id) ||
               !self.private_accessible(field.id.node) {
                self.tcx.sess.span_err(span,
                                       format!("field `{}` is private",
                                               token::get_ident(ident)))
            }
            break;
        }
    }

    // Given the ID of a method, checks to ensure it's in scope.
    fn check_static_method(&mut self, span: Span, method_id: ast::DefId,
                           name: ast::Ident) {
        // If the method is a default method, we need to use the def_id of
        // the default implementation.
        let method_id = ty::method(self.tcx, method_id).provided_source
                                                       .unwrap_or(method_id);

        let string = token::get_ident(name);
        self.report_error(self.ensure_public(span,
                                             method_id,
                                             None,
                                             format!("method `{}`", string)));
    }

    // Checks that a path is in scope.
    fn check_path(&mut self, span: Span, path_id: ast::NodeId, path: &ast::Path) {
        debug!("privacy - path {}", self.nodestr(path_id));
        let def_map = self.tcx.def_map.borrow();
        let orig_def = def_map.get().get_copy(&path_id);
        let ck = |tyname: &str| {
            let ck_public = |def: ast::DefId| {
                let name = token::get_ident(path.segments
                                                .last()
                                                .unwrap()
                                                .identifier);
                let origdid = def_id_of_def(orig_def);
                self.ensure_public(span,
                                   def,
                                   Some(origdid),
                                   format!("{} `{}`",
                                           tyname,
                                           name))
            };

            match *self.last_private_map.get(&path_id) {
                resolve::LastMod(resolve::AllPublic) => {},
                resolve::LastMod(resolve::DependsOn(def)) => {
                    self.report_error(ck_public(def));
                },
                resolve::LastImport{value_priv: value_priv,
                                    value_used: check_value,
                                    type_priv: type_priv,
                                    type_used: check_type} => {
                    // This dance with found_error is because we don't want to report
                    // a privacy error twice for the same directive.
                    let found_error = match (type_priv, check_type) {
                        (Some(resolve::DependsOn(def)), resolve::Used) => {
                            !self.report_error(ck_public(def))
                        },
                        _ => false,
                    };
                    if !found_error {
                        match (value_priv, check_value) {
                            (Some(resolve::DependsOn(def)), resolve::Used) => {
                                self.report_error(ck_public(def));
                            },
                            _ => {},
                        }
                    }
                    // If an import is not used in either namespace, we still want to check
                    // that it could be legal. Therefore we check in both namespaces and only
                    // report an error if both would be illegal. We only report one error,
                    // even if it is illegal to import from both namespaces.
                    match (value_priv, check_value, type_priv, check_type) {
                        (Some(p), resolve::Unused, None, _) |
                        (None, _, Some(p), resolve::Unused) => {
                            let p = match p {
                                resolve::AllPublic => None,
                                resolve::DependsOn(def) => ck_public(def),
                            };
                            if p.is_some() {
                                self.report_error(p);
                            }
                        },
                        (Some(v), resolve::Unused, Some(t), resolve::Unused) => {
                            let v = match v {
                                resolve::AllPublic => None,
                                resolve::DependsOn(def) => ck_public(def),
                            };
                            let t = match t {
                                resolve::AllPublic => None,
                                resolve::DependsOn(def) => ck_public(def),
                            };
                            match (v, t) {
                                (Some(_), Some(t)) => {
                                    self.report_error(Some(t));
                                },
                                _ => {},
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
        let def_map = self.tcx.def_map.borrow();
        match def_map.get().get_copy(&path_id) {
            ast::DefStaticMethod(..) => ck("static method"),
            ast::DefFn(..) => ck("function"),
            ast::DefStatic(..) => ck("static"),
            ast::DefVariant(..) => ck("variant"),
            ast::DefTy(..) => ck("type"),
            ast::DefTrait(..) => ck("trait"),
            ast::DefStruct(..) => ck("struct"),
            ast::DefMethod(_, Some(..)) => ck("trait method"),
            ast::DefMethod(..) => ck("method"),
            ast::DefMod(..) => ck("module"),
            _ => {}
        }
    }

    // Checks that a method is in scope.
    fn check_method(&mut self, span: Span, origin: &method_origin,
                    ident: ast::Ident) {
        match *origin {
            method_static(method_id) => {
                self.check_static_method(span, method_id, ident)
            }
            // Trait methods are always all public. The only controlling factor
            // is whether the trait itself is accessible or not.
            method_param(method_param { trait_id: trait_id, .. }) |
            method_object(method_object { trait_id: trait_id, .. }) => {
                self.report_error(self.ensure_public(span, trait_id, None, "source trait"));
            }
        }
    }
}

impl<'a> Visitor<()> for PrivacyVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        // Do not check privacy inside items with the resolve_unexported
        // attribute. This is used for the test runner.
        if attr::contains_name(item.attrs, "!resolve_unexported") {
            return;
        }

        let orig_curitem = replace(&mut self.curitem, item.id);
        visit::walk_item(self, item, ());
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: &ast::Expr, _: ()) {
        match expr.node {
            ast::ExprField(base, ident, _) => {
                // Method calls are now a special syntactic form,
                // so `a.b` should always be a field.
                let method_map = self.method_map.borrow();
                assert!(!method_map.get().contains_key(&expr.id));

                // With type_autoderef, make sure we don't
                // allow pointers to violate privacy
                let t = ty::type_autoderef(ty::expr_ty(self.tcx, base));
                match ty::get(t).sty {
                    ty::ty_struct(id, _) => {
                        self.check_field(expr.span, id, ident, None);
                    }
                    _ => {}
                }
            }
            ast::ExprMethodCall(_, ident, _, ref args) => {
                // see above
                let t = ty::type_autoderef(ty::expr_ty(self.tcx, args[0]));
                match ty::get(t).sty {
                    ty::ty_enum(_, _) | ty::ty_struct(_, _) => {
                        match self.method_map.borrow().get().find(&expr.id) {
                            None => {
                                self.tcx.sess.span_bug(expr.span,
                                                       "method call not in \
                                                        method map");
                            }
                            Some(origin) => {
                                debug!("(privacy checking) checking impl method");
                                self.check_method(expr.span, origin, ident);
                            }
                        }
                    }
                    _ => {}
                }
            }
            ast::ExprStruct(_, ref fields, _) => {
                match ty::get(ty::expr_ty(self.tcx, expr)).sty {
                    ty::ty_struct(id, _) => {
                        for field in (*fields).iter() {
                            self.check_field(expr.span, id, field.ident.node,
                                             None);
                        }
                    }
                    ty::ty_enum(_, _) => {
                        let def_map = self.tcx.def_map.borrow();
                        match def_map.get().get_copy(&expr.id) {
                            ast::DefVariant(enum_id, variant_id, _) => {
                                for field in fields.iter() {
                                    self.check_field(expr.span, variant_id,
                                                     field.ident.node,
                                                     Some(enum_id));
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
            _ => {}
        }

        visit::walk_expr(self, expr, ());
    }

    fn visit_view_item(&mut self, a: &ast::ViewItem, _: ()) {
        match a.node {
            ast::ViewItemExternMod(..) => {}
            ast::ViewItemUse(ref uses) => {
                for vpath in uses.iter() {
                    match vpath.node {
                        ast::ViewPathSimple(..) | ast::ViewPathGlob(..) => {}
                        ast::ViewPathList(_, ref list, _) => {
                            for pid in list.iter() {
                                debug!("privacy - list {}", pid.node.id);
                                let seg = ast::PathSegment {
                                    identifier: pid.node.name,
                                    lifetimes: opt_vec::Empty,
                                    types: opt_vec::Empty,
                                };
                                let segs = ~[seg];
                                let path = ast::Path {
                                    global: false,
                                    span: pid.span,
                                    segments: segs,
                                };
                                self.check_path(pid.span, pid.node.id, &path);
                            }
                        }
                    }
                }
            }
        }
        visit::walk_view_item(self, a, ());
    }

    fn visit_pat(&mut self, pattern: &ast::Pat, _: ()) {
        // Foreign functions do not have their patterns mapped in the def_map,
        // and there's nothing really relevant there anyway, so don't bother
        // checking privacy. If you can name the type then you can pass it to an
        // external C function anyway.
        if self.in_foreign { return }

        match pattern.node {
            ast::PatStruct(_, ref fields, _) => {
                match ty::get(ty::pat_ty(self.tcx, pattern)).sty {
                    ty::ty_struct(id, _) => {
                        for field in fields.iter() {
                            self.check_field(pattern.span, id, field.ident,
                                             None);
                        }
                    }
                    ty::ty_enum(_, _) => {
                        let def_map = self.tcx.def_map.borrow();
                        match def_map.get().find(&pattern.id) {
                            Some(&ast::DefVariant(enum_id, variant_id, _)) => {
                                for field in fields.iter() {
                                    self.check_field(pattern.span, variant_id,
                                                     field.ident, Some(enum_id));
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
            _ => {}
        }

        visit::walk_pat(self, pattern, ());
    }

    fn visit_foreign_item(&mut self, fi: &ast::ForeignItem, _: ()) {
        self.in_foreign = true;
        visit::walk_foreign_item(self, fi, ());
        self.in_foreign = false;
    }

    fn visit_path(&mut self, path: &ast::Path, id: ast::NodeId, _: ()) {
        self.check_path(path.span, id, path);
        visit::walk_path(self, path, ());
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy sanity check visitor, ensures unnecessary visibility isn't here
////////////////////////////////////////////////////////////////////////////////

struct SanePrivacyVisitor {
    tcx: ty::ctxt,
    in_fn: bool,
}

impl Visitor<()> for SanePrivacyVisitor {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        if self.in_fn {
            self.check_all_inherited(item);
        } else {
            self.check_sane_privacy(item);
        }

        let orig_in_fn = replace(&mut self.in_fn, match item.node {
            ast::ItemMod(..) => false, // modules turn privacy back on
            _ => self.in_fn,           // otherwise we inherit
        });
        visit::walk_item(self, item, ());
        self.in_fn = orig_in_fn;
    }

    fn visit_fn(&mut self, fk: &visit::FnKind, fd: &ast::FnDecl,
                b: &ast::Block, s: Span, n: ast::NodeId, _: ()) {
        // This catches both functions and methods
        let orig_in_fn = replace(&mut self.in_fn, true);
        visit::walk_fn(self, fk, fd, b, s, n, ());
        self.in_fn = orig_in_fn;
    }

    fn visit_view_item(&mut self, i: &ast::ViewItem, _: ()) {
        match i.vis {
            ast::Inherited => {}
            ast::Private => {
                self.tcx.sess.span_err(i.span, "unnecessary visibility \
                                                qualifier");
            }
            ast::Public => {
                if self.in_fn {
                    self.tcx.sess.span_err(i.span, "unnecessary `pub`, imports \
                                                    in functions are never \
                                                    reachable");
                } else {
                    match i.node {
                        ast::ViewItemExternMod(..) => {
                            self.tcx.sess.span_err(i.span, "`pub` visibility \
                                                            is not allowed");
                        }
                        _ => {}
                    }
                }
            }
        }
        visit::walk_view_item(self, i, ());
    }
}

impl SanePrivacyVisitor {
    /// Validates all of the visibility qualifers placed on the item given. This
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
        let check_not_priv = |sp: Span, vis: ast::Visibility, note: &str| {
            if vis == ast::Private {
                tcx.sess.span_err(sp, "unnecessary `priv` qualifier");
                if note.len() > 0 {
                    tcx.sess.span_note(sp, note);
                }
            }
        };
        let check_struct = |def: &@ast::StructDef,
                            vis: ast::Visibility,
                            parent_vis: Option<ast::Visibility>| {
            let public_def = match vis {
                ast::Public => true,
                ast::Inherited | ast::Private => parent_vis == Some(ast::Public),
            };
            for f in def.fields.iter() {
               match f.node.kind {
                    ast::NamedField(_, ast::Public) if public_def => {
                        tcx.sess.span_err(f.span, "unnecessary `pub` \
                                                   visibility");
                    }
                    ast::NamedField(_, ast::Private) if !public_def => {
                        tcx.sess.span_err(f.span, "unnecessary `priv` \
                                                   visibility");
                    }
                    ast::NamedField(..) | ast::UnnamedField => {}
                }
            }
        };
        match item.node {
            // implementations of traits don't need visibility qualifiers because
            // that's controlled by having the trait in scope.
            ast::ItemImpl(_, Some(..), _, ref methods) => {
                check_inherited(item.span, item.vis,
                                "visibility qualifiers have no effect on trait \
                                 impls");
                for m in methods.iter() {
                    check_inherited(m.span, m.vis, "");
                }
            }

            ast::ItemImpl(_, _, _, ref methods) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual methods instead");
                for i in methods.iter() {
                    check_not_priv(i.span, i.vis, "functions are private by \
                                                   default");
                }
            }
            ast::ItemForeignMod(ref fm) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual functions \
                                 instead");
                for i in fm.items.iter() {
                    check_not_priv(i.span, i.vis, "functions are private by \
                                                   default");
                }
            }

            ast::ItemEnum(ref def, _) => {
                for v in def.variants.iter() {
                    match v.node.vis {
                        ast::Public => {
                            if item.vis == ast::Public {
                                tcx.sess.span_err(v.span, "unnecessary `pub` \
                                                           visibility");
                            }
                        }
                        ast::Private => {
                            if item.vis != ast::Public {
                                tcx.sess.span_err(v.span, "unnecessary `priv` \
                                                           visibility");
                            }
                        }
                        ast::Inherited => {}
                    }

                    match v.node.kind {
                        ast::StructVariantKind(ref s) => {
                            check_struct(s, v.node.vis, Some(item.vis));
                        }
                        ast::TupleVariantKind(..) => {}
                    }
                }
            }

            ast::ItemStruct(ref def, _) => check_struct(def, item.vis, None),

            ast::ItemTrait(_, _, ref methods) => {
                for m in methods.iter() {
                    match *m {
                        ast::Provided(ref m) => {
                            check_inherited(m.span, m.vis,
                                            "unnecessary visibility");
                        }
                        ast::Required(..) => {}
                    }
                }
            }

            ast::ItemStatic(..) |
            ast::ItemFn(..) | ast::ItemMod(..) | ast::ItemTy(..) |
            ast::ItemMac(..) => {
                check_not_priv(item.span, item.vis, "items are private by \
                                                     default");
            }
        }
    }

    /// When inside of something like a function or a method, visibility has no
    /// control over anything so this forbids any mention of any visibility
    fn check_all_inherited(&self, item: &ast::Item) {
        let tcx = self.tcx;
        let check_inherited = |sp: Span, vis: ast::Visibility| {
            if vis != ast::Inherited {
                tcx.sess.span_err(sp, "visibility has no effect inside functions");
            }
        };
        let check_struct = |def: &@ast::StructDef| {
            for f in def.fields.iter() {
               match f.node.kind {
                    ast::NamedField(_, p) => check_inherited(f.span, p),
                    ast::UnnamedField => {}
                }
            }
        };
        check_inherited(item.span, item.vis);
        match item.node {
            ast::ItemImpl(_, _, _, ref methods) => {
                for m in methods.iter() {
                    check_inherited(m.span, m.vis);
                }
            }
            ast::ItemForeignMod(ref fm) => {
                for i in fm.items.iter() {
                    check_inherited(i.span, i.vis);
                }
            }
            ast::ItemEnum(ref def, _) => {
                for v in def.variants.iter() {
                    check_inherited(v.span, v.node.vis);

                    match v.node.kind {
                        ast::StructVariantKind(ref s) => check_struct(s),
                        ast::TupleVariantKind(..) => {}
                    }
                }
            }

            ast::ItemStruct(ref def, _) => check_struct(def),

            ast::ItemTrait(_, _, ref methods) => {
                for m in methods.iter() {
                    match *m {
                        ast::Required(..) => {}
                        ast::Provided(ref m) => check_inherited(m.span, m.vis),
                    }
                }
            }

            ast::ItemStatic(..) |
            ast::ItemFn(..) | ast::ItemMod(..) | ast::ItemTy(..) |
            ast::ItemMac(..) => {}
        }
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: &method_map,
                   exp_map2: &resolve::ExportMap2,
                   external_exports: resolve::ExternalExports,
                   last_private_map: resolve::LastPrivateMap,
                   krate: &ast::Crate) -> (ExportedItems, PublicItems) {
    // Figure out who everyone's parent is
    let mut visitor = ParentVisitor {
        parents: HashMap::new(),
        curparent: ast::DUMMY_NODE_ID,
    };
    visit::walk_crate(&mut visitor, krate, ());

    // Use the parent map to check the privacy of everything
    let mut visitor = PrivacyVisitor {
        curitem: ast::DUMMY_NODE_ID,
        in_fn: false,
        in_foreign: false,
        tcx: tcx,
        parents: visitor.parents,
        method_map: method_map,
        external_exports: external_exports,
        last_private_map: last_private_map,
    };
    visit::walk_crate(&mut visitor, krate, ());

    // Sanity check to make sure that all privacy usage and controls are
    // reasonable.
    let mut visitor = SanePrivacyVisitor {
        in_fn: false,
        tcx: tcx,
    };
    visit::walk_crate(&mut visitor, krate, ());

    tcx.sess.abort_if_errors();

    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx: tcx,
        exported_items: HashSet::new(),
        public_items: HashSet::new(),
        reexports: HashSet::new(),
        exp_map2: exp_map2,
        prev_exported: true,
        prev_public: true,
    };
    loop {
        let before = visitor.exported_items.len();
        visit::walk_crate(&mut visitor, krate, ());
        if before == visitor.exported_items.len() {
            break
        }
    }

    let EmbargoVisitor { exported_items, public_items, .. } = visitor;
    return (exported_items, public_items);
}
