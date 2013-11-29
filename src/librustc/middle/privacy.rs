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

use std::hashmap::{HashSet, HashMap};
use std::util;

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

type Context<'self> = (&'self method_map, &'self resolve::ExportMap2);

/// A set of AST nodes exported by the crate.
pub type ExportedItems = HashSet<ast::NodeId>;

////////////////////////////////////////////////////////////////////////////////
/// The parent visitor, used to determine what's the parent of what (node-wise)
////////////////////////////////////////////////////////////////////////////////

struct ParentVisitor {
    parents: HashMap<ast::NodeId, ast::NodeId>,
    curparent: ast::NodeId,
}

impl Visitor<()> for ParentVisitor {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        self.parents.insert(item.id, self.curparent);

        let prev = self.curparent;
        match item.node {
            ast::item_mod(..) => { self.curparent = item.id; }
            // Enum variants are parented to the enum definition itself beacuse
            // they inherit privacy
            ast::item_enum(ref def, _) => {
                for variant in def.variants.iter() {
                    // If variants are private, then their logical "parent" is
                    // the enclosing module because everyone in the enclosing
                    // module can still use the private variant
                    if variant.node.vis == ast::private {
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
            ast::item_trait(_, _, ref methods) if item.vis != ast::public => {
                for m in methods.iter() {
                    match *m {
                        ast::provided(ref m) => self.parents.insert(m.id, item.id),
                        ast::required(ref m) => self.parents.insert(m.id, item.id),
                    };
                }
            }

            _ => {}
        }
        visit::walk_item(self, item, ());
        self.curparent = prev;
    }

    fn visit_foreign_item(&mut self, a: @ast::foreign_item, _: ()) {
        self.parents.insert(a.id, self.curparent);
        visit::walk_foreign_item(self, a, ());
    }

    fn visit_fn(&mut self, a: &visit::fn_kind, b: &ast::fn_decl,
                c: &ast::Block, d: Span, id: ast::NodeId, _: ()) {
        // We already took care of some trait methods above, otherwise things
        // like impl methods and pub trait methods are parented to the
        // containing module, not the containing trait.
        if !self.parents.contains_key(&id) {
            self.parents.insert(id, self.curparent);
        }
        visit::walk_fn(self, a, b, c, d, id, ());
    }

    fn visit_struct_def(&mut self, s: @ast::struct_def, i: ast::Ident,
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
            let vis = match field.node.kind {
                ast::named_field(_, vis) => vis,
                ast::unnamed_field => continue
            };

            // Private fields are scoped to this module, so parent them directly
            // to the module instead of the struct. This is similar to the case
            // of private enum variants.
            if vis == ast::private {
                self.parents.insert(field.node.id, self.curparent);

            // Otherwise public fields are scoped to the visibility of the
            // struct itself
            } else {
                self.parents.insert(field.node.id, n);
            }
        }
        visit::walk_struct_def(self, s, i, g, n, ())
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the ast
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'self> {
    tcx: ty::ctxt,
    exp_map2: &'self resolve::ExportMap2,

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
}

impl<'self> Visitor<()> for EmbargoVisitor<'self> {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        let orig_all_pub = self.prev_exported;
        match item.node {
            // impls/extern blocks do not break the "public chain" because they
            // cannot have visibility qualifiers on them anyway
            ast::item_impl(..) | ast::item_foreign_mod(..) => {}

            // Private by default, hence we only retain the "public chain" if
            // `pub` is explicitly listed.
            _ => {
                self.prev_exported =
                    (orig_all_pub && item.vis == ast::public) ||
                     self.reexports.contains(&item.id);
            }
        }

        let public_first = self.prev_exported &&
                           self.exported_items.insert(item.id);

        match item.node {
            // Enum variants inherit from their parent, so if the enum is
            // public all variants are public unless they're explicitly priv
            ast::item_enum(ref def, _) if public_first => {
                for variant in def.variants.iter() {
                    if variant.node.vis != ast::private {
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
            ast::item_impl(_, _, ref ty, ref methods) => {
                let public_ty = match ty.node {
                    ast::ty_path(_, _, id) => {
                        match self.tcx.def_map.get_copy(&id) {
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
                let public_trait = tr.map_default(false, |tr| {
                    !is_local(tr.def_id) ||
                     self.exported_items.contains(&tr.def_id.node)
                });

                if public_ty || public_trait {
                    for method in methods.iter() {
                        let meth_public = match method.explicit_self.node {
                            ast::sty_static => public_ty,
                            _ => true,
                        } && method.vis == ast::public;
                        if meth_public || public_trait {
                            self.exported_items.insert(method.id);
                        }
                    }
                }
            }

            // Default methods on traits are all public so long as the trait
            // is public
            ast::item_trait(_, _, ref methods) if public_first => {
                for method in methods.iter() {
                    match *method {
                        ast::provided(ref m) => {
                            debug!("provided {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                        ast::required(ref m) => {
                            debug!("required {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                    }
                }
            }

            // Struct constructors are public if the struct is all public.
            ast::item_struct(ref def, _) if public_first => {
                match def.ctor_id {
                    Some(id) => { self.exported_items.insert(id); }
                    None => {}
                }
            }

            _ => {}
        }

        visit::walk_item(self, item, ());

        self.prev_exported = orig_all_pub;
    }

    fn visit_foreign_item(&mut self, a: @ast::foreign_item, _: ()) {
        if self.prev_exported && a.vis == ast::public {
            self.exported_items.insert(a.id);
        }
    }

    fn visit_mod(&mut self, m: &ast::_mod, _sp: Span, id: ast::NodeId, _: ()) {
        // This code is here instead of in visit_item so that the
        // crate module gets processed as well.
        if self.prev_exported {
            assert!(self.exp_map2.contains_key(&id), "wut {:?}", id);
            for export in self.exp_map2.get(&id).iter() {
                if is_local(export.def_id) && export.reexport {
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

struct PrivacyVisitor<'self> {
    tcx: ty::ctxt,
    curitem: ast::NodeId,
    in_fn: bool,
    method_map: &'self method_map,
    parents: HashMap<ast::NodeId, ast::NodeId>,
    external_exports: resolve::ExternalExports,
    last_private_map: resolve::LastPrivateMap,
}

enum PrivacyResult {
    Allowable,
    ExternallyDenied,
    DisallowedBy(ast::NodeId),
}

impl<'self> PrivacyVisitor<'self> {
    // used when debugging
    fn nodestr(&self, id: ast::NodeId) -> ~str {
        ast_map::node_id_to_str(self.tcx.items, id, token::get_ident_interner())
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
            return match self.tcx.methods.find(&did) {
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
                                    if meth.vis == ast::public {
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
            let vis = match self.tcx.items.find(&closest_private_id) {
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
                Some(&ast_map::node_method(ref m, imp, _)) => {
                    match ty::impl_trait_ref(self.tcx, imp) {
                        Some(..) => return Allowable,
                        _ if m.vis == ast::public => return Allowable,
                        _ => m.vis
                    }
                }
                Some(&ast_map::node_trait_method(..)) => {
                    return Allowable;
                }

                // This is not a method call, extract the visibility as one
                // would normally look at it
                Some(&ast_map::node_item(it, _)) => it.vis,
                Some(&ast_map::node_foreign_item(_, _, v, _)) => v,
                Some(&ast_map::node_variant(ref v, _, _)) => {
                    // sadly enum variants still inherit visibility, so only
                    // break out of this is explicitly private
                    if v.node.vis == ast::private { break }
                    ast::public // need to move up a level (to the enum)
                }
                _ => ast::public,
            };
            if vis != ast::public { break }
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

    /// Guarantee that a particular definition is public, possibly emitting an
    /// error message if it's not.
    fn ensure_public(&self, span: Span, to_check: ast::DefId,
                     source_did: Option<ast::DefId>, msg: &str) -> bool {
        match self.def_privacy(to_check) {
            ExternallyDenied => {
                self.tcx.sess.span_err(span, format!("{} is private", msg))
            }
            DisallowedBy(id) => {
                if id == source_did.unwrap_or(to_check).node {
                    self.tcx.sess.span_err(span, format!("{} is private", msg));
                    return false;
                } else {
                    self.tcx.sess.span_err(span, format!("{} is inaccessible",
                                                         msg));
                }
                match self.tcx.items.find(&id) {
                    Some(&ast_map::node_item(item, _)) => {
                        let desc = match item.node {
                            ast::item_mod(..) => "module",
                            ast::item_trait(..) => "trait",
                            _ => return false,
                        };
                        let msg = format!("{} `{}` is private", desc,
                                          token::ident_to_str(&item.ident));
                        self.tcx.sess.span_note(span, msg);
                    }
                    Some(..) | None => {}
                }
            }
            Allowable => return true
        }
        return false;
    }

    // Checks that a dereference of a univariant enum can occur.
    fn check_variant(&self, span: Span, enum_id: ast::DefId) {
        let variant_info = ty::enum_variants(self.tcx, enum_id)[0];

        match self.def_privacy(variant_info.id) {
            Allowable => {}
            ExternallyDenied | DisallowedBy(..) => {
                self.tcx.sess.span_err(span, "can only dereference enums \
                                              with a single, public variant");
            }
        }
    }

    // Checks that a field is in scope.
    // FIXME #6993: change type (and name) from Ident to Name
    fn check_field(&mut self, span: Span, id: ast::DefId, ident: ast::Ident) {
        let fields = ty::lookup_struct_fields(self.tcx, id);
        for field in fields.iter() {
            if field.name != ident.name { continue; }
            // public fields are public everywhere
            if field.vis != ast::private { break }
            if !is_local(field.id) ||
               !self.private_accessible(field.id.node) {
                self.tcx.sess.span_err(span, format!("field `{}` is private",
                                             token::ident_to_str(&ident)));
            }
            break;
        }
    }

    // Given the ID of a method, checks to ensure it's in scope.
    fn check_static_method(&mut self, span: Span, method_id: ast::DefId,
                           name: &ast::Ident) {
        // If the method is a default method, we need to use the def_id of
        // the default implementation.
        let method_id = ty::method(self.tcx, method_id).provided_source
                                                       .unwrap_or(method_id);

        self.ensure_public(span, method_id, None,
                           format!("method `{}`", token::ident_to_str(name)));
    }

    // Checks that a path is in scope.
    fn check_path(&mut self, span: Span, path_id: ast::NodeId, path: &ast::Path) {
        debug!("privacy - path {}", self.nodestr(path_id));
        let def = self.tcx.def_map.get_copy(&path_id);
        let ck = |tyname: &str| {
            let origdid = def_id_of_def(def);
            match *self.last_private_map.get(&path_id) {
                resolve::AllPublic => {},
                resolve::DependsOn(def) => {
                    let name = token::ident_to_str(&path.segments.last()
                                                        .identifier);
                    self.ensure_public(span, def, Some(origdid),
                                       format!("{} `{}`", tyname, name));
                }
            }
        };
        match self.tcx.def_map.get_copy(&path_id) {
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
                self.check_static_method(span, method_id, &ident)
            }
            // Trait methods are always all public. The only controlling factor
            // is whether the trait itself is accessible or not.
            method_param(method_param { trait_id: trait_id, .. }) |
            method_object(method_object { trait_id: trait_id, .. }) => {
                self.ensure_public(span, trait_id, None, "source trait");
            }
        }
    }
}

impl<'self> Visitor<()> for PrivacyVisitor<'self> {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        // Do not check privacy inside items with the resolve_unexported
        // attribute. This is used for the test runner.
        if attr::contains_name(item.attrs, "!resolve_unexported") {
            return;
        }

        let orig_curitem = util::replace(&mut self.curitem, item.id);
        visit::walk_item(self, item, ());
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: @ast::Expr, _: ()) {
        match expr.node {
            ast::ExprField(base, ident, _) => {
                // Method calls are now a special syntactic form,
                // so `a.b` should always be a field.
                assert!(!self.method_map.contains_key(&expr.id));

                // With type_autoderef, make sure we don't
                // allow pointers to violate privacy
                let t = ty::type_autoderef(self.tcx,
                                           ty::expr_ty(self.tcx, base));
                match ty::get(t).sty {
                    ty::ty_struct(id, _) => self.check_field(expr.span, id, ident),
                    _ => {}
                }
            }
            ast::ExprMethodCall(_, base, ident, _, _, _) => {
                // see above
                let t = ty::type_autoderef(self.tcx,
                                           ty::expr_ty(self.tcx, base));
                match ty::get(t).sty {
                    ty::ty_enum(_, _) | ty::ty_struct(_, _) => {
                        let entry = match self.method_map.find(&expr.id) {
                            None => {
                                self.tcx.sess.span_bug(expr.span,
                                                       "method call not in \
                                                        method map");
                            }
                            Some(entry) => entry
                        };
                        debug!("(privacy checking) checking impl method");
                        self.check_method(expr.span, &entry.origin, ident);
                    }
                    _ => {}
                }
            }
            ast::ExprPath(ref path) => {
                self.check_path(expr.span, expr.id, path);
            }
            ast::ExprStruct(_, ref fields, _) => {
                match ty::get(ty::expr_ty(self.tcx, expr)).sty {
                    ty::ty_struct(id, _) => {
                        for field in (*fields).iter() {
                            self.check_field(expr.span, id, field.ident.node);
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.get_copy(&expr.id) {
                            ast::DefVariant(_, variant_id, _) => {
                                for field in fields.iter() {
                                    self.check_field(expr.span, variant_id,
                                                     field.ident.node);
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
            ast::ExprUnary(_, ast::UnDeref, operand) => {
                // In *e, we need to check that if e's type is an
                // enum type t, then t's first variant is public or
                // privileged. (We can assume it has only one variant
                // since typeck already happened.)
                match ty::get(ty::expr_ty(self.tcx, operand)).sty {
                    ty::ty_enum(id, _) => {
                        self.check_variant(expr.span, id);
                    }
                    _ => { /* No check needed */ }
                }
            }
            _ => {}
        }

        visit::walk_expr(self, expr, ());
    }

    fn visit_ty(&mut self, t: &ast::Ty, _: ()) {
        match t.node {
            ast::ty_path(ref path, _, id) => self.check_path(t.span, id, path),
            _ => {}
        }
        visit::walk_ty(self, t, ());
    }

    fn visit_view_item(&mut self, a: &ast::view_item, _: ()) {
        match a.node {
            ast::view_item_extern_mod(..) => {}
            ast::view_item_use(ref uses) => {
                for vpath in uses.iter() {
                    match vpath.node {
                        ast::view_path_simple(_, ref path, id) |
                        ast::view_path_glob(ref path, id) => {
                            debug!("privacy - glob/simple {}", id);
                            self.check_path(vpath.span, id, path);
                        }
                        ast::view_path_list(_, ref list, _) => {
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
    }

    fn visit_pat(&mut self, pattern: &ast::Pat, _: ()) {
        match pattern.node {
            ast::PatStruct(_, ref fields, _) => {
                match ty::get(ty::pat_ty(self.tcx, pattern)).sty {
                    ty::ty_struct(id, _) => {
                        for field in fields.iter() {
                            self.check_field(pattern.span, id, field.ident);
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.find(&pattern.id) {
                            Some(&ast::DefVariant(_, variant_id, _)) => {
                                for field in fields.iter() {
                                    self.check_field(pattern.span, variant_id,
                                                     field.ident);
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
}

////////////////////////////////////////////////////////////////////////////////
/// The privacy sanity check visitor, ensures unnecessary visibility isn't here
////////////////////////////////////////////////////////////////////////////////

struct SanePrivacyVisitor {
    tcx: ty::ctxt,
    in_fn: bool,
}

impl Visitor<()> for SanePrivacyVisitor {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        if self.in_fn {
            self.check_all_inherited(item);
        } else {
            self.check_sane_privacy(item);
        }

        let orig_in_fn = util::replace(&mut self.in_fn, match item.node {
            ast::item_mod(..) => false, // modules turn privacy back on
            _ => self.in_fn,           // otherwise we inherit
        });
        visit::walk_item(self, item, ());
        self.in_fn = orig_in_fn;
    }

    fn visit_fn(&mut self, fk: &visit::fn_kind, fd: &ast::fn_decl,
                b: &ast::Block, s: Span, n: ast::NodeId, _: ()) {
        // This catches both functions and methods
        let orig_in_fn = util::replace(&mut self.in_fn, true);
        visit::walk_fn(self, fk, fd, b, s, n, ());
        self.in_fn = orig_in_fn;
    }
}

impl SanePrivacyVisitor {
    /// Validates all of the visibility qualifers placed on the item given. This
    /// ensures that there are no extraneous qualifiers that don't actually do
    /// anything. In theory these qualifiers wouldn't parse, but that may happen
    /// later on down the road...
    fn check_sane_privacy(&self, item: @ast::item) {
        let tcx = self.tcx;
        let check_inherited = |sp: Span, vis: ast::visibility, note: &str| {
            if vis != ast::inherited {
                tcx.sess.span_err(sp, "unnecessary visibility qualifier");
                if note.len() > 0 {
                    tcx.sess.span_note(sp, note);
                }
            }
        };
        let check_not_priv = |sp: Span, vis: ast::visibility, note: &str| {
            if vis == ast::private {
                tcx.sess.span_err(sp, "unnecessary `priv` qualifier");
                if note.len() > 0 {
                    tcx.sess.span_note(sp, note);
                }
            }
        };
        let check_struct = |def: &@ast::struct_def| {
            for f in def.fields.iter() {
               match f.node.kind {
                    ast::named_field(_, ast::public) => {
                        tcx.sess.span_err(f.span, "unnecessary `pub` \
                                                   visibility");
                    }
                    ast::named_field(_, ast::private) => {
                        // Fields should really be private by default...
                    }
                    ast::named_field(..) | ast::unnamed_field => {}
                }
            }
        };
        match item.node {
            // implementations of traits don't need visibility qualifiers because
            // that's controlled by having the trait in scope.
            ast::item_impl(_, Some(..), _, ref methods) => {
                check_inherited(item.span, item.vis,
                                "visibility qualifiers have no effect on trait \
                                 impls");
                for m in methods.iter() {
                    check_inherited(m.span, m.vis, "");
                }
            }

            ast::item_impl(_, _, _, ref methods) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual methods instead");
                for i in methods.iter() {
                    check_not_priv(i.span, i.vis, "functions are private by \
                                                   default");
                }
            }
            ast::item_foreign_mod(ref fm) => {
                check_inherited(item.span, item.vis,
                                "place qualifiers on individual functions \
                                 instead");
                for i in fm.items.iter() {
                    check_not_priv(i.span, i.vis, "functions are private by \
                                                   default");
                }
            }

            ast::item_enum(ref def, _) => {
                for v in def.variants.iter() {
                    match v.node.vis {
                        ast::public => {
                            if item.vis == ast::public {
                                tcx.sess.span_err(v.span, "unnecessary `pub` \
                                                           visibility");
                            }
                        }
                        ast::private => {
                            if item.vis != ast::public {
                                tcx.sess.span_err(v.span, "unnecessary `priv` \
                                                           visibility");
                            }
                        }
                        ast::inherited => {}
                    }

                    match v.node.kind {
                        ast::struct_variant_kind(ref s) => check_struct(s),
                        ast::tuple_variant_kind(..) => {}
                    }
                }
            }

            ast::item_struct(ref def, _) => check_struct(def),

            ast::item_trait(_, _, ref methods) => {
                for m in methods.iter() {
                    match *m {
                        ast::provided(ref m) => {
                            check_inherited(m.span, m.vis,
                                            "unnecessary visibility");
                        }
                        ast::required(..) => {}
                    }
                }
            }

            ast::item_static(..) |
            ast::item_fn(..) | ast::item_mod(..) | ast::item_ty(..) |
            ast::item_mac(..) => {
                check_not_priv(item.span, item.vis, "items are private by \
                                                     default");
            }
        }
    }

    /// When inside of something like a function or a method, visibility has no
    /// control over anything so this forbids any mention of any visibility
    fn check_all_inherited(&self, item: @ast::item) {
        let tcx = self.tcx;
        let check_inherited = |sp: Span, vis: ast::visibility| {
            if vis != ast::inherited {
                tcx.sess.span_err(sp, "visibility has no effect inside functions");
            }
        };
        let check_struct = |def: &@ast::struct_def| {
            for f in def.fields.iter() {
               match f.node.kind {
                    ast::named_field(_, p) => check_inherited(f.span, p),
                    ast::unnamed_field => {}
                }
            }
        };
        check_inherited(item.span, item.vis);
        match item.node {
            ast::item_impl(_, _, _, ref methods) => {
                for m in methods.iter() {
                    check_inherited(m.span, m.vis);
                }
            }
            ast::item_foreign_mod(ref fm) => {
                for i in fm.items.iter() {
                    check_inherited(i.span, i.vis);
                }
            }
            ast::item_enum(ref def, _) => {
                for v in def.variants.iter() {
                    check_inherited(v.span, v.node.vis);

                    match v.node.kind {
                        ast::struct_variant_kind(ref s) => check_struct(s),
                        ast::tuple_variant_kind(..) => {}
                    }
                }
            }

            ast::item_struct(ref def, _) => check_struct(def),

            ast::item_trait(_, _, ref methods) => {
                for m in methods.iter() {
                    match *m {
                        ast::required(..) => {}
                        ast::provided(ref m) => check_inherited(m.span, m.vis),
                    }
                }
            }

            ast::item_static(..) |
            ast::item_fn(..) | ast::item_mod(..) | ast::item_ty(..) |
            ast::item_mac(..) => {}
        }
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: &method_map,
                   exp_map2: &resolve::ExportMap2,
                   external_exports: resolve::ExternalExports,
                   last_private_map: resolve::LastPrivateMap,
                   crate: &ast::Crate) -> ExportedItems {
    // Figure out who everyone's parent is
    let mut visitor = ParentVisitor {
        parents: HashMap::new(),
        curparent: ast::DUMMY_NODE_ID,
    };
    visit::walk_crate(&mut visitor, crate, ());

    // Use the parent map to check the privacy of everything
    let mut visitor = PrivacyVisitor {
        curitem: ast::DUMMY_NODE_ID,
        in_fn: false,
        tcx: tcx,
        parents: visitor.parents,
        method_map: method_map,
        external_exports: external_exports,
        last_private_map: last_private_map,
    };
    visit::walk_crate(&mut visitor, crate, ());

    // Sanity check to make sure that all privacy usage and controls are
    // reasonable.
    let mut visitor = SanePrivacyVisitor {
        in_fn: false,
        tcx: tcx,
    };
    visit::walk_crate(&mut visitor, crate, ());

    tcx.sess.abort_if_errors();

    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx: tcx,
        exported_items: HashSet::new(),
        reexports: HashSet::new(),
        exp_map2: exp_map2,
        prev_exported: true,
    };
    loop {
        let before = visitor.exported_items.len();
        visit::walk_crate(&mut visitor, crate, ());
        if before == visitor.exported_items.len() {
            break
        }
    }

    return visitor.exported_items;
}
