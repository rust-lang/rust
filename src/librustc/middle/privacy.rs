// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use middle::resolve;
use middle::ty;
use middle::typeck::{method_map, method_origin, method_param};
use middle::typeck::{method_static, method_object};

use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::is_local;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::opt_vec;
use syntax::visit;
use syntax::visit::Visitor;

type Context<'self> = (&'self method_map, &'self resolve::ExportMap2);

// This visitor is used to determine the parent of all nodes in question when it
// comes to privacy. This is used to determine later on if a usage is actually
// valid or not.
struct ParentVisitor<'self> {
    parents: &'self mut HashMap<ast::NodeId, ast::NodeId>,
    curparent: ast::NodeId,
}

impl<'self> Visitor<()> for ParentVisitor<'self> {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        self.parents.insert(item.id, self.curparent);

        let prev = self.curparent;
        match item.node {
            ast::item_mod(*) => { self.curparent = item.id; }
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

// This visitor is used to determine which items of the ast are embargoed,
// otherwise known as not exported.
struct EmbargoVisitor<'self> {
    exported_items: &'self mut HashSet<ast::NodeId>,
    exp_map2: &'self resolve::ExportMap2,
    path_all_public: bool,
}

impl<'self> Visitor<()> for EmbargoVisitor<'self> {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        let orig_all_pub = self.path_all_public;
        match item.node {
            // impls/extern blocks do not break the "public chain" because they
            // cannot have visibility qualifiers on them anyway
            ast::item_impl(*) | ast::item_foreign_mod(*) => {}

            // Private by default, hence we only retain the "public chain" if
            // `pub` is explicitly listed.
            _ => {
                self.path_all_public = orig_all_pub && item.vis == ast::public;
            }
        }

        if self.path_all_public {
            self.exported_items.insert(item.id);
        }

        match item.node {
            // Enum variants inherit from their parent, so if the enum is
            // public all variants are public unless they're explicitly priv
            ast::item_enum(ref def, _) if self.path_all_public => {
                for variant in def.variants.iter() {
                    if variant.node.vis != ast::private {
                        self.exported_items.insert(variant.node.id);
                    }
                }
            }

            // Methods which are public at the source are totally public.
            ast::item_impl(_, None, _, ref methods) => {
                for method in methods.iter() {
                    let public = match method.explicit_self.node {
                        ast::sty_static => self.path_all_public,
                        _ => true,
                    } && method.vis == ast::public;
                    if public {
                        self.exported_items.insert(method.id);
                    }
                }
            }

            // Trait implementation methods are all completely public
            ast::item_impl(_, Some(*), _, ref methods) => {
                for method in methods.iter() {
                    debug2!("exporting: {}", method.id);
                    self.exported_items.insert(method.id);
                }
            }

            // Default methods on traits are all public so long as the trait is
            // public
            ast::item_trait(_, _, ref methods) if self.path_all_public => {
                for method in methods.iter() {
                    match *method {
                        ast::provided(ref m) => {
                            debug2!("provided {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                        ast::required(ref m) => {
                            debug2!("required {}", m.id);
                            self.exported_items.insert(m.id);
                        }
                    }
                }
            }

            // Default methods on traits are all public so long as the trait is
            // public
            ast::item_struct(ref def, _) if self.path_all_public => {
                match def.ctor_id {
                    Some(id) => { self.exported_items.insert(id); }
                    None => {}
                }
            }

            _ => {}
        }

        visit::walk_item(self, item, ());

        self.path_all_public = orig_all_pub;
    }

    fn visit_foreign_item(&mut self, a: @ast::foreign_item, _: ()) {
        if self.path_all_public && a.vis == ast::public {
            self.exported_items.insert(a.id);
        }
    }
}

struct PrivacyVisitor<'self> {
    tcx: ty::ctxt,
    curitem: ast::NodeId,

    // Results of previous analyses necessary for privacy checking.
    exported_items: &'self HashSet<ast::NodeId>,
    method_map: &'self method_map,
    parents: &'self HashMap<ast::NodeId, ast::NodeId>,
    external_exports: resolve::ExternalExports,
    last_private_map: resolve::LastPrivateMap,
}

impl<'self> PrivacyVisitor<'self> {
    // used when debugging
    fn nodestr(&self, id: ast::NodeId) -> ~str {
        ast_map::node_id_to_str(self.tcx.items, id, token::get_ident_interner())
    }

    // Determines whether the given definition is public from the point of view
    // of the current item.
    fn def_public(&self, did: ast::DefId) -> bool {
        if !is_local(did) {
            if self.external_exports.contains(&did) {
                debug2!("privacy - {:?} was externally exported", did);
                return true;
            }
            debug2!("privacy - is {:?} a public method", did);
            return match self.tcx.methods.find(&did) {
                Some(meth) => {
                    debug2!("privacy - well at least it's a method: {:?}", meth);
                    match meth.container {
                        ty::TraitContainer(id) => {
                            debug2!("privacy - recursing on trait {:?}", id);
                            self.def_public(id)
                        }
                        ty::ImplContainer(id) => {
                            match ty::impl_trait_ref(self.tcx, id) {
                                Some(t) => {
                                    debug2!("privacy - impl of trait {:?}", id);
                                    self.def_public(t.def_id)
                                }
                                None => {
                                    debug2!("privacy - found a method {:?}",
                                            meth.vis);
                                    meth.vis == ast::public
                                }
                            }
                        }
                    }
                }
                None => {
                    debug2!("privacy - nope, not even a method");
                    false
                }
            };
        } else if self.exported_items.contains(&did.node) {
            debug2!("privacy - exported item {}", self.nodestr(did.node));
            return true;
        }

        debug2!("privacy - local {:?} not public all the way down", did);
        // return quickly for things in the same module
        if self.parents.find(&did.node) == self.parents.find(&self.curitem) {
            debug2!("privacy - same parent, we're done here");
            return true;
        }

        // We now know that there is at least one private member between the
        // destination and the root.
        let mut closest_private_id = did.node;
        loop {
            debug2!("privacy - examining {}", self.nodestr(closest_private_id));
            let vis = match self.tcx.items.find(&closest_private_id) {
                Some(&ast_map::node_item(it, _)) => it.vis,
                Some(&ast_map::node_method(ref m, _, _)) => m.vis,
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
            closest_private_id = *self.parents.get(&closest_private_id);

            // If we reached the top, then we should have been public all the
            // way down in the first place...
            assert!(closest_private_id != ast::DUMMY_NODE_ID);
        }
        debug2!("privacy - closest priv {}", self.nodestr(closest_private_id));
        return self.private_accessible(closest_private_id);
    }

    /// For a local private node in the AST, this function will determine
    /// whether the node is accessible by the current module that iteration is
    /// inside.
    fn private_accessible(&self, id: ast::NodeId) -> bool {
        let parent = *self.parents.get(&id);
        debug2!("privacy - accessible parent {}", self.nodestr(parent));

        // After finding `did`'s closest private member, we roll ourselves back
        // to see if this private member's parent is anywhere in our ancestry.
        // By the privacy rules, we can access all of our ancestor's private
        // members, so that's why we test the parent, and not the did itself.
        let mut cur = self.curitem;
        loop {
            debug2!("privacy - questioning {}", self.nodestr(cur));
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

    // Checks that a dereference of a univariant enum can occur.
    fn check_variant(&self, span: Span, enum_id: ast::DefId) {
        let variant_info = ty::enum_variants(self.tcx, enum_id)[0];
        if !self.def_public(variant_info.id) {
            self.tcx.sess.span_err(span, "can only dereference enums \
                                          with a single, public variant");
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

        if !self.def_public(method_id) {
            debug2!("private: {:?}", method_id);
            self.tcx.sess.span_err(span, format!("method `{}` is private",
                                                 token::ident_to_str(name)));
        }
    }

    // Checks that a path is in scope.
    fn check_path(&mut self, span: Span, path_id: ast::NodeId, path: &ast::Path) {
        debug2!("privacy - path {}", self.nodestr(path_id));
        let ck = |tyname: &str| {
            let last_private = *self.last_private_map.get(&path_id);
            debug2!("privacy - {:?}", last_private);
            let public = match last_private {
                resolve::AllPublic => true,
                resolve::DependsOn(def) => self.def_public(def),
            };
            if !public {
                debug2!("denying {:?}", path);
                let name = token::ident_to_str(&path.segments.last()
                                                    .identifier);
                self.tcx.sess.span_err(span,
                                  format!("{} `{}` is private", tyname, name));
            }
        };
        match self.tcx.def_map.get_copy(&path_id) {
            ast::DefStaticMethod(*) => ck("static method"),
            ast::DefFn(*) => ck("function"),
            ast::DefStatic(*) => ck("static"),
            ast::DefVariant(*) => ck("variant"),
            ast::DefTy(*) => ck("type"),
            ast::DefTrait(*) => ck("trait"),
            ast::DefStruct(*) => ck("struct"),
            ast::DefMethod(_, Some(*)) => ck("trait method"),
            ast::DefMethod(*) => ck("method"),
            ast::DefMod(*) => ck("module"),
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
            method_param(method_param {
                trait_id: trait_id,
                method_num: method_num,
                 _
            }) |
            method_object(method_object {
                trait_id: trait_id,
                method_num: method_num,
                 _
            }) => {
                if !self.def_public(trait_id) {
                    self.tcx.sess.span_err(span, "source trait is private");
                    return;
                }
                match self.tcx.items.find(&trait_id.node) {
                    Some(&ast_map::node_item(item, _)) => {
                        match item.node {
                            ast::item_trait(_, _, ref methods) => {
                                match methods[method_num] {
                                    ast::provided(ref method) => {
                                        let def = ast::DefId {
                                            node: method.id,
                                            crate: trait_id.crate,
                                        };
                                        if self.def_public(def) { return }
                                        let msg = format!("method `{}` is \
                                                           private",
                                                          token::ident_to_str(
                                                              &method.ident));
                                        self.tcx.sess.span_err(span, msg);
                                    }
                                    ast::required(_) => {
                                        // Required methods can't be private.
                                    }
                                }
                            }
                            _ => self.tcx.sess.span_bug(span, "trait wasn't \
                                                               actually a trait?!"),
                        }
                    }
                    Some(_) => self.tcx.sess.span_bug(span, "trait wasn't an \
                                                             item?!"),
                    None => self.tcx.sess.span_bug(span, "trait item wasn't \
                                                          found in the AST \
                                                          map?!"),
                }
            }
        }
    }

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
        match item.node {
            // implementations of traits don't need visibility qualifiers because
            // that's controlled by having the trait in scope.
            ast::item_impl(_, Some(*), _, ref methods) => {
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
                }
            }

            ast::item_struct(ref def, _) => {
                for f in def.fields.iter() {
                   match f.node.kind {
                        ast::named_field(_, ast::public) => {
                            tcx.sess.span_err(f.span, "unnecessary `pub` \
                                                       visibility");
                        }
                        ast::named_field(_, ast::private) => {
                            // Fields should really be private by default...
                        }
                        ast::named_field(*) | ast::unnamed_field => {}
                    }
                }
            }

            ast::item_trait(_, _, ref methods) => {
                for m in methods.iter() {
                    match *m {
                        ast::provided(ref m) => {
                            check_inherited(m.span, m.vis,
                                            "unnecessary visibility");
                        }
                        ast::required(*) => {}
                    }
                }
            }

            ast::item_static(*) |
            ast::item_fn(*) | ast::item_mod(*) | ast::item_ty(*) |
            ast::item_mac(*) => {
                check_not_priv(item.span, item.vis, "items are private by \
                                                     default");
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

        // Disallow unnecessary visibility qualifiers
        self.check_sane_privacy(item);

        let orig_curitem = self.curitem;
        self.curitem = item.id;
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
                        debug2!("(privacy checking) checking impl method");
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
                            self.check_field(expr.span, id, field.ident);
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.get_copy(&expr.id) {
                            ast::DefVariant(_, variant_id, _) => {
                                for field in fields.iter() {
                                    self.check_field(expr.span, variant_id,
                                                     field.ident);
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
            ast::view_item_extern_mod(*) => {}
            ast::view_item_use(ref uses) => {
                for vpath in uses.iter() {
                    match vpath.node {
                        ast::view_path_simple(_, ref path, id) |
                        ast::view_path_glob(ref path, id) => {
                            debug2!("privacy - glob/simple {}", id);
                            self.check_path(vpath.span, id, path);
                        }
                        ast::view_path_list(_, ref list, _) => {
                            for pid in list.iter() {
                                debug2!("privacy - list {}", pid.node.id);
                                let seg = ast::PathSegment {
                                    identifier: pid.node.name,
                                    lifetime: None,
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

    fn visit_pat(&mut self, pattern: @ast::Pat, _: ()) {
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

pub fn check_crate(tcx: ty::ctxt,
                   method_map: &method_map,
                   exp_map2: &resolve::ExportMap2,
                   external_exports: resolve::ExternalExports,
                   last_private_map: resolve::LastPrivateMap,
                   crate: &ast::Crate) {
    let mut parents = HashMap::new();
    let mut exported_items = HashSet::new();

    // First, figure out who everyone's parent is
    {
        let mut visitor = ParentVisitor {
            parents: &mut parents,
            curparent: ast::DUMMY_NODE_ID,
        };
        visit::walk_crate(&mut visitor, crate, ());
    }

    // Next, build up the list of all exported items from this crate
    {
        // Initialize the exported items with resolve's id for the "root crate"
        // to resolve references to `super` leading to the root and such.
        exported_items.insert(0);
        let mut visitor = EmbargoVisitor {
            exported_items: &mut exported_items,
            exp_map2: exp_map2,
            path_all_public: true, // start out as public
        };
        visit::walk_crate(&mut visitor, crate, ());
    }

    // And then actually check the privacy of everything.
    {
        let mut visitor = PrivacyVisitor {
            curitem: ast::DUMMY_NODE_ID,
            tcx: tcx,
            exported_items: &exported_items,
            parents: &parents,
            method_map: method_map,
            external_exports: external_exports,
            last_private_map: last_private_map,
        };
        visit::walk_crate(&mut visitor, crate, ());
    }
}
