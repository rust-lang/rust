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

use std::hashmap::HashSet;

use metadata::csearch;
use middle::resolve::ExportMap2;
use middle::ty::{ty_struct, ty_enum};
use middle::ty;
use middle::typeck::{method_map, method_origin, method_param};
use middle::typeck::{method_static, method_object};

use std::util::ignore;
use syntax::ast::{DeclItem, Def, DefFn, DefId, DefStaticMethod};
use syntax::ast::{DefVariant, ExprField, ExprMethodCall, ExprPath};
use syntax::ast::{ExprStruct, ExprUnary, Ident, inherited, item_enum};
use syntax::ast::{item_foreign_mod, item_fn, item_impl, item_struct};
use syntax::ast::{item_trait, LOCAL_CRATE, NodeId, PatStruct, Path};
use syntax::ast::{private, provided, public, required, StmtDecl, visibility};
use syntax::ast;
use syntax::ast_map::{node_foreign_item, node_item, node_method};
use syntax::ast_map::{node_trait_method};
use syntax::ast_map;
use syntax::ast_util::{Private, Public, is_local};
use syntax::ast_util::{variant_visibility_to_privacy, visibility_to_privacy};
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit;
use syntax::visit::Visitor;
use syntax::ast::{_mod,Expr,item,Block,Pat};

// This set is a set of all item nodes which can be used by external crates if
// we're building a library. The necessary qualifications for this are that all
// items leading down to the current item (excluding an `impl`) must be `pub`.
pub type ExportedItems = HashSet<NodeId>;

type Context<'self> = (&'self method_map, &'self ExportMap2);

struct PrivacyVisitor {
    tcx: ty::ctxt,
    privileged_items: @mut ~[NodeId],

    // A set of all items which are re-exported to be used across crates
    exported_items: ExportedItems,

    // A flag as to whether the current path is public all the way down to the
    // current point or not
    path_all_public: bool,
}

impl PrivacyVisitor {
    // Adds an item to its scope.
    fn add_privileged_item(&mut self, item: @ast::item, count: &mut uint) {
        match item.node {
            item_struct(*) | item_trait(*) | item_enum(*) |
            item_fn(*) => {
                self.privileged_items.push(item.id);
                *count += 1;
            }
            item_impl(_, _, _, ref methods) => {
                for method in methods.iter() {
                    self.privileged_items.push(method.id);
                    *count += 1;
                }
                self.privileged_items.push(item.id);
                *count += 1;
            }
            item_foreign_mod(ref foreign_mod) => {
                for foreign_item in foreign_mod.items.iter() {
                    self.privileged_items.push(foreign_item.id);
                    *count += 1;
                }
            }
            _ => {}
        }
    }

    // Adds items that are privileged to this scope.
    fn add_privileged_items(&mut self, items: &[@ast::item]) -> uint {
        let mut count = 0;
        for &item in items.iter() {
            self.add_privileged_item(item, &mut count);
        }
        count
    }

    // Checks that an enum variant is in scope
    fn check_variant(&mut self, span: Span, enum_id: ast::DefId) {
        let variant_info = ty::enum_variants(self.tcx, enum_id)[0];
        let parental_privacy = if is_local(enum_id) {
            let parent_vis = ast_map::node_item_query(self.tcx.items,
                                                      enum_id.node,
                                   |it| { it.vis },
                                   ~"unbound enum parent when checking \
                                    dereference of enum type");
            visibility_to_privacy(parent_vis)
        }
        else {
            // WRONG
            Public
        };
        debug2!("parental_privacy = {:?}", parental_privacy);
        debug2!("vis = {:?}, priv = {:?}",
               variant_info.vis,
               visibility_to_privacy(variant_info.vis))
        // inherited => privacy of the enum item
        if variant_visibility_to_privacy(variant_info.vis,
                                         parental_privacy == Public)
                                         == Private {
            self.tcx.sess.span_err(span,
                "can only dereference enums \
                 with a single, public variant");
        }
    }

    // Returns true if a crate-local method is private and false otherwise.
    fn method_is_private(&mut self, span: Span, method_id: NodeId) -> bool {
        let check = |vis: visibility, container_id: DefId| {
            let mut is_private = false;
            if vis == private {
                is_private = true;
            } else if vis == public {
                is_private = false;
            } else {
                // Look up the enclosing impl.
                if container_id.crate != LOCAL_CRATE {
                    self.tcx.sess.span_bug(span,
                                      "local method isn't in local \
                                       impl?!");
                }

                match self.tcx.items.find(&container_id.node) {
                    Some(&node_item(item, _)) => {
                        match item.node {
                            item_impl(_, None, _, _)
                                    if item.vis != public => {
                                is_private = true;
                            }
                            _ => {}
                        }
                    }
                    Some(_) => {
                        self.tcx.sess.span_bug(span, "impl wasn't an item?!");
                    }
                    None => {
                        self.tcx.sess.span_bug(span, "impl wasn't in AST map?!");
                    }
                }
            }

            is_private
        };

        match self.tcx.items.find(&method_id) {
            Some(&node_method(method, impl_id, _)) => {
                check(method.vis, impl_id)
            }
            Some(&node_trait_method(trait_method, trait_id, _)) => {
                match *trait_method {
                    required(_) => check(public, trait_id),
                    provided(method) => check(method.vis, trait_id),
                }
            }
            Some(_) => {
                self.tcx.sess.span_bug(span,
                                  format!("method_is_private: method was a {}?!",
                                       ast_map::node_id_to_str(
                                            self.tcx.items,
                                            method_id,
                                           token::get_ident_interner())));
            }
            None => {
                self.tcx.sess.span_bug(span, "method not found in \
                                         AST map?!");
            }
        }
    }

    // Returns true if the given local item is private and false otherwise.
    fn local_item_is_private(&mut self, span: Span, item_id: NodeId) -> bool {
        let mut f: &fn(NodeId) -> bool = |_| false;
        f = |item_id| {
            match self.tcx.items.find(&item_id) {
                Some(&node_item(item, _)) => item.vis != public,
                Some(&node_foreign_item(*)) => false,
                Some(&node_method(method, impl_did, _)) => {
                    match method.vis {
                        private => true,
                        public => false,
                        inherited => f(impl_did.node)
                    }
                }
                Some(&node_trait_method(_, trait_did, _)) => f(trait_did.node),
                Some(_) => {
                    self.tcx.sess.span_bug(span,
                                      format!("local_item_is_private: item was \
                                            a {}?!",
                                           ast_map::node_id_to_str(
                                                self.tcx.items,
                                                item_id,
                                               token::get_ident_interner())));
                }
                None => {
                    self.tcx.sess.span_bug(span, "item not found in AST map?!");
                }
            }
        };
        f(item_id)
    }

    // Checks that a private field is in scope.
    // FIXME #6993: change type (and name) from Ident to Name
    fn check_field(&mut self, span: Span, id: ast::DefId, ident: ast::Ident) {
        let fields = ty::lookup_struct_fields(self.tcx, id);
        for field in fields.iter() {
            if field.name != ident.name { loop; }
            if field.vis == private {
                self.tcx.sess.span_err(span, format!("field `{}` is private",
                                             token::ident_to_str(&ident)));
            }
            break;
        }
    }

    // Given the ID of a method, checks to ensure it's in scope.
    fn check_method_common(&mut self, span: Span, method_id: DefId, name: &Ident) {
        // If the method is a default method, we need to use the def_id of
        // the default implementation.
        // Having to do this this is really unfortunate.
        let method_id = ty::method(self.tcx, method_id).provided_source.unwrap_or(method_id);

        if method_id.crate == LOCAL_CRATE {
            let is_private = self.method_is_private(span, method_id.node);
            let container_id = ty::method(self.tcx, method_id).container_id();
            if is_private &&
                    (container_id.crate != LOCAL_CRATE ||
                     !self.privileged_items.iter().any(|x| x == &(container_id.node))) {
                self.tcx.sess.span_err(span,
                                  format!("method `{}` is private",
                                       token::ident_to_str(name)));
            }
        } else {
            let visibility =
                csearch::get_item_visibility(self.tcx.sess.cstore, method_id);
            if visibility != public {
                self.tcx.sess.span_err(span,
                                  format!("method `{}` is private",
                                       token::ident_to_str(name)));
            }
        }
    }

    // Checks that a private path is in scope.
    fn check_path(&mut self, span: Span, def: Def, path: &Path) {
        debug2!("checking path");
        match def {
            DefStaticMethod(method_id, _, _) => {
                debug2!("found static method def, checking it");
                self.check_method_common(span,
                                         method_id,
                                         &path.segments.last().identifier)
            }
            DefFn(def_id, _) => {
                if def_id.crate == LOCAL_CRATE {
                    if self.local_item_is_private(span, def_id.node) &&
                            !self.privileged_items.iter().any(|x| x == &def_id.node) {
                        self.tcx.sess.span_err(span,
                                          format!("function `{}` is private",
                                               token::ident_to_str(
                                                &path.segments
                                                     .last()
                                                     .identifier)));
                    }
                //} else if csearch::get_item_visibility(self.tcx.sess.cstore,
                //                                       def_id) != public {
                //    self.tcx.sess.span_err(span,
                //                      format!("function `{}` is private",
                //                           token::ident_to_str(
                //                                &path.segments
                //                                     .last()
                //                                     .identifier)));
                }
                // If this is a function from a non-local crate, then the
                // privacy check is enforced during resolve. All public items
                // will be tagged as such in the crate metadata and then usage
                // of the private items will be blocked during resolve. Hence,
                // if this isn't from the local crate, nothing to check.
            }
            _ => {}
        }
    }

    // Checks that a private method is in scope.
    fn check_method(&mut self, span: Span, origin: &method_origin, ident: ast::Ident) {
        match *origin {
            method_static(method_id) => {
                self.check_method_common(span, method_id, &ident)
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
                if trait_id.crate == LOCAL_CRATE {
                    match self.tcx.items.find(&trait_id.node) {
                        Some(&node_item(item, _)) => {
                            match item.node {
                                item_trait(_, _, ref methods) => {
                                    if method_num >= (*methods).len() {
                                        self.tcx.sess.span_bug(span,
                                                               "method number out of range?!");
                                    }
                                    match (*methods)[method_num] {
                                        provided(method)
                                             if method.vis == private &&
                                             !self.privileged_items.iter()
                                             .any(|x| x == &(trait_id.node)) => {
                                            self.tcx.sess.span_err(span,
                                                              format!("method `{}` is private",
                                                                   token::ident_to_str(&method
                                                                                        .ident)));
                                        }
                                        provided(_) | required(_) => {
                                            // Required methods can't be
                                            // private.
                                        }
                                    }
                                }
                                _ => {
                                    self.tcx.sess.span_bug(span, "trait wasn't actually a trait?!");
                                }
                            }
                        }
                        Some(_) => {
                            self.tcx.sess.span_bug(span, "trait wasn't an item?!");
                        }
                        None => {
                            self.tcx.sess.span_bug(span,
                                                   "trait item wasn't found in the AST map?!");
                        }
                    }
                } else {
                    // FIXME #4732: External crates.
                }
            }
        }
    }
}

impl<'self> Visitor<Context<'self>> for PrivacyVisitor {

    fn visit_mod(&mut self, the_module:&_mod, _:Span, _:NodeId,
                 cx: Context<'self>) {

            let n_added = self.add_privileged_items(the_module.items);

            visit::walk_mod(self, the_module, cx);

            do n_added.times {
                ignore(self.privileged_items.pop());
            }
    }

    fn visit_item(&mut self, item:@item, cx: Context<'self>) {

        // Do not check privacy inside items with the resolve_unexported
        // attribute. This is used for the test runner.
        if attr::contains_name(item.attrs, "!resolve_unexported") {
            return;
        }

        // Disallow unnecessary visibility qualifiers
        check_sane_privacy(self.tcx, item);

        // Keep track of whether this item is available for export or not.
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
        debug2!("public path at {}: {}", item.id, self.path_all_public);

        if self.path_all_public {
            debug2!("all the way public {}", item.id);
            self.exported_items.insert(item.id);

            // All re-exported items in a module which is public should also be
            // public (in terms of how they should get encoded)
            match item.node {
                ast::item_mod(*) => {
                    let (_, exp_map2) = cx;
                    match exp_map2.find(&item.id) {
                        Some(exports) => {
                            for export in exports.iter() {
                                if export.reexport && is_local(export.def_id) {
                                    debug2!("found reexported {:?}", export);
                                    let id = export.def_id.node;
                                    self.exported_items.insert(id);
                                }
                            }
                        }
                        None => {}
                    }
                }
                _ => {}
            }
        }

        visit::walk_item(self, item, cx);

        self.path_all_public = orig_all_pub;
    }

    fn visit_block(&mut self, block:&Block, cx: Context<'self>) {

            // Gather up all the privileged items.
            let mut n_added = 0;
            for stmt in block.stmts.iter() {
                match stmt.node {
                    StmtDecl(decl, _) => {
                        match decl.node {
                            DeclItem(item) => {
                                self.add_privileged_item(item, &mut n_added);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }

            visit::walk_block(self, block, cx);

            do n_added.times {
                ignore(self.privileged_items.pop());
            }

    }

    fn visit_expr(&mut self, expr:@Expr, cx: Context<'self>) {
        let (method_map, _) = cx;
            match expr.node {
                ExprField(base, ident, _) => {
                    // Method calls are now a special syntactic form,
                    // so `a.b` should always be a field.
                    assert!(!method_map.contains_key(&expr.id));

                    // With type_autoderef, make sure we don't
                    // allow pointers to violate privacy
                    match ty::get(ty::type_autoderef(self.tcx, ty::expr_ty(self.tcx,
                                                          base))).sty {
                        ty_struct(id, _)
                        if id.crate != LOCAL_CRATE || !self.privileged_items.iter()
                                .any(|x| x == &(id.node)) => {
                            debug2!("(privacy checking) checking field access");
                            self.check_field(expr.span, id, ident);
                        }
                        _ => {}
                    }
                }
                ExprMethodCall(_, base, ident, _, _, _) => {
                    // Ditto
                    match ty::get(ty::type_autoderef(self.tcx, ty::expr_ty(self.tcx,
                                                          base))).sty {
                        ty_enum(id, _) |
                        ty_struct(id, _)
                        if id.crate != LOCAL_CRATE ||
                           !self.privileged_items.iter().any(|x| x == &(id.node)) => {
                            match method_map.find(&expr.id) {
                                None => {
                                    self.tcx.sess.span_bug(expr.span,
                                                      "method call not in \
                                                       method map");
                                }
                                Some(ref entry) => {
                                    debug2!("(privacy checking) checking \
                                            impl method");
                                    self.check_method(expr.span, &entry.origin, ident);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                ExprPath(ref path) => {
                    self.check_path(expr.span, self.tcx.def_map.get_copy(&expr.id), path);
                }
                ExprStruct(_, ref fields, _) => {
                    match ty::get(ty::expr_ty(self.tcx, expr)).sty {
                        ty_struct(id, _) => {
                            if id.crate != LOCAL_CRATE ||
                                    !self.privileged_items.iter().any(|x| x == &(id.node)) {
                                for field in (*fields).iter() {
                                        debug2!("(privacy checking) checking \
                                                field in struct literal");
                                    self.check_field(expr.span, id, field.ident);
                                }
                            }
                        }
                        ty_enum(id, _) => {
                            if id.crate != LOCAL_CRATE ||
                                    !self.privileged_items.iter().any(|x| x == &(id.node)) {
                                match self.tcx.def_map.get_copy(&expr.id) {
                                    DefVariant(_, variant_id, _) => {
                                        for field in (*fields).iter() {
                                                debug2!("(privacy checking) \
                                                        checking field in \
                                                        struct variant \
                                                        literal");
                                            self.check_field(expr.span, variant_id, field.ident);
                                        }
                                    }
                                    _ => {
                                        self.tcx.sess.span_bug(expr.span,
                                                          "resolve didn't \
                                                           map enum struct \
                                                           constructor to a \
                                                           variant def");
                                    }
                                }
                            }
                        }
                        _ => {
                            self.tcx.sess.span_bug(expr.span, "struct expr \
                                                          didn't have \
                                                          struct type?!");
                        }
                    }
                }
                ExprUnary(_, ast::UnDeref, operand) => {
                    // In *e, we need to check that if e's type is an
                    // enum type t, then t's first variant is public or
                    // privileged. (We can assume it has only one variant
                    // since typeck already happened.)
                    match ty::get(ty::expr_ty(self.tcx, operand)).sty {
                        ty_enum(id, _) => {
                            if id.crate != LOCAL_CRATE ||
                                !self.privileged_items.iter().any(|x| x == &(id.node)) {
                                self.check_variant(expr.span, id);
                            }
                        }
                        _ => { /* No check needed */ }
                    }
                }
                _ => {}
            }

            visit::walk_expr(self, expr, cx);

    }

    fn visit_pat(&mut self, pattern:@Pat, cx: Context<'self>) {

            match pattern.node {
                PatStruct(_, ref fields, _) => {
                    match ty::get(ty::pat_ty(self.tcx, pattern)).sty {
                        ty_struct(id, _) => {
                            if id.crate != LOCAL_CRATE ||
                                    !self.privileged_items.iter().any(|x| x == &(id.node)) {
                                for field in fields.iter() {
                                        debug2!("(privacy checking) checking \
                                                struct pattern");
                                    self.check_field(pattern.span, id, field.ident);
                                }
                            }
                        }
                        ty_enum(enum_id, _) => {
                            if enum_id.crate != LOCAL_CRATE ||
                                    !self.privileged_items.iter().any(|x| x == &enum_id.node) {
                                match self.tcx.def_map.find(&pattern.id) {
                                    Some(&DefVariant(_, variant_id, _)) => {
                                        for field in fields.iter() {
                                            debug2!("(privacy checking) \
                                                    checking field in \
                                                    struct variant pattern");
                                            self.check_field(pattern.span, variant_id, field.ident);
                                        }
                                    }
                                    _ => {
                                        self.tcx.sess.span_bug(pattern.span,
                                                          "resolve didn't \
                                                           map enum struct \
                                                           pattern to a \
                                                           variant def");
                                    }
                                }
                            }
                        }
                        _ => {
                            self.tcx.sess.span_bug(pattern.span,
                                              "struct pattern didn't have \
                                               struct type?!");
                        }
                    }
                }
                _ => {}
            }

            visit::walk_pat(self, pattern, cx);
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: &method_map,
                   exp_map2: &ExportMap2,
                   crate: &ast::Crate) -> ExportedItems {
    let privileged_items = @mut ~[];

    let mut visitor = PrivacyVisitor {
        tcx: tcx,
        privileged_items: privileged_items,
        exported_items: HashSet::new(),
        path_all_public: true, // start out as public
    };
    visit::walk_crate(&mut visitor, crate, (method_map, exp_map2));
    return visitor.exported_items;
}

/// Validates all of the visibility qualifers placed on the item given. This
/// ensures that there are no extraneous qualifiers that don't actually do
/// anything. In theory these qualifiers wouldn't parse, but that may happen
/// later on down the road...
fn check_sane_privacy(tcx: ty::ctxt, item: @ast::item) {
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
                            "visibility qualifiers have no effect on trait impls");
            for m in methods.iter() {
                check_inherited(m.span, m.vis, "");
            }
        }

        ast::item_impl(_, _, _, ref methods) => {
            check_inherited(item.span, item.vis,
                            "place qualifiers on individual methods instead");
            for i in methods.iter() {
                check_not_priv(i.span, i.vis, "functions are private by default");
            }
        }
        ast::item_foreign_mod(ref fm) => {
            check_inherited(item.span, item.vis,
                            "place qualifiers on individual functions instead");
            for i in fm.items.iter() {
                check_not_priv(i.span, i.vis, "functions are private by default");
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
            check_not_priv(item.span, item.vis, "items are private by default");
        }
    }
}
