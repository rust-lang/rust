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

use metadata::csearch;
use middle::lint;
use middle::resolve;
use middle::ty;
use middle::typeck::{MethodCall, MethodMap, MethodOrigin, MethodParam};
use middle::typeck::{MethodStatic, MethodObject};
use util::nodemap::{NodeMap, NodeSet};

use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{is_local, def_id_of_def, local_def};
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::owned_slice::OwnedSlice;
use syntax::visit;
use syntax::visit::Visitor;

type Context<'a> = (&'a MethodMap, &'a resolve::ExportMap2);

/// A set of AST nodes exported by the crate.
pub type ExportedItems = NodeSet;

/// A set of AST nodes that are fully public in the crate. This map is used for
/// documentation purposes (reexporting a private struct inlines the doc,
/// reexporting a public struct doesn't inline the doc).
pub type PublicItems = NodeSet;

/// Result of a checking operation - None => no errors were found. Some => an
/// error and contains the span and message for reporting that error and
/// optionally the same for a note about the error.
type CheckResult = Option<(Span, ~str, Option<(Span, ~str)>)>;

////////////////////////////////////////////////////////////////////////////////
/// The parent visitor, used to determine what's the parent of what (node-wise)
////////////////////////////////////////////////////////////////////////////////

struct ParentVisitor {
    parents: NodeMap<ast::NodeId>,
    curparent: ast::NodeId,
}

impl Visitor<()> for ParentVisitor {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        self.parents.insert(item.id, self.curparent);

        let prev = self.curparent;
        match item.node {
            ast::ItemMod(..) => { self.curparent = item.id; }
            // Enum variants are parented to the enum definition itself because
            // they inherit privacy
            ast::ItemEnum(ref def, _) => {
                for variant in def.variants.iter() {
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
    tcx: &'a ty::ctxt,
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
    reexports: NodeSet,

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
            ast::ItemImpl(_, _, ref ty, ref methods) => {
                let public_ty = match ty.node {
                    ast::TyPath(_, _, id) => {
                        match self.tcx.def_map.borrow().get_copy(&id) {
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
                let public_trait = tr.clone().map_or(false, |tr| {
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
            ast::ItemTrait(_, _, _, ref methods) if public_first => {
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
            assert!(exp_map2.contains_key(&id), "wut {:?}", id);
            for export in exp_map2.get(&id).iter() {
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
    tcx: &'a ty::ctxt,
    curitem: ast::NodeId,
    in_fn: bool,
    in_foreign: bool,
    parents: NodeMap<ast::NodeId>,
    external_exports: resolve::ExternalExports,
    last_private_map: resolve::LastPrivateMap,
}

enum PrivacyResult {
    Allowable,
    ExternallyDenied,
    DisallowedBy(ast::NodeId),
}

enum FieldName {
    UnnamedField(uint), // index
    // FIXME #6993: change type (and name) from Ident to Name
    NamedField(ast::Ident),
}

impl<'a> PrivacyVisitor<'a> {
    // used when debugging
    fn nodestr(&self, id: ast::NodeId) -> ~str {
        self.tcx.map.node_to_str(id).to_owned()
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

            return match self.tcx.methods.borrow().find(&did) {
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

        debug!("privacy - local {} not public all the way down",
               self.tcx.map.node_to_str(did.node));
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
                Some(ast_map::NodeVariant(..)) => {
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
            debug!("privacy - questioning {}, {:?}", self.nodestr(cur), cur);
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
                    ast::ItemImpl(_, _, ref ty, _) => {
                        let id = match ty.node {
                            ast::TyPath(_, _, id) => id,
                            _ => return Some((err_span, err_msg, None)),
                        };
                        let def = self.tcx.def_map.borrow().get_copy(&id);
                        let did = def_id_of_def(def);
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
            NamedField(ident) => {
                debug!("privacy - check named field {} in struct {}", ident.name, id);
                fields.iter().find(|f| f.name == ident.name).unwrap()
            }
            UnnamedField(idx) => fields.get(idx)
        };
        if field.vis == ast::Public ||
            (is_local(field.id) && self.private_accessible(field.id.node)) {
            return
        }

        let struct_type = ty::lookup_item_type(self.tcx, id).ty;
        let struct_desc = match ty::get(struct_type).sty {
            ty::ty_struct(_, _) => format!("struct `{}`", ty::item_path_str(self.tcx, id)),
            ty::ty_bare_fn(ty::BareFnTy { sig: ty::FnSig { output, .. }, .. }) => {
                // Struct `id` is really a struct variant of an enum,
                // and we're really looking at the variant's constructor
                // function. So get the return type for a detailed error
                // message.
                let enum_id = match ty::get(output).sty {
                    ty::ty_enum(id, _) => id,
                    _ => self.tcx.sess.span_bug(span, "enum variant doesn't \
                                                       belong to an enum")
                };
                format!("variant `{}` of enum `{}`",
                        ty::with_path(self.tcx, id, |mut p| p.last().unwrap()),
                        ty::item_path_str(self.tcx, enum_id))
            }
            _ => self.tcx.sess.span_bug(span, "can't find struct for field")
        };
        let msg = match name {
            NamedField(name) => format!("field `{}` of {} is private",
                                        token::get_ident(name), struct_desc),
            UnnamedField(idx) => format!("field \\#{} of {} is private",
                                         idx + 1, struct_desc),
        };
        self.tcx.sess.span_err(span, msg);
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
        let orig_def = self.tcx.def_map.borrow().get_copy(&path_id);
        let ck = |tyname: &str| {
            let ck_public = |def: ast::DefId| {
                let name = token::get_ident(path.segments
                                                .last()
                                                .unwrap()
                                                .identifier);
                let origdid = def_id_of_def(orig_def);
                self.ensure_public(span, def, Some(origdid),
                                   format!("{} `{}`", tyname, name))
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
                    // If an import is not used in either namespace, we still
                    // want to check that it could be legal. Therefore we check
                    // in both namespaces and only report an error if both would
                    // be illegal. We only report one error, even if it is
                    // illegal to import from both namespaces.
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
        match self.tcx.def_map.borrow().get_copy(&path_id) {
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
    fn check_method(&mut self, span: Span, origin: MethodOrigin,
                    ident: ast::Ident) {
        match origin {
            MethodStatic(method_id) => {
                self.check_static_method(span, method_id, ident)
            }
            // Trait methods are always all public. The only controlling factor
            // is whether the trait itself is accessible or not.
            MethodParam(MethodParam { trait_id: trait_id, .. }) |
            MethodObject(MethodObject { trait_id: trait_id, .. }) => {
                self.report_error(self.ensure_public(span, trait_id, None,
                                                     "source trait"));
            }
        }
    }
}

impl<'a> Visitor<()> for PrivacyVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        // Do not check privacy inside items with the resolve_unexported
        // attribute. This is used for the test runner.
        if attr::contains_name(item.attrs.as_slice(), "!resolve_unexported") {
            return;
        }

        let orig_curitem = replace(&mut self.curitem, item.id);
        visit::walk_item(self, item, ());
        self.curitem = orig_curitem;
    }

    fn visit_expr(&mut self, expr: &ast::Expr, _: ()) {
        match expr.node {
            ast::ExprField(base, ident, _) => {
                match ty::get(ty::expr_ty_adjusted(self.tcx, base)).sty {
                    ty::ty_struct(id, _) => {
                        self.check_field(expr.span, id, NamedField(ident));
                    }
                    _ => {}
                }
            }
            ast::ExprMethodCall(ident, _, _) => {
                let method_call = MethodCall::expr(expr.id);
                match self.tcx.method_map.borrow().find(&method_call) {
                    None => {
                        self.tcx.sess.span_bug(expr.span,
                                                "method call not in \
                                                method map");
                    }
                    Some(method) => {
                        debug!("(privacy checking) checking impl method");
                        self.check_method(expr.span, method.origin, ident.node);
                    }
                }
            }
            ast::ExprStruct(_, ref fields, _) => {
                match ty::get(ty::expr_ty(self.tcx, expr)).sty {
                    ty::ty_struct(id, _) => {
                        for field in (*fields).iter() {
                            self.check_field(expr.span, id,
                                             NamedField(field.ident.node));
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.borrow().get_copy(&expr.id) {
                            ast::DefVariant(_, variant_id, _) => {
                                for field in fields.iter() {
                                    self.check_field(expr.span, variant_id,
                                                     NamedField(field.ident.node));
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
            ast::ExprPath(..) => {
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
                match self.tcx.def_map.borrow().find(&expr.id) {
                    Some(&ast::DefStruct(did)) => {
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
                    // Tuple struct constructors across crates are identified as
                    // DefFn types, so we explicitly handle that case here.
                    Some(&ast::DefFn(did, _)) if !is_local(did) => {
                        match csearch::get_tuple_struct_definition_if_ctor(
                                    &self.tcx.sess.cstore, did) {
                            Some(did) => guard(did),
                            None => {}
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        visit::walk_expr(self, expr, ());
    }

    fn visit_view_item(&mut self, a: &ast::ViewItem, _: ()) {
        match a.node {
            ast::ViewItemExternCrate(..) => {}
            ast::ViewItemUse(ref vpath) => {
                match vpath.node {
                    ast::ViewPathSimple(..) | ast::ViewPathGlob(..) => {}
                    ast::ViewPathList(_, ref list, _) => {
                        for pid in list.iter() {
                            debug!("privacy - list {}", pid.node.id);
                            let seg = ast::PathSegment {
                                identifier: pid.node.name,
                                lifetimes: Vec::new(),
                                types: OwnedSlice::empty(),
                            };
                            let segs = vec!(seg);
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
                            self.check_field(pattern.span, id,
                                             NamedField(field.ident));
                        }
                    }
                    ty::ty_enum(_, _) => {
                        match self.tcx.def_map.borrow().find(&pattern.id) {
                            Some(&ast::DefVariant(_, variant_id, _)) => {
                                for field in fields.iter() {
                                    self.check_field(pattern.span, variant_id,
                                                     NamedField(field.ident));
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
                match ty::get(ty::pat_ty(self.tcx, pattern)).sty {
                    ty::ty_struct(id, _) => {
                        for (i, field) in fields.iter().enumerate() {
                            match field.node {
                                ast::PatWild(..) => continue,
                                _ => {}
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

struct SanePrivacyVisitor<'a> {
    tcx: &'a ty::ctxt,
    in_fn: bool,
}

impl<'a> Visitor<()> for SanePrivacyVisitor<'a> {
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
            ast::Public => {
                if self.in_fn {
                    self.tcx.sess.span_err(i.span, "unnecessary `pub`, imports \
                                                    in functions are never \
                                                    reachable");
                } else {
                    match i.node {
                        ast::ViewItemExternCrate(..) => {
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

impl<'a> SanePrivacyVisitor<'a> {
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
            ast::ItemImpl(_, Some(..), _, ref methods) => {
                check_inherited(item.span, item.vis,
                                "visibility qualifiers have no effect on trait \
                                 impls");
                for m in methods.iter() {
                    check_inherited(m.span, m.vis, "");
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
                for v in def.variants.iter() {
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

            ast::ItemStatic(..) | ast::ItemStruct(..) |
            ast::ItemFn(..) | ast::ItemMod(..) | ast::ItemTy(..) |
            ast::ItemMac(..) => {}
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
        let check_struct = |def: &@ast::StructDef| {
            for f in def.fields.iter() {
               match f.node.kind {
                    ast::NamedField(_, p) => check_inherited(tcx, f.span, p),
                    ast::UnnamedField(..) => {}
                }
            }
        };
        check_inherited(tcx, item.span, item.vis);
        match item.node {
            ast::ItemImpl(_, _, _, ref methods) => {
                for m in methods.iter() {
                    check_inherited(tcx, m.span, m.vis);
                }
            }
            ast::ItemForeignMod(ref fm) => {
                for i in fm.items.iter() {
                    check_inherited(tcx, i.span, i.vis);
                }
            }
            ast::ItemEnum(ref def, _) => {
                for v in def.variants.iter() {
                    check_inherited(tcx, v.span, v.node.vis);

                    match v.node.kind {
                        ast::StructVariantKind(ref s) => check_struct(s),
                        ast::TupleVariantKind(..) => {}
                    }
                }
            }

            ast::ItemStruct(ref def, _) => check_struct(def),

            ast::ItemTrait(_, _, _, ref methods) => {
                for m in methods.iter() {
                    match *m {
                        ast::Required(..) => {}
                        ast::Provided(ref m) => check_inherited(tcx, m.span,
                                                                m.vis),
                    }
                }
            }

            ast::ItemStatic(..) |
            ast::ItemFn(..) | ast::ItemMod(..) | ast::ItemTy(..) |
            ast::ItemMac(..) => {}
        }
    }
}

struct VisiblePrivateTypesVisitor<'a> {
    tcx: &'a ty::ctxt,
    exported_items: &'a ExportedItems,
    public_items: &'a PublicItems,
}

struct CheckTypeForPrivatenessVisitor<'a, 'b> {
    inner: &'b VisiblePrivateTypesVisitor<'a>,
    /// whether the type refers to private types.
    contains_private: bool,
    /// whether we've recurred at all (i.e. if we're pointing at the
    /// first type on which visit_ty was called).
    at_outer_type: bool,
    // whether that first type is a public path.
    outer_type_is_public_path: bool,
}

impl<'a> VisiblePrivateTypesVisitor<'a> {
    fn path_is_private_type(&self, path_id: ast::NodeId) -> bool {
        let did = match self.tcx.def_map.borrow().find_copy(&path_id) {
            // `int` etc. (None doesn't seem to occur.)
            None | Some(ast::DefPrimTy(..)) => return false,
            Some(def) => def_id_of_def(def)
        };
        // A path can only be private if:
        // it's in this crate...
        is_local(did) &&
            // ... it's not exported (obviously) ...
            !self.exported_items.contains(&did.node) &&
            // .. and it corresponds to a type in the AST (this returns None for
            // type parameters)
            self.tcx.map.find(did.node).is_some()
    }

    fn trait_is_public(&self, trait_id: ast::NodeId) -> bool {
        // FIXME: this would preferably be using `exported_items`, but all
        // traits are exported currently (see `EmbargoVisitor.exported_trait`)
        self.public_items.contains(&trait_id)
    }
}

impl<'a, 'b> Visitor<()> for CheckTypeForPrivatenessVisitor<'a, 'b> {
    fn visit_ty(&mut self, ty: &ast::Ty, _: ()) {
        match ty.node {
            ast::TyPath(_, _, path_id) => {
                if self.inner.path_is_private_type(path_id) {
                    self.contains_private = true;
                    // found what we're looking for so let's stop
                    // working.
                    return
                } else if self.at_outer_type {
                    self.outer_type_is_public_path = true;
                }
            }
            _ => {}
        }
        self.at_outer_type = false;
        visit::walk_ty(self, ty, ())
    }

    // don't want to recurse into [, .. expr]
    fn visit_expr(&mut self, _: &ast::Expr, _: ()) {}
}

impl<'a> Visitor<()> for VisiblePrivateTypesVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        match item.node {
            // contents of a private mod can be reexported, so we need
            // to check internals.
            ast::ItemMod(_) => {}

            // An `extern {}` doesn't introduce a new privacy
            // namespace (the contents have their own privacies).
            ast::ItemForeignMod(_) => {}

            ast::ItemTrait(..) if !self.trait_is_public(item.id) => return,

            // impls need some special handling to try to offer useful
            // error messages without (too many) false positives
            // (i.e. we could just return here to not check them at
            // all, or some worse estimation of whether an impl is
            // publically visible.
            ast::ItemImpl(ref g, ref trait_ref, self_, ref methods) => {
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
                    visitor.visit_ty(self_, ());
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
                    methods.iter().any(|m| self.exported_items.contains(&m.id));

                if !self_contains_private &&
                        not_private_trait &&
                        trait_or_some_public_method {

                    visit::walk_generics(self, g, ());

                    match *trait_ref {
                        None => {
                            for method in methods.iter() {
                                visit::walk_method_helper(self, *method, ())
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
                            visit::walk_trait_ref_helper(self, tr, ())
                        }
                    }
                } else if trait_ref.is_none() && self_is_public_path {
                    // impl Public<Private> { ... }. Any public static
                    // methods will be visible as `Public::foo`.
                    let mut found_pub_static = false;
                    for method in methods.iter() {
                        if method.explicit_self.node == ast::SelfStatic &&
                            self.exported_items.contains(&method.id) {
                            found_pub_static = true;
                            visit::walk_method_helper(self, *method, ());
                        }
                    }
                    if found_pub_static {
                        visit::walk_generics(self, g, ())
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
        visit::walk_item(self, item, ());
    }

    fn visit_foreign_item(&mut self, item: &ast::ForeignItem, _: ()) {
        if self.exported_items.contains(&item.id) {
            visit::walk_foreign_item(self, item, ())
        }
    }

    fn visit_fn(&mut self,
                fk: &visit::FnKind, fd: &ast::FnDecl, b: &ast::Block, s: Span, id: ast::NodeId,
                _: ()) {
        // needs special handling for methods.
        if self.exported_items.contains(&id) {
            visit::walk_fn(self, fk, fd, b, s, id, ());
        }
    }

    fn visit_ty(&mut self, t: &ast::Ty, _: ()) {
        match t.node {
            ast::TyPath(ref p, _, path_id) => {
                if self.path_is_private_type(path_id) {
                    self.tcx.sess.add_lint(lint::VisiblePrivateTypes,
                                           path_id, p.span,
                                           "private type in exported type signature".to_owned());
                }
            }
            _ => {}
        }
        visit::walk_ty(self, t, ())
    }

    fn visit_variant(&mut self, v: &ast::Variant, g: &ast::Generics, _: ()) {
        if self.exported_items.contains(&v.node.id) {
            visit::walk_variant(self, v, g, ());
        }
    }

    fn visit_struct_field(&mut self, s: &ast::StructField, _: ()) {
        match s.node.kind {
            ast::NamedField(_, ast::Public)  => {
                visit::walk_struct_field(self, s, ());
            }
            _ => {}
        }
    }


    // we don't need to introspect into these at all: an
    // expression/block context can't possibly contain exported
    // things, and neither do view_items. (Making them no-ops stops us
    // from traversing the whole AST without having to be super
    // careful about our `walk_...` calls above.)
    fn visit_view_item(&mut self, _: &ast::ViewItem, _: ()) {}
    fn visit_block(&mut self, _: &ast::Block, _: ()) {}
    fn visit_expr(&mut self, _: &ast::Expr, _: ()) {}
}

pub fn check_crate(tcx: &ty::ctxt,
                   exp_map2: &resolve::ExportMap2,
                   external_exports: resolve::ExternalExports,
                   last_private_map: resolve::LastPrivateMap,
                   krate: &ast::Crate) -> (ExportedItems, PublicItems) {
    // Figure out who everyone's parent is
    let mut visitor = ParentVisitor {
        parents: NodeMap::new(),
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
        exported_items: NodeSet::new(),
        public_items: NodeSet::new(),
        reexports: NodeSet::new(),
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

    {
        let mut visitor = VisiblePrivateTypesVisitor {
            tcx: tcx,
            exported_items: &exported_items,
            public_items: &public_items
        };
        visit::walk_crate(&mut visitor, krate, ());
    }
    return (exported_items, public_items);
}
