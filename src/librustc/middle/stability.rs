// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use util::nodemap::{NodeMap, DefIdMap};
use syntax::codemap::Span;
use syntax::{attr, visit};
use syntax::ast;
use syntax::ast::{Attribute, Block, Crate, DefId, FnDecl, NodeId, Variant};
use syntax::ast::{Item, RequiredMethod, ProvidedMethod, TraitItem};
use syntax::ast::{TypeMethod, Method, Generics, StructDef, StructField};
use syntax::ast::{Ident, TypeTraitItem};
use syntax::ast_util::is_local;
use syntax::attr::Stability;
use syntax::visit::{FnKind, FkMethod, Visitor};
use middle::ty;
use metadata::csearch;

use std::mem::replace;

/// A stability index, giving the stability level for items and methods.
pub struct Index {
    // stability for crate-local items; unmarked stability == no entry
    local: NodeMap<Stability>,
    // cache for extern-crate items; unmarked stability == entry with None
    extern_cache: DefIdMap<Option<Stability>>
}

// A private tree-walker for producing an Index.
struct Annotator {
    index: Index,
    parent: Option<Stability>
}

impl Annotator {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate(&mut self, id: NodeId, attrs: &Vec<Attribute>, f: |&mut Annotator|) {
        match attr::find_stability(attrs.as_slice()) {
            Some(stab) => {
                self.index.local.insert(id, stab.clone());
                let parent = replace(&mut self.parent, Some(stab));
                f(self);
                self.parent = parent;
            }
            None => {
                self.parent.clone().map(|stab| self.index.local.insert(id, stab));
                f(self);
            }
        }
    }
}

impl<'v> Visitor<'v> for Annotator {
    fn visit_item(&mut self, i: &Item) {
        self.annotate(i.id, &i.attrs, |v| visit::walk_item(v, i));
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl,
                b: &'v Block, s: Span, _: NodeId) {
        match fk {
            FkMethod(_, _, meth) => {
                self.annotate(meth.id, &meth.attrs, |v| visit::walk_fn(v, fk, fd, b, s));
            }
            _ => visit::walk_fn(self, fk, fd, b, s)
        }
    }

    fn visit_trait_item(&mut self, t: &TraitItem) {
        let (id, attrs) = match *t {
            RequiredMethod(TypeMethod {id, ref attrs, ..}) => (id, attrs),

            // work around lack of pattern matching for @ types
            ProvidedMethod(ref method) => {
                match **method {
                    Method {attrs: ref attrs, id: id, ..} => (id, attrs),
                }
            }

            TypeTraitItem(ref typedef) => (typedef.id, &typedef.attrs),
        };
        self.annotate(id, attrs, |v| visit::walk_trait_item(v, t));
    }

    fn visit_variant(&mut self, var: &Variant, g: &'v Generics) {
        self.annotate(var.node.id, &var.node.attrs, |v| visit::walk_variant(v, var, g))
    }

    fn visit_struct_def(&mut self, s: &StructDef, _: Ident, _: &Generics, _: NodeId) {
        match s.ctor_id {
            Some(id) => self.annotate(id, &vec![], |v| visit::walk_struct_def(v, s)),
            None => visit::walk_struct_def(self, s)
        }
    }

    fn visit_struct_field(&mut self, s: &StructField) {
        self.annotate(s.node.id, &s.node.attrs, |v| visit::walk_struct_field(v, s));
    }
}

impl Index {
    /// Construct the stability index for a crate being compiled.
    pub fn build(krate: &Crate) -> Index {
        let mut annotator = Annotator {
            index: Index {
                local: NodeMap::new(),
                extern_cache: DefIdMap::new()
            },
            parent: None
        };
        annotator.annotate(ast::CRATE_NODE_ID, &krate.attrs, |v| visit::walk_crate(v, krate));
        annotator.index
    }
}

/// Lookup the stability for a node, loading external crate
/// metadata as necessary.
pub fn lookup(tcx: &ty::ctxt, id: DefId) -> Option<Stability> {
    // is this definition the implementation of a trait method?
    match ty::trait_item_of_item(tcx, id) {
        Some(ty::MethodTraitItemId(trait_method_id))
                if trait_method_id != id => {
            lookup(tcx, trait_method_id)
        }
        _ if is_local(id) => {
            tcx.stability.borrow().local.find_copy(&id.node)
        }
        _ => {
            let stab = csearch::get_stability(&tcx.sess.cstore, id);
            let mut index = tcx.stability.borrow_mut();
            (*index).extern_cache.insert(id, stab.clone());
            stab
        }
    }
}
