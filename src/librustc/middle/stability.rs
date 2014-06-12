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
use syntax::ast::{Attribute, Block, Crate, DefId, FnDecl, NodeId, Variant};
use syntax::ast::{Item, Required, Provided, TraitMethod, TypeMethod, Method};
use syntax::ast::{Generics, StructDef, Ident};
use syntax::ast_util::is_local;
use syntax::attr::Stability;
use syntax::visit::{FnKind, FkMethod, Visitor};
use metadata::{cstore, csearch};

/// A stability index, giving the stability level for items and methods.
pub struct Index {
    // stability for crate-local items; unmarked stability == no entry
    local: NodeMap<Stability>,
    // cache for extern-crate items; unmarked stability == entry with None
    extern_cache: DefIdMap<Option<Stability>>
}

// A private tree-walker for producing an Index.
struct Annotator {
    index: Index
}

impl Annotator {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and returned.
    fn annotate(&mut self, id: NodeId, attrs: &[Attribute],
                parent: Option<Stability>) -> Option<Stability> {
        match attr::find_stability(attrs).or(parent) {
            Some(stab) => {
                self.index.local.insert(id, stab.clone());
                Some(stab)
            }
            None => None
        }
    }
}

impl Visitor<Option<Stability>> for Annotator {
    fn visit_item(&mut self, i: &Item, parent: Option<Stability>) {
        let stab = self.annotate(i.id, i.attrs.as_slice(), parent);
        visit::walk_item(self, i, stab)
    }

    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block,
                s: Span, _: NodeId, parent: Option<Stability>) {
        let stab = match *fk {
            FkMethod(_, _, meth) =>
                self.annotate(meth.id, meth.attrs.as_slice(), parent),
            _ => parent
        };
        visit::walk_fn(self, fk, fd, b, s, stab)
    }

    fn visit_trait_method(&mut self, t: &TraitMethod, parent: Option<Stability>) {
        let stab = match *t {
            Required(TypeMethod {attrs: ref attrs, id: id, ..}) =>
                self.annotate(id, attrs.as_slice(), parent),

            // work around lack of pattern matching for @ types
            Provided(method) => match *method {
                Method {attrs: ref attrs, id: id, ..} =>
                    self.annotate(id, attrs.as_slice(), parent)
            }
        };
        visit::walk_trait_method(self, t, stab)
    }

    fn visit_variant(&mut self, v: &Variant, g: &Generics, parent: Option<Stability>) {
        let stab = self.annotate(v.node.id, v.node.attrs.as_slice(), parent);
        visit::walk_variant(self, v, g, stab)
    }

    fn visit_struct_def(&mut self, s: &StructDef, _: Ident, _: &Generics,
                        _: NodeId, parent: Option<Stability>) {
        s.ctor_id.map(|id| self.annotate(id, &[], parent.clone()));
        visit::walk_struct_def(self, s, parent)
    }
}

impl Index {
    /// Construct the stability index for a crate being compiled.
    pub fn build(krate: &Crate) -> Index {
        let mut annotator = Annotator {
            index: Index {
                local: NodeMap::new(),
                extern_cache: DefIdMap::new()
            }
        };
        visit::walk_crate(&mut annotator, krate,
                          attr::find_stability(krate.attrs.as_slice()));
        annotator.index
    }

    /// Lookup the stability for a node, loading external crate
    /// metadata as necessary.
    pub fn lookup(&mut self, cstore: &cstore::CStore, id: DefId) -> Option<Stability> {
        if is_local(id) {
            self.lookup_local(id.node)
        } else {
            let stab = csearch::get_stability(cstore, id);
            self.extern_cache.insert(id, stab.clone());
            stab
        }
    }

    /// Lookup the stability for a local node without loading any external crates
    pub fn lookup_local(&self, id: NodeId) -> Option<Stability> {
        self.local.find_copy(&id)
    }
}
