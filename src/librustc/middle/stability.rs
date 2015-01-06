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

use middle::ty;
use metadata::csearch;
use syntax::codemap::Span;
use syntax::{attr, visit};
use syntax::ast;
use syntax::ast::{Attribute, Block, Crate, DefId, FnDecl, NodeId, Variant};
use syntax::ast::{Item, RequiredMethod, ProvidedMethod, TraitItem};
use syntax::ast::{TypeMethod, Method, Generics, StructField, TypeTraitItem};
use syntax::ast_util::is_local;
use syntax::attr::Stability;
use syntax::visit::{FnKind, FkMethod, Visitor};
use util::nodemap::{NodeMap, DefIdMap};
use util::ppaux::Repr;

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
    fn annotate<F>(&mut self, id: NodeId, use_parent: bool,
                   attrs: &Vec<Attribute>, f: F) where
        F: FnOnce(&mut Annotator),
    {
        match attr::find_stability(attrs.as_slice()) {
            Some(stab) => {
                self.index.local.insert(id, stab.clone());

                // Don't inherit #[stable]
                if stab.level != attr::Stable {
                    let parent = replace(&mut self.parent, Some(stab));
                    f(self);
                    self.parent = parent;
                } else {
                    f(self);
                }
            }
            None => {
                if use_parent {
                    self.parent.clone().map(|stab| self.index.local.insert(id, stab));
                }
                f(self);
            }
        }
    }
}

impl<'v> Visitor<'v> for Annotator {
    fn visit_item(&mut self, i: &Item) {
        // FIXME (#18969): the following is a hack around the fact
        // that we cannot currently annotate the stability of
        // `deriving`.  Basically, we do *not* allow stability
        // inheritance on trait implementations, so that derived
        // implementations appear to be unannotated. This then allows
        // derived implementations to be automatically tagged with the
        // stability of the trait. This is WRONG, but expedient to get
        // libstd stabilized for the 1.0 release.
        let use_parent = match i.node {
            ast::ItemImpl(_, _, _, Some(_), _, _) => false,
            _ => true,
        };

        self.annotate(i.id, use_parent, &i.attrs, |v| visit::walk_item(v, i));

        if let ast::ItemStruct(ref sd, _) = i.node {
            sd.ctor_id.map(|id| {
                self.annotate(id, true, &i.attrs, |_| {})
            });
        }
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, _: &'v FnDecl,
                _: &'v Block, _: Span, _: NodeId) {
        if let FkMethod(_, _, meth) = fk {
            // Methods are not already annotated, so we annotate it
            self.annotate(meth.id, true, &meth.attrs, |_| {});
        }
        // Items defined in a function body have no reason to have
        // a stability attribute, so we don't recurse.
    }

    fn visit_trait_item(&mut self, t: &TraitItem) {
        let (id, attrs) = match *t {
            RequiredMethod(TypeMethod {id, ref attrs, ..}) => (id, attrs),

            // work around lack of pattern matching for @ types
            ProvidedMethod(ref method) => {
                match **method {
                    Method {ref attrs, id, ..} => (id, attrs),
                }
            }

            TypeTraitItem(ref typedef) => (typedef.ty_param.id, &typedef.attrs),
        };
        self.annotate(id, true, attrs, |v| visit::walk_trait_item(v, t));
    }

    fn visit_variant(&mut self, var: &Variant, g: &'v Generics) {
        self.annotate(var.node.id, true, &var.node.attrs,
                      |v| visit::walk_variant(v, var, g))
    }

    fn visit_struct_field(&mut self, s: &StructField) {
        self.annotate(s.node.id, true, &s.node.attrs,
                      |v| visit::walk_struct_field(v, s));
    }

    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        self.annotate(i.id, true, &i.attrs, |_| {});
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
        annotator.annotate(ast::CRATE_NODE_ID, true, &krate.attrs,
                           |v| visit::walk_crate(v, krate));
        annotator.index
    }
}

/// Lookup the stability for a node, loading external crate
/// metadata as necessary.
pub fn lookup(tcx: &ty::ctxt, id: DefId) -> Option<Stability> {
    debug!("lookup(id={})",
           id.repr(tcx));

    // is this definition the implementation of a trait method?
    match ty::trait_item_of_item(tcx, id) {
        Some(ty::MethodTraitItemId(trait_method_id)) if trait_method_id != id => {
            debug!("lookup: trait_method_id={:?}", trait_method_id);
            return lookup(tcx, trait_method_id)
        }
        _ => {}
    }

    let item_stab = if is_local(id) {
        tcx.stability.borrow().local.get(&id.node).cloned()
    } else {
        let stab = csearch::get_stability(&tcx.sess.cstore, id);
        let mut index = tcx.stability.borrow_mut();
        (*index).extern_cache.insert(id, stab.clone());
        stab
    };

    item_stab.or_else(|| {
        if let Some(trait_id) = ty::trait_id_of_impl(tcx, id) {
            // FIXME (#18969): for the time being, simply use the
            // stability of the trait to determine the stability of any
            // unmarked impls for it. See FIXME above for more details.

            debug!("lookup: trait_id={:?}", trait_id);
            lookup(tcx, trait_id)
        } else {
            None
        }
    })
}

pub fn is_staged_api(tcx: &ty::ctxt, id: DefId) -> bool {
    match ty::trait_item_of_item(tcx, id) {
        Some(ty::MethodTraitItemId(trait_method_id))
            if trait_method_id != id => {
                is_staged_api(tcx, trait_method_id)
            }
        _ if is_local(id) => {
            // Unused case
            unreachable!()
        }
        _ => {
            csearch::is_staged_api(&tcx.sess.cstore, id)
        }
    }
}
