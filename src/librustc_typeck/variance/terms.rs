// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Representing terms
//
// Terms are structured as a straightforward tree. Rather than rely on
// GC, we allocate terms out of a bounded arena (the lifetime of this
// arena is the lifetime 'a that is threaded around).
//
// We assign a unique index to each type/region parameter whose variance
// is to be inferred. We refer to such variables as "inferreds". An
// `InferredIndex` is a newtype'd int representing the index of such
// a variable.

use arena::TypedArena;
use dep_graph::DepTrackingMapConfig;
use rustc::ty::{self, TyCtxt};
use rustc::ty::maps::ItemVariances;
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use util::nodemap::NodeMap;

use self::VarianceTerm::*;

pub type VarianceTermPtr<'a> = &'a VarianceTerm<'a>;

#[derive(Copy, Clone, Debug)]
pub struct InferredIndex(pub usize);

#[derive(Copy, Clone)]
pub enum VarianceTerm<'a> {
    ConstantTerm(ty::Variance),
    TransformTerm(VarianceTermPtr<'a>, VarianceTermPtr<'a>),
    InferredTerm(InferredIndex),
}

impl<'a> fmt::Debug for VarianceTerm<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ConstantTerm(c1) => write!(f, "{:?}", c1),
            TransformTerm(v1, v2) => write!(f, "({:?} \u{00D7} {:?})", v1, v2),
            InferredTerm(id) => {
                write!(f, "[{}]", {
                    let InferredIndex(i) = id;
                    i
                })
            }
        }
    }
}

// The first pass over the crate simply builds up the set of inferreds.

pub struct TermsContext<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub arena: &'a TypedArena<VarianceTerm<'a>>,

    pub empty_variances: Rc<Vec<ty::Variance>>,

    // For marker types, UnsafeCell, and other lang items where
    // variance is hardcoded, records the item-id and the hardcoded
    // variance.
    pub lang_items: Vec<(ast::NodeId, Vec<ty::Variance>)>,

    // Maps from the node id of a type/generic parameter to the
    // corresponding inferred index.
    pub inferred_map: NodeMap<InferredIndex>,

    // Maps from an InferredIndex to the info for that variable.
    pub inferred_infos: Vec<InferredInfo<'a>>,
}

pub struct InferredInfo<'a> {
    pub item_id: ast::NodeId,
    pub index: usize,
    pub param_id: ast::NodeId,
    pub term: VarianceTermPtr<'a>,

    // Initial value to use for this parameter when inferring
    // variance. For most parameters, this is Bivariant. But for lang
    // items and input type parameters on traits, it is different.
    pub initial_variance: ty::Variance,
}

pub fn determine_parameters_to_be_inferred<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                     arena: &'a mut TypedArena<VarianceTerm<'a>>)
                                                     -> TermsContext<'a, 'tcx> {
    let mut terms_cx = TermsContext {
        tcx: tcx,
        arena: arena,
        inferred_map: NodeMap(),
        inferred_infos: Vec::new(),

        lang_items: lang_items(tcx),

        // cache and share the variance struct used for items with
        // no type/region parameters
        empty_variances: Rc::new(vec![]),
    };

    // See README.md for a discussion on dep-graph management.
    tcx.visit_all_item_likes_in_krate(|def_id| ItemVariances::to_dep_node(&def_id), &mut terms_cx);

    terms_cx
}

fn lang_items(tcx: TyCtxt) -> Vec<(ast::NodeId, Vec<ty::Variance>)> {
    let all = vec![
        (tcx.lang_items.phantom_data(), vec![ty::Covariant]),
        (tcx.lang_items.unsafe_cell_type(), vec![ty::Invariant]),

        // Deprecated:
        (tcx.lang_items.covariant_type(), vec![ty::Covariant]),
        (tcx.lang_items.contravariant_type(), vec![ty::Contravariant]),
        (tcx.lang_items.invariant_type(), vec![ty::Invariant]),
        (tcx.lang_items.covariant_lifetime(), vec![ty::Covariant]),
        (tcx.lang_items.contravariant_lifetime(), vec![ty::Contravariant]),
        (tcx.lang_items.invariant_lifetime(), vec![ty::Invariant]),

        ];

    all.into_iter() // iterating over (Option<DefId>, Variance)
       .filter(|&(ref d,_)| d.is_some())
       .map(|(d, v)| (d.unwrap(), v)) // (DefId, Variance)
       .filter_map(|(d, v)| tcx.map.as_local_node_id(d).map(|n| (n, v))) // (NodeId, Variance)
       .collect()
}

impl<'a, 'tcx> TermsContext<'a, 'tcx> {
    fn add_inferreds_for_item(&mut self,
                              item_id: ast::NodeId,
                              has_self: bool,
                              generics: &hir::Generics) {
        //! Add "inferreds" for the generic parameters declared on this
        //! item. This has a lot of annoying parameters because we are
        //! trying to drive this from the AST, rather than the
        //! ty::Generics, so that we can get span info -- but this
        //! means we must accommodate syntactic distinctions.
        //!

        // NB: In the code below for writing the results back into the
        // tcx, we rely on the fact that all inferreds for a particular
        // item are assigned continuous indices.

        let inferreds_on_entry = self.num_inferred();

        if has_self {
            self.add_inferred(item_id, 0, item_id);
        }

        for (i, p) in generics.lifetimes.iter().enumerate() {
            let id = p.lifetime.id;
            let i = has_self as usize + i;
            self.add_inferred(item_id, i, id);
        }

        for (i, p) in generics.ty_params.iter().enumerate() {
            let i = has_self as usize + generics.lifetimes.len() + i;
            self.add_inferred(item_id, i, p.id);
        }

        // If this item has no type or lifetime parameters,
        // then there are no variances to infer, so just
        // insert an empty entry into the variance map.
        // Arguably we could just leave the map empty in this
        // case but it seems cleaner to be able to distinguish
        // "invalid item id" from "item id with no
        // parameters".
        if self.num_inferred() == inferreds_on_entry {
            let item_def_id = self.tcx.map.local_def_id(item_id);
            let newly_added = self.tcx
                .item_variance_map
                .borrow_mut()
                .insert(item_def_id, self.empty_variances.clone())
                .is_none();
            assert!(newly_added);
        }
    }

    fn add_inferred(&mut self, item_id: ast::NodeId, index: usize, param_id: ast::NodeId) {
        let inf_index = InferredIndex(self.inferred_infos.len());
        let term = self.arena.alloc(InferredTerm(inf_index));
        let initial_variance = self.pick_initial_variance(item_id, index);
        self.inferred_infos.push(InferredInfo {
            item_id: item_id,
            index: index,
            param_id: param_id,
            term: term,
            initial_variance: initial_variance,
        });
        let newly_added = self.inferred_map.insert(param_id, inf_index).is_none();
        assert!(newly_added);

        debug!("add_inferred(item_path={}, \
                item_id={}, \
                index={}, \
                param_id={}, \
                inf_index={:?}, \
                initial_variance={:?})",
               self.tcx.item_path_str(self.tcx.map.local_def_id(item_id)),
               item_id,
               index,
               param_id,
               inf_index,
               initial_variance);
    }

    fn pick_initial_variance(&self, item_id: ast::NodeId, index: usize) -> ty::Variance {
        match self.lang_items.iter().find(|&&(n, _)| n == item_id) {
            Some(&(_, ref variances)) => variances[index],
            None => ty::Bivariant,
        }
    }

    pub fn num_inferred(&self) -> usize {
        self.inferred_infos.len()
    }
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for TermsContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        debug!("add_inferreds for item {}",
               self.tcx.map.node_to_string(item.id));

        match item.node {
            hir::ItemEnum(_, ref generics) |
            hir::ItemStruct(_, ref generics) |
            hir::ItemUnion(_, ref generics) => {
                self.add_inferreds_for_item(item.id, false, generics);
            }
            hir::ItemTrait(_, ref generics, ..) => {
                // Note: all inputs for traits are ultimately
                // constrained to be invariant. See `visit_item` in
                // the impl for `ConstraintContext` in `constraints.rs`.
                self.add_inferreds_for_item(item.id, true, generics);
            }

            hir::ItemExternCrate(_) |
            hir::ItemUse(..) |
            hir::ItemDefaultImpl(..) |
            hir::ItemImpl(..) |
            hir::ItemStatic(..) |
            hir::ItemConst(..) |
            hir::ItemFn(..) |
            hir::ItemMod(..) |
            hir::ItemForeignMod(..) |
            hir::ItemTy(..) => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
