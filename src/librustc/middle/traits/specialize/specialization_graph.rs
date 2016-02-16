// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell;
use std::rc::Rc;

use super::{Overlap, specializes};

use middle::cstore::CrateStore;
use middle::def_id::DefId;
use middle::infer;
use middle::traits;
use middle::ty::{self, ImplOrTraitItem, TraitDef, TypeFoldable};
use syntax::ast::Name;
use util::nodemap::DefIdMap;

/// A per-trait graph of impls in specialization order. At the moment, this
/// graph forms a tree rooted with the trait itself, with all other nodes
/// representing impls, and parent-child relationships representing
/// specializations.
///
/// The graph provides two key services:
///
/// - Construction, which implicitly checks for overlapping impls (i.e., impls
///   that overlap but where neither specializes the other -- an artifact of the
///   simple "chain" rule.
///
/// - Parent extraction. In particular, the graph can give you the *immediate*
///   parents of a given specializing impl, which is needed for extracting
///   default items amongst other thigns. In the simple "chain" rule, every impl
///   has at most one parent.
pub struct Graph {
    // all impls have a parent; the "root" impls have as their parent the def_id
    // of the trait
    parent: DefIdMap<DefId>,

    // the "root" impls are found by looking up the trait's def_id.
    children: DefIdMap<Vec<DefId>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            parent: Default::default(),
            children: Default::default(),
        }
    }

    /// Insert a local impl into the specialization graph. If an existing impl
    /// conflicts with it (has overlap, but neither specializes the other),
    /// information about the area of overlap is returned in the `Err`.
    pub fn insert<'a, 'tcx>(&mut self,
                            tcx: &'a ty::ctxt<'tcx>,
                            impl_def_id: DefId)
                            -> Result<(), Overlap<'a, 'tcx>> {
        assert!(impl_def_id.is_local());

        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let trait_def_id = trait_ref.def_id;

        // if the reference itself contains an earlier error (e.g., due to a
        // resolution failure), then we just insert the impl at the top level of
        // the graph and claim that there's no overlap (in order to supress
        // bogus errors).
        if trait_ref.references_error() {
            debug!("Inserting dummy node for erroneous TraitRef {:?}, \
                    impl_def_id={:?}, trait_def_id={:?}",
                   trait_ref, impl_def_id, trait_def_id);

            self.parent.insert(impl_def_id, trait_def_id);
            self.children.entry(trait_def_id).or_insert(vec![]).push(impl_def_id);
            return Ok(());
        }

        let mut parent = trait_def_id;

        // Ugly hack around borrowck limitations. Assigned only in the case
        // where we bump downward an existing node in the graph.
        let child_to_insert;

        'descend: loop {
            let mut possible_siblings = self.children.entry(parent).or_insert(vec![]);

            for slot in possible_siblings.iter_mut() {
                let possible_sibling = *slot;

                let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, None);
                let overlap = traits::overlapping_impls(&infcx, possible_sibling, impl_def_id);

                if let Some(trait_ref) = overlap {
                    let le = specializes(tcx, impl_def_id, possible_sibling);
                    let ge = specializes(tcx, possible_sibling, impl_def_id);

                    if le && !ge {
                        // the impl specializes possible_sibling
                        parent = possible_sibling;
                        continue 'descend;
                    } else if ge && !le {
                        // possible_sibling specializes the impl
                        *slot = impl_def_id;
                        self.parent.insert(possible_sibling, impl_def_id);
                        // we have to defer the insertion, because we can't
                        // relinquish the borrow of `self.children`
                        child_to_insert = possible_sibling;
                        break 'descend;
                    } else {
                        // overlap, but no specialization; error out
                        return Err(Overlap {
                            with_impl: possible_sibling,
                            on_trait_ref: trait_ref,
                            in_context: infcx,
                        });
                    }
                }
            }

            // no overlap with any potential siblings, so add as a new sibling
            self.parent.insert(impl_def_id, parent);
            possible_siblings.push(impl_def_id);
            return Ok(());
        }

        self.children.insert(impl_def_id, vec![child_to_insert]);
        Ok(())
    }

    /// Insert cached metadata mapping from a child impl back to its parent.
    pub fn record_impl_from_cstore(&mut self, parent: DefId, child: DefId) {
        if self.parent.insert(child, parent).is_some() {
            panic!("When recording an impl from the crate store, information about its parent \
                    was already present.");
        }

        self.children.entry(parent).or_insert(vec![]).push(child);
    }

    /// The parent of a given impl, which is the def id of the trait when the
    /// impl is a "specialization root".
    pub fn parent(&self, child: DefId) -> DefId {
        *self.parent.get(&child).unwrap()
    }
}

#[derive(Debug, Copy, Clone)]
/// A node in the specialization graph is either an impl or a trait
/// definition; either can serve as a source of item definitions.
/// There is always exactly one trait definition node: the root.
pub enum Node {
    Impl(DefId),
    Trait(DefId),
}

impl Node {
    pub fn is_from_trait(&self) -> bool {
        match *self {
            Node::Trait(..) => true,
            _ => false,
        }
    }

    /// Iterate over the items defined directly by the given (impl or trait) node.
    pub fn items<'a, 'tcx>(&self, tcx: &'a ty::ctxt<'tcx>) -> NodeItems<'a, 'tcx> {
        match *self {
            Node::Impl(impl_def_id) => {
                NodeItems::Impl {
                    tcx: tcx,
                    items: cell::Ref::map(tcx.impl_items.borrow(),
                                          |impl_items| &impl_items[&impl_def_id]),
                    idx: 0,
                }
            }
            Node::Trait(trait_def_id) => {
                NodeItems::Trait {
                    items: tcx.trait_items(trait_def_id).clone(),
                    idx: 0,
                }
            }
        }
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            Node::Impl(did) => did,
            Node::Trait(did) => did,
        }
    }
}

/// An iterator over the items defined within a trait or impl.
pub enum NodeItems<'a, 'tcx: 'a> {
    Impl {
        tcx: &'a ty::ctxt<'tcx>,
        items: cell::Ref<'a, Vec<ty::ImplOrTraitItemId>>,
        idx: usize,
    },
    Trait {
        items: Rc<Vec<ImplOrTraitItem<'tcx>>>,
        idx: usize,
    },
}

impl<'a, 'tcx> Iterator for NodeItems<'a, 'tcx> {
    type Item = ImplOrTraitItem<'tcx>;
    fn next(&mut self) -> Option<ImplOrTraitItem<'tcx>> {
        match *self {
            NodeItems::Impl { tcx, ref items, ref mut idx } => {
                let items_table = tcx.impl_or_trait_items.borrow();
                if *idx < items.len() {
                    let item_def_id = items[*idx].def_id();
                    let item = items_table[&item_def_id].clone();
                    *idx += 1;
                    Some(item)
                } else {
                    None
                }
            }
            NodeItems::Trait { ref items, ref mut idx } => {
                if *idx < items.len() {
                    let item = items[*idx].clone();
                    *idx += 1;
                    Some(item)
                } else {
                    None
                }
            }
        }
    }
}

pub struct Ancestors<'a, 'tcx: 'a> {
    trait_def: &'a TraitDef<'tcx>,
    current_source: Option<Node>,
}

impl<'a, 'tcx> Iterator for Ancestors<'a, 'tcx> {
    type Item = Node;
    fn next(&mut self) -> Option<Node> {
        let cur = self.current_source.take();
        if let Some(Node::Impl(cur_impl)) = cur {
            let parent = self.trait_def.specialization_graph.borrow().parent(cur_impl);
            if parent == self.trait_def.def_id() {
                self.current_source = Some(Node::Trait(parent));
            } else {
                self.current_source = Some(Node::Impl(parent));
            }
        }
        cur
    }
}

pub struct NodeItem<T> {
    pub node: Node,
    pub item: T,
}

impl<T> NodeItem<T> {
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> NodeItem<U> {
        NodeItem {
            node: self.node,
            item: f(self.item),
        }
    }
}

pub struct TypeDefs<'a, 'tcx: 'a> {
    // generally only invoked once or twice, so the box doesn't hurt
    iter: Box<Iterator<Item = NodeItem<Rc<ty::AssociatedType<'tcx>>>> + 'a>,
}

impl<'a, 'tcx> Iterator for TypeDefs<'a, 'tcx> {
    type Item = NodeItem<Rc<ty::AssociatedType<'tcx>>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct FnDefs<'a, 'tcx: 'a> {
    // generally only invoked once or twice, so the box doesn't hurt
    iter: Box<Iterator<Item = NodeItem<Rc<ty::Method<'tcx>>>> + 'a>,
}

impl<'a, 'tcx> Iterator for FnDefs<'a, 'tcx> {
    type Item = NodeItem<Rc<ty::Method<'tcx>>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct ConstDefs<'a, 'tcx: 'a> {
    // generally only invoked once or twice, so the box doesn't hurt
    iter: Box<Iterator<Item = NodeItem<Rc<ty::AssociatedConst<'tcx>>>> + 'a>,
}

impl<'a, 'tcx> Iterator for ConstDefs<'a, 'tcx> {
    type Item = NodeItem<Rc<ty::AssociatedConst<'tcx>>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, 'tcx> Ancestors<'a, 'tcx> {
    /// Seach the items from the given ancestors, returning each type definition
    /// with the given name.
    pub fn type_defs(self, tcx: &'a ty::ctxt<'tcx>, name: Name) -> TypeDefs<'a, 'tcx> {
        let iter = self.flat_map(move |node| {
            node.items(tcx)
                .filter_map(move |item| {
                    if let ty::TypeTraitItem(assoc_ty) = item {
                        if assoc_ty.name == name {
                            return Some(NodeItem {
                                node: node,
                                item: assoc_ty,
                            });
                        }
                    }
                    None
                })

        });
        TypeDefs { iter: Box::new(iter) }
    }

    /// Seach the items from the given ancestors, returning each fn definition
    /// with the given name.
    pub fn fn_defs(self, tcx: &'a ty::ctxt<'tcx>, name: Name) -> FnDefs<'a, 'tcx> {
        let iter = self.flat_map(move |node| {
            node.items(tcx)
                .filter_map(move |item| {
                    if let ty::MethodTraitItem(method) = item {
                        if method.name == name {
                            return Some(NodeItem {
                                node: node,
                                item: method,
                            });
                        }
                    }
                    None
                })

        });
        FnDefs { iter: Box::new(iter) }
    }

    /// Seach the items from the given ancestors, returning each const
    /// definition with the given name.
    pub fn const_defs(self, tcx: &'a ty::ctxt<'tcx>, name: Name) -> ConstDefs<'a, 'tcx> {
        let iter = self.flat_map(move |node| {
            node.items(tcx)
                .filter_map(move |item| {
                    if let ty::ConstTraitItem(konst) = item {
                        if konst.name == name {
                            return Some(NodeItem {
                                node: node,
                                item: konst,
                            });
                        }
                    }
                    None
                })

        });
        ConstDefs { iter: Box::new(iter) }
    }
}

/// Walk up the specialization ancestors of a given impl, starting with that
/// impl itself.
pub fn ancestors<'a, 'tcx>(trait_def: &'a TraitDef<'tcx>,
                           start_from_impl: DefId)
                           -> Ancestors<'a, 'tcx> {
    Ancestors {
        trait_def: trait_def,
        current_source: Some(Node::Impl(start_from_impl)),
    }
}
