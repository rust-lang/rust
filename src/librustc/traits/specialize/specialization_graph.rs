// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

use super::{OverlapError, specializes};

use hir::def_id::DefId;
use traits::{self, Reveal};
use ty::{self, TyCtxt, ImplOrTraitItem, TraitDef, TypeFoldable};
use ty::fast_reject::{self, SimplifiedType};
use syntax::ast::Name;
use util::nodemap::{DefIdMap, FnvHashMap};

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
    children: DefIdMap<Children>,
}

/// Children of a given impl, grouped into blanket/non-blanket varieties as is
/// done in `TraitDef`.
struct Children {
    // Impls of a trait (or specializations of a given impl). To allow for
    // quicker lookup, the impls are indexed by a simplified version of their
    // `Self` type: impls with a simplifiable `Self` are stored in
    // `nonblanket_impls` keyed by it, while all other impls are stored in
    // `blanket_impls`.
    //
    // A similar division is used within `TraitDef`, but the lists there collect
    // together *all* the impls for a trait, and are populated prior to building
    // the specialization graph.

    /// Impls of the trait.
    nonblanket_impls: FnvHashMap<fast_reject::SimplifiedType, Vec<DefId>>,

    /// Blanket impls associated with the trait.
    blanket_impls: Vec<DefId>,
}

/// The result of attempting to insert an impl into a group of children.
enum Inserted {
    /// The impl was inserted as a new child in this group of children.
    BecameNewSibling,

    /// The impl replaced an existing impl that specializes it.
    Replaced(DefId),

    /// The impl is a specialization of an existing child.
    ShouldRecurseOn(DefId),
}

impl<'a, 'gcx, 'tcx> Children {
    fn new() -> Children {
        Children {
            nonblanket_impls: FnvHashMap(),
            blanket_impls: vec![],
        }
    }

    /// Insert an impl into this set of children without comparing to any existing impls
    fn insert_blindly(&mut self,
                      tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      impl_def_id: DefId) {
        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        if let Some(sty) = fast_reject::simplify_type(tcx, trait_ref.self_ty(), false) {
            self.nonblanket_impls.entry(sty).or_insert(vec![]).push(impl_def_id)
        } else {
            self.blanket_impls.push(impl_def_id)
        }
    }

    /// Attempt to insert an impl into this set of children, while comparing for
    /// specialiation relationships.
    fn insert(&mut self,
              tcx: TyCtxt<'a, 'gcx, 'tcx>,
              impl_def_id: DefId,
              simplified_self: Option<SimplifiedType>)
              -> Result<Inserted, OverlapError>
    {
        for slot in match simplified_self {
            Some(sty) => self.filtered_mut(sty),
            None => self.iter_mut(),
        } {
            let possible_sibling = *slot;

            let tcx = tcx.global_tcx();
            let (le, ge) = tcx.infer_ctxt(None, None, Reveal::ExactMatch).enter(|infcx| {
                let overlap = traits::overlapping_impls(&infcx,
                                                        possible_sibling,
                                                        impl_def_id);
                if let Some(impl_header) = overlap {
                    let le = specializes(tcx, impl_def_id, possible_sibling);
                    let ge = specializes(tcx, possible_sibling, impl_def_id);

                    if le == ge {
                        // overlap, but no specialization; error out
                        let trait_ref = impl_header.trait_ref.unwrap();
                        let self_ty = trait_ref.self_ty();
                        Err(OverlapError {
                            with_impl: possible_sibling,
                            trait_desc: trait_ref.to_string(),
                            // only report the Self type if it has at least
                            // some outer concrete shell; otherwise, it's
                            // not adding much information.
                            self_desc: if self_ty.has_concrete_skeleton() {
                                Some(self_ty.to_string())
                            } else {
                                None
                            }
                        })
                    } else {
                        Ok((le, ge))
                    }
                } else {
                    Ok((false, false))
                }
            })?;

            if le && !ge {
                debug!("descending as child of TraitRef {:?}",
                       tcx.impl_trait_ref(possible_sibling).unwrap());

                // the impl specializes possible_sibling
                return Ok(Inserted::ShouldRecurseOn(possible_sibling));
            } else if ge && !le {
                debug!("placing as parent of TraitRef {:?}",
                       tcx.impl_trait_ref(possible_sibling).unwrap());

                    // possible_sibling specializes the impl
                    *slot = impl_def_id;
                return Ok(Inserted::Replaced(possible_sibling));
            } else {
                // no overlap (error bailed already via ?)
            }
        }

        // no overlap with any potential siblings, so add as a new sibling
        debug!("placing as new sibling");
        self.insert_blindly(tcx, impl_def_id);
        Ok(Inserted::BecameNewSibling)
    }

    fn iter_mut(&'a mut self) -> Box<Iterator<Item = &'a mut DefId> + 'a> {
        let nonblanket = self.nonblanket_impls.iter_mut().flat_map(|(_, v)| v.iter_mut());
        Box::new(self.blanket_impls.iter_mut().chain(nonblanket))
    }

    fn filtered_mut(&'a mut self, sty: SimplifiedType)
                    -> Box<Iterator<Item = &'a mut DefId> + 'a> {
        let nonblanket = self.nonblanket_impls.entry(sty).or_insert(vec![]).iter_mut();
        Box::new(self.blanket_impls.iter_mut().chain(nonblanket))
    }
}

impl<'a, 'gcx, 'tcx> Graph {
    pub fn new() -> Graph {
        Graph {
            parent: Default::default(),
            children: Default::default(),
        }
    }

    /// Insert a local impl into the specialization graph. If an existing impl
    /// conflicts with it (has overlap, but neither specializes the other),
    /// information about the area of overlap is returned in the `Err`.
    pub fn insert(&mut self,
                  tcx: TyCtxt<'a, 'gcx, 'tcx>,
                  impl_def_id: DefId)
                  -> Result<(), OverlapError> {
        assert!(impl_def_id.is_local());

        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let trait_def_id = trait_ref.def_id;

        debug!("insert({:?}): inserting TraitRef {:?} into specialization graph",
               impl_def_id, trait_ref);

        // if the reference itself contains an earlier error (e.g., due to a
        // resolution failure), then we just insert the impl at the top level of
        // the graph and claim that there's no overlap (in order to supress
        // bogus errors).
        if trait_ref.references_error() {
            debug!("insert: inserting dummy node for erroneous TraitRef {:?}, \
                    impl_def_id={:?}, trait_def_id={:?}",
                   trait_ref, impl_def_id, trait_def_id);

            self.parent.insert(impl_def_id, trait_def_id);
            self.children.entry(trait_def_id).or_insert(Children::new())
                .insert_blindly(tcx, impl_def_id);
            return Ok(());
        }

        let mut parent = trait_def_id;
        let simplified = fast_reject::simplify_type(tcx, trait_ref.self_ty(), false);

        // Descend the specialization tree, where `parent` is the current parent node
        loop {
            use self::Inserted::*;

            let insert_result = self.children.entry(parent).or_insert(Children::new())
                .insert(tcx, impl_def_id, simplified)?;

            match insert_result {
                BecameNewSibling => {
                    break;
                }
                Replaced(new_child) => {
                    self.parent.insert(new_child, impl_def_id);
                    let mut new_children = Children::new();
                    new_children.insert_blindly(tcx, new_child);
                    self.children.insert(impl_def_id, new_children);
                    break;
                }
                ShouldRecurseOn(new_parent) => {
                    parent = new_parent;
                }
            }
        }

        self.parent.insert(impl_def_id, parent);
        Ok(())
    }

    /// Insert cached metadata mapping from a child impl back to its parent.
    pub fn record_impl_from_cstore(&mut self,
                                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                   parent: DefId,
                                   child: DefId) {
        if self.parent.insert(child, parent).is_some() {
            bug!("When recording an impl from the crate store, information about its parent \
                  was already present.");
        }

        self.children.entry(parent).or_insert(Children::new()).insert_blindly(tcx, child);
    }

    /// The parent of a given impl, which is the def id of the trait when the
    /// impl is a "specialization root".
    pub fn parent(&self, child: DefId) -> DefId {
        *self.parent.get(&child).unwrap()
    }
}

/// A node in the specialization graph is either an impl or a trait
/// definition; either can serve as a source of item definitions.
/// There is always exactly one trait definition node: the root.
#[derive(Debug, Copy, Clone)]
pub enum Node {
    Impl(DefId),
    Trait(DefId),
}

impl<'a, 'gcx, 'tcx> Node {
    pub fn is_from_trait(&self) -> bool {
        match *self {
            Node::Trait(..) => true,
            _ => false,
        }
    }

    /// Iterate over the items defined directly by the given (impl or trait) node.
    pub fn items(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> NodeItems<'a, 'gcx> {
        NodeItems {
            tcx: tcx.global_tcx(),
            items: tcx.impl_or_trait_items(self.def_id()),
            idx: 0,
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
pub struct NodeItems<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    items: Rc<Vec<DefId>>,
    idx: usize
}

impl<'a, 'tcx> Iterator for NodeItems<'a, 'tcx> {
    type Item = ImplOrTraitItem<'tcx>;
    fn next(&mut self) -> Option<ImplOrTraitItem<'tcx>> {
        if self.idx < self.items.len() {
            let item_def_id = self.items[self.idx];
            let items_table = self.tcx.impl_or_trait_items.borrow();
            let item = items_table[&item_def_id].clone();
            self.idx += 1;
            Some(item)
        } else {
            None
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

impl<'a, 'gcx, 'tcx> Ancestors<'a, 'tcx> {
    /// Search the items from the given ancestors, returning each type definition
    /// with the given name.
    pub fn type_defs(self, tcx: TyCtxt<'a, 'gcx, 'tcx>, name: Name) -> TypeDefs<'a, 'gcx> {
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

    /// Search the items from the given ancestors, returning each fn definition
    /// with the given name.
    pub fn fn_defs(self, tcx: TyCtxt<'a, 'gcx, 'tcx>, name: Name) -> FnDefs<'a, 'gcx> {
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

    /// Search the items from the given ancestors, returning each const
    /// definition with the given name.
    pub fn const_defs(self, tcx: TyCtxt<'a, 'gcx, 'tcx>, name: Name) -> ConstDefs<'a, 'gcx> {
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
