// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::OverlapError;

use hir::def_id::DefId;
use ich::{self, StableHashingContext};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};
use rustc_data_structures::graph::{Graph as DataStructuresGraph, NodeIndex, INCOMING};
use traits::{self, ObligationCause, TraitEngine};
use ty::{self, TyCtxt, TypeFoldable};
use ty::fast_reject::{self, SimplifiedType};
use rustc_data_structures::sync::Lrc;
use syntax::ast::{Name, DUMMY_NODE_ID};
use util::captures::Captures;
use util::nodemap::{DefIdMap, FxHashMap};
use super::FulfillmentContext;
use std::collections::HashSet;
use syntax_pos::DUMMY_SP;

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
///   default items amongst other things. In the simple "chain" rule, every impl
///   has at most one parent.
#[derive(RustcEncodable, RustcDecodable)]
pub struct Graph {
    // all impls have one or more parents; the "root" impls have as their parent the def_id
    // of the trait
    parent: DefIdMap<Vec<DefId>>,

    // the "root" impls are found by looking up the trait's def_id.
    children: DefIdMap<Children>,
}

/// Children of a given impl, grouped into blanket/non-blanket varieties as is
/// done in `TraitDef`.
#[derive(RustcEncodable, RustcDecodable)]
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
    nonblanket_impls: FxHashMap<fast_reject::SimplifiedType, Vec<DefId>>,

    /// Blanket impls associated with the trait.
    blanket_impls: Vec<DefId>,
}

/// The result of attempting to insert an impl into a group of children.
enum Inserted {
    /// The impl was inserted as a new child in this group of children.
    BecameNewSibling(Option<OverlapError>),

    /// The impl replaced an existing impl that specializes it.
    Replaced(DefId),

    /// The impl is a specialization of an existing child.
    ShouldRecurseOn(DefId),
}

impl<'a, 'gcx, 'tcx> Children {
    fn new() -> Children {
        Children {
            nonblanket_impls: FxHashMap(),
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
    /// specialization relationships.
    fn insert(&mut self,
              tcx: TyCtxt<'a, 'gcx, 'tcx>,
              impl_def_id: DefId,
              simplified_self: Option<SimplifiedType>)
              -> Result<Vec<Inserted>, OverlapError>
    {
        let mut last_lint = None;

        // Nodes could specialize more than one parent
        // so every impl must visited in order to properly place it
        let mut inserted_results = Vec::new();

        for slot in match simplified_self {
            Some(sty) => self.filtered_mut(sty),
            None => self.iter_mut(),
        } {
            let possible_sibling = *slot;

            let overlap_error = |overlap: traits::coherence::OverlapResult,
                                 used_to_be_allowed: bool| {
                // overlap, but no specialization; error out
                let trait_ref = overlap.impl_header.trait_ref.unwrap();
                let self_ty = trait_ref.self_ty();
                OverlapError {
                    with_impl: possible_sibling,
                    trait_desc: trait_ref.to_string(),
                    // only report the Self type if it has at least
                    // some outer concrete shell; otherwise, it's
                    // not adding much information.
                    self_desc: if self_ty.has_concrete_skeleton() {
                        Some(self_ty.to_string())
                    } else {
                        None
                    },
                    intercrate_ambiguity_causes: overlap.intercrate_ambiguity_causes,
                    used_to_be_allowed: used_to_be_allowed,
                }
            };

            let tcx = tcx.global_tcx();
            let (le, ge, overlap_er) = traits::overlapping_impls(
                tcx,
                possible_sibling,
                impl_def_id,
                traits::IntercrateMode::Issue43355,
                |overlap| {
                    if tcx.impls_are_allowed_to_overlap(impl_def_id, possible_sibling) {
                        return Ok((false, false, None));
                    }

                    let le = tcx.specializes((impl_def_id, possible_sibling));
                    let ge = tcx.specializes((possible_sibling, impl_def_id));

                    if le == ge {
                        Ok((true, true, Some(overlap_error(overlap, false))))
                    } else {
                        Ok((le, ge, None))
                    }
                },
                || Ok((false, false, None)),
            )?;

            if le && !ge {
                debug!("descending as child of TraitRef {:?}",
                       tcx.impl_trait_ref(possible_sibling).unwrap());

                // the impl specializes possible_sibling
                inserted_results.push(Inserted::ShouldRecurseOn(possible_sibling));
            } else if ge && !le {
                debug!("placing as parent of TraitRef {:?}",
                       tcx.impl_trait_ref(possible_sibling).unwrap());

                // possible_sibling specializes the impl
                *slot = impl_def_id;
                inserted_results.push(Inserted::Replaced(possible_sibling));
            } else if ge && le {
                last_lint = overlap_er;
            } else {
                if !tcx.impls_are_allowed_to_overlap(impl_def_id, possible_sibling) {
                    traits::overlapping_impls(
                        tcx,
                        possible_sibling,
                        impl_def_id,
                        traits::IntercrateMode::Fixed,
                        |overlap| last_lint = Some(overlap_error(overlap, true)),
                        || (),
                    );
                }

                // no overlap (error bailed already via ?)
            }
        }

        if inserted_results.len() > 0 {
            return Ok(inserted_results);
        }

        // no overlap with any potential siblings, so add as a new sibling
        debug!("placing as new sibling");
        self.insert_blindly(tcx, impl_def_id);
        Ok(vec![Inserted::BecameNewSibling(last_lint)])
    }

    fn iter_mut(&'a mut self) -> Box<dyn Iterator<Item = &'a mut DefId> + 'a> {
        let nonblanket = self.nonblanket_impls.iter_mut().flat_map(|(_, v)| v.iter_mut());
        Box::new(self.blanket_impls.iter_mut().chain(nonblanket))
    }

    fn filtered_mut(&'a mut self, sty: SimplifiedType)
                    -> Box<dyn Iterator<Item = &'a mut DefId> + 'a> {
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
                  -> Result<Option<Vec<OverlapError>>, OverlapError> {
        assert!(impl_def_id.is_local());

        let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
        let trait_def_id = trait_ref.def_id;

        debug!("insert({:?}): inserting TraitRef {:?} into specialization graph",
               impl_def_id, trait_ref);

        // if the reference itself contains an earlier error (e.g., due to a
        // resolution failure), then we just insert the impl at the top level of
        // the graph and claim that there's no overlap (in order to suppress
        // bogus errors).
        if trait_ref.references_error() {
            debug!("insert: inserting dummy node for erroneous TraitRef {:?}, \
                    impl_def_id={:?}, trait_def_id={:?}",
                   trait_ref, impl_def_id, trait_def_id);

            self.parent_insert(impl_def_id, trait_def_id);
            self.children.entry(trait_def_id).or_insert(Children::new())
                .insert_blindly(tcx, impl_def_id);
            return Ok(None);
        }

        let parent = trait_def_id;
        let simplified = fast_reject::simplify_type(tcx, trait_ref.self_ty(), false);

        let mut last_lints = vec![];
        // Recusively descend the specialization tree, where `parent` is the current parent node
        self.recursive_insert(parent, tcx, impl_def_id, simplified, &mut last_lints);
        Ok(Some(last_lints))
    }

    fn recursive_insert(
        &mut self,
        parent: DefId,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        impl_def_id: DefId,
        simplified: Option<SimplifiedType>,
        last_lints: &mut Vec<OverlapError>,
    ) {
        use self::Inserted::*;
        match self.children
            .entry(parent)
            .or_insert(Children::new())
            .insert(tcx, impl_def_id, simplified)
        {
            Ok(insert_results) => for insert_result in insert_results {
                match insert_result {
                    BecameNewSibling(opt_lint) => {
                        if opt_lint.is_some() {
                            last_lints.push(opt_lint.unwrap());
                        }
                        self.parent_insert(impl_def_id, parent);
                    }
                    Replaced(new_child) => {
                        self.parent_insert(new_child, impl_def_id);
                        let mut new_children = Children::new();
                        new_children.insert_blindly(tcx, new_child);
                        self.children.insert(impl_def_id, new_children);
                        self.parent_insert(impl_def_id, parent);
                    }
                    ShouldRecurseOn(new_parent) => {
                        self.recursive_insert(new_parent, tcx, impl_def_id, simplified, last_lints);
                    }
                }
            },
            _ => {}
        }
    }

    fn parent_insert(&mut self, key: DefId, value: DefId) -> Option<DefId> {
        if self.parent.contains_key(&key) {
            let mut impl_vec = self.parent.get(&key).unwrap().clone();
            impl_vec.push(value);
            self.parent.insert(key, impl_vec);
            Some(value)
        } else {
            if self.parent.insert(key, vec![value]).is_some() {
                Some(value)
            } else {
                None
            }
        }
    }

    /// Insert cached metadata mapping from a child impl back to its parent.
    pub fn record_impl_from_cstore(&mut self,
                                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                   parent: DefId,
                                   child: DefId) {
        if self.parent_insert(child, parent).is_some() {
            bug!(
                "When recording an impl from the crate store, information about its parent \
                 was already present."
            );
        }

        self.children.entry(parent).or_insert(Children::new()).insert_blindly(tcx, child);
    }

    /// Returns the def-id of the parent impl(s) for a given impl.
    /// An impl A has a parent impl B if A matches a strict subset of the types that B matches.
    pub fn parents(&self, child: DefId) -> Vec<DefId> {
        self.parent.get(&child).unwrap().clone()
    }

    pub fn build_graph(&self) -> (DataStructuresGraph<DefId, String>, DefIdMap<NodeIndex>) {
        let mut sg_graph: DataStructuresGraph<DefId, String> = DataStructuresGraph::new();
        let mut nodes_idx = Default::default();
        for (key, val) in self.parent.iter() {
            let idx = self.node_idx(&mut sg_graph, &mut nodes_idx, *key);
            for parent in val {
                let pidx = self.node_idx(&mut sg_graph, &mut nodes_idx, *parent);
                sg_graph.add_edge(idx, pidx, format!("fromt {:?} to {:?}", key, parent));
                debug!("from {:?} to {:?}", key, parent);
            }
        }

        (sg_graph, nodes_idx)
    }

    /// Return true if impl1 and impl2 are allowed to overlap:
    /// They have an intersection impl
    pub fn check_overlap(
        &self,
        sg_graph: &DataStructuresGraph<DefId, String>,
        nodes_idx: &DefIdMap<NodeIndex>,
        impl1: DefId,
        impl2: DefId,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ) -> bool {

        let impl1_idx = *nodes_idx.get(&impl1).unwrap();
        let impl2_idx = *nodes_idx.get(&impl2).unwrap();

        // two impls are allowed to overlap if they have a mutual postdominator in the graph.
        // That postdominator is the specializing impl.
        let impl1_incoming_nodes = sg_graph.nodes_in_postorder(INCOMING, impl1_idx);
        let impl2_incoming_nodes = sg_graph.nodes_in_postorder(INCOMING, impl2_idx);

        if impl1_incoming_nodes[0] == impl2_incoming_nodes[0] {

            // a mutual postdominator has been found but the intersection impls
            // could not be complete.
            // The query below tries to answer to the following
            // Does the exist a set of types such that
            //  - impl A applies and impl B applies
            //  - but impl C does not apply
            let mut result = true;
            let int_impl = sg_graph.node_data(impl1_incoming_nodes[0]);
            tcx.infer_ctxt().enter(|infcx| {

                let param_env1 = tcx.param_env(impl1);
                let param_env2 = tcx.param_env(impl2);

                let mut combined_param_envs_vec =
                    param_env1
                        .caller_bounds
                        .iter()
                        .chain(param_env2.caller_bounds.iter())
                        .map(|p| *p)
                        .collect::<Vec<_>>();

                let combined_param_envs: HashSet<_> =
                    combined_param_envs_vec
                        .drain(..)
                        .collect();

                // Combine the param envs of the overlapping impls into a single param env.
                let param_env = ty::ParamEnv::new(
                    tcx.intern_predicates(
                        combined_param_envs
                            .into_iter()
                            .collect::<Vec<_>>()
                            .as_slice()
                    ),
                    param_env1.reveal
                );

                let predicates = tcx.predicates_of(*int_impl);
                let ty = tcx.type_of(*int_impl);

                let mut fulfill_cx = FulfillmentContext::new();
                for predicate in predicates.predicates {
                    if let ty::Predicate::Trait(trait_predicate) = predicate {
                        fulfill_cx.register_bound(
                            &infcx,
                            param_env,
                            ty,
                            trait_predicate.skip_binder().trait_ref.def_id,
                            ObligationCause::misc(DUMMY_SP, DUMMY_NODE_ID),
                        );
                    }
                }

                fulfill_cx.select_all_or_error(&infcx).unwrap_or_else(|_| {
                    result = false;
                });
            });

            result
        } else {
            false
        }
    }

    fn node_idx(
        &self,
        sg_graph: &mut DataStructuresGraph<DefId, String>,
        nodes_idx: &mut DefIdMap<NodeIndex>,
        node: DefId,
    ) -> NodeIndex {
        if nodes_idx.get(&node).is_some() {
            *nodes_idx.get(&node).unwrap()
        } else {
            let idx = sg_graph.add_node(node);
            nodes_idx.insert(node, idx);
            idx
        }
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
    pub fn items(
        &self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
    ) -> impl Iterator<Item = ty::AssociatedItem> + 'a {
        tcx.associated_items(self.def_id())
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            Node::Impl(did) => did,
            Node::Trait(did) => did,
        }
    }
}

pub struct Ancestors {
    trait_def_id: DefId,
    specialization_graph: Lrc<Graph>,
    current_source: Option<Vec<Node>>,
}

impl Iterator for Ancestors {
    type Item = Node;
    fn next(&mut self) -> Option<Node> {
        // Visit and return the graph nodes from bottom to top
        // When multiple parents are found return each one of them
        // prior to move up in the graph
        let cur = self.current_source.take();
        match cur {
            Some(mut cur_vec) => {
                let next_value = cur_vec[0];
                if let Node::Impl(cur_impl) = next_value {
                    let parents = self.specialization_graph.parents(cur_impl);

                    let mut ncur = vec![];
                    ncur.append(&mut cur_vec[1..].to_vec());
                    for parent in parents {
                        let node = if parent == self.trait_def_id {
                            Node::Trait(parent)
                        } else {
                            Node::Impl(parent)
                        };

                        if ncur.iter().find(|n| n.def_id() == node.def_id()).is_none() {
                            ncur.push(node);
                        }
                    }

                    self.current_source = Some(ncur);
                }

                Some(next_value)
            }
            None => None,
        }
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

impl<'a, 'gcx, 'tcx> Ancestors {
    /// Search the items from the given ancestors, returning each definition
    /// with the given name and the given kind.
    #[inline] // FIXME(#35870) Avoid closures being unexported due to impl Trait.
    pub fn defs(
        self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        trait_item_name: Name,
        trait_item_kind: ty::AssociatedKind,
        trait_def_id: DefId,
    ) -> impl Iterator<Item = NodeItem<ty::AssociatedItem>> + Captures<'gcx> + Captures<'tcx> + 'a {
        self.flat_map(move |node| {
            node.items(tcx).filter(move |impl_item| {
                impl_item.kind == trait_item_kind &&
                tcx.hygienic_eq(impl_item.name, trait_item_name, trait_def_id)
            }).map(move |item| NodeItem { node: node, item: item })
        })
    }
}

/// Walk up the specialization ancestors of a given impl, starting with that
/// impl itself.
pub fn ancestors(tcx: TyCtxt,
                 trait_def_id: DefId,
                 start_from_impl: DefId)
                 -> Ancestors {
    let specialization_graph = tcx.specialization_graph_of(trait_def_id);
    Ancestors {
        trait_def_id,
        specialization_graph,
        current_source: Some(vec![Node::Impl(start_from_impl)]),
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Children {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let Children {
            ref nonblanket_impls,
            ref blanket_impls,
        } = *self;

        ich::hash_stable_trait_impls(hcx, hasher, blanket_impls, nonblanket_impls);
    }
}

impl_stable_hash_for!(struct self::Graph {
    parent,
    children
});
