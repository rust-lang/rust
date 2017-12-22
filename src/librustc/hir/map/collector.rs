// Copyright 2015-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;

use dep_graph::{DepGraph, DepKind, DepNodeIndex};
use hir::intravisit::{Visitor, NestedVisitorMap};
use hir::svh::Svh;
use middle::cstore::CrateStore;
use session::CrateDisambiguator;
use std::iter::repeat;
use syntax::ast::{NodeId, CRATE_NODE_ID};
use syntax_pos::Span;

use ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};

/// A Visitor that walks over the HIR and collects Nodes into a HIR map
pub(super) struct NodeCollector<'a, 'hir> {
    /// The crate
    krate: &'hir Crate,
    /// The node map
    map: Vec<MapEntry<'hir>>,
    /// The parent of this node
    parent_node: NodeId,

    // These fields keep track of the currently relevant DepNodes during
    // the visitor's traversal.
    current_dep_node_owner: DefIndex,
    current_signature_dep_index: DepNodeIndex,
    current_full_dep_index: DepNodeIndex,
    currently_in_body: bool,

    dep_graph: &'a DepGraph,
    definitions: &'a definitions::Definitions,

    hcx: StableHashingContext<'a>,

    // We are collecting DepNode::HirBody hashes here so we can compute the
    // crate hash from then later on.
    hir_body_nodes: Vec<(DefPathHash, DepNodeIndex)>,
}

impl<'a, 'hir> NodeCollector<'a, 'hir> {
    pub(super) fn root(krate: &'hir Crate,
                       dep_graph: &'a DepGraph,
                       definitions: &'a definitions::Definitions,
                       hcx: StableHashingContext<'a>)
                -> NodeCollector<'a, 'hir> {
        let root_mod_def_path_hash = definitions.def_path_hash(CRATE_DEF_INDEX);

        // Allocate DepNodes for the root module
        let (root_mod_sig_dep_index, root_mod_full_dep_index);
        {
            let Crate {
                ref module,
                // Crate attributes are not copied over to the root `Mod`, so hash
                // them explicitly here.
                ref attrs,
                span,

                // These fields are handled separately:
                exported_macros: _,
                items: _,
                trait_items: _,
                impl_items: _,
                bodies: _,
                trait_impls: _,
                trait_auto_impl: _,
                body_ids: _,
            } = *krate;

            root_mod_sig_dep_index = dep_graph.with_task(
                root_mod_def_path_hash.to_dep_node(DepKind::Hir),
                &hcx,
                HirItemLike { item_like: (module, attrs, span), hash_bodies: false },
                identity_fn
            ).1;
            root_mod_full_dep_index = dep_graph.with_task(
                root_mod_def_path_hash.to_dep_node(DepKind::HirBody),
                &hcx,
                HirItemLike { item_like: (module, attrs, span), hash_bodies: true },
                identity_fn
            ).1;
        }

        {
            dep_graph.with_task(
                DepNode::new_no_params(DepKind::AllLocalTraitImpls),
                &hcx,
                &krate.trait_impls,
                identity_fn
            );
        }

        let hir_body_nodes = vec![(root_mod_def_path_hash, root_mod_full_dep_index)];

        let mut collector = NodeCollector {
            krate,
            map: vec![],
            parent_node: CRATE_NODE_ID,
            current_signature_dep_index: root_mod_sig_dep_index,
            current_full_dep_index: root_mod_full_dep_index,
            current_dep_node_owner: CRATE_DEF_INDEX,
            currently_in_body: false,
            dep_graph,
            definitions,
            hcx,
            hir_body_nodes,
        };
        collector.insert_entry(CRATE_NODE_ID, RootCrate(root_mod_sig_dep_index));

        collector
    }

    pub(super) fn finalize_and_compute_crate_hash(self,
                                                  crate_disambiguator: CrateDisambiguator,
                                                  cstore: &CrateStore,
                                                  commandline_args_hash: u64)
                                                  -> (Vec<MapEntry<'hir>>, Svh) {
        let mut node_hashes: Vec<_> = self
            .hir_body_nodes
            .iter()
            .map(|&(def_path_hash, dep_node_index)| {
                (def_path_hash, self.dep_graph.fingerprint_of(dep_node_index))
            })
            .collect();

        node_hashes.sort_unstable_by(|&(ref d1, _), &(ref d2, _)| d1.cmp(d2));

        let mut upstream_crates: Vec<_> = cstore.crates_untracked().iter().map(|&cnum| {
            let name = cstore.crate_name_untracked(cnum).as_str();
            let disambiguator = cstore.crate_disambiguator_untracked(cnum)
                                      .to_fingerprint();
            let hash = cstore.crate_hash_untracked(cnum);
            (name, disambiguator, hash)
        }).collect();

        upstream_crates.sort_unstable_by(|&(name1, dis1, _), &(name2, dis2, _)| {
            (name1, dis1).cmp(&(name2, dis2))
        });

        let (_, crate_dep_node_index) = self
            .dep_graph
            .with_task(DepNode::new_no_params(DepKind::Krate),
                       &self.hcx,
                       ((node_hashes, upstream_crates),
                        (commandline_args_hash,
                         crate_disambiguator.to_fingerprint())),
                       identity_fn);

        let svh = Svh::new(self.dep_graph
                               .fingerprint_of(crate_dep_node_index)
                               .to_smaller_hash());
        (self.map, svh)
    }

    fn insert_entry(&mut self, id: NodeId, entry: MapEntry<'hir>) {
        debug!("hir_map: {:?} => {:?}", id, entry);
        let len = self.map.len();
        if id.as_usize() >= len {
            self.map.extend(repeat(NotPresent).take(id.as_usize() - len + 1));
        }
        self.map[id.as_usize()] = entry;
    }

    fn insert(&mut self, id: NodeId, node: Node<'hir>) {
        let parent = self.parent_node;
        let dep_node_index = if self.currently_in_body {
            self.current_full_dep_index
        } else {
            self.current_signature_dep_index
        };

        let entry = match node {
            NodeItem(n) => EntryItem(parent, dep_node_index, n),
            NodeForeignItem(n) => EntryForeignItem(parent, dep_node_index, n),
            NodeTraitItem(n) => EntryTraitItem(parent, dep_node_index, n),
            NodeImplItem(n) => EntryImplItem(parent, dep_node_index, n),
            NodeVariant(n) => EntryVariant(parent, dep_node_index, n),
            NodeField(n) => EntryField(parent, dep_node_index, n),
            NodeExpr(n) => EntryExpr(parent, dep_node_index, n),
            NodeStmt(n) => EntryStmt(parent, dep_node_index, n),
            NodeTy(n) => EntryTy(parent, dep_node_index, n),
            NodeTraitRef(n) => EntryTraitRef(parent, dep_node_index, n),
            NodeBinding(n) => EntryBinding(parent, dep_node_index, n),
            NodePat(n) => EntryPat(parent, dep_node_index, n),
            NodeBlock(n) => EntryBlock(parent, dep_node_index, n),
            NodeStructCtor(n) => EntryStructCtor(parent, dep_node_index, n),
            NodeLifetime(n) => EntryLifetime(parent, dep_node_index, n),
            NodeTyParam(n) => EntryTyParam(parent, dep_node_index, n),
            NodeVisibility(n) => EntryVisibility(parent, dep_node_index, n),
            NodeLocal(n) => EntryLocal(parent, dep_node_index, n),
            NodeMacroDef(n) => EntryMacroDef(dep_node_index, n),
        };

        // Make sure that the DepNode of some node coincides with the HirId
        // owner of that node.
        if cfg!(debug_assertions) {
            let hir_id_owner = self.definitions.node_to_hir_id(id).owner;

            if hir_id_owner != self.current_dep_node_owner {
                let node_str = match self.definitions.opt_def_index(id) {
                    Some(def_index) => {
                        self.definitions.def_path(def_index).to_string_no_crate()
                    }
                    None => format!("{:?}", node)
                };

                bug!("inconsistent DepNode for `{}`: \
                      current_dep_node_owner={}, hir_id.owner={}",
                    node_str,
                    self.definitions
                        .def_path(self.current_dep_node_owner)
                        .to_string_no_crate(),
                    self.definitions.def_path(hir_id_owner).to_string_no_crate())
            }
        }

        self.insert_entry(id, entry);

    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_id: NodeId, f: F) {
        let parent_node = self.parent_node;
        self.parent_node = parent_id;
        f(self);
        self.parent_node = parent_node;
    }

    fn with_dep_node_owner<T: HashStable<StableHashingContext<'a>>,
                           F: FnOnce(&mut Self)>(&mut self,
                                                 dep_node_owner: DefIndex,
                                                 item_like: &T,
                                                 f: F) {
        let prev_owner = self.current_dep_node_owner;
        let prev_signature_dep_index = self.current_signature_dep_index;
        let prev_full_dep_index = self.current_full_dep_index;
        let prev_in_body = self.currently_in_body;

        let def_path_hash = self.definitions.def_path_hash(dep_node_owner);

        self.current_signature_dep_index = self.dep_graph.with_task(
            def_path_hash.to_dep_node(DepKind::Hir),
            &self.hcx,
            HirItemLike { item_like, hash_bodies: false },
            identity_fn
        ).1;

        self.current_full_dep_index = self.dep_graph.with_task(
            def_path_hash.to_dep_node(DepKind::HirBody),
            &self.hcx,
            HirItemLike { item_like, hash_bodies: true },
            identity_fn
        ).1;

        self.hir_body_nodes.push((def_path_hash, self.current_full_dep_index));

        self.current_dep_node_owner = dep_node_owner;
        self.currently_in_body = false;
        f(self);
        self.currently_in_body = prev_in_body;
        self.current_dep_node_owner = prev_owner;
        self.current_full_dep_index = prev_full_dep_index;
        self.current_signature_dep_index = prev_signature_dep_index;
    }
}

impl<'a, 'hir> Visitor<'hir> for NodeCollector<'a, 'hir> {
    /// Because we want to track parent items and so forth, enable
    /// deep walking so that we walk nested items in the context of
    /// their outer items.

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'hir> {
        panic!("visit_nested_xxx must be manually implemented in this visitor")
    }

    fn visit_nested_item(&mut self, item: ItemId) {
        debug!("visit_nested_item: {:?}", item);
        self.visit_item(self.krate.item(item.id));
    }

    fn visit_nested_trait_item(&mut self, item_id: TraitItemId) {
        self.visit_trait_item(self.krate.trait_item(item_id));
    }

    fn visit_nested_impl_item(&mut self, item_id: ImplItemId) {
        self.visit_impl_item(self.krate.impl_item(item_id));
    }

    fn visit_nested_body(&mut self, id: BodyId) {
        let prev_in_body = self.currently_in_body;
        self.currently_in_body = true;
        self.visit_body(self.krate.body(id));
        self.currently_in_body = prev_in_body;
    }

    fn visit_item(&mut self, i: &'hir Item) {
        debug!("visit_item: {:?}", i);
        debug_assert_eq!(i.hir_id.owner,
                         self.definitions.opt_def_index(i.id).unwrap());
        self.with_dep_node_owner(i.hir_id.owner, i, |this| {
            this.insert(i.id, NodeItem(i));
            this.with_parent(i.id, |this| {
                match i.node {
                    ItemStruct(ref struct_def, _) => {
                        // If this is a tuple-like struct, register the constructor.
                        if !struct_def.is_struct() {
                            this.insert(struct_def.id(), NodeStructCtor(struct_def));
                        }
                    }
                    _ => {}
                }
                intravisit::walk_item(this, i);
            });
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &'hir ForeignItem) {
        self.insert(foreign_item.id, NodeForeignItem(foreign_item));

        self.with_parent(foreign_item.id, |this| {
            intravisit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generics(&mut self, generics: &'hir Generics) {
        for ty_param in generics.ty_params() {
            self.insert(ty_param.id, NodeTyParam(ty_param));
        }

        intravisit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'hir TraitItem) {
        debug_assert_eq!(ti.hir_id.owner,
                         self.definitions.opt_def_index(ti.id).unwrap());
        self.with_dep_node_owner(ti.hir_id.owner, ti, |this| {
            this.insert(ti.id, NodeTraitItem(ti));

            this.with_parent(ti.id, |this| {
                intravisit::walk_trait_item(this, ti);
            });
        });
    }

    fn visit_impl_item(&mut self, ii: &'hir ImplItem) {
        debug_assert_eq!(ii.hir_id.owner,
                         self.definitions.opt_def_index(ii.id).unwrap());
        self.with_dep_node_owner(ii.hir_id.owner, ii, |this| {
            this.insert(ii.id, NodeImplItem(ii));

            this.with_parent(ii.id, |this| {
                intravisit::walk_impl_item(this, ii);
            });
        });
    }

    fn visit_pat(&mut self, pat: &'hir Pat) {
        let node = if let PatKind::Binding(..) = pat.node {
            NodeBinding(pat)
        } else {
            NodePat(pat)
        };
        self.insert(pat.id, node);

        self.with_parent(pat.id, |this| {
            intravisit::walk_pat(this, pat);
        });
    }

    fn visit_expr(&mut self, expr: &'hir Expr) {
        self.insert(expr.id, NodeExpr(expr));

        self.with_parent(expr.id, |this| {
            intravisit::walk_expr(this, expr);
        });
    }

    fn visit_stmt(&mut self, stmt: &'hir Stmt) {
        let id = stmt.node.id();
        self.insert(id, NodeStmt(stmt));

        self.with_parent(id, |this| {
            intravisit::walk_stmt(this, stmt);
        });
    }

    fn visit_ty(&mut self, ty: &'hir Ty) {
        self.insert(ty.id, NodeTy(ty));

        self.with_parent(ty.id, |this| {
            intravisit::walk_ty(this, ty);
        });
    }

    fn visit_trait_ref(&mut self, tr: &'hir TraitRef) {
        self.insert(tr.ref_id, NodeTraitRef(tr));

        self.with_parent(tr.ref_id, |this| {
            intravisit::walk_trait_ref(this, tr);
        });
    }

    fn visit_fn(&mut self, fk: intravisit::FnKind<'hir>, fd: &'hir FnDecl,
                b: BodyId, s: Span, id: NodeId) {
        assert_eq!(self.parent_node, id);
        intravisit::walk_fn(self, fk, fd, b, s, id);
    }

    fn visit_block(&mut self, block: &'hir Block) {
        self.insert(block.id, NodeBlock(block));
        self.with_parent(block.id, |this| {
            intravisit::walk_block(this, block);
        });
    }

    fn visit_local(&mut self, l: &'hir Local) {
        self.insert(l.id, NodeLocal(l));
        self.with_parent(l.id, |this| {
            intravisit::walk_local(this, l)
        })
    }

    fn visit_lifetime(&mut self, lifetime: &'hir Lifetime) {
        self.insert(lifetime.id, NodeLifetime(lifetime));
    }

    fn visit_vis(&mut self, visibility: &'hir Visibility) {
        match *visibility {
            Visibility::Public |
            Visibility::Crate |
            Visibility::Inherited => {}
            Visibility::Restricted { id, .. } => {
                self.insert(id, NodeVisibility(visibility));
                self.with_parent(id, |this| {
                    intravisit::walk_vis(this, visibility);
                });
            }
        }
    }

    fn visit_macro_def(&mut self, macro_def: &'hir MacroDef) {
        let def_index = self.definitions.opt_def_index(macro_def.id).unwrap();

        self.with_dep_node_owner(def_index, macro_def, |this| {
            this.insert(macro_def.id, NodeMacroDef(macro_def));
        });
    }

    fn visit_variant(&mut self, v: &'hir Variant, g: &'hir Generics, item_id: NodeId) {
        let id = v.node.data.id();
        self.insert(id, NodeVariant(v));
        self.with_parent(id, |this| {
            intravisit::walk_variant(this, v, g, item_id);
        });
    }

    fn visit_struct_field(&mut self, field: &'hir StructField) {
        self.insert(field.id, NodeField(field));
        self.with_parent(field.id, |this| {
            intravisit::walk_struct_field(this, field);
        });
    }

    fn visit_trait_item_ref(&mut self, ii: &'hir TraitItemRef) {
        // Do not visit the duplicate information in TraitItemRef. We want to
        // map the actual nodes, not the duplicate ones in the *Ref.
        let TraitItemRef {
            id,
            name: _,
            kind: _,
            span: _,
            defaultness: _,
        } = *ii;

        self.visit_nested_trait_item(id);
    }

    fn visit_impl_item_ref(&mut self, ii: &'hir ImplItemRef) {
        // Do not visit the duplicate information in ImplItemRef. We want to
        // map the actual nodes, not the duplicate ones in the *Ref.
        let ImplItemRef {
            id,
            name: _,
            kind: _,
            span: _,
            vis: _,
            defaultness: _,
        } = *ii;

        self.visit_nested_impl_item(id);
    }
}

// We use this with DepGraph::with_task(). Since we are handling only input
// values here, the "task" computing them just passes them through.
fn identity_fn<T>(_: &StableHashingContext, item_like: T) -> T {
    item_like
}

// This is a wrapper structure that allows determining if span values within
// the wrapped item should be hashed or not.
struct HirItemLike<T> {
    item_like: T,
    hash_bodies: bool,
}

impl<'hir, T> HashStable<StableHashingContext<'hir>> for HirItemLike<T>
    where T: HashStable<StableHashingContext<'hir>>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(self.hash_bodies, |hcx| {
            self.item_like.hash_stable(hcx, hasher);
        });
    }
}
