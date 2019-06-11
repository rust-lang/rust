use super::*;
use crate::dep_graph::{DepGraph, DepKind, DepNodeIndex};
use crate::hir;
use crate::hir::map::HirEntryMap;
use crate::hir::def_id::{LOCAL_CRATE, CrateNum};
use crate::hir::intravisit::{Visitor, NestedVisitorMap};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::indexed_vec::IndexVec;
use crate::ich::Fingerprint;
use crate::middle::cstore::CrateStore;
use crate::session::CrateDisambiguator;
use crate::session::Session;
use crate::util::nodemap::FxHashMap;
use syntax::ast::NodeId;
use syntax::source_map::SourceMap;
use syntax_pos::Span;
use std::iter::repeat;

use crate::ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};

/// A visitor that walks over the HIR and collects `Node`s into a HIR map.
pub(super) struct NodeCollector<'a, 'hir> {
    /// The crate
    krate: &'hir Crate,

    /// Source map
    source_map: &'a SourceMap,

    /// The node map
    map: HirEntryMap<'hir>,
    /// The parent of this node
    parent_node: hir::HirId,

    // These fields keep track of the currently relevant DepNodes during
    // the visitor's traversal.
    current_dep_node_owner: DefIndex,
    current_signature_dep_index: DepNodeIndex,
    current_full_dep_index: DepNodeIndex,
    currently_in_body: bool,

    dep_graph: &'a DepGraph,
    definitions: &'a definitions::Definitions,
    hir_to_node_id: &'a FxHashMap<HirId, NodeId>,

    hcx: StableHashingContext<'a>,

    // We are collecting `DepNode::HirBody` hashes here so we can compute the
    // crate hash from then later on.
    hir_body_nodes: Vec<(DefPathHash, Fingerprint)>,
}

fn input_dep_node_and_hash<I>(
    dep_graph: &DepGraph,
    hcx: &mut StableHashingContext<'_>,
    dep_node: DepNode,
    input: I,
) -> (DepNodeIndex, Fingerprint)
where
    I: for<'a> HashStable<StableHashingContext<'a>>,
{
    let dep_node_index = dep_graph.input_task(dep_node, &mut *hcx, &input).1;

    let hash = if dep_graph.is_fully_enabled() {
        dep_graph.fingerprint_of(dep_node_index)
    } else {
        let mut stable_hasher = StableHasher::new();
        input.hash_stable(hcx, &mut stable_hasher);
        stable_hasher.finish()
    };

    (dep_node_index, hash)
}

fn alloc_hir_dep_nodes<I>(
    dep_graph: &DepGraph,
    hcx: &mut StableHashingContext<'_>,
    def_path_hash: DefPathHash,
    item_like: I,
    hir_body_nodes: &mut Vec<(DefPathHash, Fingerprint)>,
) -> (DepNodeIndex, DepNodeIndex)
where
    I: for<'a> HashStable<StableHashingContext<'a>>,
{
    let sig = dep_graph.input_task(
        def_path_hash.to_dep_node(DepKind::Hir),
        &mut *hcx,
        HirItemLike { item_like: &item_like, hash_bodies: false },
    ).1;
    let (full, hash) = input_dep_node_and_hash(
        dep_graph,
        hcx,
        def_path_hash.to_dep_node(DepKind::HirBody),
        HirItemLike { item_like: &item_like, hash_bodies: true },
    );
    hir_body_nodes.push((def_path_hash, hash));
    (sig, full)
}

impl<'a, 'hir> NodeCollector<'a, 'hir> {
    pub(super) fn root(sess: &'a Session,
                       krate: &'hir Crate,
                       dep_graph: &'a DepGraph,
                       definitions: &'a definitions::Definitions,
                       hir_to_node_id: &'a FxHashMap<HirId, NodeId>,
                       mut hcx: StableHashingContext<'a>)
                -> NodeCollector<'a, 'hir> {
        let root_mod_def_path_hash = definitions.def_path_hash(CRATE_DEF_INDEX);

        let mut hir_body_nodes = Vec::new();

        // Allocate `DepNode`s for the root module.
        let (root_mod_sig_dep_index, root_mod_full_dep_index) = {
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
                body_ids: _,
                modules: _,
            } = *krate;

            alloc_hir_dep_nodes(
                dep_graph,
                &mut hcx,
                root_mod_def_path_hash,
                (module, attrs, span),
                &mut hir_body_nodes,
            )
        };

        {
            dep_graph.input_task(
                DepNode::new_no_params(DepKind::AllLocalTraitImpls),
                &mut hcx,
                &krate.trait_impls,
            );
        }

        let mut collector = NodeCollector {
            krate,
            source_map: sess.source_map(),
            map: vec![None; definitions.def_index_count()],
            parent_node: hir::CRATE_HIR_ID,
            current_signature_dep_index: root_mod_sig_dep_index,
            current_full_dep_index: root_mod_full_dep_index,
            current_dep_node_owner: CRATE_DEF_INDEX,
            currently_in_body: false,
            dep_graph,
            definitions,
            hir_to_node_id,
            hcx,
            hir_body_nodes,
        };
        collector.insert_entry(hir::CRATE_HIR_ID, Entry {
            parent: hir::CRATE_HIR_ID,
            dep_node: root_mod_sig_dep_index,
            node: Node::Crate,
        });

        collector
    }

    pub(super) fn finalize_and_compute_crate_hash(mut self,
                                                  crate_disambiguator: CrateDisambiguator,
                                                  cstore: &dyn CrateStore,
                                                  commandline_args_hash: u64)
                                                  -> (HirEntryMap<'hir>, Svh)
    {
        self.hir_body_nodes.sort_unstable_by_key(|bn| bn.0);

        let node_hashes = self
            .hir_body_nodes
            .iter()
            .fold(Fingerprint::ZERO, |combined_fingerprint, &(def_path_hash, fingerprint)| {
                combined_fingerprint.combine(def_path_hash.0.combine(fingerprint))
            });

        let mut upstream_crates: Vec<_> = cstore.crates_untracked().iter().map(|&cnum| {
            let name = cstore.crate_name_untracked(cnum).as_str();
            let disambiguator = cstore.crate_disambiguator_untracked(cnum).to_fingerprint();
            let hash = cstore.crate_hash_untracked(cnum);
            (name, disambiguator, hash)
        }).collect();

        upstream_crates.sort_unstable_by_key(|&(name, dis, _)| (name, dis));

        // We hash the final, remapped names of all local source files so we
        // don't have to include the path prefix remapping commandline args.
        // If we included the full mapping in the SVH, we could only have
        // reproducible builds by compiling from the same directory. So we just
        // hash the result of the mapping instead of the mapping itself.
        let mut source_file_names: Vec<_> = self
            .source_map
            .files()
            .iter()
            .filter(|source_file| CrateNum::from_u32(source_file.crate_of_origin) == LOCAL_CRATE)
            .map(|source_file| source_file.name_hash)
            .collect();

        source_file_names.sort_unstable();

        let crate_hash_input = (
            ((node_hashes, upstream_crates), source_file_names),
            (commandline_args_hash, crate_disambiguator.to_fingerprint())
        );

        let (_, crate_hash) = input_dep_node_and_hash(
            self.dep_graph,
            &mut self.hcx,
            DepNode::new_no_params(DepKind::Krate),
            crate_hash_input,
        );

        let svh = Svh::new(crate_hash.to_smaller_hash());
        (self.map, svh)
    }

    fn insert_entry(&mut self, id: HirId, entry: Entry<'hir>) {
        debug!("hir_map: {:?} => {:?}", id, entry);
        let local_map = &mut self.map[id.owner.index()];
        let i = id.local_id.as_u32() as usize;
        if local_map.is_none() {
            *local_map = Some(IndexVec::with_capacity(i + 1));
        }
        let local_map = local_map.as_mut().unwrap();
        let len = local_map.len();
        if i >= len {
            local_map.extend(repeat(None).take(i - len + 1));
        }
        local_map[id.local_id] = Some(entry);
    }

    fn insert(&mut self, span: Span, hir_id: HirId, node: Node<'hir>) {
        let entry = Entry {
            parent: self.parent_node,
            dep_node: if self.currently_in_body {
                self.current_full_dep_index
            } else {
                self.current_signature_dep_index
            },
            node,
        };

        // Make sure that the DepNode of some node coincides with the HirId
        // owner of that node.
        if cfg!(debug_assertions) {
            let node_id = self.hir_to_node_id[&hir_id];
            assert_eq!(self.definitions.node_to_hir_id(node_id), hir_id);

            if hir_id.owner != self.current_dep_node_owner {
                let node_str = match self.definitions.opt_def_index(node_id) {
                    Some(def_index) => {
                        self.definitions.def_path(def_index).to_string_no_crate()
                    }
                    None => format!("{:?}", node)
                };

                let forgot_str = if hir_id == crate::hir::DUMMY_HIR_ID {
                    format!("\nMaybe you forgot to lower the node id {:?}?", node_id)
                } else {
                    String::new()
                };

                span_bug!(
                    span,
                    "inconsistent DepNode at `{:?}` for `{}`: \
                     current_dep_node_owner={} ({:?}), hir_id.owner={} ({:?}){}",
                    self.source_map.span_to_string(span),
                    node_str,
                    self.definitions
                        .def_path(self.current_dep_node_owner)
                        .to_string_no_crate(),
                    self.current_dep_node_owner,
                    self.definitions.def_path(hir_id.owner).to_string_no_crate(),
                    hir_id.owner,
                    forgot_str,
                )
            }
        }

        self.insert_entry(hir_id, entry);
    }

    fn with_parent<F: FnOnce(&mut Self)>(
        &mut self,
        parent_node_id: HirId,
        f: F,
    ) {
        let parent_node = self.parent_node;
        self.parent_node = parent_node_id;
        f(self);
        self.parent_node = parent_node;
    }

    fn with_dep_node_owner<T: for<'b> HashStable<StableHashingContext<'b>>,
                           F: FnOnce(&mut Self)>(&mut self,
                                                 dep_node_owner: DefIndex,
                                                 item_like: &T,
                                                 f: F) {
        let prev_owner = self.current_dep_node_owner;
        let prev_signature_dep_index = self.current_signature_dep_index;
        let prev_full_dep_index = self.current_full_dep_index;
        let prev_in_body = self.currently_in_body;

        let def_path_hash = self.definitions.def_path_hash(dep_node_owner);

        let (signature_dep_index, full_dep_index) = alloc_hir_dep_nodes(
            self.dep_graph,
            &mut self.hcx,
            def_path_hash,
            item_like,
            &mut self.hir_body_nodes,
        );
        self.current_signature_dep_index = signature_dep_index;
        self.current_full_dep_index = full_dep_index;

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
                         self.definitions.opt_def_index(self.hir_to_node_id[&i.hir_id]).unwrap());
        self.with_dep_node_owner(i.hir_id.owner, i, |this| {
            this.insert(i.span, i.hir_id, Node::Item(i));
            this.with_parent(i.hir_id, |this| {
                if let ItemKind::Struct(ref struct_def, _) = i.node {
                    // If this is a tuple or unit-like struct, register the constructor.
                    if let Some(ctor_hir_id) = struct_def.ctor_hir_id() {
                        this.insert(i.span, ctor_hir_id, Node::Ctor(struct_def));
                    }
                }
                intravisit::walk_item(this, i);
            });
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &'hir ForeignItem) {
        self.insert(foreign_item.span, foreign_item.hir_id, Node::ForeignItem(foreign_item));

        self.with_parent(foreign_item.hir_id, |this| {
            intravisit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generic_param(&mut self, param: &'hir GenericParam) {
        self.insert(param.span, param.hir_id, Node::GenericParam(param));
        intravisit::walk_generic_param(self, param);
    }

    fn visit_trait_item(&mut self, ti: &'hir TraitItem) {
        debug_assert_eq!(ti.hir_id.owner,
                         self.definitions.opt_def_index(self.hir_to_node_id[&ti.hir_id]).unwrap());
        self.with_dep_node_owner(ti.hir_id.owner, ti, |this| {
            this.insert(ti.span, ti.hir_id, Node::TraitItem(ti));

            this.with_parent(ti.hir_id, |this| {
                intravisit::walk_trait_item(this, ti);
            });
        });
    }

    fn visit_impl_item(&mut self, ii: &'hir ImplItem) {
        debug_assert_eq!(ii.hir_id.owner,
                         self.definitions.opt_def_index(self.hir_to_node_id[&ii.hir_id]).unwrap());
        self.with_dep_node_owner(ii.hir_id.owner, ii, |this| {
            this.insert(ii.span, ii.hir_id, Node::ImplItem(ii));

            this.with_parent(ii.hir_id, |this| {
                intravisit::walk_impl_item(this, ii);
            });
        });
    }

    fn visit_pat(&mut self, pat: &'hir Pat) {
        let node = if let PatKind::Binding(..) = pat.node {
            Node::Binding(pat)
        } else {
            Node::Pat(pat)
        };
        self.insert(pat.span, pat.hir_id, node);

        self.with_parent(pat.hir_id, |this| {
            intravisit::walk_pat(this, pat);
        });
    }

    fn visit_arm(&mut self, arm: &'hir Arm) {
        let node = Node::Arm(arm);

        self.insert(arm.span, arm.hir_id, node);

        self.with_parent(arm.hir_id, |this| {
            intravisit::walk_arm(this, arm);
        });
    }

    fn visit_anon_const(&mut self, constant: &'hir AnonConst) {
        self.insert(DUMMY_SP, constant.hir_id, Node::AnonConst(constant));

        self.with_parent(constant.hir_id, |this| {
            intravisit::walk_anon_const(this, constant);
        });
    }

    fn visit_expr(&mut self, expr: &'hir Expr) {
        self.insert(expr.span, expr.hir_id, Node::Expr(expr));

        self.with_parent(expr.hir_id, |this| {
            intravisit::walk_expr(this, expr);
        });
    }

    fn visit_stmt(&mut self, stmt: &'hir Stmt) {
        self.insert(stmt.span, stmt.hir_id, Node::Stmt(stmt));

        self.with_parent(stmt.hir_id, |this| {
            intravisit::walk_stmt(this, stmt);
        });
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'hir PathSegment) {
        if let Some(hir_id) = path_segment.hir_id {
            self.insert(path_span, hir_id, Node::PathSegment(path_segment));
        }
        intravisit::walk_path_segment(self, path_span, path_segment);
    }

    fn visit_ty(&mut self, ty: &'hir Ty) {
        self.insert(ty.span, ty.hir_id, Node::Ty(ty));

        self.with_parent(ty.hir_id, |this| {
            intravisit::walk_ty(this, ty);
        });
    }

    fn visit_trait_ref(&mut self, tr: &'hir TraitRef) {
        self.insert(tr.path.span, tr.hir_ref_id, Node::TraitRef(tr));

        self.with_parent(tr.hir_ref_id, |this| {
            intravisit::walk_trait_ref(this, tr);
        });
    }

    fn visit_fn(&mut self, fk: intravisit::FnKind<'hir>, fd: &'hir FnDecl,
                b: BodyId, s: Span, id: HirId) {
        assert_eq!(self.parent_node, id);
        intravisit::walk_fn(self, fk, fd, b, s, id);
    }

    fn visit_block(&mut self, block: &'hir Block) {
        self.insert(block.span, block.hir_id, Node::Block(block));
        self.with_parent(block.hir_id, |this| {
            intravisit::walk_block(this, block);
        });
    }

    fn visit_local(&mut self, l: &'hir Local) {
        self.insert(l.span, l.hir_id, Node::Local(l));
        self.with_parent(l.hir_id, |this| {
            intravisit::walk_local(this, l)
        })
    }

    fn visit_lifetime(&mut self, lifetime: &'hir Lifetime) {
        self.insert(lifetime.span, lifetime.hir_id, Node::Lifetime(lifetime));
    }

    fn visit_vis(&mut self, visibility: &'hir Visibility) {
        match visibility.node {
            VisibilityKind::Public |
            VisibilityKind::Crate(_) |
            VisibilityKind::Inherited => {}
            VisibilityKind::Restricted { hir_id, .. } => {
                self.insert(visibility.span, hir_id, Node::Visibility(visibility));
                self.with_parent(hir_id, |this| {
                    intravisit::walk_vis(this, visibility);
                });
            }
        }
    }

    fn visit_macro_def(&mut self, macro_def: &'hir MacroDef) {
        let node_id = self.hir_to_node_id[&macro_def.hir_id];
        let def_index = self.definitions.opt_def_index(node_id).unwrap();

        self.with_dep_node_owner(def_index, macro_def, |this| {
            this.insert(macro_def.span, macro_def.hir_id, Node::MacroDef(macro_def));
        });
    }

    fn visit_variant(&mut self, v: &'hir Variant, g: &'hir Generics, item_id: HirId) {
        self.insert(v.span, v.node.id, Node::Variant(v));
        self.with_parent(v.node.id, |this| {
            // Register the constructor of this variant.
            if let Some(ctor_hir_id) = v.node.data.ctor_hir_id() {
                this.insert(v.span, ctor_hir_id, Node::Ctor(&v.node.data));
            }
            intravisit::walk_variant(this, v, g, item_id);
        });
    }

    fn visit_struct_field(&mut self, field: &'hir StructField) {
        self.insert(field.span, field.hir_id, Node::Field(field));
        self.with_parent(field.hir_id, |this| {
            intravisit::walk_struct_field(this, field);
        });
    }

    fn visit_trait_item_ref(&mut self, ii: &'hir TraitItemRef) {
        // Do not visit the duplicate information in TraitItemRef. We want to
        // map the actual nodes, not the duplicate ones in the *Ref.
        let TraitItemRef {
            id,
            ident: _,
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
            ident: _,
            kind: _,
            span: _,
            vis: _,
            defaultness: _,
        } = *ii;

        self.visit_nested_impl_item(id);
    }
}

// This is a wrapper structure that allows determining if span values within
// the wrapped item should be hashed or not.
struct HirItemLike<T> {
    item_like: T,
    hash_bodies: bool,
}

impl<'hir, T> HashStable<StableHashingContext<'hir>> for HirItemLike<T>
where
    T: HashStable<StableHashingContext<'hir>>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'hir>,
                                          hasher: &mut StableHasher<W>) {
        hcx.while_hashing_hir_bodies(self.hash_bodies, |hcx| {
            self.item_like.hash_stable(hcx, hasher);
        });
    }
}
