use crate::arena::Arena;
use crate::hir::map::{Entry, HirOwnerData, Map};
use crate::hir::{Owner, OwnerNodes, ParentedNode};
use crate::ich::StableHashingContext;
use crate::middle::cstore::CrateStore;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_hir as hir;
use rustc_hir::def_id::CRATE_DEF_INDEX;
use rustc_hir::def_id::{LocalDefId, LOCAL_CRATE};
use rustc_hir::definitions::{self, DefPathHash};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_index::vec::{Idx, IndexVec};
use rustc_session::{CrateDisambiguator, Session};
use rustc_span::source_map::SourceMap;
use rustc_span::{Span, Symbol, DUMMY_SP};

use std::iter::repeat;

/// A visitor that walks over the HIR and collects `Node`s into a HIR map.
pub(super) struct NodeCollector<'a, 'hir> {
    arena: &'hir Arena<'hir>,

    /// The crate
    krate: &'hir Crate<'hir>,

    /// Source map
    source_map: &'a SourceMap,

    map: IndexVec<LocalDefId, HirOwnerData<'hir>>,

    /// The parent of this node
    parent_node: hir::HirId,

    current_dep_node_owner: LocalDefId,

    definitions: &'a definitions::Definitions,

    hcx: StableHashingContext<'a>,

    // We are collecting HIR hashes here so we can compute the
    // crate hash from them later on.
    hir_body_nodes: Vec<(DefPathHash, Fingerprint)>,
}

fn insert_vec_map<K: Idx, V: Clone>(map: &mut IndexVec<K, Option<V>>, k: K, v: V) {
    let i = k.index();
    let len = map.len();
    if i >= len {
        map.extend(repeat(None).take(i - len + 1));
    }
    map[k] = Some(v);
}

fn hash(
    hcx: &mut StableHashingContext<'_>,
    input: impl for<'a> HashStable<StableHashingContext<'a>>,
) -> Fingerprint {
    let mut stable_hasher = StableHasher::new();
    input.hash_stable(hcx, &mut stable_hasher);
    stable_hasher.finish()
}

fn hash_body(
    hcx: &mut StableHashingContext<'_>,
    def_path_hash: DefPathHash,
    item_like: impl for<'a> HashStable<StableHashingContext<'a>>,
    hir_body_nodes: &mut Vec<(DefPathHash, Fingerprint)>,
) -> Fingerprint {
    let hash = hash(hcx, HirItemLike { item_like: &item_like });
    hir_body_nodes.push((def_path_hash, hash));
    hash
}

fn upstream_crates(cstore: &dyn CrateStore) -> Vec<(Symbol, Fingerprint, Svh)> {
    let mut upstream_crates: Vec<_> = cstore
        .crates_untracked()
        .iter()
        .map(|&cnum| {
            let name = cstore.crate_name_untracked(cnum);
            let disambiguator = cstore.crate_disambiguator_untracked(cnum).to_fingerprint();
            let hash = cstore.crate_hash_untracked(cnum);
            (name, disambiguator, hash)
        })
        .collect();
    upstream_crates.sort_unstable_by_key(|&(name, dis, _)| (name.as_str(), dis));
    upstream_crates
}

impl<'a, 'hir> NodeCollector<'a, 'hir> {
    pub(super) fn root(
        sess: &'a Session,
        arena: &'hir Arena<'hir>,
        krate: &'hir Crate<'hir>,
        definitions: &'a definitions::Definitions,
        mut hcx: StableHashingContext<'a>,
    ) -> NodeCollector<'a, 'hir> {
        let root_mod_def_path_hash =
            definitions.def_path_hash(LocalDefId { local_def_index: CRATE_DEF_INDEX });

        let mut hir_body_nodes = Vec::new();

        let hash = {
            let Crate {
                ref item,
                // These fields are handled separately:
                exported_macros: _,
                non_exported_macro_attrs: _,
                items: _,
                trait_items: _,
                impl_items: _,
                foreign_items: _,
                bodies: _,
                trait_impls: _,
                body_ids: _,
                modules: _,
                proc_macros: _,
                trait_map: _,
            } = *krate;

            hash_body(&mut hcx, root_mod_def_path_hash, item, &mut hir_body_nodes)
        };

        let mut collector = NodeCollector {
            arena,
            krate,
            source_map: sess.source_map(),
            parent_node: hir::CRATE_HIR_ID,
            current_dep_node_owner: LocalDefId { local_def_index: CRATE_DEF_INDEX },
            definitions,
            hcx,
            hir_body_nodes,
            map: (0..definitions.def_index_count())
                .map(|_| HirOwnerData { signature: None, with_bodies: None })
                .collect(),
        };
        collector.insert_entry(
            hir::CRATE_HIR_ID,
            Entry { parent: hir::CRATE_HIR_ID, node: Node::Crate(&krate.item) },
            hash,
        );

        collector
    }

    pub(super) fn finalize_and_compute_crate_hash(
        mut self,
        crate_disambiguator: CrateDisambiguator,
        cstore: &dyn CrateStore,
        commandline_args_hash: u64,
    ) -> (IndexVec<LocalDefId, HirOwnerData<'hir>>, Svh) {
        // Insert bodies into the map
        for (id, body) in self.krate.bodies.iter() {
            let bodies = &mut self.map[id.hir_id.owner].with_bodies.as_mut().unwrap().bodies;
            assert!(bodies.insert(id.hir_id.local_id, body).is_none());
        }

        self.hir_body_nodes.sort_unstable_by_key(|bn| bn.0);

        let node_hashes = self.hir_body_nodes.iter().fold(
            Fingerprint::ZERO,
            |combined_fingerprint, &(def_path_hash, fingerprint)| {
                combined_fingerprint.combine(def_path_hash.0.combine(fingerprint))
            },
        );

        let upstream_crates = upstream_crates(cstore);

        // We hash the final, remapped names of all local source files so we
        // don't have to include the path prefix remapping commandline args.
        // If we included the full mapping in the SVH, we could only have
        // reproducible builds by compiling from the same directory. So we just
        // hash the result of the mapping instead of the mapping itself.
        let mut source_file_names: Vec<_> = self
            .source_map
            .files()
            .iter()
            .filter(|source_file| source_file.cnum == LOCAL_CRATE)
            .map(|source_file| source_file.name_hash)
            .collect();

        source_file_names.sort_unstable();

        let crate_hash_input = (
            ((node_hashes, upstream_crates), source_file_names),
            (commandline_args_hash, crate_disambiguator.to_fingerprint()),
        );

        let mut stable_hasher = StableHasher::new();
        crate_hash_input.hash_stable(&mut self.hcx, &mut stable_hasher);
        let crate_hash: Fingerprint = stable_hasher.finish();

        let svh = Svh::new(crate_hash.to_smaller_hash());
        (self.map, svh)
    }

    fn insert_entry(&mut self, id: HirId, entry: Entry<'hir>, hash: Fingerprint) {
        let i = id.local_id.as_u32() as usize;

        let arena = self.arena;

        let data = &mut self.map[id.owner];

        if data.with_bodies.is_none() {
            data.with_bodies = Some(arena.alloc(OwnerNodes {
                hash,
                nodes: IndexVec::new(),
                bodies: FxHashMap::default(),
            }));
        }

        let nodes = data.with_bodies.as_mut().unwrap();

        if i == 0 {
            // Overwrite the dummy hash with the real HIR owner hash.
            nodes.hash = hash;

            // FIXME: feature(impl_trait_in_bindings) broken and trigger this assert
            //assert!(data.signature.is_none());

            data.signature =
                Some(self.arena.alloc(Owner { parent: entry.parent, node: entry.node }));
        } else {
            assert_eq!(entry.parent.owner, id.owner);
            insert_vec_map(
                &mut nodes.nodes,
                id.local_id,
                ParentedNode { parent: entry.parent.local_id, node: entry.node },
            );
        }
    }

    fn insert(&mut self, span: Span, hir_id: HirId, node: Node<'hir>) {
        self.insert_with_hash(span, hir_id, node, Fingerprint::ZERO)
    }

    fn insert_with_hash(&mut self, span: Span, hir_id: HirId, node: Node<'hir>, hash: Fingerprint) {
        let entry = Entry { parent: self.parent_node, node };

        // Make sure that the DepNode of some node coincides with the HirId
        // owner of that node.
        if cfg!(debug_assertions) {
            if hir_id.owner != self.current_dep_node_owner {
                let node_str = match self.definitions.opt_hir_id_to_local_def_id(hir_id) {
                    Some(def_id) => self.definitions.def_path(def_id).to_string_no_crate_verbose(),
                    None => format!("{:?}", node),
                };

                span_bug!(
                    span,
                    "inconsistent DepNode at `{:?}` for `{}`: \
                     current_dep_node_owner={} ({:?}), hir_id.owner={} ({:?})",
                    self.source_map.span_to_string(span),
                    node_str,
                    self.definitions
                        .def_path(self.current_dep_node_owner)
                        .to_string_no_crate_verbose(),
                    self.current_dep_node_owner,
                    self.definitions.def_path(hir_id.owner).to_string_no_crate_verbose(),
                    hir_id.owner,
                )
            }
        }

        self.insert_entry(hir_id, entry, hash);
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_node_id: HirId, f: F) {
        let parent_node = self.parent_node;
        self.parent_node = parent_node_id;
        f(self);
        self.parent_node = parent_node;
    }

    fn with_dep_node_owner<
        T: for<'b> HashStable<StableHashingContext<'b>>,
        F: FnOnce(&mut Self, Fingerprint),
    >(
        &mut self,
        dep_node_owner: LocalDefId,
        item_like: &T,
        f: F,
    ) {
        let prev_owner = self.current_dep_node_owner;

        let def_path_hash = self.definitions.def_path_hash(dep_node_owner);

        let hash = hash_body(&mut self.hcx, def_path_hash, item_like, &mut self.hir_body_nodes);

        self.current_dep_node_owner = dep_node_owner;
        f(self, hash);
        self.current_dep_node_owner = prev_owner;
    }
}

impl<'a, 'hir> Visitor<'hir> for NodeCollector<'a, 'hir> {
    type Map = Map<'hir>;

    /// Because we want to track parent items and so forth, enable
    /// deep walking so that we walk nested items in the context of
    /// their outer items.

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        panic!("`visit_nested_xxx` must be manually implemented in this visitor");
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

    fn visit_nested_foreign_item(&mut self, foreign_id: ForeignItemId) {
        self.visit_foreign_item(self.krate.foreign_item(foreign_id));
    }

    fn visit_nested_body(&mut self, id: BodyId) {
        self.visit_body(self.krate.body(id));
    }

    fn visit_param(&mut self, param: &'hir Param<'hir>) {
        let node = Node::Param(param);
        self.insert(param.pat.span, param.hir_id, node);
        self.with_parent(param.hir_id, |this| {
            intravisit::walk_param(this, param);
        });
    }

    fn visit_item(&mut self, i: &'hir Item<'hir>) {
        debug!("visit_item: {:?}", i);
        debug_assert_eq!(
            i.hir_id.owner,
            self.definitions.opt_hir_id_to_local_def_id(i.hir_id).unwrap()
        );
        self.with_dep_node_owner(i.hir_id.owner, i, |this, hash| {
            this.insert_with_hash(i.span, i.hir_id, Node::Item(i), hash);
            this.with_parent(i.hir_id, |this| {
                if let ItemKind::Struct(ref struct_def, _) = i.kind {
                    // If this is a tuple or unit-like struct, register the constructor.
                    if let Some(ctor_hir_id) = struct_def.ctor_hir_id() {
                        this.insert(i.span, ctor_hir_id, Node::Ctor(struct_def));
                    }
                }
                intravisit::walk_item(this, i);
            });
        });
    }

    fn visit_foreign_item(&mut self, fi: &'hir ForeignItem<'hir>) {
        debug_assert_eq!(
            fi.hir_id.owner,
            self.definitions.opt_hir_id_to_local_def_id(fi.hir_id).unwrap()
        );
        self.with_dep_node_owner(fi.hir_id.owner, fi, |this, hash| {
            this.insert_with_hash(fi.span, fi.hir_id, Node::ForeignItem(fi), hash);

            this.with_parent(fi.hir_id, |this| {
                intravisit::walk_foreign_item(this, fi);
            });
        });
    }

    fn visit_generic_param(&mut self, param: &'hir GenericParam<'hir>) {
        if let hir::GenericParamKind::Type {
            synthetic: Some(hir::SyntheticTyParamKind::ImplTrait),
            ..
        } = param.kind
        {
            debug_assert_eq!(
                param.hir_id.owner,
                self.definitions.opt_hir_id_to_local_def_id(param.hir_id).unwrap()
            );
            self.with_dep_node_owner(param.hir_id.owner, param, |this, hash| {
                this.insert_with_hash(param.span, param.hir_id, Node::GenericParam(param), hash);

                this.with_parent(param.hir_id, |this| {
                    intravisit::walk_generic_param(this, param);
                });
            });
        } else {
            self.insert(param.span, param.hir_id, Node::GenericParam(param));
            intravisit::walk_generic_param(self, param);
        }
    }

    fn visit_trait_item(&mut self, ti: &'hir TraitItem<'hir>) {
        debug_assert_eq!(
            ti.hir_id.owner,
            self.definitions.opt_hir_id_to_local_def_id(ti.hir_id).unwrap()
        );
        self.with_dep_node_owner(ti.hir_id.owner, ti, |this, hash| {
            this.insert_with_hash(ti.span, ti.hir_id, Node::TraitItem(ti), hash);

            this.with_parent(ti.hir_id, |this| {
                intravisit::walk_trait_item(this, ti);
            });
        });
    }

    fn visit_impl_item(&mut self, ii: &'hir ImplItem<'hir>) {
        debug_assert_eq!(
            ii.hir_id.owner,
            self.definitions.opt_hir_id_to_local_def_id(ii.hir_id).unwrap()
        );
        self.with_dep_node_owner(ii.hir_id.owner, ii, |this, hash| {
            this.insert_with_hash(ii.span, ii.hir_id, Node::ImplItem(ii), hash);

            this.with_parent(ii.hir_id, |this| {
                intravisit::walk_impl_item(this, ii);
            });
        });
    }

    fn visit_pat(&mut self, pat: &'hir Pat<'hir>) {
        let node =
            if let PatKind::Binding(..) = pat.kind { Node::Binding(pat) } else { Node::Pat(pat) };
        self.insert(pat.span, pat.hir_id, node);

        self.with_parent(pat.hir_id, |this| {
            intravisit::walk_pat(this, pat);
        });
    }

    fn visit_arm(&mut self, arm: &'hir Arm<'hir>) {
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

    fn visit_expr(&mut self, expr: &'hir Expr<'hir>) {
        self.insert(expr.span, expr.hir_id, Node::Expr(expr));

        self.with_parent(expr.hir_id, |this| {
            intravisit::walk_expr(this, expr);
        });
    }

    fn visit_stmt(&mut self, stmt: &'hir Stmt<'hir>) {
        self.insert(stmt.span, stmt.hir_id, Node::Stmt(stmt));

        self.with_parent(stmt.hir_id, |this| {
            intravisit::walk_stmt(this, stmt);
        });
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'hir PathSegment<'hir>) {
        if let Some(hir_id) = path_segment.hir_id {
            self.insert(path_span, hir_id, Node::PathSegment(path_segment));
        }
        intravisit::walk_path_segment(self, path_span, path_segment);
    }

    fn visit_ty(&mut self, ty: &'hir Ty<'hir>) {
        self.insert(ty.span, ty.hir_id, Node::Ty(ty));

        self.with_parent(ty.hir_id, |this| {
            intravisit::walk_ty(this, ty);
        });
    }

    fn visit_trait_ref(&mut self, tr: &'hir TraitRef<'hir>) {
        self.insert(tr.path.span, tr.hir_ref_id, Node::TraitRef(tr));

        self.with_parent(tr.hir_ref_id, |this| {
            intravisit::walk_trait_ref(this, tr);
        });
    }

    fn visit_fn(
        &mut self,
        fk: intravisit::FnKind<'hir>,
        fd: &'hir FnDecl<'hir>,
        b: BodyId,
        s: Span,
        id: HirId,
    ) {
        assert_eq!(self.parent_node, id);
        intravisit::walk_fn(self, fk, fd, b, s, id);
    }

    fn visit_block(&mut self, block: &'hir Block<'hir>) {
        self.insert(block.span, block.hir_id, Node::Block(block));
        self.with_parent(block.hir_id, |this| {
            intravisit::walk_block(this, block);
        });
    }

    fn visit_local(&mut self, l: &'hir Local<'hir>) {
        self.insert(l.span, l.hir_id, Node::Local(l));
        self.with_parent(l.hir_id, |this| intravisit::walk_local(this, l))
    }

    fn visit_lifetime(&mut self, lifetime: &'hir Lifetime) {
        self.insert(lifetime.span, lifetime.hir_id, Node::Lifetime(lifetime));
    }

    fn visit_vis(&mut self, visibility: &'hir Visibility<'hir>) {
        match visibility.node {
            VisibilityKind::Public | VisibilityKind::Crate(_) | VisibilityKind::Inherited => {}
            VisibilityKind::Restricted { hir_id, .. } => {
                self.insert(visibility.span, hir_id, Node::Visibility(visibility));
                self.with_parent(hir_id, |this| {
                    intravisit::walk_vis(this, visibility);
                });
            }
        }
    }

    fn visit_macro_def(&mut self, macro_def: &'hir MacroDef<'hir>) {
        // Exported macros are visited directly from the crate root,
        // so they do not have `parent_node` set.
        // Find the correct enclosing module from their DefKey.
        let def_key = self.definitions.def_key(macro_def.hir_id.owner);
        let parent = def_key.parent.map_or(hir::CRATE_HIR_ID, |local_def_index| {
            self.definitions.local_def_id_to_hir_id(LocalDefId { local_def_index })
        });
        self.with_parent(parent, |this| {
            this.with_dep_node_owner(macro_def.hir_id.owner, macro_def, |this, hash| {
                this.insert_with_hash(
                    macro_def.span,
                    macro_def.hir_id,
                    Node::MacroDef(macro_def),
                    hash,
                );
            })
        });
    }

    fn visit_variant(&mut self, v: &'hir Variant<'hir>, g: &'hir Generics<'hir>, item_id: HirId) {
        self.insert(v.span, v.id, Node::Variant(v));
        self.with_parent(v.id, |this| {
            // Register the constructor of this variant.
            if let Some(ctor_hir_id) = v.data.ctor_hir_id() {
                this.insert(v.span, ctor_hir_id, Node::Ctor(&v.data));
            }
            intravisit::walk_variant(this, v, g, item_id);
        });
    }

    fn visit_struct_field(&mut self, field: &'hir StructField<'hir>) {
        self.insert(field.span, field.hir_id, Node::Field(field));
        self.with_parent(field.hir_id, |this| {
            intravisit::walk_struct_field(this, field);
        });
    }

    fn visit_trait_item_ref(&mut self, ii: &'hir TraitItemRef) {
        // Do not visit the duplicate information in TraitItemRef. We want to
        // map the actual nodes, not the duplicate ones in the *Ref.
        let TraitItemRef { id, ident: _, kind: _, span: _, defaultness: _ } = *ii;

        self.visit_nested_trait_item(id);
    }

    fn visit_impl_item_ref(&mut self, ii: &'hir ImplItemRef<'hir>) {
        // Do not visit the duplicate information in ImplItemRef. We want to
        // map the actual nodes, not the duplicate ones in the *Ref.
        let ImplItemRef { id, ident: _, kind: _, span: _, vis: _, defaultness: _ } = *ii;

        self.visit_nested_impl_item(id);
    }

    fn visit_foreign_item_ref(&mut self, fi: &'hir ForeignItemRef<'hir>) {
        // Do not visit the duplicate information in ForeignItemRef. We want to
        // map the actual nodes, not the duplicate ones in the *Ref.
        let ForeignItemRef { id, ident: _, span: _, vis: _ } = *fi;

        self.visit_nested_foreign_item(id);
    }
}

struct HirItemLike<T> {
    item_like: T,
}

impl<'hir, T> HashStable<StableHashingContext<'hir>> for HirItemLike<T>
where
    T: HashStable<StableHashingContext<'hir>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'hir>, hasher: &mut StableHasher) {
        hcx.while_hashing_hir_bodies(true, |hcx| {
            self.item_like.hash_stable(hcx, hasher);
        });
    }
}
