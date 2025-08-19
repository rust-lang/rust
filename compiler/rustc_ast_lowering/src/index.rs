use intravisit::InferKind;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_hir as hir;
use rustc_hir::def_id::{LocalDefId, LocalDefIdMap};
use rustc_hir::intravisit::Visitor;
use rustc_hir::*;
use rustc_index::IndexVec;
use rustc_middle::span_bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument};

/// A visitor that walks over the HIR and collects `Node`s into a HIR map.
struct NodeCollector<'a, 'hir> {
    tcx: TyCtxt<'hir>,

    bodies: &'a SortedMap<ItemLocalId, &'hir Body<'hir>>,

    /// Outputs
    nodes: IndexVec<ItemLocalId, ParentedNode<'hir>>,
    parenting: LocalDefIdMap<ItemLocalId>,

    /// The parent of this node
    parent_node: ItemLocalId,

    owner: OwnerId,
}

#[instrument(level = "debug", skip(tcx, bodies))]
pub(super) fn index_hir<'hir>(
    tcx: TyCtxt<'hir>,
    item: hir::OwnerNode<'hir>,
    bodies: &SortedMap<ItemLocalId, &'hir Body<'hir>>,
    num_nodes: usize,
) -> (IndexVec<ItemLocalId, ParentedNode<'hir>>, LocalDefIdMap<ItemLocalId>) {
    let err_node = ParentedNode { parent: ItemLocalId::ZERO, node: Node::Err(item.span()) };
    let mut nodes = IndexVec::from_elem_n(err_node, num_nodes);
    // This node's parent should never be accessed: the owner's parent is computed by the
    // hir_owner_parent query. Make it invalid (= ItemLocalId::MAX) to force an ICE whenever it is
    // used.
    nodes[ItemLocalId::ZERO] = ParentedNode { parent: ItemLocalId::INVALID, node: item.into() };
    let mut collector = NodeCollector {
        tcx,
        owner: item.def_id(),
        parent_node: ItemLocalId::ZERO,
        nodes,
        bodies,
        parenting: Default::default(),
    };

    match item {
        OwnerNode::Crate(citem) => {
            collector.visit_mod(citem, citem.spans.inner_span, hir::CRATE_HIR_ID)
        }
        OwnerNode::Item(item) => collector.visit_item(item),
        OwnerNode::TraitItem(item) => collector.visit_trait_item(item),
        OwnerNode::ImplItem(item) => collector.visit_impl_item(item),
        OwnerNode::ForeignItem(item) => collector.visit_foreign_item(item),
        OwnerNode::Synthetic => unreachable!(),
    };

    for (local_id, node) in collector.nodes.iter_enumerated() {
        if let Node::Err(span) = node.node {
            let hir_id = HirId { owner: item.def_id(), local_id };
            let msg = format!("ID {hir_id} not encountered when visiting item HIR");
            tcx.dcx().span_delayed_bug(span, msg);
        }
    }

    (collector.nodes, collector.parenting)
}

impl<'a, 'hir> NodeCollector<'a, 'hir> {
    #[instrument(level = "debug", skip(self))]
    fn insert(&mut self, span: Span, hir_id: HirId, node: Node<'hir>) {
        debug_assert_eq!(self.owner, hir_id.owner);
        debug_assert_ne!(hir_id.local_id.as_u32(), 0);
        debug_assert_ne!(hir_id.local_id, self.parent_node);

        // Make sure that the DepNode of some node coincides with the HirId
        // owner of that node.
        if cfg!(debug_assertions) {
            if hir_id.owner != self.owner {
                span_bug!(
                    span,
                    "inconsistent HirId at `{:?}` for `{node:?}`: \
                     current_dep_node_owner={} ({:?}), hir_id.owner={} ({:?})",
                    self.tcx.sess.source_map().span_to_diagnostic_string(span),
                    self.tcx
                        .definitions_untracked()
                        .def_path(self.owner.def_id)
                        .to_string_no_crate_verbose(),
                    self.owner,
                    self.tcx
                        .definitions_untracked()
                        .def_path(hir_id.owner.def_id)
                        .to_string_no_crate_verbose(),
                    hir_id.owner,
                )
            }
            if self.tcx.sess.opts.incremental.is_some()
                && span.parent().is_none()
                && !span.is_dummy()
            {
                span_bug!(span, "span without a parent: {:#?}, {node:?}", span.data())
            }
        }

        self.nodes[hir_id.local_id] = ParentedNode { parent: self.parent_node, node };
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_node_id: HirId, f: F) {
        debug_assert_eq!(parent_node_id.owner, self.owner);
        let parent_node = self.parent_node;
        self.parent_node = parent_node_id.local_id;
        f(self);
        self.parent_node = parent_node;
    }

    fn insert_nested(&mut self, item: LocalDefId) {
        if self.parent_node != ItemLocalId::ZERO {
            self.parenting.insert(item, self.parent_node);
        }
    }
}

impl<'a, 'hir> Visitor<'hir> for NodeCollector<'a, 'hir> {
    /// Because we want to track parent items and so forth, enable
    /// deep walking so that we walk nested items in the context of
    /// their outer items.

    fn visit_nested_item(&mut self, item: ItemId) {
        debug!("visit_nested_item: {:?}", item);
        self.insert_nested(item.owner_id.def_id);
    }

    fn visit_nested_trait_item(&mut self, item_id: TraitItemId) {
        self.insert_nested(item_id.owner_id.def_id);
    }

    fn visit_nested_impl_item(&mut self, item_id: ImplItemId) {
        self.insert_nested(item_id.owner_id.def_id);
    }

    fn visit_nested_foreign_item(&mut self, foreign_id: ForeignItemId) {
        self.insert_nested(foreign_id.owner_id.def_id);
    }

    fn visit_nested_body(&mut self, id: BodyId) {
        debug_assert_eq!(id.hir_id.owner, self.owner);
        let body = self.bodies[&id.hir_id.local_id];
        self.visit_body(body);
    }

    fn visit_param(&mut self, param: &'hir Param<'hir>) {
        let node = Node::Param(param);
        self.insert(param.pat.span, param.hir_id, node);
        self.with_parent(param.hir_id, |this| {
            intravisit::walk_param(this, param);
        });
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_item(&mut self, i: &'hir Item<'hir>) {
        debug_assert_eq!(i.owner_id, self.owner);
        self.with_parent(i.hir_id(), |this| {
            if let ItemKind::Struct(_, _, struct_def) = &i.kind
                // If this is a tuple or unit-like struct, register the constructor.
                && let Some(ctor_hir_id) = struct_def.ctor_hir_id()
            {
                this.insert(i.span, ctor_hir_id, Node::Ctor(struct_def));
            }
            intravisit::walk_item(this, i);
        });
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_foreign_item(&mut self, fi: &'hir ForeignItem<'hir>) {
        debug_assert_eq!(fi.owner_id, self.owner);
        self.with_parent(fi.hir_id(), |this| {
            intravisit::walk_foreign_item(this, fi);
        });
    }

    fn visit_generic_param(&mut self, param: &'hir GenericParam<'hir>) {
        self.insert(param.span, param.hir_id, Node::GenericParam(param));
        intravisit::walk_generic_param(self, param);
    }

    fn visit_const_param_default(&mut self, param: HirId, ct: &'hir ConstArg<'hir>) {
        self.with_parent(param, |this| {
            intravisit::walk_const_param_default(this, ct);
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_trait_item(&mut self, ti: &'hir TraitItem<'hir>) {
        debug_assert_eq!(ti.owner_id, self.owner);
        self.with_parent(ti.hir_id(), |this| {
            intravisit::walk_trait_item(this, ti);
        });
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_impl_item(&mut self, ii: &'hir ImplItem<'hir>) {
        debug_assert_eq!(ii.owner_id, self.owner);
        self.with_parent(ii.hir_id(), |this| {
            intravisit::walk_impl_item(this, ii);
        });
    }

    fn visit_pat(&mut self, pat: &'hir Pat<'hir>) {
        self.insert(pat.span, pat.hir_id, Node::Pat(pat));

        self.with_parent(pat.hir_id, |this| {
            intravisit::walk_pat(this, pat);
        });
    }

    fn visit_pat_expr(&mut self, expr: &'hir PatExpr<'hir>) {
        self.insert(expr.span, expr.hir_id, Node::PatExpr(expr));

        self.with_parent(expr.hir_id, |this| {
            intravisit::walk_pat_expr(this, expr);
        });
    }

    fn visit_pat_field(&mut self, field: &'hir PatField<'hir>) {
        self.insert(field.span, field.hir_id, Node::PatField(field));
        self.with_parent(field.hir_id, |this| {
            intravisit::walk_pat_field(this, field);
        });
    }

    fn visit_arm(&mut self, arm: &'hir Arm<'hir>) {
        let node = Node::Arm(arm);

        self.insert(arm.span, arm.hir_id, node);

        self.with_parent(arm.hir_id, |this| {
            intravisit::walk_arm(this, arm);
        });
    }

    fn visit_opaque_ty(&mut self, opaq: &'hir OpaqueTy<'hir>) {
        self.insert(opaq.span, opaq.hir_id, Node::OpaqueTy(opaq));

        self.with_parent(opaq.hir_id, |this| {
            intravisit::walk_opaque_ty(this, opaq);
        });
    }

    fn visit_anon_const(&mut self, constant: &'hir AnonConst) {
        self.insert(constant.span, constant.hir_id, Node::AnonConst(constant));

        self.with_parent(constant.hir_id, |this| {
            intravisit::walk_anon_const(this, constant);
        });
    }

    fn visit_inline_const(&mut self, constant: &'hir ConstBlock) {
        self.insert(DUMMY_SP, constant.hir_id, Node::ConstBlock(constant));

        self.with_parent(constant.hir_id, |this| {
            intravisit::walk_inline_const(this, constant);
        });
    }

    fn visit_expr(&mut self, expr: &'hir Expr<'hir>) {
        self.insert(expr.span, expr.hir_id, Node::Expr(expr));

        self.with_parent(expr.hir_id, |this| {
            intravisit::walk_expr(this, expr);
        });
    }

    fn visit_expr_field(&mut self, field: &'hir ExprField<'hir>) {
        self.insert(field.span, field.hir_id, Node::ExprField(field));
        self.with_parent(field.hir_id, |this| {
            intravisit::walk_expr_field(this, field);
        });
    }

    fn visit_stmt(&mut self, stmt: &'hir Stmt<'hir>) {
        self.insert(stmt.span, stmt.hir_id, Node::Stmt(stmt));

        self.with_parent(stmt.hir_id, |this| {
            intravisit::walk_stmt(this, stmt);
        });
    }

    fn visit_path_segment(&mut self, path_segment: &'hir PathSegment<'hir>) {
        // FIXME: walk path segment with `path_segment.hir_id` parent.
        self.insert(path_segment.ident.span, path_segment.hir_id, Node::PathSegment(path_segment));
        intravisit::walk_path_segment(self, path_segment);
    }

    fn visit_ty(&mut self, ty: &'hir Ty<'hir, AmbigArg>) {
        self.insert(ty.span, ty.hir_id, Node::Ty(ty.as_unambig_ty()));

        self.with_parent(ty.hir_id, |this| {
            intravisit::walk_ty(this, ty);
        });
    }

    fn visit_const_arg(&mut self, const_arg: &'hir ConstArg<'hir, AmbigArg>) {
        self.insert(
            const_arg.as_unambig_ct().span(),
            const_arg.hir_id,
            Node::ConstArg(const_arg.as_unambig_ct()),
        );

        self.with_parent(const_arg.hir_id, |this| {
            intravisit::walk_const_arg(this, const_arg);
        });
    }

    fn visit_infer(
        &mut self,
        inf_id: HirId,
        inf_span: Span,
        kind: InferKind<'hir>,
    ) -> Self::Result {
        match kind {
            InferKind::Ty(ty) => self.insert(inf_span, inf_id, Node::Ty(ty)),
            InferKind::Const(ct) => self.insert(inf_span, inf_id, Node::ConstArg(ct)),
            InferKind::Ambig(inf) => self.insert(inf_span, inf_id, Node::Infer(inf)),
        }

        self.visit_id(inf_id);
    }

    fn visit_trait_ref(&mut self, tr: &'hir TraitRef<'hir>) {
        self.insert(tr.path.span, tr.hir_ref_id, Node::TraitRef(tr));

        self.with_parent(tr.hir_ref_id, |this| {
            intravisit::walk_trait_ref(this, tr);
        });
    }

    fn visit_block(&mut self, block: &'hir Block<'hir>) {
        self.insert(block.span, block.hir_id, Node::Block(block));
        self.with_parent(block.hir_id, |this| {
            intravisit::walk_block(this, block);
        });
    }

    fn visit_local(&mut self, l: &'hir LetStmt<'hir>) {
        self.insert(l.span, l.hir_id, Node::LetStmt(l));
        self.with_parent(l.hir_id, |this| {
            intravisit::walk_local(this, l);
        })
    }

    fn visit_lifetime(&mut self, lifetime: &'hir Lifetime) {
        self.insert(lifetime.ident.span, lifetime.hir_id, Node::Lifetime(lifetime));
    }

    fn visit_variant(&mut self, v: &'hir Variant<'hir>) {
        self.insert(v.span, v.hir_id, Node::Variant(v));
        self.with_parent(v.hir_id, |this| {
            // Register the constructor of this variant.
            if let Some(ctor_hir_id) = v.data.ctor_hir_id() {
                this.insert(v.span, ctor_hir_id, Node::Ctor(&v.data));
            }
            intravisit::walk_variant(this, v);
        });
    }

    fn visit_field_def(&mut self, field: &'hir FieldDef<'hir>) {
        self.insert(field.span, field.hir_id, Node::Field(field));
        self.with_parent(field.hir_id, |this| {
            intravisit::walk_field_def(this, field);
        });
    }

    fn visit_assoc_item_constraint(&mut self, constraint: &'hir AssocItemConstraint<'hir>) {
        self.insert(constraint.span, constraint.hir_id, Node::AssocItemConstraint(constraint));
        self.with_parent(constraint.hir_id, |this| {
            intravisit::walk_assoc_item_constraint(this, constraint)
        })
    }

    fn visit_trait_item_ref(&mut self, id: &'hir TraitItemId) {
        self.visit_nested_trait_item(*id);
    }

    fn visit_impl_item_ref(&mut self, id: &'hir ImplItemId) {
        self.visit_nested_impl_item(*id);
    }

    fn visit_foreign_item_ref(&mut self, id: &'hir ForeignItemId) {
        self.visit_nested_foreign_item(*id);
    }

    fn visit_where_predicate(&mut self, predicate: &'hir WherePredicate<'hir>) {
        self.insert(predicate.span, predicate.hir_id, Node::WherePredicate(predicate));
        self.with_parent(predicate.hir_id, |this| {
            intravisit::walk_where_predicate(this, predicate)
        });
    }

    fn visit_pattern_type_pattern(&mut self, pat: &'hir hir::TyPat<'hir>) {
        self.insert(pat.span, pat.hir_id, Node::TyPat(pat));

        self.with_parent(pat.hir_id, |this| {
            intravisit::walk_ty_pat(this, pat);
        });
    }

    fn visit_precise_capturing_arg(
        &mut self,
        arg: &'hir PreciseCapturingArg<'hir>,
    ) -> Self::Result {
        match arg {
            PreciseCapturingArg::Lifetime(_) => {
                // This is represented as a `Node::Lifetime`, intravisit will get to it below.
            }
            PreciseCapturingArg::Param(param) => self.insert(
                param.ident.span,
                param.hir_id,
                Node::PreciseCapturingNonLifetimeArg(param),
            ),
        }
        intravisit::walk_precise_capturing_arg(self, arg);
    }
}
