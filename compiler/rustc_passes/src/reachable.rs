// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::Node;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::middle::privacy;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, DefIdTree, TyCtxt};
use rustc_session::config::CrateType;
use rustc_target::spec::abi::Abi;

// Returns true if the given item must be inlined because it may be
// monomorphized or it was marked with `#[inline]`. This will only return
// true for functions.
fn item_might_be_inlined(tcx: TyCtxt<'_>, item: &hir::Item<'_>, attrs: &CodegenFnAttrs) -> bool {
    if attrs.requests_inline() {
        return true;
    }

    match item.kind {
        hir::ItemKind::Fn(ref sig, ..) if sig.header.is_const() => true,
        hir::ItemKind::Impl { .. } | hir::ItemKind::Fn(..) => {
            let generics = tcx.generics_of(item.def_id);
            generics.requires_monomorphization(tcx)
        }
        _ => false,
    }
}

fn method_might_be_inlined(
    tcx: TyCtxt<'_>,
    impl_item: &hir::ImplItem<'_>,
    impl_src: LocalDefId,
) -> bool {
    let codegen_fn_attrs = tcx.codegen_fn_attrs(impl_item.hir_id().owner.to_def_id());
    let generics = tcx.generics_of(impl_item.def_id);
    if codegen_fn_attrs.requests_inline() || generics.requires_monomorphization(tcx) {
        return true;
    }
    if let hir::ImplItemKind::Fn(method_sig, _) = &impl_item.kind {
        if method_sig.header.is_const() {
            return true;
        }
    }
    match tcx.hir().find(tcx.hir().local_def_id_to_hir_id(impl_src)) {
        Some(Node::Item(item)) => item_might_be_inlined(tcx, &item, codegen_fn_attrs),
        Some(..) | None => span_bug!(impl_item.span, "impl did is not an item"),
    }
}

// Information needed while computing reachability.
struct ReachableContext<'tcx> {
    // The type context.
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    // The set of items which must be exported in the linkage sense.
    reachable_symbols: FxHashSet<LocalDefId>,
    // A worklist of item IDs. Each item ID in this worklist will be inlined
    // and will be scanned for further references.
    // FIXME(eddyb) benchmark if this would be faster as a `VecDeque`.
    worklist: Vec<LocalDefId>,
    // Whether any output of this compilation is a library
    any_library: bool,
}

impl<'tcx> Visitor<'tcx> for ReachableContext<'tcx> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_maybe_typeck_results =
            self.maybe_typeck_results.replace(self.tcx.typeck_body(body));
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let res = match expr.kind {
            hir::ExprKind::Path(ref qpath) => {
                Some(self.typeck_results().qpath_res(qpath, expr.hir_id))
            }
            hir::ExprKind::MethodCall(..) => self
                .typeck_results()
                .type_dependent_def(expr.hir_id)
                .map(|(kind, def_id)| Res::Def(kind, def_id)),
            _ => None,
        };

        if let Some(res) = res {
            if let Some(def_id) = res.opt_def_id().and_then(|def_id| def_id.as_local()) {
                if self.def_id_represents_local_inlined_item(def_id.to_def_id()) {
                    self.worklist.push(def_id);
                } else {
                    match res {
                        // If this path leads to a constant, then we need to
                        // recurse into the constant to continue finding
                        // items that are reachable.
                        Res::Def(DefKind::Const | DefKind::AssocConst, _) => {
                            self.worklist.push(def_id);
                        }

                        // If this wasn't a static, then the destination is
                        // surely reachable.
                        _ => {
                            self.reachable_symbols.insert(def_id);
                        }
                    }
                }
            }
        }

        intravisit::walk_expr(self, expr)
    }
}

impl<'tcx> ReachableContext<'tcx> {
    /// Gets the type-checking results for the current body.
    /// As this will ICE if called outside bodies, only call when working with
    /// `Expr` or `Pat` nodes (they are guaranteed to be found only in bodies).
    #[track_caller]
    fn typeck_results(&self) -> &'tcx ty::TypeckResults<'tcx> {
        self.maybe_typeck_results
            .expect("`ReachableContext::typeck_results` called outside of body")
    }

    // Returns true if the given def ID represents a local item that is
    // eligible for inlining and false otherwise.
    fn def_id_represents_local_inlined_item(&self, def_id: DefId) -> bool {
        let hir_id = match def_id.as_local() {
            Some(def_id) => self.tcx.hir().local_def_id_to_hir_id(def_id),
            None => {
                return false;
            }
        };

        match self.tcx.hir().find(hir_id) {
            Some(Node::Item(item)) => match item.kind {
                hir::ItemKind::Fn(..) => {
                    item_might_be_inlined(self.tcx, &item, self.tcx.codegen_fn_attrs(def_id))
                }
                _ => false,
            },
            Some(Node::TraitItem(trait_method)) => match trait_method.kind {
                hir::TraitItemKind::Const(_, ref default) => default.is_some(),
                hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(_)) => true,
                hir::TraitItemKind::Fn(_, hir::TraitFn::Required(_))
                | hir::TraitItemKind::Type(..) => false,
            },
            Some(Node::ImplItem(impl_item)) => {
                match impl_item.kind {
                    hir::ImplItemKind::Const(..) => true,
                    hir::ImplItemKind::Fn(..) => {
                        let attrs = self.tcx.codegen_fn_attrs(def_id);
                        let generics = self.tcx.generics_of(def_id);
                        if generics.requires_monomorphization(self.tcx) || attrs.requests_inline() {
                            true
                        } else {
                            let impl_did = self.tcx.hir().get_parent_did(hir_id);
                            // Check the impl. If the generics on the self
                            // type of the impl require inlining, this method
                            // does too.
                            match self.tcx.hir().expect_item(impl_did).kind {
                                hir::ItemKind::Impl { .. } => {
                                    let generics = self.tcx.generics_of(impl_did);
                                    generics.requires_monomorphization(self.tcx)
                                }
                                _ => false,
                            }
                        }
                    }
                    hir::ImplItemKind::TyAlias(_) => false,
                }
            }
            Some(_) => false,
            None => false, // This will happen for default methods.
        }
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    fn propagate(&mut self) {
        let mut scanned = FxHashSet::default();
        while let Some(search_item) = self.worklist.pop() {
            if !scanned.insert(search_item) {
                continue;
            }

            if let Some(ref item) =
                self.tcx.hir().find(self.tcx.hir().local_def_id_to_hir_id(search_item))
            {
                self.propagate_node(item, search_item);
            }
        }
    }

    fn propagate_node(&mut self, node: &Node<'tcx>, search_item: LocalDefId) {
        if !self.any_library {
            // If we are building an executable, only explicitly extern
            // types need to be exported.
            let reachable =
                if let Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, ..), .. })
                | Node::ImplItem(hir::ImplItem {
                    kind: hir::ImplItemKind::Fn(sig, ..), ..
                }) = *node
                {
                    sig.header.abi != Abi::Rust
                } else {
                    false
                };
            let codegen_attrs = self.tcx.codegen_fn_attrs(search_item);
            let is_extern = codegen_attrs.contains_extern_indicator();
            let std_internal =
                codegen_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL);
            if reachable || is_extern || std_internal {
                self.reachable_symbols.insert(search_item);
            }
        } else {
            // If we are building a library, then reachable symbols will
            // continue to participate in linkage after this product is
            // produced. In this case, we traverse the ast node, recursing on
            // all reachable nodes from this one.
            self.reachable_symbols.insert(search_item);
        }

        match *node {
            Node::Item(item) => {
                match item.kind {
                    hir::ItemKind::Fn(.., body) => {
                        if item_might_be_inlined(
                            self.tcx,
                            &item,
                            self.tcx.codegen_fn_attrs(item.def_id),
                        ) {
                            self.visit_nested_body(body);
                        }
                    }

                    // Reachable constants will be inlined into other crates
                    // unconditionally, so we need to make sure that their
                    // contents are also reachable.
                    hir::ItemKind::Const(_, init) | hir::ItemKind::Static(_, _, init) => {
                        self.visit_nested_body(init);
                    }

                    // These are normal, nothing reachable about these
                    // inherently and their children are already in the
                    // worklist, as determined by the privacy pass
                    hir::ItemKind::ExternCrate(_)
                    | hir::ItemKind::Use(..)
                    | hir::ItemKind::OpaqueTy(..)
                    | hir::ItemKind::TyAlias(..)
                    | hir::ItemKind::Macro(..)
                    | hir::ItemKind::Mod(..)
                    | hir::ItemKind::ForeignMod { .. }
                    | hir::ItemKind::Impl { .. }
                    | hir::ItemKind::Trait(..)
                    | hir::ItemKind::TraitAlias(..)
                    | hir::ItemKind::Struct(..)
                    | hir::ItemKind::Enum(..)
                    | hir::ItemKind::Union(..)
                    | hir::ItemKind::GlobalAsm(..) => {}
                }
            }
            Node::TraitItem(trait_method) => {
                match trait_method.kind {
                    hir::TraitItemKind::Const(_, None)
                    | hir::TraitItemKind::Fn(_, hir::TraitFn::Required(_)) => {
                        // Keep going, nothing to get exported
                    }
                    hir::TraitItemKind::Const(_, Some(body_id))
                    | hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(body_id)) => {
                        self.visit_nested_body(body_id);
                    }
                    hir::TraitItemKind::Type(..) => {}
                }
            }
            Node::ImplItem(impl_item) => match impl_item.kind {
                hir::ImplItemKind::Const(_, body) => {
                    self.visit_nested_body(body);
                }
                hir::ImplItemKind::Fn(_, body) => {
                    let impl_def_id =
                        self.tcx.parent(search_item.to_def_id()).unwrap().expect_local();
                    if method_might_be_inlined(self.tcx, impl_item, impl_def_id) {
                        self.visit_nested_body(body)
                    }
                }
                hir::ImplItemKind::TyAlias(_) => {}
            },
            Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure(.., body, _, _), .. }) => {
                self.visit_nested_body(body);
            }
            // Nothing to recurse on for these
            Node::ForeignItem(_)
            | Node::Variant(_)
            | Node::Ctor(..)
            | Node::Field(_)
            | Node::Ty(_)
            | Node::Crate(_) => {}
            _ => {
                bug!(
                    "found unexpected node kind in worklist: {} ({:?})",
                    self.tcx
                        .hir()
                        .node_to_string(self.tcx.hir().local_def_id_to_hir_id(search_item)),
                    node,
                );
            }
        }
    }
}

// Some methods from non-exported (completely private) trait impls still have to be
// reachable if they are called from inlinable code. Generally, it's not known until
// monomorphization if a specific trait impl item can be reachable or not. So, we
// conservatively mark all of them as reachable.
// FIXME: One possible strategy for pruning the reachable set is to avoid marking impl
// items of non-exported traits (or maybe all local traits?) unless their respective
// trait items are used from inlinable code through method call syntax or UFCS, or their
// trait is a lang item.
struct CollectPrivateImplItemsVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    access_levels: &'a privacy::AccessLevels,
    worklist: &'a mut Vec<LocalDefId>,
}

impl CollectPrivateImplItemsVisitor<'_, '_> {
    fn push_to_worklist_if_has_custom_linkage(&mut self, def_id: LocalDefId) {
        // Anything which has custom linkage gets thrown on the worklist no
        // matter where it is in the crate, along with "special std symbols"
        // which are currently akin to allocator symbols.
        let codegen_attrs = self.tcx.codegen_fn_attrs(def_id);
        if codegen_attrs.contains_extern_indicator()
            || codegen_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
        {
            self.worklist.push(def_id);
        }
    }
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for CollectPrivateImplItemsVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        self.push_to_worklist_if_has_custom_linkage(item.def_id);

        // We need only trait impls here, not inherent impls, and only non-exported ones
        if let hir::ItemKind::Impl(hir::Impl { of_trait: Some(ref trait_ref), ref items, .. }) =
            item.kind
        {
            if !self.access_levels.is_reachable(item.def_id) {
                // FIXME(#53488) remove `let`
                let tcx = self.tcx;
                self.worklist.extend(items.iter().map(|ii_ref| ii_ref.id.def_id));

                let trait_def_id = match trait_ref.path.res {
                    Res::Def(DefKind::Trait, def_id) => def_id,
                    _ => unreachable!(),
                };

                if !trait_def_id.is_local() {
                    return;
                }

                self.worklist.extend(
                    tcx.provided_trait_methods(trait_def_id)
                        .map(|assoc| assoc.def_id.expect_local()),
                );
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) {
        self.push_to_worklist_if_has_custom_linkage(impl_item.def_id);
    }

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {
        // We never export foreign functions as they have no body to export.
    }
}

fn reachable_set<'tcx>(tcx: TyCtxt<'tcx>, (): ()) -> FxHashSet<LocalDefId> {
    let access_levels = &tcx.privacy_access_levels(());

    let any_library =
        tcx.sess.crate_types().iter().any(|ty| {
            *ty == CrateType::Rlib || *ty == CrateType::Dylib || *ty == CrateType::ProcMacro
        });
    let mut reachable_context = ReachableContext {
        tcx,
        maybe_typeck_results: None,
        reachable_symbols: Default::default(),
        worklist: Vec::new(),
        any_library,
    };

    // Step 1: Seed the worklist with all nodes which were found to be public as
    //         a result of the privacy pass along with all local lang items and impl items.
    //         If other crates link to us, they're going to expect to be able to
    //         use the lang items, so we need to be sure to mark them as
    //         exported.
    reachable_context.worklist.extend(access_levels.map.keys());
    for item in tcx.lang_items().items().iter() {
        if let Some(def_id) = *item {
            if let Some(def_id) = def_id.as_local() {
                reachable_context.worklist.push(def_id);
            }
        }
    }
    {
        let mut collect_private_impl_items = CollectPrivateImplItemsVisitor {
            tcx,
            access_levels,
            worklist: &mut reachable_context.worklist,
        };
        tcx.hir().visit_all_item_likes(&mut collect_private_impl_items);
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    reachable_context.propagate();

    debug!("Inline reachability shows: {:?}", reachable_context.reachable_symbols);

    // Return the set of reachable symbols.
    reachable_context.reachable_symbols
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { reachable_set, ..*providers };
}
