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
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::Node;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::middle::privacy::{self, Level};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
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
            let generics = tcx.generics_of(item.owner_id);
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
    let generics = tcx.generics_of(impl_item.owner_id);
    if codegen_fn_attrs.requests_inline() || generics.requires_monomorphization(tcx) {
        return true;
    }
    if let hir::ImplItemKind::Fn(method_sig, _) = &impl_item.kind {
        if method_sig.header.is_const() {
            return true;
        }
    }
    match tcx.hir().find_by_def_id(impl_src) {
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

        if let Some(res) = res && let Some(def_id) = res.opt_def_id().and_then(|el| el.as_local()) {
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

        intravisit::walk_expr(self, expr)
    }

    fn visit_inline_asm(&mut self, asm: &'tcx hir::InlineAsm<'tcx>, id: hir::HirId) {
        for (op, _) in asm.operands {
            if let hir::InlineAsmOperand::SymStatic { def_id, .. } = op {
                if let Some(def_id) = def_id.as_local() {
                    self.reachable_symbols.insert(def_id);
                }
            }
        }
        intravisit::walk_inline_asm(self, asm, id);
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
        let Some(def_id) = def_id.as_local() else {
            return false;
        };

        match self.tcx.hir().find_by_def_id(def_id) {
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
            Some(Node::ImplItem(impl_item)) => match impl_item.kind {
                hir::ImplItemKind::Const(..) => true,
                hir::ImplItemKind::Fn(..) => {
                    let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                    let impl_did = self.tcx.hir().get_parent_item(hir_id);
                    method_might_be_inlined(self.tcx, impl_item, impl_did.def_id)
                }
                hir::ImplItemKind::Type(_) => false,
            },
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

            if let Some(ref item) = self.tcx.hir().find_by_def_id(search_item) {
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
            let codegen_attrs = if self.tcx.def_kind(search_item).has_codegen_attrs() {
                self.tcx.codegen_fn_attrs(search_item)
            } else {
                CodegenFnAttrs::EMPTY
            };
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
                            self.tcx.codegen_fn_attrs(item.owner_id),
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
                    let impl_def_id = self.tcx.local_parent(search_item);
                    if method_might_be_inlined(self.tcx, impl_item, impl_def_id) {
                        self.visit_nested_body(body)
                    }
                }
                hir::ImplItemKind::Type(_) => {}
            },
            Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { body, .. }),
                ..
            }) => {
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

fn check_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    id: hir::ItemId,
    worklist: &mut Vec<LocalDefId>,
    effective_visibilities: &privacy::EffectiveVisibilities,
) {
    if has_custom_linkage(tcx, id.owner_id.def_id) {
        worklist.push(id.owner_id.def_id);
    }

    if !matches!(tcx.def_kind(id.owner_id), DefKind::Impl { of_trait: true }) {
        return;
    }

    // We need only trait impls here, not inherent impls, and only non-exported ones
    if effective_visibilities.is_reachable(id.owner_id.def_id) {
        return;
    }

    let items = tcx.associated_item_def_ids(id.owner_id);
    worklist.extend(items.iter().map(|ii_ref| ii_ref.expect_local()));

    let Some(trait_def_id) = tcx.trait_id_of_impl(id.owner_id.to_def_id()) else {
        unreachable!();
    };

    if !trait_def_id.is_local() {
        return;
    }

    worklist
        .extend(tcx.provided_trait_methods(trait_def_id).map(|assoc| assoc.def_id.expect_local()));
}

fn has_custom_linkage(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // Anything which has custom linkage gets thrown on the worklist no
    // matter where it is in the crate, along with "special std symbols"
    // which are currently akin to allocator symbols.
    if !tcx.def_kind(def_id).has_codegen_attrs() {
        return false;
    }
    let codegen_attrs = tcx.codegen_fn_attrs(def_id);
    codegen_attrs.contains_extern_indicator()
        || codegen_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
        // FIXME(nbdd0121): `#[used]` are marked as reachable here so it's picked up by
        // `linked_symbols` in cg_ssa. They won't be exported in binary or cdylib due to their
        // `SymbolExportLevel::Rust` export level but may end up being exported in dylibs.
        || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED)
        || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
}

fn reachable_set(tcx: TyCtxt<'_>, (): ()) -> FxHashSet<LocalDefId> {
    let effective_visibilities = &tcx.effective_visibilities(());

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
    reachable_context.worklist = effective_visibilities
        .iter()
        .filter_map(|(&id, effective_vis)| {
            effective_vis.is_public_at_level(Level::ReachableThroughImplTrait).then_some(id)
        })
        .collect::<Vec<_>>();

    for (_, def_id) in tcx.lang_items().iter() {
        if let Some(def_id) = def_id.as_local() {
            reachable_context.worklist.push(def_id);
        }
    }
    {
        // Some methods from non-exported (completely private) trait impls still have to be
        // reachable if they are called from inlinable code. Generally, it's not known until
        // monomorphization if a specific trait impl item can be reachable or not. So, we
        // conservatively mark all of them as reachable.
        // FIXME: One possible strategy for pruning the reachable set is to avoid marking impl
        // items of non-exported traits (or maybe all local traits?) unless their respective
        // trait items are used from inlinable code through method call syntax or UFCS, or their
        // trait is a lang item.
        let crate_items = tcx.hir_crate_items(());

        for id in crate_items.items() {
            check_item(tcx, id, &mut reachable_context.worklist, effective_visibilities);
        }

        for id in crate_items.impl_items() {
            if has_custom_linkage(tcx, id.owner_id.def_id) {
                reachable_context.worklist.push(id.owner_id.def_id);
            }
        }
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
