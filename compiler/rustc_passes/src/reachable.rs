//! Finds local items that are "reachable", which means that other crates need access to their
//! compiled code or their *runtime* MIR. (Compile-time MIR is always encoded anyway, so we don't
//! worry about that here.)
//!
//! An item is "reachable" if codegen that happens in downstream crates can end up referencing this
//! item. This obviously includes all public items. However, some of these items cannot be codegen'd
//! (because they are generic), and for some the compiled code is not sufficient (because we want to
//! cross-crate inline them). These items "need cross-crate MIR". When a reachable function `f`
//! needs cross-crate MIR, then its MIR may be codegen'd in a downstream crate, and hence items it
//! mentions need to be considered reachable.
//!
//! Furthermore, if a `const`/`const fn` is reachable, then it can return pointers to other items,
//! making those reachable as well. For instance, consider a `const fn` returning a pointer to an
//! otherwise entirely private function: if a downstream crate calls that `const fn` to compute the
//! initial value of a `static`, then it needs to generate a direct reference to this function --
//! i.e., the function is directly reachable from that downstream crate! Hence we have to recurse
//! into `const` and `const fn`.
//!
//! Conversely, reachability *stops* when it hits a monomorphic non-`const` function that we do not
//! want to cross-crate inline. That function will just be codegen'd in this crate, which means the
//! monomorphization collector will consider it a root and then do another graph traversal to
//! codegen everything called by this function -- but that's a very different graph from what we are
//! considering here as at that point, everything is monomorphic.

use hir::def_id::LocalDefIdSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::middle::privacy::{self, Level};
use rustc_middle::mir::interpret::{ConstAllocation, ErrorHandled, GlobalAlloc};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, ExistentialTraitRef, TyCtxt};
use rustc_privacy::DefIdVisitor;
use rustc_session::config::CrateType;
use tracing::debug;

/// Determines whether this item is recursive for reachability. See `is_recursively_reachable_local`
/// below for details.
fn recursively_reachable(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.generics_of(def_id).requires_monomorphization(tcx)
        || tcx.cross_crate_inlinable(def_id)
        || tcx.is_const_fn(def_id)
}

// Information needed while computing reachability.
struct ReachableContext<'tcx> {
    // The type context.
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    // The set of items which must be exported in the linkage sense.
    reachable_symbols: LocalDefIdSet,
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
        let body = self.tcx.hir_body(body);
        self.visit_body(body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let res = match expr.kind {
            hir::ExprKind::Path(ref qpath) => {
                // This covers fn ptr casts but also "non-method" calls.
                Some(self.typeck_results().qpath_res(qpath, expr.hir_id))
            }
            hir::ExprKind::MethodCall(..) => {
                // Method calls don't involve a full "path", so we need to determine the callee
                // based on the receiver type.
                // If this is a method call on a generic type, we might not be able to find the
                // callee. That's why `reachable_set` also adds all potential callees for such
                // calls, i.e. all trait impl items, to the reachable set. So here we only worry
                // about the calls we can identify.
                self.typeck_results()
                    .type_dependent_def(expr.hir_id)
                    .map(|(kind, def_id)| Res::Def(kind, def_id))
            }
            hir::ExprKind::Closure(&hir::Closure { def_id, .. }) => {
                self.reachable_symbols.insert(def_id);
                None
            }
            _ => None,
        };

        if let Some(res) = res {
            self.propagate_item(res);
        }

        intravisit::walk_expr(self, expr)
    }

    fn visit_inline_asm(&mut self, asm: &'tcx hir::InlineAsm<'tcx>, id: hir::HirId) {
        for (op, _) in asm.operands {
            if let hir::InlineAsmOperand::SymStatic { def_id, .. } = op
                && let Some(def_id) = def_id.as_local()
            {
                self.reachable_symbols.insert(def_id);
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

    /// Returns true if the given def ID represents a local item that is recursive for reachability,
    /// i.e. whether everything mentioned in here also needs to be considered reachable.
    ///
    /// There are two reasons why an item may be recursively reachable:
    /// - It needs cross-crate MIR (see the module-level doc comment above).
    /// - It is a `const` or `const fn`. This is *not* because we need the MIR to interpret them
    ///   (MIR for const-eval and MIR for codegen is separate, and MIR for const-eval is always
    ///   encoded). Instead, it is because `const fn` can create `fn()` pointers to other items
    ///   which end up in the evaluated result of the constant and can then be called from other
    ///   crates. Those items must be considered reachable.
    fn is_recursively_reachable_local(&self, def_id: DefId) -> bool {
        let Some(def_id) = def_id.as_local() else {
            return false;
        };

        match self.tcx.hir_node_by_def_id(def_id) {
            Node::Item(item) => match item.kind {
                hir::ItemKind::Fn { .. } => recursively_reachable(self.tcx, def_id.into()),
                _ => false,
            },
            Node::TraitItem(trait_method) => match trait_method.kind {
                hir::TraitItemKind::Const(_, ref default) => default.is_some(),
                hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(_)) => true,
                hir::TraitItemKind::Fn(_, hir::TraitFn::Required(_))
                | hir::TraitItemKind::Type(..) => false,
            },
            Node::ImplItem(impl_item) => match impl_item.kind {
                hir::ImplItemKind::Const(..) => true,
                hir::ImplItemKind::Fn(..) => {
                    recursively_reachable(self.tcx, impl_item.hir_id().owner.to_def_id())
                }
                hir::ImplItemKind::Type(_) => false,
            },
            Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure(..), .. }) => true,
            _ => false,
        }
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    fn propagate(&mut self) {
        let mut scanned = LocalDefIdSet::default();
        while let Some(search_item) = self.worklist.pop() {
            if !scanned.insert(search_item) {
                continue;
            }

            self.propagate_node(&self.tcx.hir_node_by_def_id(search_item), search_item);
        }
    }

    fn propagate_node(&mut self, node: &Node<'tcx>, search_item: LocalDefId) {
        if !self.any_library {
            // If we are building an executable, only explicitly extern
            // types need to be exported.
            let codegen_attrs = if self.tcx.def_kind(search_item).has_codegen_attrs() {
                self.tcx.codegen_fn_attrs(search_item)
            } else {
                CodegenFnAttrs::EMPTY
            };
            let is_extern = codegen_attrs.contains_extern_indicator();
            if is_extern {
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
                    hir::ItemKind::Fn { body, .. } => {
                        if recursively_reachable(self.tcx, item.owner_id.into()) {
                            self.visit_nested_body(body);
                        }
                    }

                    hir::ItemKind::Const(_, _, _, init) => {
                        // Only things actually ending up in the final constant value are reachable
                        // for codegen. Everything else is only needed during const-eval, so even if
                        // const-eval happens in a downstream crate, all they need is
                        // `mir_for_ctfe`.
                        match self.tcx.const_eval_poly_to_alloc(item.owner_id.def_id.into()) {
                            Ok(alloc) => {
                                let alloc = self.tcx.global_alloc(alloc.alloc_id).unwrap_memory();
                                self.propagate_from_alloc(alloc);
                            }
                            // We can't figure out which value the constant will evaluate to. In
                            // lieu of that, we have to consider everything mentioned in the const
                            // initializer reachable, since it *may* end up in the final value.
                            Err(ErrorHandled::TooGeneric(_)) => self.visit_nested_body(init),
                            // If there was an error evaluating the const, nothing can be reachable
                            // via it, and anyway compilation will fail.
                            Err(ErrorHandled::Reported(..)) => {}
                        }
                    }
                    hir::ItemKind::Static(..) => {
                        if let Ok(alloc) = self.tcx.eval_static_initializer(item.owner_id.def_id) {
                            self.propagate_from_alloc(alloc);
                        }
                    }

                    // These are normal, nothing reachable about these
                    // inherently and their children are already in the
                    // worklist, as determined by the privacy pass
                    hir::ItemKind::ExternCrate(..)
                    | hir::ItemKind::Use(..)
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
                    | hir::ItemKind::GlobalAsm { .. } => {}
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
                    if recursively_reachable(self.tcx, impl_item.hir_id().owner.to_def_id()) {
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
            | Node::Crate(_)
            | Node::Synthetic
            | Node::OpaqueTy(..) => {}
            _ => {
                bug!(
                    "found unexpected node kind in worklist: {} ({:?})",
                    self.tcx.hir_id_to_string(self.tcx.local_def_id_to_hir_id(search_item)),
                    node,
                );
            }
        }
    }

    /// Finds things to add to `reachable_symbols` within allocations.
    /// In contrast to visit_nested_body this ignores things that were only needed to evaluate
    /// the allocation.
    fn propagate_from_alloc(&mut self, alloc: ConstAllocation<'tcx>) {
        if !self.any_library {
            return;
        }
        for (_, prov) in alloc.0.provenance().ptrs().iter() {
            match self.tcx.global_alloc(prov.alloc_id()) {
                GlobalAlloc::Static(def_id) => {
                    self.propagate_item(Res::Def(self.tcx.def_kind(def_id), def_id))
                }
                GlobalAlloc::Function { instance, .. } => {
                    // Manually visit to actually see the instance's `DefId`. Type visitors won't see it
                    self.propagate_item(Res::Def(
                        self.tcx.def_kind(instance.def_id()),
                        instance.def_id(),
                    ));
                    self.visit(instance.args);
                }
                GlobalAlloc::VTable(ty, dyn_ty) => {
                    self.visit(ty);
                    // Manually visit to actually see the trait's `DefId`. Type visitors won't see it
                    if let Some(trait_ref) = dyn_ty.principal() {
                        let ExistentialTraitRef { def_id, args, .. } = trait_ref.skip_binder();
                        self.visit_def_id(def_id, "", &"");
                        self.visit(args);
                    }
                }
                GlobalAlloc::TypeId { ty, .. } => self.visit(ty),
                GlobalAlloc::Memory(alloc) => self.propagate_from_alloc(alloc),
            }
        }
    }

    fn propagate_item(&mut self, res: Res) {
        let Res::Def(kind, def_id) = res else { return };
        let Some(def_id) = def_id.as_local() else { return };
        match kind {
            DefKind::Static { nested: true, .. } => {
                // This is the main purpose of this function: add the def_id we find
                // to `reachable_symbols`.
                if self.reachable_symbols.insert(def_id) {
                    if let Ok(alloc) = self.tcx.eval_static_initializer(def_id) {
                        // This cannot cause infinite recursion, because we abort by inserting into the
                        // work list once we hit a normal static. Nested statics, even if they somehow
                        // become recursive, are also not infinitely recursing, because of the
                        // `reachable_symbols` check above.
                        // We still need to protect against stack overflow due to deeply nested statics.
                        ensure_sufficient_stack(|| self.propagate_from_alloc(alloc));
                    }
                }
            }
            // Reachable constants and reachable statics can have their contents inlined
            // into other crates. Mark them as reachable and recurse into their body.
            DefKind::Const | DefKind::AssocConst | DefKind::Static { .. } => {
                self.worklist.push(def_id);
            }
            _ => {
                if self.is_recursively_reachable_local(def_id.to_def_id()) {
                    self.worklist.push(def_id);
                } else {
                    self.reachable_symbols.insert(def_id);
                }
            }
        }
    }
}

impl<'tcx> DefIdVisitor<'tcx> for ReachableContext<'tcx> {
    type Result = ();

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_def_id(
        &mut self,
        def_id: DefId,
        _kind: &str,
        _descr: &dyn std::fmt::Display,
    ) -> Self::Result {
        self.propagate_item(Res::Def(self.tcx.def_kind(def_id), def_id))
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
        // FIXME(nbdd0121): `#[used]` are marked as reachable here so it's picked up by
        // `linked_symbols` in cg_ssa. They won't be exported in binary or cdylib due to their
        // `SymbolExportLevel::Rust` export level but may end up being exported in dylibs.
        || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER)
        || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
}

/// See module-level doc comment above.
fn reachable_set(tcx: TyCtxt<'_>, (): ()) -> LocalDefIdSet {
    let effective_visibilities = &tcx.effective_visibilities(());

    let any_library = tcx.crate_types().iter().any(|ty| {
        *ty == CrateType::Rlib
            || *ty == CrateType::Dylib
            || *ty == CrateType::ProcMacro
            || *ty == CrateType::Sdylib
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
        // As explained above, we have to mark all functions called from reachable
        // `item_might_be_inlined` items as reachable. The issue is, when those functions are
        // generic and call a trait method, we have no idea where that call goes! So, we
        // conservatively mark all trait impl items as reachable.
        // FIXME: One possible strategy for pruning the reachable set is to avoid marking impl
        // items of non-exported traits (or maybe all local traits?) unless their respective
        // trait items are used from inlinable code through method call syntax or UFCS, or their
        // trait is a lang item.
        // (But if you implement this, don't forget to take into account that vtables can also
        // make trait methods reachable!)
        let crate_items = tcx.hir_crate_items(());

        for id in crate_items.free_items() {
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

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { reachable_set, ..*providers };
}
