// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.

use crate::hir::{CodegenFnAttrs, CodegenFnAttrFlags};
use crate::hir::Node;
use crate::hir::def::{Res, DefKind};
use crate::hir::def_id::{DefId, CrateNum};
use rustc_data_structures::sync::Lrc;
use crate::ty::{self, TyCtxt};
use crate::ty::query::Providers;
use crate::middle::privacy;
use crate::session::config;
use crate::util::nodemap::{HirIdSet, FxHashSet};

use rustc_target::spec::abi::Abi;
use rustc_macros::HashStable;
use crate::hir;
use crate::hir::def_id::LOCAL_CRATE;
use crate::hir::intravisit::{Visitor, NestedVisitorMap};
use crate::hir::itemlikevisit::ItemLikeVisitor;
use crate::hir::intravisit;

// Returns true if the given item must be inlined because it may be
// monomorphized or it was marked with `#[inline]`. This will only return
// true for functions.
fn item_might_be_inlined(tcx: TyCtxt<'tcx>, item: &hir::Item, attrs: CodegenFnAttrs) -> bool {
    if attrs.requests_inline() {
        return true
    }

    match item.node {
        hir::ItemKind::Impl(..) |
        hir::ItemKind::Fn(..) => {
            let generics = tcx.generics_of(tcx.hir().local_def_id_from_hir_id(item.hir_id));
            generics.requires_monomorphization(tcx)
        }
        _ => false,
    }
}

fn method_might_be_inlined(
    tcx: TyCtxt<'_>,
    impl_item: &hir::ImplItem,
    impl_src: DefId,
) -> bool {
    let codegen_fn_attrs = tcx.codegen_fn_attrs(impl_item.hir_id.owner_def_id());
    let generics = tcx.generics_of(tcx.hir().local_def_id_from_hir_id(impl_item.hir_id));
    if codegen_fn_attrs.requests_inline() || generics.requires_monomorphization(tcx) {
        return true
    }
    if let Some(impl_hir_id) = tcx.hir().as_local_hir_id(impl_src) {
        match tcx.hir().find(impl_hir_id) {
            Some(Node::Item(item)) =>
                item_might_be_inlined(tcx, &item, codegen_fn_attrs),
            Some(..) | None =>
                span_bug!(impl_item.span, "impl did is not an item")
        }
    } else {
        span_bug!(impl_item.span, "found a foreign impl as a parent of a local method")
    }
}

// Information needed while computing reachability.
struct ReachableContext<'a, 'tcx> {
    // The type context.
    tcx: TyCtxt<'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    // The set of items which must be exported in the linkage sense.
    reachable_symbols: HirIdSet,
    // A worklist of item IDs. Each item ID in this worklist will be inlined
    // and will be scanned for further references.
    worklist: Vec<hir::HirId>,
    // Whether any output of this compilation is a library
    any_library: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for ReachableContext<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_tables = self.tables;
        self.tables = self.tcx.body_tables(body);
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.tables = old_tables;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        let res = match expr.node {
            hir::ExprKind::Path(ref qpath) => {
                Some(self.tables.qpath_res(qpath, expr.hir_id))
            }
            hir::ExprKind::MethodCall(..) => {
                self.tables.type_dependent_def(expr.hir_id)
                    .map(|(kind, def_id)| Res::Def(kind, def_id))
            }
            _ => None
        };

        match res {
            Some(Res::Local(hir_id)) => {
                self.reachable_symbols.insert(hir_id);
            }
            Some(res) => {
                if let Some((hir_id, def_id)) = res.opt_def_id().and_then(|def_id| {
                    self.tcx.hir().as_local_hir_id(def_id).map(|hir_id| (hir_id, def_id))
                }) {
                    if self.def_id_represents_local_inlined_item(def_id) {
                        self.worklist.push(hir_id);
                    } else {
                        match res {
                            // If this path leads to a constant, then we need to
                            // recurse into the constant to continue finding
                            // items that are reachable.
                            Res::Def(DefKind::Const, _) | Res::Def(DefKind::AssocConst, _) => {
                                self.worklist.push(hir_id);
                            }

                            // If this wasn't a static, then the destination is
                            // surely reachable.
                            _ => {
                                self.reachable_symbols.insert(hir_id);
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr)
    }
}

impl<'a, 'tcx> ReachableContext<'a, 'tcx> {
    // Returns true if the given def ID represents a local item that is
    // eligible for inlining and false otherwise.
    fn def_id_represents_local_inlined_item(&self, def_id: DefId) -> bool {
        let hir_id = match self.tcx.hir().as_local_hir_id(def_id) {
            Some(hir_id) => hir_id,
            None => { return false; }
        };

        match self.tcx.hir().find(hir_id) {
            Some(Node::Item(item)) => {
                match item.node {
                    hir::ItemKind::Fn(..) =>
                        item_might_be_inlined(self.tcx, &item, self.tcx.codegen_fn_attrs(def_id)),
                    _ => false,
                }
            }
            Some(Node::TraitItem(trait_method)) => {
                match trait_method.node {
                    hir::TraitItemKind::Const(_, ref default) => default.is_some(),
                    hir::TraitItemKind::Method(_, hir::TraitMethod::Provided(_)) => true,
                    hir::TraitItemKind::Method(_, hir::TraitMethod::Required(_)) |
                    hir::TraitItemKind::Type(..) => false,
                }
            }
            Some(Node::ImplItem(impl_item)) => {
                match impl_item.node {
                    hir::ImplItemKind::Const(..) => true,
                    hir::ImplItemKind::Method(..) => {
                        let attrs = self.tcx.codegen_fn_attrs(def_id);
                        let generics = self.tcx.generics_of(def_id);
                        if generics.requires_monomorphization(self.tcx) || attrs.requests_inline() {
                            true
                        } else {
                            let impl_did = self.tcx
                                               .hir()
                                               .get_parent_did(hir_id);
                            // Check the impl. If the generics on the self
                            // type of the impl require inlining, this method
                            // does too.
                            let impl_hir_id = self.tcx.hir().as_local_hir_id(impl_did).unwrap();
                            match self.tcx.hir().expect_item(impl_hir_id).node {
                                hir::ItemKind::Impl(..) => {
                                    let generics = self.tcx.generics_of(impl_did);
                                    generics.requires_monomorphization(self.tcx)
                                }
                                _ => false
                            }
                        }
                    }
                    hir::ImplItemKind::Existential(..) |
                    hir::ImplItemKind::Type(_) => false,
                }
            }
            Some(_) => false,
            None => false   // This will happen for default methods.
        }
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    fn propagate(&mut self) {
        let mut scanned = FxHashSet::default();
        while let Some(search_item) = self.worklist.pop() {
            if !scanned.insert(search_item) {
                continue
            }

            if let Some(ref item) = self.tcx.hir().find(search_item) {
                self.propagate_node(item, search_item);
            }
        }
    }

    fn propagate_node(&mut self, node: &Node<'tcx>,
                      search_item: hir::HirId) {
        if !self.any_library {
            // If we are building an executable, only explicitly extern
            // types need to be exported.
            if let Node::Item(item) = *node {
                let reachable = if let hir::ItemKind::Fn(_, header, ..) = item.node {
                    header.abi != Abi::Rust
                } else {
                    false
                };
                let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                let codegen_attrs = self.tcx.codegen_fn_attrs(def_id);
                let is_extern = codegen_attrs.contains_extern_indicator();
                let std_internal = codegen_attrs.flags.contains(
                    CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL);
                if reachable || is_extern || std_internal {
                    self.reachable_symbols.insert(search_item);
                }
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
                match item.node {
                    hir::ItemKind::Fn(.., body) => {
                        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                        if item_might_be_inlined(self.tcx,
                                                 &item,
                                                 self.tcx.codegen_fn_attrs(def_id)) {
                            self.visit_nested_body(body);
                        }
                    }

                    // Reachable constants will be inlined into other crates
                    // unconditionally, so we need to make sure that their
                    // contents are also reachable.
                    hir::ItemKind::Const(_, init) => {
                        self.visit_nested_body(init);
                    }

                    // These are normal, nothing reachable about these
                    // inherently and their children are already in the
                    // worklist, as determined by the privacy pass
                    hir::ItemKind::ExternCrate(_) |
                    hir::ItemKind::Use(..) |
                    hir::ItemKind::Existential(..) |
                    hir::ItemKind::Ty(..) |
                    hir::ItemKind::Static(..) |
                    hir::ItemKind::Mod(..) |
                    hir::ItemKind::ForeignMod(..) |
                    hir::ItemKind::Impl(..) |
                    hir::ItemKind::Trait(..) |
                    hir::ItemKind::TraitAlias(..) |
                    hir::ItemKind::Struct(..) |
                    hir::ItemKind::Enum(..) |
                    hir::ItemKind::Union(..) |
                    hir::ItemKind::GlobalAsm(..) => {}
                }
            }
            Node::TraitItem(trait_method) => {
                match trait_method.node {
                    hir::TraitItemKind::Const(_, None) |
                    hir::TraitItemKind::Method(_, hir::TraitMethod::Required(_)) => {
                        // Keep going, nothing to get exported
                    }
                    hir::TraitItemKind::Const(_, Some(body_id)) |
                    hir::TraitItemKind::Method(_, hir::TraitMethod::Provided(body_id)) => {
                        self.visit_nested_body(body_id);
                    }
                    hir::TraitItemKind::Type(..) => {}
                }
            }
            Node::ImplItem(impl_item) => {
                match impl_item.node {
                    hir::ImplItemKind::Const(_, body) => {
                        self.visit_nested_body(body);
                    }
                    hir::ImplItemKind::Method(_, body) => {
                        let did = self.tcx.hir().get_parent_did(search_item);
                        if method_might_be_inlined(self.tcx, impl_item, did) {
                            self.visit_nested_body(body)
                        }
                    }
                    hir::ImplItemKind::Existential(..) |
                    hir::ImplItemKind::Type(_) => {}
                }
            }
            Node::Expr(&hir::Expr { node: hir::ExprKind::Closure(.., body, _, _), .. }) => {
                self.visit_nested_body(body);
            }
            // Nothing to recurse on for these
            Node::ForeignItem(_) |
            Node::Variant(_) |
            Node::Ctor(..) |
            Node::Field(_) |
            Node::Ty(_) |
            Node::MacroDef(_) => {}
            _ => {
                bug!(
                    "found unexpected node kind in worklist: {} ({:?})",
                    self.tcx.hir().node_to_string(search_item),
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
    worklist: &'a mut Vec<hir::HirId>,
}

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for CollectPrivateImplItemsVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        // Anything which has custom linkage gets thrown on the worklist no
        // matter where it is in the crate, along with "special std symbols"
        // which are currently akin to allocator symbols.
        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
        let codegen_attrs = self.tcx.codegen_fn_attrs(def_id);
        if codegen_attrs.contains_extern_indicator() ||
            codegen_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL) {
            self.worklist.push(item.hir_id);
        }

        // We need only trait impls here, not inherent impls, and only non-exported ones
        if let hir::ItemKind::Impl(.., Some(ref trait_ref), _, ref impl_item_refs) = item.node {
            if !self.access_levels.is_reachable(item.hir_id) {
                self.worklist.extend(impl_item_refs.iter().map(|ii_ref| ii_ref.id.hir_id));

                let trait_def_id = match trait_ref.path.res {
                    Res::Def(DefKind::Trait, def_id) => def_id,
                    _ => unreachable!()
                };

                if !trait_def_id.is_local() {
                    return
                }

                let provided_trait_methods = self.tcx.provided_trait_methods(trait_def_id);
                self.worklist.reserve(provided_trait_methods.len());
                for default_method in provided_trait_methods {
                    let hir_id = self.tcx
                                     .hir()
                                     .as_local_hir_id(default_method.def_id)
                                     .unwrap();
                    self.worklist.push(hir_id);
                }
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
        // processed in visit_item above
    }
}

// We introduce a new-type here, so we can have a specialized HashStable
// implementation for it.
#[derive(Clone, HashStable)]
pub struct ReachableSet(pub Lrc<HirIdSet>);

fn reachable_set(tcx: TyCtxt<'_>, crate_num: CrateNum) -> ReachableSet {
    debug_assert!(crate_num == LOCAL_CRATE);

    let access_levels = &tcx.privacy_access_levels(LOCAL_CRATE);

    let any_library = tcx.sess.crate_types.borrow().iter().any(|ty| {
        *ty == config::CrateType::Rlib || *ty == config::CrateType::Dylib ||
        *ty == config::CrateType::ProcMacro
    });
    let mut reachable_context = ReachableContext {
        tcx,
        tables: &ty::TypeckTables::empty(None),
        reachable_symbols: Default::default(),
        worklist: Vec::new(),
        any_library,
    };

    // Step 1: Seed the worklist with all nodes which were found to be public as
    //         a result of the privacy pass along with all local lang items and impl items.
    //         If other crates link to us, they're going to expect to be able to
    //         use the lang items, so we need to be sure to mark them as
    //         exported.
    reachable_context.worklist.extend(
        access_levels.map.iter().map(|(id, _)| *id));
    for item in tcx.lang_items().items().iter() {
        if let Some(did) = *item {
            if let Some(hir_id) = tcx.hir().as_local_hir_id(did) {
                reachable_context.worklist.push(hir_id);
            }
        }
    }
    {
        let mut collect_private_impl_items = CollectPrivateImplItemsVisitor {
            tcx,
            access_levels,
            worklist: &mut reachable_context.worklist,
        };
        tcx.hir().krate().visit_all_item_likes(&mut collect_private_impl_items);
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    reachable_context.propagate();

    debug!("Inline reachability shows: {:?}", reachable_context.reachable_symbols);

    // Return the set of reachable symbols.
    ReachableSet(Lrc::new(reachable_context.reachable_symbols))
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        reachable_set,
        ..*providers
    };
}
