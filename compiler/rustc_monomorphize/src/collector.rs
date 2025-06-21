//! Mono Item Collection
//! ====================
//!
//! This module is responsible for discovering all items that will contribute
//! to code generation of the crate. The important part here is that it not only
//! needs to find syntax-level items (functions, structs, etc) but also all
//! their monomorphized instantiations. Every non-generic, non-const function
//! maps to one LLVM artifact. Every generic function can produce
//! from zero to N artifacts, depending on the sets of type arguments it
//! is instantiated with.
//! This also applies to generic items from other crates: A generic definition
//! in crate X might produce monomorphizations that are compiled into crate Y.
//! We also have to collect these here.
//!
//! The following kinds of "mono items" are handled here:
//!
//! - Functions
//! - Methods
//! - Closures
//! - Statics
//! - Drop glue
//!
//! The following things also result in LLVM artifacts, but are not collected
//! here, since we instantiate them locally on demand when needed in a given
//! codegen unit:
//!
//! - Constants
//! - VTables
//! - Object Shims
//!
//! The main entry point is `collect_crate_mono_items`, at the bottom of this file.
//!
//! General Algorithm
//! -----------------
//! Let's define some terms first:
//!
//! - A "mono item" is something that results in a function or global in
//!   the LLVM IR of a codegen unit. Mono items do not stand on their
//!   own, they can use other mono items. For example, if function
//!   `foo()` calls function `bar()` then the mono item for `foo()`
//!   uses the mono item for function `bar()`. In general, the
//!   definition for mono item A using a mono item B is that
//!   the LLVM artifact produced for A uses the LLVM artifact produced
//!   for B.
//!
//! - Mono items and the uses between them form a directed graph,
//!   where the mono items are the nodes and uses form the edges.
//!   Let's call this graph the "mono item graph".
//!
//! - The mono item graph for a program contains all mono items
//!   that are needed in order to produce the complete LLVM IR of the program.
//!
//! The purpose of the algorithm implemented in this module is to build the
//! mono item graph for the current crate. It runs in two phases:
//!
//! 1. Discover the roots of the graph by traversing the HIR of the crate.
//! 2. Starting from the roots, find uses by inspecting the MIR
//!    representation of the item corresponding to a given node, until no more
//!    new nodes are found.
//!
//! ### Discovering roots
//! The roots of the mono item graph correspond to the public non-generic
//! syntactic items in the source code. We find them by walking the HIR of the
//! crate, and whenever we hit upon a public function, method, or static item,
//! we create a mono item consisting of the items DefId and, since we only
//! consider non-generic items, an empty type-parameters set. (In eager
//! collection mode, during incremental compilation, all non-generic functions
//! are considered as roots, as well as when the `-Clink-dead-code` option is
//! specified. Functions marked `#[no_mangle]` and functions called by inlinable
//! functions also always act as roots.)
//!
//! ### Finding uses
//! Given a mono item node, we can discover uses by inspecting its MIR. We walk
//! the MIR to find other mono items used by each mono item. Since the mono
//! item we are currently at is always monomorphic, we also know the concrete
//! type arguments of its used mono items. The specific forms a use can take in
//! MIR are quite diverse. Here is an overview:
//!
//! #### Calling Functions/Methods
//! The most obvious way for one mono item to use another is a
//! function or method call (represented by a CALL terminator in MIR). But
//! calls are not the only thing that might introduce a use between two
//! function mono items, and as we will see below, they are just a
//! specialization of the form described next, and consequently will not get any
//! special treatment in the algorithm.
//!
//! #### Taking a reference to a function or method
//! A function does not need to actually be called in order to be used by
//! another function. It suffices to just take a reference in order to introduce
//! an edge. Consider the following example:
//!
//! ```
//! # use core::fmt::Display;
//! fn print_val<T: Display>(x: T) {
//!     println!("{}", x);
//! }
//!
//! fn call_fn(f: &dyn Fn(i32), x: i32) {
//!     f(x);
//! }
//!
//! fn main() {
//!     let print_i32 = print_val::<i32>;
//!     call_fn(&print_i32, 0);
//! }
//! ```
//! The MIR of none of these functions will contain an explicit call to
//! `print_val::<i32>`. Nonetheless, in order to mono this program, we need
//! an instance of this function. Thus, whenever we encounter a function or
//! method in operand position, we treat it as a use of the current
//! mono item. Calls are just a special case of that.
//!
//! #### Drop glue
//! Drop glue mono items are introduced by MIR drop-statements. The
//! generated mono item will have additional drop-glue item uses if the
//! type to be dropped contains nested values that also need to be dropped. It
//! might also have a function item use for the explicit `Drop::drop`
//! implementation of its type.
//!
//! #### Unsizing Casts
//! A subtle way of introducing use edges is by casting to a trait object.
//! Since the resulting wide-pointer contains a reference to a vtable, we need to
//! instantiate all dyn-compatible methods of the trait, as we need to store
//! pointers to these functions even if they never get called anywhere. This can
//! be seen as a special case of taking a function reference.
//!
//!
//! Interaction with Cross-Crate Inlining
//! -------------------------------------
//! The binary of a crate will not only contain machine code for the items
//! defined in the source code of that crate. It will also contain monomorphic
//! instantiations of any extern generic functions and of functions marked with
//! `#[inline]`.
//! The collection algorithm handles this more or less mono. If it is
//! about to create a mono item for something with an external `DefId`,
//! it will take a look if the MIR for that item is available, and if so just
//! proceed normally. If the MIR is not available, it assumes that the item is
//! just linked to and no node is created; which is exactly what we want, since
//! no machine code should be generated in the current crate for such an item.
//!
//! Eager and Lazy Collection Strategy
//! ----------------------------------
//! Mono item collection can be performed with one of two strategies:
//!
//! - Lazy strategy means that items will only be instantiated when actually
//!   used. The goal is to produce the least amount of machine code
//!   possible.
//!
//! - Eager strategy is meant to be used in conjunction with incremental compilation
//!   where a stable set of mono items is more important than a minimal
//!   one. Thus, eager strategy will instantiate drop-glue for every drop-able type
//!   in the crate, even if no drop call for that type exists (yet). It will
//!   also instantiate default implementations of trait methods, something that
//!   otherwise is only done on demand.
//!
//! Collection-time const evaluation and "mentioned" items
//! ------------------------------------------------------
//!
//! One important role of collection is to evaluate all constants that are used by all the items
//! which are being collected. Codegen can then rely on only encountering constants that evaluate
//! successfully, and if a constant fails to evaluate, the collector has much better context to be
//! able to show where this constant comes up.
//!
//! However, the exact set of "used" items (collected as described above), and therefore the exact
//! set of used constants, can depend on optimizations. Optimizing away dead code may optimize away
//! a function call that uses a failing constant, so an unoptimized build may fail where an
//! optimized build succeeds. This is undesirable.
//!
//! To avoid this, the collector has the concept of "mentioned" items. Some time during the MIR
//! pipeline, before any optimization-level-dependent optimizations, we compute a list of all items
//! that syntactically appear in the code. These are considered "mentioned", and even if they are in
//! dead code and get optimized away (which makes them no longer "used"), they are still
//! "mentioned". For every used item, the collector ensures that all mentioned items, recursively,
//! do not use a failing constant. This is reflected via the [`CollectionMode`], which determines
//! whether we are visiting a used item or merely a mentioned item.
//!
//! The collector and "mentioned items" gathering (which lives in `rustc_mir_transform::mentioned_items`)
//! need to stay in sync in the following sense:
//!
//! - For every item that the collector gather that could eventually lead to build failure (most
//!   likely due to containing a constant that fails to evaluate), a corresponding mentioned item
//!   must be added. This should use the exact same strategy as the ecollector to make sure they are
//!   in sync. However, while the collector works on monomorphized types, mentioned items are
//!   collected on generic MIR -- so any time the collector checks for a particular type (such as
//!   `ty::FnDef`), we have to just onconditionally add this as a mentioned item.
//! - In `visit_mentioned_item`, we then do with that mentioned item exactly what the collector
//!   would have done during regular MIR visiting. Basically you can think of the collector having
//!   two stages, a pre-monomorphization stage and a post-monomorphization stage (usually quite
//!   literally separated by a call to `self.monomorphize`); the pre-monomorphizationn stage is
//!   duplicated in mentioned items gathering and the post-monomorphization stage is duplicated in
//!   `visit_mentioned_item`.
//! - Finally, as a performance optimization, the collector should fill `used_mentioned_item` during
//!   its MIR traversal with exactly what mentioned item gathering would have added in the same
//!   situation. This detects mentioned items that have *not* been optimized away and hence don't
//!   need a dedicated traversal.
//!
//! Open Issues
//! -----------
//! Some things are not yet fully implemented in the current version of this
//! module.
//!
//! ### Const Fns
//! Ideally, no mono item should be generated for const fns unless there
//! is a call to them that cannot be evaluated at compile time. At the moment
//! this is not implemented however: a mono item will be produced
//! regardless of whether it is actually needed or not.

use std::cell::OnceCell;
use std::path::PathBuf;

use rustc_attr_data_structures::InlineAttr;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::{MTLock, par_for_each_in};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::{AllocId, ErrorHandled, GlobalAlloc, Scalar};
use rustc_middle::mir::mono::{CollectionMode, InstantiationMode, MonoItem};
use rustc_middle::mir::visit::Visitor as MirVisitor;
use rustc_middle::mir::{self, Location, MentionedItem, traversal};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::adjustment::{CustomCoerceUnsized, PointerCoercion};
use rustc_middle::ty::layout::ValidityRequirement;
use rustc_middle::ty::print::{shrunk_instance_name, with_no_trimmed_paths};
use rustc_middle::ty::{
    self, GenericArgs, GenericParamDefKind, Instance, InstanceKind, Ty, TyCtxt, TypeFoldable,
    TypeVisitableExt, VtblEntry,
};
use rustc_middle::util::Providers;
use rustc_middle::{bug, span_bug};
use rustc_session::Limit;
use rustc_session::config::{DebugInfo, EntryFnType};
use rustc_span::source_map::{Spanned, dummy_spanned, respan};
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument, trace};

use crate::errors::{self, EncounteredErrorWhileInstantiating, NoOptimizedMir, RecursionLimit};

#[derive(PartialEq)]
pub(crate) enum MonoItemCollectionStrategy {
    Eager,
    Lazy,
}

/// The state that is shared across the concurrent threads that are doing collection.
struct SharedState<'tcx> {
    /// Items that have been or are currently being recursively collected.
    visited: MTLock<UnordSet<MonoItem<'tcx>>>,
    /// Items that have been or are currently being recursively treated as "mentioned", i.e., their
    /// consts are evaluated but nothing is added to the collection.
    mentioned: MTLock<UnordSet<MonoItem<'tcx>>>,
    /// Which items are being used where, for better errors.
    usage_map: MTLock<UsageMap<'tcx>>,
}

pub(crate) struct UsageMap<'tcx> {
    // Maps every mono item to the mono items used by it.
    pub used_map: UnordMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>>,

    // Maps every mono item to the mono items that use it.
    user_map: UnordMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>>,
}

impl<'tcx> UsageMap<'tcx> {
    fn new() -> UsageMap<'tcx> {
        UsageMap { used_map: Default::default(), user_map: Default::default() }
    }

    fn record_used<'a>(&mut self, user_item: MonoItem<'tcx>, used_items: &'a MonoItems<'tcx>)
    where
        'tcx: 'a,
    {
        for used_item in used_items.items() {
            self.user_map.entry(used_item).or_default().push(user_item);
        }

        assert!(self.used_map.insert(user_item, used_items.items().collect()).is_none());
    }

    pub(crate) fn get_user_items(&self, item: MonoItem<'tcx>) -> &[MonoItem<'tcx>] {
        self.user_map.get(&item).map(|items| items.as_slice()).unwrap_or(&[])
    }

    /// Internally iterate over all inlined items used by `item`.
    pub(crate) fn for_each_inlined_used_item<F>(
        &self,
        tcx: TyCtxt<'tcx>,
        item: MonoItem<'tcx>,
        mut f: F,
    ) where
        F: FnMut(MonoItem<'tcx>),
    {
        let used_items = self.used_map.get(&item).unwrap();
        for used_item in used_items.iter() {
            let is_inlined = used_item.instantiation_mode(tcx) == InstantiationMode::LocalCopy;
            if is_inlined {
                f(*used_item);
            }
        }
    }
}

struct MonoItems<'tcx> {
    // We want a set of MonoItem + Span where trying to re-insert a MonoItem with a different Span
    // is ignored. Map does that, but it looks odd.
    items: FxIndexMap<MonoItem<'tcx>, Span>,
}

impl<'tcx> MonoItems<'tcx> {
    fn new() -> Self {
        Self { items: FxIndexMap::default() }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn push(&mut self, item: Spanned<MonoItem<'tcx>>) {
        // Insert only if the entry does not exist. A normal insert would stomp the first span that
        // got inserted.
        self.items.entry(item.node).or_insert(item.span);
    }

    fn items(&self) -> impl Iterator<Item = MonoItem<'tcx>> {
        self.items.keys().cloned()
    }
}

impl<'tcx> IntoIterator for MonoItems<'tcx> {
    type Item = Spanned<MonoItem<'tcx>>;
    type IntoIter = impl Iterator<Item = Spanned<MonoItem<'tcx>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().map(|(item, span)| respan(span, item))
    }
}

impl<'tcx> Extend<Spanned<MonoItem<'tcx>>> for MonoItems<'tcx> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Spanned<MonoItem<'tcx>>>,
    {
        for item in iter {
            self.push(item)
        }
    }
}

fn collect_items_root<'tcx>(
    tcx: TyCtxt<'tcx>,
    starting_item: Spanned<MonoItem<'tcx>>,
    state: &SharedState<'tcx>,
    recursion_limit: Limit,
) {
    if !state.visited.lock_mut().insert(starting_item.node) {
        // We've been here already, no need to search again.
        return;
    }
    let mut recursion_depths = DefIdMap::default();
    collect_items_rec(
        tcx,
        starting_item,
        state,
        &mut recursion_depths,
        recursion_limit,
        CollectionMode::UsedItems,
    );
}

/// Collect all monomorphized items reachable from `starting_point`, and emit a note diagnostic if a
/// post-monomorphization error is encountered during a collection step.
///
/// `mode` determined whether we are scanning for [used items][CollectionMode::UsedItems]
/// or [mentioned items][CollectionMode::MentionedItems].
#[instrument(skip(tcx, state, recursion_depths, recursion_limit), level = "debug")]
fn collect_items_rec<'tcx>(
    tcx: TyCtxt<'tcx>,
    starting_item: Spanned<MonoItem<'tcx>>,
    state: &SharedState<'tcx>,
    recursion_depths: &mut DefIdMap<usize>,
    recursion_limit: Limit,
    mode: CollectionMode,
) {
    let mut used_items = MonoItems::new();
    let mut mentioned_items = MonoItems::new();
    let recursion_depth_reset;

    // Post-monomorphization errors MVP
    //
    // We can encounter errors while monomorphizing an item, but we don't have a good way of
    // showing a complete stack of spans ultimately leading to collecting the erroneous one yet.
    // (It's also currently unclear exactly which diagnostics and information would be interesting
    // to report in such cases)
    //
    // This leads to suboptimal error reporting: a post-monomorphization error (PME) will be
    // shown with just a spanned piece of code causing the error, without information on where
    // it was called from. This is especially obscure if the erroneous mono item is in a
    // dependency. See for example issue #85155, where, before minimization, a PME happened two
    // crates downstream from libcore's stdarch, without a way to know which dependency was the
    // cause.
    //
    // If such an error occurs in the current crate, its span will be enough to locate the
    // source. If the cause is in another crate, the goal here is to quickly locate which mono
    // item in the current crate is ultimately responsible for causing the error.
    //
    // To give at least _some_ context to the user: while collecting mono items, we check the
    // error count. If it has changed, a PME occurred, and we trigger some diagnostics about the
    // current step of mono items collection.
    //
    // FIXME: don't rely on global state, instead bubble up errors. Note: this is very hard to do.
    let error_count = tcx.dcx().err_count();

    // In `mentioned_items` we collect items that were mentioned in this MIR but possibly do not
    // need to be monomorphized. This is done to ensure that optimizing away function calls does not
    // hide const-eval errors that those calls would otherwise have triggered.
    match starting_item.node {
        MonoItem::Static(def_id) => {
            recursion_depth_reset = None;

            // Statics always get evaluated (which is possible because they can't be generic), so for
            // `MentionedItems` collection there's nothing to do here.
            if mode == CollectionMode::UsedItems {
                let instance = Instance::mono(tcx, def_id);

                // Sanity check whether this ended up being collected accidentally
                debug_assert!(tcx.should_codegen_locally(instance));

                let DefKind::Static { nested, .. } = tcx.def_kind(def_id) else { bug!() };
                // Nested statics have no type.
                if !nested {
                    let ty = instance.ty(tcx, ty::TypingEnv::fully_monomorphized());
                    visit_drop_use(tcx, ty, true, starting_item.span, &mut used_items);
                }

                if let Ok(alloc) = tcx.eval_static_initializer(def_id) {
                    for &prov in alloc.inner().provenance().ptrs().values() {
                        collect_alloc(tcx, prov.alloc_id(), &mut used_items);
                    }
                }

                if tcx.needs_thread_local_shim(def_id) {
                    used_items.push(respan(
                        starting_item.span,
                        MonoItem::Fn(Instance {
                            def: InstanceKind::ThreadLocalShim(def_id),
                            args: GenericArgs::empty(),
                        }),
                    ));
                }
            }

            // mentioned_items stays empty since there's no codegen for statics. statics don't get
            // optimized, and if they did then the const-eval interpreter would have to worry about
            // mentioned_items.
        }
        MonoItem::Fn(instance) => {
            // Sanity check whether this ended up being collected accidentally
            debug_assert!(tcx.should_codegen_locally(instance));

            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(
                tcx,
                instance,
                starting_item.span,
                recursion_depths,
                recursion_limit,
            ));

            rustc_data_structures::stack::ensure_sufficient_stack(|| {
                let (used, mentioned) = tcx.items_of_instance((instance, mode));
                used_items.extend(used.into_iter().copied());
                mentioned_items.extend(mentioned.into_iter().copied());
            });
        }
        MonoItem::GlobalAsm(item_id) => {
            assert!(
                mode == CollectionMode::UsedItems,
                "should never encounter global_asm when collecting mentioned items"
            );
            recursion_depth_reset = None;

            let item = tcx.hir_item(item_id);
            if let hir::ItemKind::GlobalAsm { asm, .. } = item.kind {
                for (op, op_sp) in asm.operands {
                    match *op {
                        hir::InlineAsmOperand::Const { .. } => {
                            // Only constants which resolve to a plain integer
                            // are supported. Therefore the value should not
                            // depend on any other items.
                        }
                        hir::InlineAsmOperand::SymFn { expr } => {
                            let fn_ty = tcx.typeck(item_id.owner_id).expr_ty(expr);
                            visit_fn_use(tcx, fn_ty, false, *op_sp, &mut used_items);
                        }
                        hir::InlineAsmOperand::SymStatic { path: _, def_id } => {
                            let instance = Instance::mono(tcx, def_id);
                            if tcx.should_codegen_locally(instance) {
                                trace!("collecting static {:?}", def_id);
                                used_items.push(dummy_spanned(MonoItem::Static(def_id)));
                            }
                        }
                        hir::InlineAsmOperand::In { .. }
                        | hir::InlineAsmOperand::Out { .. }
                        | hir::InlineAsmOperand::InOut { .. }
                        | hir::InlineAsmOperand::SplitInOut { .. }
                        | hir::InlineAsmOperand::Label { .. } => {
                            span_bug!(*op_sp, "invalid operand type for global_asm!")
                        }
                    }
                }
            } else {
                span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
            }

            // mention_items stays empty as nothing gets optimized here.
        }
    };

    // Check for PMEs and emit a diagnostic if one happened. To try to show relevant edges of the
    // mono item graph.
    if tcx.dcx().err_count() > error_count
        && starting_item.node.is_generic_fn()
        && starting_item.node.is_user_defined()
    {
        let formatted_item = with_no_trimmed_paths!(starting_item.node.to_string());
        tcx.dcx().emit_note(EncounteredErrorWhileInstantiating {
            span: starting_item.span,
            formatted_item,
        });
    }
    // Only updating `usage_map` for used items as otherwise we may be inserting the same item
    // multiple times (if it is first 'mentioned' and then later actuall used), and the usage map
    // logic does not like that.
    // This is part of the output of collection and hence only relevant for "used" items.
    // ("Mentioned" items are only considered internally during collection.)
    if mode == CollectionMode::UsedItems {
        state.usage_map.lock_mut().record_used(starting_item.node, &used_items);
    }

    {
        let mut visited = OnceCell::default();
        if mode == CollectionMode::UsedItems {
            used_items
                .items
                .retain(|k, _| visited.get_mut_or_init(|| state.visited.lock_mut()).insert(*k));
        }

        let mut mentioned = OnceCell::default();
        mentioned_items.items.retain(|k, _| {
            !visited.get_or_init(|| state.visited.lock()).contains(k)
                && mentioned.get_mut_or_init(|| state.mentioned.lock_mut()).insert(*k)
        });
    }
    if mode == CollectionMode::MentionedItems {
        assert!(used_items.is_empty(), "'mentioned' collection should never encounter used items");
    } else {
        for used_item in used_items {
            collect_items_rec(
                tcx,
                used_item,
                state,
                recursion_depths,
                recursion_limit,
                CollectionMode::UsedItems,
            );
        }
    }

    // Walk over mentioned items *after* used items, so that if an item is both mentioned and used then
    // the loop above has fully collected it, so this loop will skip it.
    for mentioned_item in mentioned_items {
        collect_items_rec(
            tcx,
            mentioned_item,
            state,
            recursion_depths,
            recursion_limit,
            CollectionMode::MentionedItems,
        );
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }
}

fn check_recursion_limit<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    span: Span,
    recursion_depths: &mut DefIdMap<usize>,
    recursion_limit: Limit,
) -> (DefId, usize) {
    let def_id = instance.def_id();
    let recursion_depth = recursion_depths.get(&def_id).cloned().unwrap_or(0);
    debug!(" => recursion depth={}", recursion_depth);

    let adjusted_recursion_depth = if tcx.is_lang_item(def_id, LangItem::DropInPlace) {
        // HACK: drop_in_place creates tight monomorphization loops. Give
        // it more margin.
        recursion_depth / 4
    } else {
        recursion_depth
    };

    // Code that needs to instantiate the same function recursively
    // more than the recursion limit is assumed to be causing an
    // infinite expansion.
    if !recursion_limit.value_within_limit(adjusted_recursion_depth) {
        let def_span = tcx.def_span(def_id);
        let def_path_str = tcx.def_path_str(def_id);
        let (shrunk, written_to_path) = shrunk_instance_name(tcx, instance);
        let mut path = PathBuf::new();
        let was_written = if let Some(written_to_path) = written_to_path {
            path = written_to_path;
            true
        } else {
            false
        };
        tcx.dcx().emit_fatal(RecursionLimit {
            span,
            shrunk,
            def_span,
            def_path_str,
            was_written,
            path,
        });
    }

    recursion_depths.insert(def_id, recursion_depth + 1);

    (def_id, recursion_depth)
}

struct MirUsedCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    used_items: &'a mut MonoItems<'tcx>,
    /// See the comment in `collect_items_of_instance` for the purpose of this set.
    /// Note that this contains *not-monomorphized* items!
    used_mentioned_items: &'a mut UnordSet<MentionedItem<'tcx>>,
    instance: Instance<'tcx>,
}

impl<'a, 'tcx> MirUsedCollector<'a, 'tcx> {
    fn monomorphize<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        trace!("monomorphize: self.instance={:?}", self.instance);
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(value),
        )
    }

    /// Evaluates a *not yet monomorphized* constant.
    fn eval_constant(
        &mut self,
        constant: &mir::ConstOperand<'tcx>,
    ) -> Option<mir::ConstValue<'tcx>> {
        let const_ = self.monomorphize(constant.const_);
        // Evaluate the constant. This makes const eval failure a collection-time error (rather than
        // a codegen-time error). rustc stops after collection if there was an error, so this
        // ensures codegen never has to worry about failing consts.
        // (codegen relies on this and ICEs will happen if this is violated.)
        match const_.eval(self.tcx, ty::TypingEnv::fully_monomorphized(), constant.span) {
            Ok(v) => Some(v),
            Err(ErrorHandled::TooGeneric(..)) => span_bug!(
                constant.span,
                "collection encountered polymorphic constant: {:?}",
                const_
            ),
            Err(err @ ErrorHandled::Reported(..)) => {
                err.emit_note(self.tcx);
                return None;
            }
        }
    }
}

impl<'a, 'tcx> MirVisitor<'tcx> for MirUsedCollector<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        debug!("visiting rvalue {:?}", *rvalue);

        let span = self.body.source_info(location).span;

        match *rvalue {
            // When doing an cast from a regular pointer to a wide pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _)
                | mir::CastKind::PointerCoercion(PointerCoercion::DynStar, _),
                ref operand,
                target_ty,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                // *Before* monomorphizing, record that we already handled this mention.
                self.used_mentioned_items
                    .insert(MentionedItem::UnsizeCast { source_ty, target_ty });
                let target_ty = self.monomorphize(target_ty);
                let source_ty = self.monomorphize(source_ty);
                let (source_ty, target_ty) =
                    find_tails_for_unsizing(self.tcx.at(span), source_ty, target_ty);
                // This could also be a different Unsize instruction, like
                // from a fixed sized array to a slice. But we are only
                // interested in things that produce a vtable.
                if (target_ty.is_trait() && !source_ty.is_trait())
                    || (target_ty.is_dyn_star() && !source_ty.is_dyn_star())
                {
                    create_mono_items_for_vtable_methods(
                        self.tcx,
                        target_ty,
                        source_ty,
                        span,
                        self.used_items,
                    );
                }
            }
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, _),
                ref operand,
                _,
            ) => {
                let fn_ty = operand.ty(self.body, self.tcx);
                // *Before* monomorphizing, record that we already handled this mention.
                self.used_mentioned_items.insert(MentionedItem::Fn(fn_ty));
                let fn_ty = self.monomorphize(fn_ty);
                visit_fn_use(self.tcx, fn_ty, false, span, self.used_items);
            }
            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_), _),
                ref operand,
                _,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                // *Before* monomorphizing, record that we already handled this mention.
                self.used_mentioned_items.insert(MentionedItem::Closure(source_ty));
                let source_ty = self.monomorphize(source_ty);
                if let ty::Closure(def_id, args) = *source_ty.kind() {
                    let instance =
                        Instance::resolve_closure(self.tcx, def_id, args, ty::ClosureKind::FnOnce);
                    if self.tcx.should_codegen_locally(instance) {
                        self.used_items.push(create_fn_mono_item(self.tcx, instance, span));
                    }
                } else {
                    bug!()
                }
            }
            mir::Rvalue::ThreadLocalRef(def_id) => {
                assert!(self.tcx.is_thread_local_static(def_id));
                let instance = Instance::mono(self.tcx, def_id);
                if self.tcx.should_codegen_locally(instance) {
                    trace!("collecting thread-local static {:?}", def_id);
                    self.used_items.push(respan(span, MonoItem::Static(def_id)));
                }
            }
            _ => { /* not interesting */ }
        }

        self.super_rvalue(rvalue, location);
    }

    /// This does not walk the MIR of the constant as that is not needed for codegen, all we need is
    /// to ensure that the constant evaluates successfully and walk the result.
    #[instrument(skip(self), level = "debug")]
    fn visit_const_operand(&mut self, constant: &mir::ConstOperand<'tcx>, _location: Location) {
        // No `super_constant` as we don't care about `visit_ty`/`visit_ty_const`.
        let Some(val) = self.eval_constant(constant) else { return };
        collect_const_value(self.tcx, val, self.used_items);
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        debug!("visiting terminator {:?} @ {:?}", terminator, location);
        let source = self.body.source_info(location).span;

        let tcx = self.tcx;
        let push_mono_lang_item = |this: &mut Self, lang_item: LangItem| {
            let instance = Instance::mono(tcx, tcx.require_lang_item(lang_item, source));
            if tcx.should_codegen_locally(instance) {
                this.used_items.push(create_fn_mono_item(tcx, instance, source));
            }
        };

        match terminator.kind {
            mir::TerminatorKind::Call { ref func, .. }
            | mir::TerminatorKind::TailCall { ref func, .. } => {
                let callee_ty = func.ty(self.body, tcx);
                // *Before* monomorphizing, record that we already handled this mention.
                self.used_mentioned_items.insert(MentionedItem::Fn(callee_ty));
                let callee_ty = self.monomorphize(callee_ty);
                visit_fn_use(self.tcx, callee_ty, true, source, &mut self.used_items)
            }
            mir::TerminatorKind::Drop { ref place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                // *Before* monomorphizing, record that we already handled this mention.
                self.used_mentioned_items.insert(MentionedItem::Drop(ty));
                let ty = self.monomorphize(ty);
                visit_drop_use(self.tcx, ty, true, source, self.used_items);
            }
            mir::TerminatorKind::InlineAsm { ref operands, .. } => {
                for op in operands {
                    match *op {
                        mir::InlineAsmOperand::SymFn { ref value } => {
                            let fn_ty = value.const_.ty();
                            // *Before* monomorphizing, record that we already handled this mention.
                            self.used_mentioned_items.insert(MentionedItem::Fn(fn_ty));
                            let fn_ty = self.monomorphize(fn_ty);
                            visit_fn_use(self.tcx, fn_ty, false, source, self.used_items);
                        }
                        mir::InlineAsmOperand::SymStatic { def_id } => {
                            let instance = Instance::mono(self.tcx, def_id);
                            if self.tcx.should_codegen_locally(instance) {
                                trace!("collecting asm sym static {:?}", def_id);
                                self.used_items.push(respan(source, MonoItem::Static(def_id)));
                            }
                        }
                        _ => {}
                    }
                }
            }
            mir::TerminatorKind::Assert { ref msg, .. } => match &**msg {
                mir::AssertKind::BoundsCheck { .. } => {
                    push_mono_lang_item(self, LangItem::PanicBoundsCheck);
                }
                mir::AssertKind::MisalignedPointerDereference { .. } => {
                    push_mono_lang_item(self, LangItem::PanicMisalignedPointerDereference);
                }
                mir::AssertKind::NullPointerDereference => {
                    push_mono_lang_item(self, LangItem::PanicNullPointerDereference);
                }
                _ => {
                    push_mono_lang_item(self, msg.panic_function());
                }
            },
            mir::TerminatorKind::UnwindTerminate(reason) => {
                push_mono_lang_item(self, reason.lang_item());
            }
            mir::TerminatorKind::Goto { .. }
            | mir::TerminatorKind::SwitchInt { .. }
            | mir::TerminatorKind::UnwindResume
            | mir::TerminatorKind::Return
            | mir::TerminatorKind::Unreachable => {}
            mir::TerminatorKind::CoroutineDrop
            | mir::TerminatorKind::Yield { .. }
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::FalseUnwind { .. } => bug!(),
        }

        if let Some(mir::UnwindAction::Terminate(reason)) = terminator.unwind() {
            push_mono_lang_item(self, reason.lang_item());
        }

        self.super_terminator(terminator, location);
    }
}

fn visit_drop_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    let instance = Instance::resolve_drop_in_place(tcx, ty);
    visit_instance_use(tcx, instance, is_direct_call, source, output);
}

/// For every call of this function in the visitor, make sure there is a matching call in the
/// `mentioned_items` pass!
fn visit_fn_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    if let ty::FnDef(def_id, args) = *ty.kind() {
        let instance = if is_direct_call {
            ty::Instance::expect_resolve(
                tcx,
                ty::TypingEnv::fully_monomorphized(),
                def_id,
                args,
                source,
            )
        } else {
            match ty::Instance::resolve_for_fn_ptr(
                tcx,
                ty::TypingEnv::fully_monomorphized(),
                def_id,
                args,
            ) {
                Some(instance) => instance,
                _ => bug!("failed to resolve instance for {ty}"),
            }
        };
        visit_instance_use(tcx, instance, is_direct_call, source, output);
    }
}

fn visit_instance_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    debug!("visit_item_use({:?}, is_direct_call={:?})", instance, is_direct_call);
    if !tcx.should_codegen_locally(instance) {
        return;
    }
    if let Some(intrinsic) = tcx.intrinsic(instance.def_id()) {
        if let Some(_requirement) = ValidityRequirement::from_intrinsic(intrinsic.name) {
            // The intrinsics assert_inhabited, assert_zero_valid, and assert_mem_uninitialized_valid will
            // be lowered in codegen to nothing or a call to panic_nounwind. So if we encounter any
            // of those intrinsics, we need to include a mono item for panic_nounwind, else we may try to
            // codegen a call to that function without generating code for the function itself.
            let def_id = tcx.require_lang_item(LangItem::PanicNounwind, source);
            let panic_instance = Instance::mono(tcx, def_id);
            if tcx.should_codegen_locally(panic_instance) {
                output.push(create_fn_mono_item(tcx, panic_instance, source));
            }
        } else if !intrinsic.must_be_overridden {
            // Codegen the fallback body of intrinsics with fallback bodies.
            // We explicitly skip this otherwise to ensure we get a linker error
            // if anyone tries to call this intrinsic and the codegen backend did not
            // override the implementation.
            let instance = ty::Instance::new_raw(instance.def_id(), instance.args);
            if tcx.should_codegen_locally(instance) {
                output.push(create_fn_mono_item(tcx, instance, source));
            }
        }
    }

    match instance.def {
        ty::InstanceKind::Virtual(..) | ty::InstanceKind::Intrinsic(_) => {
            if !is_direct_call {
                bug!("{:?} being reified", instance);
            }
        }
        ty::InstanceKind::ThreadLocalShim(..) => {
            bug!("{:?} being reified", instance);
        }
        ty::InstanceKind::DropGlue(_, None) => {
            // Don't need to emit noop drop glue if we are calling directly.
            //
            // Note that we also optimize away the call to visit_instance_use in vtable construction
            // (see create_mono_items_for_vtable_methods).
            if !is_direct_call {
                output.push(create_fn_mono_item(tcx, instance, source));
            }
        }
        ty::InstanceKind::DropGlue(_, Some(_))
        | ty::InstanceKind::FutureDropPollShim(..)
        | ty::InstanceKind::AsyncDropGlue(_, _)
        | ty::InstanceKind::AsyncDropGlueCtorShim(_, _)
        | ty::InstanceKind::VTableShim(..)
        | ty::InstanceKind::ReifyShim(..)
        | ty::InstanceKind::ClosureOnceShim { .. }
        | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
        | ty::InstanceKind::Item(..)
        | ty::InstanceKind::FnPtrShim(..)
        | ty::InstanceKind::CloneShim(..)
        | ty::InstanceKind::FnPtrAddrShim(..) => {
            output.push(create_fn_mono_item(tcx, instance, source));
        }
    }
}

/// Returns `true` if we should codegen an instance in the local crate, or returns `false` if we
/// can just link to the upstream crate and therefore don't need a mono item.
fn should_codegen_locally<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let Some(def_id) = instance.def.def_id_if_not_guaranteed_local_codegen() else {
        return true;
    };

    if tcx.is_foreign_item(def_id) {
        // Foreign items are always linked against, there's no way of instantiating them.
        return false;
    }

    if tcx.def_kind(def_id).has_codegen_attrs()
        && matches!(tcx.codegen_fn_attrs(def_id).inline, InlineAttr::Force { .. })
    {
        // `#[rustc_force_inline]` items should never be codegened. This should be caught by
        // the MIR validator.
        tcx.dcx().delayed_bug("attempt to codegen `#[rustc_force_inline]` item");
    }

    if def_id.is_local() {
        // Local items cannot be referred to locally without monomorphizing them locally.
        return true;
    }

    if tcx.is_reachable_non_generic(def_id) || instance.upstream_monomorphization(tcx).is_some() {
        // We can link to the item in question, no instance needed in this crate.
        return false;
    }

    if let DefKind::Static { .. } = tcx.def_kind(def_id) {
        // We cannot monomorphize statics from upstream crates.
        return false;
    }

    if !tcx.is_mir_available(def_id) {
        tcx.dcx().emit_fatal(NoOptimizedMir {
            span: tcx.def_span(def_id),
            crate_name: tcx.crate_name(def_id.krate),
            instance: instance.to_string(),
        });
    }

    true
}

/// For a given pair of source and target type that occur in an unsizing coercion,
/// this function finds the pair of types that determines the vtable linking
/// them.
///
/// For example, the source type might be `&SomeStruct` and the target type
/// might be `&dyn SomeTrait` in a cast like:
///
/// ```rust,ignore (not real code)
/// let src: &SomeStruct = ...;
/// let target = src as &dyn SomeTrait;
/// ```
///
/// Then the output of this function would be (SomeStruct, SomeTrait) since for
/// constructing the `target` wide-pointer we need the vtable for that pair.
///
/// Things can get more complicated though because there's also the case where
/// the unsized type occurs as a field:
///
/// ```rust
/// struct ComplexStruct<T: ?Sized> {
///    a: u32,
///    b: f64,
///    c: T
/// }
/// ```
///
/// In this case, if `T` is sized, `&ComplexStruct<T>` is a thin pointer. If `T`
/// is unsized, `&SomeStruct` is a wide pointer, and the vtable it points to is
/// for the pair of `T` (which is a trait) and the concrete type that `T` was
/// originally coerced from:
///
/// ```rust,ignore (not real code)
/// let src: &ComplexStruct<SomeStruct> = ...;
/// let target = src as &ComplexStruct<dyn SomeTrait>;
/// ```
///
/// Again, we want this `find_vtable_types_for_unsizing()` to provide the pair
/// `(SomeStruct, SomeTrait)`.
///
/// Finally, there is also the case of custom unsizing coercions, e.g., for
/// smart pointers such as `Rc` and `Arc`.
fn find_tails_for_unsizing<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> (Ty<'tcx>, Ty<'tcx>) {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    debug_assert!(!source_ty.has_param(), "{source_ty} should be fully monomorphic");
    debug_assert!(!target_ty.has_param(), "{target_ty} should be fully monomorphic");

    match (source_ty.kind(), target_ty.kind()) {
        (
            &ty::Ref(_, source_pointee, _),
            &ty::Ref(_, target_pointee, _) | &ty::RawPtr(target_pointee, _),
        )
        | (&ty::RawPtr(source_pointee, _), &ty::RawPtr(target_pointee, _)) => {
            tcx.struct_lockstep_tails_for_codegen(source_pointee, target_pointee, typing_env)
        }

        // `Box<T>` could go through the ADT code below, b/c it'll unpeel to `Unique<T>`,
        // and eventually bottom out in a raw ref, but we can micro-optimize it here.
        (_, _)
            if let Some(source_boxed) = source_ty.boxed_ty()
                && let Some(target_boxed) = target_ty.boxed_ty() =>
        {
            tcx.struct_lockstep_tails_for_codegen(source_boxed, target_boxed, typing_env)
        }

        (&ty::Adt(source_adt_def, source_args), &ty::Adt(target_adt_def, target_args)) => {
            assert_eq!(source_adt_def, target_adt_def);
            let CustomCoerceUnsized::Struct(coerce_index) =
                match crate::custom_coerce_unsize_info(tcx, source_ty, target_ty) {
                    Ok(ccu) => ccu,
                    Err(e) => {
                        let e = Ty::new_error(tcx.tcx, e);
                        return (e, e);
                    }
                };
            let coerce_field = &source_adt_def.non_enum_variant().fields[coerce_index];
            // We're getting a possibly unnormalized type, so normalize it.
            let source_field =
                tcx.normalize_erasing_regions(typing_env, coerce_field.ty(*tcx, source_args));
            let target_field =
                tcx.normalize_erasing_regions(typing_env, coerce_field.ty(*tcx, target_args));
            find_tails_for_unsizing(tcx, source_field, target_field)
        }

        // `T` as `dyn* Trait` unsizes *directly*.
        //
        // FIXME(dyn_star): This case is a bit awkward, b/c we're not really computing
        // a tail here. We probably should handle this separately in the *caller* of
        // this function, rather than returning something that is semantically different
        // than what we return above.
        (_, &ty::Dynamic(_, _, ty::DynStar)) => (source_ty, target_ty),

        _ => bug!(
            "find_vtable_types_for_unsizing: invalid coercion {:?} -> {:?}",
            source_ty,
            target_ty
        ),
    }
}

#[instrument(skip(tcx), level = "debug", ret)]
fn create_fn_mono_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    source: Span,
) -> Spanned<MonoItem<'tcx>> {
    let def_id = instance.def_id();
    if tcx.sess.opts.unstable_opts.profile_closures
        && def_id.is_local()
        && tcx.is_closure_like(def_id)
    {
        crate::util::dump_closure_profile(tcx, instance);
    }

    respan(source, MonoItem::Fn(instance))
}

/// Creates a `MonoItem` for each method that is referenced by the vtable for
/// the given trait/impl pair.
fn create_mono_items_for_vtable_methods<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ty: Ty<'tcx>,
    impl_ty: Ty<'tcx>,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    assert!(!trait_ty.has_escaping_bound_vars() && !impl_ty.has_escaping_bound_vars());

    let ty::Dynamic(trait_ty, ..) = trait_ty.kind() else {
        bug!("create_mono_items_for_vtable_methods: {trait_ty:?} not a trait type");
    };
    if let Some(principal) = trait_ty.principal() {
        let trait_ref =
            tcx.instantiate_bound_regions_with_erased(principal.with_self_ty(tcx, impl_ty));
        assert!(!trait_ref.has_escaping_bound_vars());

        // Walk all methods of the trait, including those of its supertraits
        let entries = tcx.vtable_entries(trait_ref);
        debug!(?entries);
        let methods = entries
            .iter()
            .filter_map(|entry| match entry {
                VtblEntry::MetadataDropInPlace
                | VtblEntry::MetadataSize
                | VtblEntry::MetadataAlign
                | VtblEntry::Vacant => None,
                VtblEntry::TraitVPtr(_) => {
                    // all super trait items already covered, so skip them.
                    None
                }
                VtblEntry::Method(instance) => {
                    Some(*instance).filter(|instance| tcx.should_codegen_locally(*instance))
                }
            })
            .map(|item| create_fn_mono_item(tcx, item, source));
        output.extend(methods);
    }

    // Also add the destructor, if it's necessary.
    //
    // This matches the check in vtable_allocation_provider in middle/ty/vtable.rs,
    // if we don't need drop we're not adding an actual pointer to the vtable.
    if impl_ty.needs_drop(tcx, ty::TypingEnv::fully_monomorphized()) {
        visit_drop_use(tcx, impl_ty, false, source, output);
    }
}

/// Scans the CTFE alloc in order to find function pointers and statics that must be monomorphized.
fn collect_alloc<'tcx>(tcx: TyCtxt<'tcx>, alloc_id: AllocId, output: &mut MonoItems<'tcx>) {
    match tcx.global_alloc(alloc_id) {
        GlobalAlloc::Static(def_id) => {
            assert!(!tcx.is_thread_local_static(def_id));
            let instance = Instance::mono(tcx, def_id);
            if tcx.should_codegen_locally(instance) {
                trace!("collecting static {:?}", def_id);
                output.push(dummy_spanned(MonoItem::Static(def_id)));
            }
        }
        GlobalAlloc::Memory(alloc) => {
            trace!("collecting {:?} with {:#?}", alloc_id, alloc);
            let ptrs = alloc.inner().provenance().ptrs();
            // avoid `ensure_sufficient_stack` in the common case of "no pointers"
            if !ptrs.is_empty() {
                rustc_data_structures::stack::ensure_sufficient_stack(move || {
                    for &prov in ptrs.values() {
                        collect_alloc(tcx, prov.alloc_id(), output);
                    }
                });
            }
        }
        GlobalAlloc::Function { instance, .. } => {
            if tcx.should_codegen_locally(instance) {
                trace!("collecting {:?} with {:#?}", alloc_id, instance);
                output.push(create_fn_mono_item(tcx, instance, DUMMY_SP));
            }
        }
        GlobalAlloc::VTable(ty, dyn_ty) => {
            let alloc_id = tcx.vtable_allocation((
                ty,
                dyn_ty
                    .principal()
                    .map(|principal| tcx.instantiate_bound_regions_with_erased(principal)),
            ));
            collect_alloc(tcx, alloc_id, output)
        }
    }
}

/// Scans the MIR in order to find function calls, closures, and drop-glue.
///
/// Anything that's found is added to `output`. Furthermore the "mentioned items" of the MIR are returned.
#[instrument(skip(tcx), level = "debug")]
fn collect_items_of_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    mode: CollectionMode,
) -> (MonoItems<'tcx>, MonoItems<'tcx>) {
    // This item is getting monomorphized, do mono-time checks.
    tcx.ensure_ok().check_mono_item(instance);

    let body = tcx.instance_mir(instance.def);
    // Naively, in "used" collection mode, all functions get added to *both* `used_items` and
    // `mentioned_items`. Mentioned items processing will then notice that they have already been
    // visited, but at that point each mentioned item has been monomorphized, added to the
    // `mentioned_items` worklist, and checked in the global set of visited items. To remove that
    // overhead, we have a special optimization that avoids adding items to `mentioned_items` when
    // they are already added in `used_items`. We could just scan `used_items`, but that's a linear
    // scan and not very efficient. Furthermore we can only do that *after* monomorphizing the
    // mentioned item. So instead we collect all pre-monomorphized `MentionedItem` that were already
    // added to `used_items` in a hash set, which can efficiently query in the
    // `body.mentioned_items` loop below without even having to monomorphize the item.
    let mut used_items = MonoItems::new();
    let mut mentioned_items = MonoItems::new();
    let mut used_mentioned_items = Default::default();
    let mut collector = MirUsedCollector {
        tcx,
        body,
        used_items: &mut used_items,
        used_mentioned_items: &mut used_mentioned_items,
        instance,
    };

    if mode == CollectionMode::UsedItems {
        if tcx.sess.opts.debuginfo == DebugInfo::Full {
            for var_debug_info in &body.var_debug_info {
                collector.visit_var_debug_info(var_debug_info);
            }
        }
        for (bb, data) in traversal::mono_reachable(body, tcx, instance) {
            collector.visit_basic_block_data(bb, data)
        }
    }

    // Always visit all `required_consts`, so that we evaluate them and abort compilation if any of
    // them errors.
    for const_op in body.required_consts() {
        if let Some(val) = collector.eval_constant(const_op) {
            collect_const_value(tcx, val, &mut mentioned_items);
        }
    }

    // Always gather mentioned items. We try to avoid processing items that we have already added to
    // `used_items` above.
    for item in body.mentioned_items() {
        if !collector.used_mentioned_items.contains(&item.node) {
            let item_mono = collector.monomorphize(item.node);
            visit_mentioned_item(tcx, &item_mono, item.span, &mut mentioned_items);
        }
    }

    (used_items, mentioned_items)
}

fn items_of_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    (instance, mode): (Instance<'tcx>, CollectionMode),
) -> (&'tcx [Spanned<MonoItem<'tcx>>], &'tcx [Spanned<MonoItem<'tcx>>]) {
    let (used_items, mentioned_items) = collect_items_of_instance(tcx, instance, mode);

    let used_items = tcx.arena.alloc_from_iter(used_items);
    let mentioned_items = tcx.arena.alloc_from_iter(mentioned_items);

    (used_items, mentioned_items)
}

/// `item` must be already monomorphized.
#[instrument(skip(tcx, span, output), level = "debug")]
fn visit_mentioned_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &MentionedItem<'tcx>,
    span: Span,
    output: &mut MonoItems<'tcx>,
) {
    match *item {
        MentionedItem::Fn(ty) => {
            if let ty::FnDef(def_id, args) = *ty.kind() {
                let instance = Instance::expect_resolve(
                    tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    def_id,
                    args,
                    span,
                );
                // `visit_instance_use` was written for "used" item collection but works just as well
                // for "mentioned" item collection.
                // We can set `is_direct_call`; that just means we'll skip a bunch of shims that anyway
                // can't have their own failing constants.
                visit_instance_use(tcx, instance, /*is_direct_call*/ true, span, output);
            }
        }
        MentionedItem::Drop(ty) => {
            visit_drop_use(tcx, ty, /*is_direct_call*/ true, span, output);
        }
        MentionedItem::UnsizeCast { source_ty, target_ty } => {
            let (source_ty, target_ty) =
                find_tails_for_unsizing(tcx.at(span), source_ty, target_ty);
            // This could also be a different Unsize instruction, like
            // from a fixed sized array to a slice. But we are only
            // interested in things that produce a vtable.
            if (target_ty.is_trait() && !source_ty.is_trait())
                || (target_ty.is_dyn_star() && !source_ty.is_dyn_star())
            {
                create_mono_items_for_vtable_methods(tcx, target_ty, source_ty, span, output);
            }
        }
        MentionedItem::Closure(source_ty) => {
            if let ty::Closure(def_id, args) = *source_ty.kind() {
                let instance =
                    Instance::resolve_closure(tcx, def_id, args, ty::ClosureKind::FnOnce);
                if tcx.should_codegen_locally(instance) {
                    output.push(create_fn_mono_item(tcx, instance, span));
                }
            } else {
                bug!()
            }
        }
    }
}

#[instrument(skip(tcx, output), level = "debug")]
fn collect_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    value: mir::ConstValue<'tcx>,
    output: &mut MonoItems<'tcx>,
) {
    match value {
        mir::ConstValue::Scalar(Scalar::Ptr(ptr, _size)) => {
            collect_alloc(tcx, ptr.provenance.alloc_id(), output)
        }
        mir::ConstValue::Indirect { alloc_id, .. } => collect_alloc(tcx, alloc_id, output),
        mir::ConstValue::Slice { data, meta: _ } => {
            for &prov in data.inner().provenance().ptrs().values() {
                collect_alloc(tcx, prov.alloc_id(), output);
            }
        }
        _ => {}
    }
}

//=-----------------------------------------------------------------------------
// Root Collection
//=-----------------------------------------------------------------------------

// Find all non-generic items by walking the HIR. These items serve as roots to
// start monomorphizing from.
#[instrument(skip(tcx, mode), level = "debug")]
fn collect_roots(tcx: TyCtxt<'_>, mode: MonoItemCollectionStrategy) -> Vec<MonoItem<'_>> {
    debug!("collecting roots");
    let mut roots = MonoItems::new();

    {
        let entry_fn = tcx.entry_fn(());

        debug!("collect_roots: entry_fn = {:?}", entry_fn);

        let mut collector = RootCollector { tcx, strategy: mode, entry_fn, output: &mut roots };

        let crate_items = tcx.hir_crate_items(());

        for id in crate_items.free_items() {
            collector.process_item(id);
        }

        for id in crate_items.impl_items() {
            collector.process_impl_item(id);
        }

        for id in crate_items.nested_bodies() {
            collector.process_nested_body(id);
        }

        collector.push_extra_entry_roots();
    }

    // We can only codegen items that are instantiable - items all of
    // whose predicates hold. Luckily, items that aren't instantiable
    // can't actually be used, so we can just skip codegenning them.
    roots
        .into_iter()
        .filter_map(|Spanned { node: mono_item, .. }| {
            mono_item.is_instantiable(tcx).then_some(mono_item)
        })
        .collect()
}

struct RootCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    strategy: MonoItemCollectionStrategy,
    output: &'a mut MonoItems<'tcx>,
    entry_fn: Option<(DefId, EntryFnType)>,
}

impl<'v> RootCollector<'_, 'v> {
    fn process_item(&mut self, id: hir::ItemId) {
        match self.tcx.def_kind(id.owner_id) {
            DefKind::Enum | DefKind::Struct | DefKind::Union => {
                if self.strategy == MonoItemCollectionStrategy::Eager
                    && !self.tcx.generics_of(id.owner_id).requires_monomorphization(self.tcx)
                {
                    debug!("RootCollector: ADT drop-glue for `{id:?}`",);
                    let id_args =
                        ty::GenericArgs::for_item(self.tcx, id.owner_id.to_def_id(), |param, _| {
                            match param.kind {
                                GenericParamDefKind::Lifetime => {
                                    self.tcx.lifetimes.re_erased.into()
                                }
                                GenericParamDefKind::Type { .. }
                                | GenericParamDefKind::Const { .. } => {
                                    unreachable!(
                                        "`own_requires_monomorphization` check means that \
                                we should have no type/const params"
                                    )
                                }
                            }
                        });

                    // This type is impossible to instantiate, so we should not try to
                    // generate a `drop_in_place` instance for it.
                    if self.tcx.instantiate_and_check_impossible_predicates((
                        id.owner_id.to_def_id(),
                        id_args,
                    )) {
                        return;
                    }

                    let ty =
                        self.tcx.type_of(id.owner_id.to_def_id()).instantiate(self.tcx, id_args);
                    assert!(!ty.has_non_region_param());
                    visit_drop_use(self.tcx, ty, true, DUMMY_SP, self.output);
                }
            }
            DefKind::GlobalAsm => {
                debug!(
                    "RootCollector: ItemKind::GlobalAsm({})",
                    self.tcx.def_path_str(id.owner_id)
                );
                self.output.push(dummy_spanned(MonoItem::GlobalAsm(id)));
            }
            DefKind::Static { .. } => {
                let def_id = id.owner_id.to_def_id();
                debug!("RootCollector: ItemKind::Static({})", self.tcx.def_path_str(def_id));
                self.output.push(dummy_spanned(MonoItem::Static(def_id)));
            }
            DefKind::Const => {
                // Const items only generate mono items if they are actually used somewhere.
                // Just declaring them is insufficient.

                // But even just declaring them must collect the items they refer to
                // unless their generics require monomorphization.
                if !self.tcx.generics_of(id.owner_id).own_requires_monomorphization()
                    && let Ok(val) = self.tcx.const_eval_poly(id.owner_id.to_def_id())
                {
                    collect_const_value(self.tcx, val, self.output);
                }
            }
            DefKind::Impl { .. } => {
                if self.strategy == MonoItemCollectionStrategy::Eager {
                    create_mono_items_for_default_impls(self.tcx, id, self.output);
                }
            }
            DefKind::Fn => {
                self.push_if_root(id.owner_id.def_id);
            }
            _ => {}
        }
    }

    fn process_impl_item(&mut self, id: hir::ImplItemId) {
        if matches!(self.tcx.def_kind(id.owner_id), DefKind::AssocFn) {
            self.push_if_root(id.owner_id.def_id);
        }
    }

    fn process_nested_body(&mut self, def_id: LocalDefId) {
        match self.tcx.def_kind(def_id) {
            DefKind::Closure => {
                if self.strategy == MonoItemCollectionStrategy::Eager
                    && !self
                        .tcx
                        .generics_of(self.tcx.typeck_root_def_id(def_id.to_def_id()))
                        .requires_monomorphization(self.tcx)
                {
                    let instance = match *self.tcx.type_of(def_id).instantiate_identity().kind() {
                        ty::Closure(def_id, args)
                        | ty::Coroutine(def_id, args)
                        | ty::CoroutineClosure(def_id, args) => {
                            Instance::new_raw(def_id, self.tcx.erase_regions(args))
                        }
                        _ => unreachable!(),
                    };
                    let Ok(instance) = self.tcx.try_normalize_erasing_regions(
                        ty::TypingEnv::fully_monomorphized(),
                        instance,
                    ) else {
                        // Don't ICE on an impossible-to-normalize closure.
                        return;
                    };
                    let mono_item = create_fn_mono_item(self.tcx, instance, DUMMY_SP);
                    if mono_item.node.is_instantiable(self.tcx) {
                        self.output.push(mono_item);
                    }
                }
            }
            _ => {}
        }
    }

    fn is_root(&self, def_id: LocalDefId) -> bool {
        !self.tcx.generics_of(def_id).requires_monomorphization(self.tcx)
            && match self.strategy {
                MonoItemCollectionStrategy::Eager => {
                    !matches!(self.tcx.codegen_fn_attrs(def_id).inline, InlineAttr::Force { .. })
                }
                MonoItemCollectionStrategy::Lazy => {
                    self.entry_fn.and_then(|(id, _)| id.as_local()) == Some(def_id)
                        || self.tcx.is_reachable_non_generic(def_id)
                        || self
                            .tcx
                            .codegen_fn_attrs(def_id)
                            .flags
                            .contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
                }
            }
    }

    /// If `def_id` represents a root, pushes it onto the list of
    /// outputs. (Note that all roots must be monomorphic.)
    #[instrument(skip(self), level = "debug")]
    fn push_if_root(&mut self, def_id: LocalDefId) {
        if self.is_root(def_id) {
            debug!("found root");

            let instance = Instance::mono(self.tcx, def_id.to_def_id());
            self.output.push(create_fn_mono_item(self.tcx, instance, DUMMY_SP));
        }
    }

    /// As a special case, when/if we encounter the
    /// `main()` function, we also have to generate a
    /// monomorphized copy of the start lang item based on
    /// the return type of `main`. This is not needed when
    /// the user writes their own `start` manually.
    fn push_extra_entry_roots(&mut self) {
        let Some((main_def_id, EntryFnType::Main { .. })) = self.entry_fn else {
            return;
        };

        let Some(start_def_id) = self.tcx.lang_items().start_fn() else {
            self.tcx.dcx().emit_fatal(errors::StartNotFound);
        };
        let main_ret_ty = self.tcx.fn_sig(main_def_id).no_bound_vars().unwrap().output();

        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = self.tcx.normalize_erasing_regions(
            ty::TypingEnv::fully_monomorphized(),
            main_ret_ty.no_bound_vars().unwrap(),
        );

        let start_instance = Instance::expect_resolve(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            start_def_id,
            self.tcx.mk_args(&[main_ret_ty.into()]),
            DUMMY_SP,
        );

        self.output.push(create_fn_mono_item(self.tcx, start_instance, DUMMY_SP));
    }
}

#[instrument(level = "debug", skip(tcx, output))]
fn create_mono_items_for_default_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: hir::ItemId,
    output: &mut MonoItems<'tcx>,
) {
    let Some(impl_) = tcx.impl_trait_header(item.owner_id) else {
        return;
    };

    if matches!(impl_.polarity, ty::ImplPolarity::Negative) {
        return;
    }

    if tcx.generics_of(item.owner_id).own_requires_monomorphization() {
        return;
    }

    // Lifetimes never affect trait selection, so we are allowed to eagerly
    // instantiate an instance of an impl method if the impl (and method,
    // which we check below) is only parameterized over lifetime. In that case,
    // we use the ReErased, which has no lifetime information associated with
    // it, to validate whether or not the impl is legal to instantiate at all.
    let only_region_params = |param: &ty::GenericParamDef, _: &_| match param.kind {
        GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
        GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
            unreachable!(
                "`own_requires_monomorphization` check means that \
                we should have no type/const params"
            )
        }
    };
    let impl_args = GenericArgs::for_item(tcx, item.owner_id.to_def_id(), only_region_params);
    let trait_ref = impl_.trait_ref.instantiate(tcx, impl_args);

    // Unlike 'lazy' monomorphization that begins by collecting items transitively
    // called by `main` or other global items, when eagerly monomorphizing impl
    // items, we never actually check that the predicates of this impl are satisfied
    // in a empty param env (i.e. with no assumptions).
    //
    // Even though this impl has no type or const generic parameters, because we don't
    // consider higher-ranked predicates such as `for<'a> &'a mut [u8]: Copy` to
    // be trivially false. We must now check that the impl has no impossible-to-satisfy
    // predicates.
    if tcx.instantiate_and_check_impossible_predicates((item.owner_id.to_def_id(), impl_args)) {
        return;
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let trait_ref = tcx.normalize_erasing_regions(typing_env, trait_ref);
    let overridden_methods = tcx.impl_item_implementor_ids(item.owner_id);
    for method in tcx.provided_trait_methods(trait_ref.def_id) {
        if overridden_methods.contains_key(&method.def_id) {
            continue;
        }

        if tcx.generics_of(method.def_id).own_requires_monomorphization() {
            continue;
        }

        // As mentioned above, the method is legal to eagerly instantiate if it
        // only has lifetime generic parameters. This is validated by calling
        // `own_requires_monomorphization` on both the impl and method.
        let args = trait_ref.args.extend_to(tcx, method.def_id, only_region_params);
        let instance = ty::Instance::expect_resolve(tcx, typing_env, method.def_id, args, DUMMY_SP);

        let mono_item = create_fn_mono_item(tcx, instance, DUMMY_SP);
        if mono_item.node.is_instantiable(tcx) && tcx.should_codegen_locally(instance) {
            output.push(mono_item);
        }
    }
}

//=-----------------------------------------------------------------------------
// Top-level entry point, tying it all together
//=-----------------------------------------------------------------------------

#[instrument(skip(tcx, strategy), level = "debug")]
pub(crate) fn collect_crate_mono_items<'tcx>(
    tcx: TyCtxt<'tcx>,
    strategy: MonoItemCollectionStrategy,
) -> (Vec<MonoItem<'tcx>>, UsageMap<'tcx>) {
    let _prof_timer = tcx.prof.generic_activity("monomorphization_collector");

    let roots = tcx
        .sess
        .time("monomorphization_collector_root_collections", || collect_roots(tcx, strategy));

    debug!("building mono item graph, beginning at roots");

    let state = SharedState {
        visited: MTLock::new(UnordSet::default()),
        mentioned: MTLock::new(UnordSet::default()),
        usage_map: MTLock::new(UsageMap::new()),
    };
    let recursion_limit = tcx.recursion_limit();

    tcx.sess.time("monomorphization_collector_graph_walk", || {
        par_for_each_in(roots, |root| {
            collect_items_root(tcx, dummy_spanned(*root), &state, recursion_limit);
        });
    });

    // The set of MonoItems was created in an inherently indeterministic order because
    // of parallelism. We sort it here to ensure that the output is deterministic.
    let mono_items = tcx.with_stable_hashing_context(move |ref hcx| {
        state.visited.into_inner().into_sorted(hcx, true)
    });

    (mono_items, state.usage_map.into_inner())
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.hooks.should_codegen_locally = should_codegen_locally;
    providers.items_of_instance = items_of_instance;
}
