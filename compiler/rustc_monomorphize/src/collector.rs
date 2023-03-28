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
//!
//! General Algorithm
//! -----------------
//! Let's define some terms first:
//!
//! - A "mono item" is something that results in a function or global in
//!   the LLVM IR of a codegen unit. Mono items do not stand on their
//!   own, they can reference other mono items. For example, if function
//!   `foo()` calls function `bar()` then the mono item for `foo()`
//!   references the mono item for function `bar()`. In general, the
//!   definition for mono item A referencing a mono item B is that
//!   the LLVM artifact produced for A references the LLVM artifact produced
//!   for B.
//!
//! - Mono items and the references between them form a directed graph,
//!   where the mono items are the nodes and references form the edges.
//!   Let's call this graph the "mono item graph".
//!
//! - The mono item graph for a program contains all mono items
//!   that are needed in order to produce the complete LLVM IR of the program.
//!
//! The purpose of the algorithm implemented in this module is to build the
//! mono item graph for the current crate. It runs in two phases:
//!
//! 1. Discover the roots of the graph by traversing the HIR of the crate.
//! 2. Starting from the roots, find neighboring nodes by inspecting the MIR
//!    representation of the item corresponding to a given node, until no more
//!    new nodes are found.
//!
//! ### Discovering roots
//!
//! The roots of the mono item graph correspond to the public non-generic
//! syntactic items in the source code. We find them by walking the HIR of the
//! crate, and whenever we hit upon a public function, method, or static item,
//! we create a mono item consisting of the items DefId and, since we only
//! consider non-generic items, an empty type-substitution set. (In eager
//! collection mode, during incremental compilation, all non-generic functions
//! are considered as roots, as well as when the `-Clink-dead-code` option is
//! specified. Functions marked `#[no_mangle]` and functions called by inlinable
//! functions also always act as roots.)
//!
//! ### Finding neighbor nodes
//! Given a mono item node, we can discover neighbors by inspecting its
//! MIR. We walk the MIR and any time we hit upon something that signifies a
//! reference to another mono item, we have found a neighbor. Since the
//! mono item we are currently at is always monomorphic, we also know the
//! concrete type arguments of its neighbors, and so all neighbors again will be
//! monomorphic. The specific forms a reference to a neighboring node can take
//! in MIR are quite diverse. Here is an overview:
//!
//! #### Calling Functions/Methods
//! The most obvious form of one mono item referencing another is a
//! function or method call (represented by a CALL terminator in MIR). But
//! calls are not the only thing that might introduce a reference between two
//! function mono items, and as we will see below, they are just a
//! specialization of the form described next, and consequently will not get any
//! special treatment in the algorithm.
//!
//! #### Taking a reference to a function or method
//! A function does not need to actually be called in order to be a neighbor of
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
//! method in operand position, we treat it as a neighbor of the current
//! mono item. Calls are just a special case of that.
//!
//! #### Drop glue
//! Drop glue mono items are introduced by MIR drop-statements. The
//! generated mono item will again have drop-glue item neighbors if the
//! type to be dropped contains nested values that also need to be dropped. It
//! might also have a function item neighbor for the explicit `Drop::drop`
//! implementation of its type.
//!
//! #### Unsizing Casts
//! A subtle way of introducing neighbor edges is by casting to a trait object.
//! Since the resulting fat-pointer contains a reference to a vtable, we need to
//! instantiate all object-safe methods of the trait, as we need to store
//! pointers to these functions even if they never get called anywhere. This can
//! be seen as a special case of taking a function reference.
//!
//! #### Boxes
//! Since `Box` expression have special compiler support, no explicit calls to
//! `exchange_malloc()` and `box_free()` may show up in MIR, even if the
//! compiler will generate them. We have to observe `Rvalue::Box` expressions
//! and Box-typed drop-statements for that purpose.
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
//! Eager and Lazy Collection Mode
//! ------------------------------
//! Mono item collection can be performed in one of two modes:
//!
//! - Lazy mode means that items will only be instantiated when actually
//!   referenced. The goal is to produce the least amount of machine code
//!   possible.
//!
//! - Eager mode is meant to be used in conjunction with incremental compilation
//!   where a stable set of mono items is more important than a minimal
//!   one. Thus, eager mode will instantiate drop-glue for every drop-able type
//!   in the crate, even if no drop call for that type exists (yet). It will
//!   also instantiate default implementations of trait methods, something that
//!   otherwise is only done on demand.
//!
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

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::{par_for_each_in, MTLock, MTRef};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdMap, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_index::bit_set::GrowableBitSet;
use rustc_middle::mir::interpret::{AllocId, ConstValue};
use rustc_middle::mir::interpret::{ErrorHandled, GlobalAlloc, Scalar};
use rustc_middle::mir::mono::{InstantiationMode, MonoItem};
use rustc_middle::mir::visit::Visitor as MirVisitor;
use rustc_middle::mir::{self, Local, Location};
use rustc_middle::ty::adjustment::{CustomCoerceUnsized, PointerCast};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::query::TyCtxtAt;
use rustc_middle::ty::subst::{GenericArgKind, InternalSubsts};
use rustc_middle::ty::{
    self, GenericParamDefKind, Instance, Ty, TyCtxt, TypeFoldable, TypeVisitableExt, VtblEntry,
};
use rustc_middle::{middle::codegen_fn_attrs::CodegenFnAttrFlags, mir::visit::TyContext};
use rustc_session::config::EntryFnType;
use rustc_session::lint::builtin::LARGE_ASSIGNMENTS;
use rustc_session::Limit;
use rustc_span::source_map::{dummy_spanned, respan, Span, Spanned, DUMMY_SP};
use rustc_target::abi::Size;
use std::ops::Range;
use std::path::PathBuf;

use crate::errors::{
    EncounteredErrorWhileInstantiating, LargeAssignmentsLint, RecursionLimit, TypeLengthLimit,
};

#[derive(PartialEq)]
pub enum MonoItemCollectionMode {
    Eager,
    Lazy,
}

/// Maps every mono item to all mono items it references in its
/// body.
pub struct InliningMap<'tcx> {
    // Maps a source mono item to the range of mono items
    // accessed by it.
    // The range selects elements within the `targets` vecs.
    index: FxHashMap<MonoItem<'tcx>, Range<usize>>,
    targets: Vec<MonoItem<'tcx>>,

    // Contains one bit per mono item in the `targets` field. That bit
    // is true if that mono item needs to be inlined into every CGU.
    inlines: GrowableBitSet<usize>,
}

/// Struct to store mono items in each collecting and if they should
/// be inlined. We call `instantiation_mode` to get their inlining
/// status when inserting new elements, which avoids calling it in
/// `inlining_map.lock_mut()`. See the `collect_items_rec` implementation
/// below.
struct MonoItems<'tcx> {
    // If this is false, we do not need to compute whether items
    // will need to be inlined.
    compute_inlining: bool,

    // The TyCtxt used to determine whether the a item should
    // be inlined.
    tcx: TyCtxt<'tcx>,

    // The collected mono items. The bool field in each element
    // indicates whether this element should be inlined.
    items: Vec<(Spanned<MonoItem<'tcx>>, bool /*inlined*/)>,
}

impl<'tcx> MonoItems<'tcx> {
    #[inline]
    fn push(&mut self, item: Spanned<MonoItem<'tcx>>) {
        self.extend([item]);
    }

    #[inline]
    fn extend<T: IntoIterator<Item = Spanned<MonoItem<'tcx>>>>(&mut self, iter: T) {
        self.items.extend(iter.into_iter().map(|mono_item| {
            let inlined = if !self.compute_inlining {
                false
            } else {
                mono_item.node.instantiation_mode(self.tcx) == InstantiationMode::LocalCopy
            };
            (mono_item, inlined)
        }))
    }
}

impl<'tcx> InliningMap<'tcx> {
    fn new() -> InliningMap<'tcx> {
        InliningMap {
            index: FxHashMap::default(),
            targets: Vec::new(),
            inlines: GrowableBitSet::with_capacity(1024),
        }
    }

    fn record_accesses<'a>(
        &mut self,
        source: MonoItem<'tcx>,
        new_targets: &'a [(Spanned<MonoItem<'tcx>>, bool)],
    ) where
        'tcx: 'a,
    {
        let start_index = self.targets.len();
        let new_items_count = new_targets.len();
        let new_items_count_total = new_items_count + self.targets.len();

        self.targets.reserve(new_items_count);
        self.inlines.ensure(new_items_count_total);

        for (i, (Spanned { node: mono_item, .. }, inlined)) in new_targets.into_iter().enumerate() {
            self.targets.push(*mono_item);
            if *inlined {
                self.inlines.insert(i + start_index);
            }
        }

        let end_index = self.targets.len();
        assert!(self.index.insert(source, start_index..end_index).is_none());
    }

    /// Internally iterate over all items referenced by `source` which will be
    /// made available for inlining.
    pub fn with_inlining_candidates<F>(&self, source: MonoItem<'tcx>, mut f: F)
    where
        F: FnMut(MonoItem<'tcx>),
    {
        if let Some(range) = self.index.get(&source) {
            for (i, candidate) in self.targets[range.clone()].iter().enumerate() {
                if self.inlines.contains(range.start + i) {
                    f(*candidate);
                }
            }
        }
    }

    /// Internally iterate over all items and the things each accesses.
    pub fn iter_accesses<F>(&self, mut f: F)
    where
        F: FnMut(MonoItem<'tcx>, &[MonoItem<'tcx>]),
    {
        for (&accessor, range) in &self.index {
            f(accessor, &self.targets[range.clone()])
        }
    }
}

#[instrument(skip(tcx, mode), level = "debug")]
pub fn collect_crate_mono_items(
    tcx: TyCtxt<'_>,
    mode: MonoItemCollectionMode,
) -> (FxHashSet<MonoItem<'_>>, InliningMap<'_>) {
    let _prof_timer = tcx.prof.generic_activity("monomorphization_collector");

    let roots =
        tcx.sess.time("monomorphization_collector_root_collections", || collect_roots(tcx, mode));

    debug!("building mono item graph, beginning at roots");

    let mut visited = MTLock::new(FxHashSet::default());
    let mut inlining_map = MTLock::new(InliningMap::new());
    let recursion_limit = tcx.recursion_limit();

    {
        let visited: MTRef<'_, _> = &mut visited;
        let inlining_map: MTRef<'_, _> = &mut inlining_map;

        tcx.sess.time("monomorphization_collector_graph_walk", || {
            par_for_each_in(roots, |root| {
                let mut recursion_depths = DefIdMap::default();
                collect_items_rec(
                    tcx,
                    dummy_spanned(root),
                    visited,
                    &mut recursion_depths,
                    recursion_limit,
                    inlining_map,
                );
            });
        });
    }

    (visited.into_inner(), inlining_map.into_inner())
}

// Find all non-generic items by walking the HIR. These items serve as roots to
// start monomorphizing from.
#[instrument(skip(tcx, mode), level = "debug")]
fn collect_roots(tcx: TyCtxt<'_>, mode: MonoItemCollectionMode) -> Vec<MonoItem<'_>> {
    debug!("collecting roots");
    let mut roots = MonoItems { compute_inlining: false, tcx, items: Vec::new() };

    {
        let entry_fn = tcx.entry_fn(());

        debug!("collect_roots: entry_fn = {:?}", entry_fn);

        let mut collector = RootCollector { tcx, mode, entry_fn, output: &mut roots };

        let crate_items = tcx.hir_crate_items(());

        for id in crate_items.items() {
            collector.process_item(id);
        }

        for id in crate_items.impl_items() {
            collector.process_impl_item(id);
        }

        collector.push_extra_entry_roots();
    }

    // We can only codegen items that are instantiable - items all of
    // whose predicates hold. Luckily, items that aren't instantiable
    // can't actually be used, so we can just skip codegenning them.
    roots
        .items
        .into_iter()
        .filter_map(|(Spanned { node: mono_item, .. }, _)| {
            mono_item.is_instantiable(tcx).then_some(mono_item)
        })
        .collect()
}

/// Collect all monomorphized items reachable from `starting_point`, and emit a note diagnostic if a
/// post-monorphization error is encountered during a collection step.
#[instrument(skip(tcx, visited, recursion_depths, recursion_limit, inlining_map), level = "debug")]
fn collect_items_rec<'tcx>(
    tcx: TyCtxt<'tcx>,
    starting_point: Spanned<MonoItem<'tcx>>,
    visited: MTRef<'_, MTLock<FxHashSet<MonoItem<'tcx>>>>,
    recursion_depths: &mut DefIdMap<usize>,
    recursion_limit: Limit,
    inlining_map: MTRef<'_, MTLock<InliningMap<'tcx>>>,
) {
    if !visited.lock_mut().insert(starting_point.node) {
        // We've been here already, no need to search again.
        return;
    }

    let mut neighbors = MonoItems { compute_inlining: true, tcx, items: Vec::new() };
    let recursion_depth_reset;

    //
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
    let error_count = tcx.sess.diagnostic().err_count();

    match starting_point.node {
        MonoItem::Static(def_id) => {
            let instance = Instance::mono(tcx, def_id);

            // Sanity check whether this ended up being collected accidentally
            debug_assert!(should_codegen_locally(tcx, &instance));

            let ty = instance.ty(tcx, ty::ParamEnv::reveal_all());
            visit_drop_use(tcx, ty, true, starting_point.span, &mut neighbors);

            recursion_depth_reset = None;

            if let Ok(alloc) = tcx.eval_static_initializer(def_id) {
                for &id in alloc.inner().provenance().ptrs().values() {
                    collect_miri(tcx, id, &mut neighbors);
                }
            }
        }
        MonoItem::Fn(instance) => {
            // Sanity check whether this ended up being collected accidentally
            debug_assert!(should_codegen_locally(tcx, &instance));

            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(
                tcx,
                instance,
                starting_point.span,
                recursion_depths,
                recursion_limit,
            ));
            check_type_length_limit(tcx, instance);

            rustc_data_structures::stack::ensure_sufficient_stack(|| {
                collect_neighbours(tcx, instance, &mut neighbors);
            });
        }
        MonoItem::GlobalAsm(item_id) => {
            recursion_depth_reset = None;

            let item = tcx.hir().item(item_id);
            if let hir::ItemKind::GlobalAsm(asm) = item.kind {
                for (op, op_sp) in asm.operands {
                    match op {
                        hir::InlineAsmOperand::Const { .. } => {
                            // Only constants which resolve to a plain integer
                            // are supported. Therefore the value should not
                            // depend on any other items.
                        }
                        hir::InlineAsmOperand::SymFn { anon_const } => {
                            let fn_ty =
                                tcx.typeck_body(anon_const.body).node_type(anon_const.hir_id);
                            visit_fn_use(tcx, fn_ty, false, *op_sp, &mut neighbors);
                        }
                        hir::InlineAsmOperand::SymStatic { path: _, def_id } => {
                            let instance = Instance::mono(tcx, *def_id);
                            if should_codegen_locally(tcx, &instance) {
                                trace!("collecting static {:?}", def_id);
                                neighbors.push(dummy_spanned(MonoItem::Static(*def_id)));
                            }
                        }
                        hir::InlineAsmOperand::In { .. }
                        | hir::InlineAsmOperand::Out { .. }
                        | hir::InlineAsmOperand::InOut { .. }
                        | hir::InlineAsmOperand::SplitInOut { .. } => {
                            span_bug!(*op_sp, "invalid operand type for global_asm!")
                        }
                    }
                }
            } else {
                span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
            }
        }
    }

    // Check for PMEs and emit a diagnostic if one happened. To try to show relevant edges of the
    // mono item graph.
    if tcx.sess.diagnostic().err_count() > error_count
        && starting_point.node.is_generic_fn()
        && starting_point.node.is_user_defined()
    {
        let formatted_item = with_no_trimmed_paths!(starting_point.node.to_string());
        tcx.sess.emit_note(EncounteredErrorWhileInstantiating {
            span: starting_point.span,
            formatted_item,
        });
    }
    inlining_map.lock_mut().record_accesses(starting_point.node, &neighbors.items);

    for (neighbour, _) in neighbors.items {
        collect_items_rec(tcx, neighbour, visited, recursion_depths, recursion_limit, inlining_map);
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }
}

/// Format instance name that is already known to be too long for rustc.
/// Show only the first 2 types if it is longer than 32 characters to avoid blasting
/// the user's terminal with thousands of lines of type-name.
///
/// If the type name is longer than before+after, it will be written to a file.
fn shrunk_instance_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
) -> (String, Option<PathBuf>) {
    let s = instance.to_string();

    // Only use the shrunk version if it's really shorter.
    // This also avoids the case where before and after slices overlap.
    if s.chars().nth(33).is_some() {
        let shrunk = format!("{}", ty::ShortInstance(instance, 4));
        if shrunk == s {
            return (s, None);
        }

        let path = tcx.output_filenames(()).temp_path_ext("long-type.txt", None);
        let written_to_path = std::fs::write(&path, s).ok().map(|_| path);

        (shrunk, written_to_path)
    } else {
        (s, None)
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

    let adjusted_recursion_depth = if Some(def_id) == tcx.lang_items().drop_in_place_fn() {
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
        let (shrunk, written_to_path) = shrunk_instance_name(tcx, &instance);
        let mut path = PathBuf::new();
        let was_written = if let Some(written_to_path) = written_to_path {
            path = written_to_path;
            Some(())
        } else {
            None
        };
        tcx.sess.emit_fatal(RecursionLimit {
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

fn check_type_length_limit<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let type_length = instance
        .substs
        .iter()
        .flat_map(|arg| arg.walk())
        .filter(|arg| match arg.unpack() {
            GenericArgKind::Type(_) | GenericArgKind::Const(_) => true,
            GenericArgKind::Lifetime(_) => false,
        })
        .count();
    debug!(" => type length={}", type_length);

    // Rust code can easily create exponentially-long types using only a
    // polynomial recursion depth. Even with the default recursion
    // depth, you can easily get cases that take >2^60 steps to run,
    // which means that rustc basically hangs.
    //
    // Bail out in these cases to avoid that bad user experience.
    if !tcx.type_length_limit().value_within_limit(type_length) {
        let (shrunk, written_to_path) = shrunk_instance_name(tcx, &instance);
        let span = tcx.def_span(instance.def_id());
        let mut path = PathBuf::new();
        let was_written = if written_to_path.is_some() {
            path = written_to_path.unwrap();
            Some(())
        } else {
            None
        };
        tcx.sess.emit_fatal(TypeLengthLimit { span, shrunk, was_written, path, type_length });
    }
}

struct MirNeighborCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    output: &'a mut MonoItems<'tcx>,
    instance: Instance<'tcx>,
}

impl<'a, 'tcx> MirNeighborCollector<'a, 'tcx> {
    pub fn monomorphize<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        debug!("monomorphize: self.instance={:?}", self.instance);
        self.instance.subst_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::ParamEnv::reveal_all(),
            value,
        )
    }
}

impl<'a, 'tcx> MirVisitor<'tcx> for MirNeighborCollector<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        debug!("visiting rvalue {:?}", *rvalue);

        let span = self.body.source_info(location).span;

        match *rvalue {
            // When doing an cast from a regular pointer to a fat pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::Unsize),
                ref operand,
                target_ty,
            )
            | mir::Rvalue::Cast(mir::CastKind::DynStar, ref operand, target_ty) => {
                let target_ty = self.monomorphize(target_ty);
                let source_ty = operand.ty(self.body, self.tcx);
                let source_ty = self.monomorphize(source_ty);
                let (source_ty, target_ty) =
                    find_vtable_types_for_unsizing(self.tcx.at(span), source_ty, target_ty);
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
                        self.output,
                    );
                }
            }
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::ReifyFnPointer),
                ref operand,
                _,
            ) => {
                let fn_ty = operand.ty(self.body, self.tcx);
                let fn_ty = self.monomorphize(fn_ty);
                visit_fn_use(self.tcx, fn_ty, false, span, &mut self.output);
            }
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::ClosureFnPointer(_)),
                ref operand,
                _,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                let source_ty = self.monomorphize(source_ty);
                match *source_ty.kind() {
                    ty::Closure(def_id, substs) => {
                        let instance = Instance::resolve_closure(
                            self.tcx,
                            def_id,
                            substs,
                            ty::ClosureKind::FnOnce,
                        )
                        .expect("failed to normalize and resolve closure during codegen");
                        if should_codegen_locally(self.tcx, &instance) {
                            self.output.push(create_fn_mono_item(self.tcx, instance, span));
                        }
                    }
                    _ => bug!(),
                }
            }
            mir::Rvalue::ThreadLocalRef(def_id) => {
                assert!(self.tcx.is_thread_local_static(def_id));
                let instance = Instance::mono(self.tcx, def_id);
                if should_codegen_locally(self.tcx, &instance) {
                    trace!("collecting thread-local static {:?}", def_id);
                    self.output.push(respan(span, MonoItem::Static(def_id)));
                }
            }
            _ => { /* not interesting */ }
        }

        self.super_rvalue(rvalue, location);
    }

    /// This does not walk the constant, as it has been handled entirely here and trying
    /// to walk it would attempt to evaluate the `ty::Const` inside, which doesn't necessarily
    /// work, as some constants cannot be represented in the type system.
    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>, location: Location) {
        let literal = self.monomorphize(constant.literal);
        let val = match literal {
            mir::ConstantKind::Val(val, _) => val,
            mir::ConstantKind::Ty(ct) => match ct.kind() {
                ty::ConstKind::Value(val) => self.tcx.valtree_to_const_val((ct.ty(), val)),
                ty::ConstKind::Unevaluated(ct) => {
                    debug!(?ct);
                    let param_env = ty::ParamEnv::reveal_all();
                    match self.tcx.const_eval_resolve(param_env, ct.expand(), None) {
                        // The `monomorphize` call should have evaluated that constant already.
                        Ok(val) => val,
                        Err(ErrorHandled::Reported(_)) => return,
                        Err(ErrorHandled::TooGeneric) => span_bug!(
                            self.body.source_info(location).span,
                            "collection encountered polymorphic constant: {:?}",
                            literal
                        ),
                    }
                }
                _ => return,
            },
            mir::ConstantKind::Unevaluated(uv, _) => {
                let param_env = ty::ParamEnv::reveal_all();
                match self.tcx.const_eval_resolve(param_env, uv, None) {
                    // The `monomorphize` call should have evaluated that constant already.
                    Ok(val) => val,
                    Err(ErrorHandled::Reported(_)) => return,
                    Err(ErrorHandled::TooGeneric) => span_bug!(
                        self.body.source_info(location).span,
                        "collection encountered polymorphic constant: {:?}",
                        literal
                    ),
                }
            }
        };
        collect_const_value(self.tcx, val, self.output);
        MirVisitor::visit_ty(self, literal.ty(), TyContext::Location(location));
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        debug!("visiting terminator {:?} @ {:?}", terminator, location);
        let source = self.body.source_info(location).span;

        let tcx = self.tcx;
        match terminator.kind {
            mir::TerminatorKind::Call { ref func, .. } => {
                let callee_ty = func.ty(self.body, tcx);
                let callee_ty = self.monomorphize(callee_ty);
                visit_fn_use(self.tcx, callee_ty, true, source, &mut self.output)
            }
            mir::TerminatorKind::Drop { ref place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                let ty = self.monomorphize(ty);
                visit_drop_use(self.tcx, ty, true, source, self.output);
            }
            mir::TerminatorKind::InlineAsm { ref operands, .. } => {
                for op in operands {
                    match *op {
                        mir::InlineAsmOperand::SymFn { ref value } => {
                            let fn_ty = self.monomorphize(value.literal.ty());
                            visit_fn_use(self.tcx, fn_ty, false, source, &mut self.output);
                        }
                        mir::InlineAsmOperand::SymStatic { def_id } => {
                            let instance = Instance::mono(self.tcx, def_id);
                            if should_codegen_locally(self.tcx, &instance) {
                                trace!("collecting asm sym static {:?}", def_id);
                                self.output.push(respan(source, MonoItem::Static(def_id)));
                            }
                        }
                        _ => {}
                    }
                }
            }
            mir::TerminatorKind::Assert { ref msg, .. } => {
                let lang_item = match msg {
                    mir::AssertKind::BoundsCheck { .. } => LangItem::PanicBoundsCheck,
                    _ => LangItem::Panic,
                };
                let instance = Instance::mono(tcx, tcx.require_lang_item(lang_item, Some(source)));
                if should_codegen_locally(tcx, &instance) {
                    self.output.push(create_fn_mono_item(tcx, instance, source));
                }
            }
            mir::TerminatorKind::Abort { .. } => {
                let instance = Instance::mono(
                    tcx,
                    tcx.require_lang_item(LangItem::PanicCannotUnwind, Some(source)),
                );
                if should_codegen_locally(tcx, &instance) {
                    self.output.push(create_fn_mono_item(tcx, instance, source));
                }
            }
            mir::TerminatorKind::Goto { .. }
            | mir::TerminatorKind::SwitchInt { .. }
            | mir::TerminatorKind::Resume
            | mir::TerminatorKind::Return
            | mir::TerminatorKind::Unreachable => {}
            mir::TerminatorKind::GeneratorDrop
            | mir::TerminatorKind::Yield { .. }
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::FalseUnwind { .. } => bug!(),
        }

        self.super_terminator(terminator, location);
    }

    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
        let limit = self.tcx.move_size_limit().0;
        if limit == 0 {
            return;
        }
        let limit = Size::from_bytes(limit);
        let ty = operand.ty(self.body, self.tcx);
        let ty = self.monomorphize(ty);
        let layout = self.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty));
        if let Ok(layout) = layout {
            if layout.size > limit {
                debug!(?layout);
                let source_info = self.body.source_info(location);
                debug!(?source_info);
                let lint_root = source_info.scope.lint_root(&self.body.source_scopes);
                debug!(?lint_root);
                let Some(lint_root) = lint_root else {
                    // This happens when the issue is in a function from a foreign crate that
                    // we monomorphized in the current crate. We can't get a `HirId` for things
                    // in other crates.
                    // FIXME: Find out where to report the lint on. Maybe simply crate-level lint root
                    // but correct span? This would make the lint at least accept crate-level lint attributes.
                    return;
                };
                self.tcx.emit_spanned_lint(
                    LARGE_ASSIGNMENTS,
                    lint_root,
                    source_info.span,
                    LargeAssignmentsLint {
                        span: source_info.span,
                        size: layout.size.bytes(),
                        limit: limit.bytes(),
                    },
                )
            }
        }
    }

    fn visit_local(
        &mut self,
        _place_local: Local,
        _context: mir::visit::PlaceContext,
        _location: Location,
    ) {
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

fn visit_fn_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    is_direct_call: bool,
    source: Span,
    output: &mut MonoItems<'tcx>,
) {
    if let ty::FnDef(def_id, substs) = *ty.kind() {
        let instance = if is_direct_call {
            ty::Instance::expect_resolve(tcx, ty::ParamEnv::reveal_all(), def_id, substs)
        } else {
            match ty::Instance::resolve_for_fn_ptr(tcx, ty::ParamEnv::reveal_all(), def_id, substs)
            {
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
    if !should_codegen_locally(tcx, &instance) {
        return;
    }

    match instance.def {
        ty::InstanceDef::Virtual(..) | ty::InstanceDef::Intrinsic(_) => {
            if !is_direct_call {
                bug!("{:?} being reified", instance);
            }
        }
        ty::InstanceDef::DropGlue(_, None) => {
            // Don't need to emit noop drop glue if we are calling directly.
            if !is_direct_call {
                output.push(create_fn_mono_item(tcx, instance, source));
            }
        }
        ty::InstanceDef::DropGlue(_, Some(_))
        | ty::InstanceDef::VTableShim(..)
        | ty::InstanceDef::ReifyShim(..)
        | ty::InstanceDef::ClosureOnceShim { .. }
        | ty::InstanceDef::Item(..)
        | ty::InstanceDef::FnPtrShim(..)
        | ty::InstanceDef::CloneShim(..)
        | ty::InstanceDef::FnPtrAddrShim(..) => {
            output.push(create_fn_mono_item(tcx, instance, source));
        }
    }
}

/// Returns `true` if we should codegen an instance in the local crate, or returns `false` if we
/// can just link to the upstream crate and therefore don't need a mono item.
fn should_codegen_locally<'tcx>(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>) -> bool {
    let Some(def_id) = instance.def.def_id_if_not_guaranteed_local_codegen() else {
        return true;
    };

    if tcx.is_foreign_item(def_id) {
        // Foreign items are always linked against, there's no way of instantiating them.
        return false;
    }

    if def_id.is_local() {
        // Local items cannot be referred to locally without monomorphizing them locally.
        return true;
    }

    if tcx.is_reachable_non_generic(def_id)
        || instance.polymorphize(tcx).upstream_monomorphization(tcx).is_some()
    {
        // We can link to the item in question, no instance needed in this crate.
        return false;
    }

    if let DefKind::Static(_) = tcx.def_kind(def_id) {
        // We cannot monomorphize statics from upstream crates.
        return false;
    }

    if !tcx.is_mir_available(def_id) {
        bug!("no MIR available for {:?}", def_id);
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
/// constructing the `target` fat-pointer we need the vtable for that pair.
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
/// is unsized, `&SomeStruct` is a fat pointer, and the vtable it points to is
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
fn find_vtable_types_for_unsizing<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> (Ty<'tcx>, Ty<'tcx>) {
    let ptr_vtable = |inner_source: Ty<'tcx>, inner_target: Ty<'tcx>| {
        let param_env = ty::ParamEnv::reveal_all();
        let type_has_metadata = |ty: Ty<'tcx>| -> bool {
            if ty.is_sized(tcx.tcx, param_env) {
                return false;
            }
            let tail = tcx.struct_tail_erasing_lifetimes(ty, param_env);
            match tail.kind() {
                ty::Foreign(..) => false,
                ty::Str | ty::Slice(..) | ty::Dynamic(..) => true,
                _ => bug!("unexpected unsized tail: {:?}", tail),
            }
        };
        if type_has_metadata(inner_source) {
            (inner_source, inner_target)
        } else {
            tcx.struct_lockstep_tails_erasing_lifetimes(inner_source, inner_target, param_env)
        }
    };

    match (&source_ty.kind(), &target_ty.kind()) {
        (&ty::Ref(_, a, _), &ty::Ref(_, b, _) | &ty::RawPtr(ty::TypeAndMut { ty: b, .. }))
        | (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }), &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            ptr_vtable(*a, *b)
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            ptr_vtable(source_ty.boxed_ty(), target_ty.boxed_ty())
        }

        // T as dyn* Trait
        (_, &ty::Dynamic(_, _, ty::DynStar)) => ptr_vtable(source_ty, target_ty),

        (&ty::Adt(source_adt_def, source_substs), &ty::Adt(target_adt_def, target_substs)) => {
            assert_eq!(source_adt_def, target_adt_def);

            let CustomCoerceUnsized::Struct(coerce_index) =
                crate::custom_coerce_unsize_info(tcx, source_ty, target_ty);

            let source_fields = &source_adt_def.non_enum_variant().fields;
            let target_fields = &target_adt_def.non_enum_variant().fields;

            assert!(
                coerce_index < source_fields.len() && source_fields.len() == target_fields.len()
            );

            find_vtable_types_for_unsizing(
                tcx,
                source_fields[coerce_index].ty(*tcx, source_substs),
                target_fields[coerce_index].ty(*tcx, target_substs),
            )
        }
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
    if tcx.sess.opts.unstable_opts.profile_closures && def_id.is_local() && tcx.is_closure(def_id) {
        crate::util::dump_closure_profile(tcx, instance);
    }

    respan(source, MonoItem::Fn(instance.polymorphize(tcx)))
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

    if let ty::Dynamic(ref trait_ty, ..) = trait_ty.kind() {
        if let Some(principal) = trait_ty.principal() {
            let poly_trait_ref = principal.with_self_ty(tcx, impl_ty);
            assert!(!poly_trait_ref.has_escaping_bound_vars());

            // Walk all methods of the trait, including those of its supertraits
            let entries = tcx.vtable_entries(poly_trait_ref);
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
                        Some(*instance).filter(|instance| should_codegen_locally(tcx, instance))
                    }
                })
                .map(|item| create_fn_mono_item(tcx, item, source));
            output.extend(methods);
        }

        // Also add the destructor.
        visit_drop_use(tcx, impl_ty, false, source, output);
    }
}

//=-----------------------------------------------------------------------------
// Root Collection
//=-----------------------------------------------------------------------------

struct RootCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    mode: MonoItemCollectionMode,
    output: &'a mut MonoItems<'tcx>,
    entry_fn: Option<(DefId, EntryFnType)>,
}

impl<'v> RootCollector<'_, 'v> {
    fn process_item(&mut self, id: hir::ItemId) {
        match self.tcx.def_kind(id.owner_id) {
            DefKind::Enum | DefKind::Struct | DefKind::Union => {
                if self.mode == MonoItemCollectionMode::Eager
                    && self.tcx.generics_of(id.owner_id).count() == 0
                {
                    debug!("RootCollector: ADT drop-glue for `{id:?}`",);

                    let ty = self.tcx.type_of(id.owner_id.to_def_id()).no_bound_vars().unwrap();
                    visit_drop_use(self.tcx, ty, true, DUMMY_SP, self.output);
                }
            }
            DefKind::GlobalAsm => {
                debug!(
                    "RootCollector: ItemKind::GlobalAsm({})",
                    self.tcx.def_path_str(id.owner_id.to_def_id())
                );
                self.output.push(dummy_spanned(MonoItem::GlobalAsm(id)));
            }
            DefKind::Static(..) => {
                debug!(
                    "RootCollector: ItemKind::Static({})",
                    self.tcx.def_path_str(id.owner_id.to_def_id())
                );
                self.output.push(dummy_spanned(MonoItem::Static(id.owner_id.to_def_id())));
            }
            DefKind::Const => {
                // const items only generate mono items if they are
                // actually used somewhere. Just declaring them is insufficient.

                // but even just declaring them must collect the items they refer to
                if let Ok(val) = self.tcx.const_eval_poly(id.owner_id.to_def_id()) {
                    collect_const_value(self.tcx, val, &mut self.output);
                }
            }
            DefKind::Impl { .. } => {
                if self.mode == MonoItemCollectionMode::Eager {
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

    fn is_root(&self, def_id: LocalDefId) -> bool {
        !item_requires_monomorphization(self.tcx, def_id)
            && match self.mode {
                MonoItemCollectionMode::Eager => true,
                MonoItemCollectionMode::Lazy => {
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

        let start_def_id = self.tcx.require_lang_item(LangItem::Start, None);
        let main_ret_ty = self.tcx.fn_sig(main_def_id).no_bound_vars().unwrap().output();

        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = self.tcx.normalize_erasing_regions(
            ty::ParamEnv::reveal_all(),
            main_ret_ty.no_bound_vars().unwrap(),
        );

        let start_instance = Instance::resolve(
            self.tcx,
            ty::ParamEnv::reveal_all(),
            start_def_id,
            self.tcx.mk_substs(&[main_ret_ty.into()]),
        )
        .unwrap()
        .unwrap();

        self.output.push(create_fn_mono_item(self.tcx, start_instance, DUMMY_SP));
    }
}

fn item_requires_monomorphization(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let generics = tcx.generics_of(def_id);
    generics.requires_monomorphization(tcx)
}

#[instrument(level = "debug", skip(tcx, output))]
fn create_mono_items_for_default_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: hir::ItemId,
    output: &mut MonoItems<'tcx>,
) {
    let polarity = tcx.impl_polarity(item.owner_id);
    if matches!(polarity, ty::ImplPolarity::Negative) {
        return;
    }

    if tcx.generics_of(item.owner_id).own_requires_monomorphization() {
        return;
    }

    let Some(trait_ref) = tcx.impl_trait_ref(item.owner_id) else {
        return;
    };

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
    let impl_substs = InternalSubsts::for_item(tcx, item.owner_id.to_def_id(), only_region_params);
    let trait_ref = trait_ref.subst(tcx, impl_substs);

    // Unlike 'lazy' monomorphization that begins by collecting items transitively
    // called by `main` or other global items, when eagerly monomorphizing impl
    // items, we never actually check that the predicates of this impl are satisfied
    // in a empty reveal-all param env (i.e. with no assumptions).
    //
    // Even though this impl has no type or const substitutions, because we don't
    // consider higher-ranked predicates such as `for<'a> &'a mut [u8]: Copy` to
    // be trivially false. We must now check that the impl has no impossible-to-satisfy
    // predicates.
    if tcx.subst_and_check_impossible_predicates((item.owner_id.to_def_id(), impl_substs)) {
        return;
    }

    let param_env = ty::ParamEnv::reveal_all();
    let trait_ref = tcx.normalize_erasing_regions(param_env, trait_ref);
    let overridden_methods = tcx.impl_item_implementor_ids(item.owner_id);
    for method in tcx.provided_trait_methods(trait_ref.def_id) {
        if overridden_methods.contains_key(&method.def_id) {
            continue;
        }

        if tcx.generics_of(method.def_id).own_requires_monomorphization() {
            continue;
        }

        // As mentioned above, the method is legal to eagerly instantiate if it
        // only has lifetime substitutions. This is validated by
        let substs = trait_ref.substs.extend_to(tcx, method.def_id, only_region_params);
        let instance = ty::Instance::expect_resolve(tcx, param_env, method.def_id, substs);

        let mono_item = create_fn_mono_item(tcx, instance, DUMMY_SP);
        if mono_item.node.is_instantiable(tcx) && should_codegen_locally(tcx, &instance) {
            output.push(mono_item);
        }
    }
}

/// Scans the miri alloc in order to find function calls, closures, and drop-glue.
fn collect_miri<'tcx>(tcx: TyCtxt<'tcx>, alloc_id: AllocId, output: &mut MonoItems<'tcx>) {
    match tcx.global_alloc(alloc_id) {
        GlobalAlloc::Static(def_id) => {
            assert!(!tcx.is_thread_local_static(def_id));
            let instance = Instance::mono(tcx, def_id);
            if should_codegen_locally(tcx, &instance) {
                trace!("collecting static {:?}", def_id);
                output.push(dummy_spanned(MonoItem::Static(def_id)));
            }
        }
        GlobalAlloc::Memory(alloc) => {
            trace!("collecting {:?} with {:#?}", alloc_id, alloc);
            for &inner in alloc.inner().provenance().ptrs().values() {
                rustc_data_structures::stack::ensure_sufficient_stack(|| {
                    collect_miri(tcx, inner, output);
                });
            }
        }
        GlobalAlloc::Function(fn_instance) => {
            if should_codegen_locally(tcx, &fn_instance) {
                trace!("collecting {:?} with {:#?}", alloc_id, fn_instance);
                output.push(create_fn_mono_item(tcx, fn_instance, DUMMY_SP));
            }
        }
        GlobalAlloc::VTable(ty, trait_ref) => {
            let alloc_id = tcx.vtable_allocation((ty, trait_ref));
            collect_miri(tcx, alloc_id, output)
        }
    }
}

/// Scans the MIR in order to find function calls, closures, and drop-glue.
#[instrument(skip(tcx, output), level = "debug")]
fn collect_neighbours<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    output: &mut MonoItems<'tcx>,
) {
    let body = tcx.instance_mir(instance.def);
    MirNeighborCollector { tcx, body: &body, output, instance }.visit_body(&body);
}

#[instrument(skip(tcx, output), level = "debug")]
fn collect_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    value: ConstValue<'tcx>,
    output: &mut MonoItems<'tcx>,
) {
    match value {
        ConstValue::Scalar(Scalar::Ptr(ptr, _size)) => collect_miri(tcx, ptr.provenance, output),
        ConstValue::Slice { data: alloc, start: _, end: _ } | ConstValue::ByRef { alloc, .. } => {
            for &id in alloc.inner().provenance().ptrs().values() {
                collect_miri(tcx, id, output);
            }
        }
        _ => {}
    }
}
