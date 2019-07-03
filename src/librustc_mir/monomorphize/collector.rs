//! Mono Item Collection
//! ===========================
//!
//! This module is responsible for discovering all items that will contribute to
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
//! - Vtables
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
//! The roots of the mono item graph correspond to the non-generic
//! syntactic items in the source code. We find them by walking the HIR of the
//! crate, and whenever we hit upon a function, method, or static item, we
//! create a mono item consisting of the items DefId and, since we only
//! consider non-generic items, an empty type-substitution set.
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
//! specialized of the form described next, and consequently will don't get any
//! special treatment in the algorithm.
//!
//! #### Taking a reference to a function or method
//! A function does not need to actually be called in order to be a neighbor of
//! another function. It suffices to just take a reference in order to introduce
//! an edge. Consider the following example:
//!
//! ```rust
//! fn print_val<T: Display>(x: T) {
//!     println!("{}", x);
//! }
//!
//! fn call_fn(f: &Fn(i32), x: i32) {
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
//! #### Closures
//! In a way, closures are a simple case. Since every closure object needs to be
//! constructed somewhere, we can reliably discover them by observing
//! `RValue::Aggregate` expressions with `AggregateKind::Closure`. This is also
//! true for closures inlined from other crates.
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
//! instantiate all object-save methods of the trait, as we need to store
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
//!   in the crate, even of no drop call for that type exists (yet). It will
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

use rustc::hir::{self, CodegenFnAttrFlags};
use rustc::hir::itemlikevisit::ItemLikeVisitor;

use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::mir::interpret::{AllocId, ConstValue};
use rustc::middle::lang_items::{ExchangeMallocFnLangItem, StartFnLangItem};
use rustc::ty::subst::{InternalSubsts, SubstsRef};
use rustc::ty::{self, TypeFoldable, Ty, TyCtxt, GenericParamDefKind, Instance};
use rustc::ty::print::obsolete::DefPathBasedNames;
use rustc::ty::adjustment::{CustomCoerceUnsized, PointerCast};
use rustc::session::config::EntryFnType;
use rustc::mir::{self, Location, PlaceBase, Promoted, Static, StaticKind};
use rustc::mir::visit::Visitor as MirVisitor;
use rustc::mir::mono::{MonoItem, InstantiationMode};
use rustc::mir::interpret::{Scalar, GlobalId, GlobalAlloc, ErrorHandled};

use crate::monomorphize;
use rustc::util::nodemap::{FxHashSet, FxHashMap, DefIdMap};
use rustc::util::common::time;

use rustc_data_structures::bit_set::GrowableBitSet;
use rustc_data_structures::sync::{MTRef, MTLock, ParallelIterator, par_iter};

use std::iter;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum MonoItemCollectionMode {
    Eager,
    Lazy
}

/// Maps every mono item to all mono items it references in its
/// body.
pub struct InliningMap<'tcx> {
    // Maps a source mono item to the range of mono items
    // accessed by it.
    // The two numbers in the tuple are the start (inclusive) and
    // end index (exclusive) within the `targets` vecs.
    index: FxHashMap<MonoItem<'tcx>, (usize, usize)>,
    targets: Vec<MonoItem<'tcx>>,

    // Contains one bit per mono item in the `targets` field. That bit
    // is true if that mono item needs to be inlined into every CGU.
    inlines: GrowableBitSet<usize>,
}

impl<'tcx> InliningMap<'tcx> {

    fn new() -> InliningMap<'tcx> {
        InliningMap {
            index: FxHashMap::default(),
            targets: Vec::new(),
            inlines: GrowableBitSet::with_capacity(1024),
        }
    }

    fn record_accesses<I>(&mut self,
                          source: MonoItem<'tcx>,
                          new_targets: I)
        where I: Iterator<Item=(MonoItem<'tcx>, bool)> + ExactSizeIterator
    {
        assert!(!self.index.contains_key(&source));

        let start_index = self.targets.len();
        let new_items_count = new_targets.len();
        let new_items_count_total = new_items_count + self.targets.len();

        self.targets.reserve(new_items_count);
        self.inlines.ensure(new_items_count_total);

        for (i, (target, inline)) in new_targets.enumerate() {
            self.targets.push(target);
            if inline {
                self.inlines.insert(i + start_index);
            }
        }

        let end_index = self.targets.len();
        self.index.insert(source, (start_index, end_index));
    }

    // Internally iterate over all items referenced by `source` which will be
    // made available for inlining.
    pub fn with_inlining_candidates<F>(&self, source: MonoItem<'tcx>, mut f: F)
        where F: FnMut(MonoItem<'tcx>)
    {
        if let Some(&(start_index, end_index)) = self.index.get(&source) {
            for (i, candidate) in self.targets[start_index .. end_index]
                                      .iter()
                                      .enumerate() {
                if self.inlines.contains(start_index + i) {
                    f(*candidate);
                }
            }
        }
    }

    // Internally iterate over all items and the things each accesses.
    pub fn iter_accesses<F>(&self, mut f: F)
        where F: FnMut(MonoItem<'tcx>, &[MonoItem<'tcx>])
    {
        for (&accessor, &(start_index, end_index)) in &self.index {
            f(accessor, &self.targets[start_index .. end_index])
        }
    }
}

pub fn collect_crate_mono_items(
    tcx: TyCtxt<'_>,
    mode: MonoItemCollectionMode,
) -> (FxHashSet<MonoItem<'_>>, InliningMap<'_>) {
    let roots = time(tcx.sess, "collecting roots", || {
        collect_roots(tcx, mode)
    });

    debug!("Building mono item graph, beginning at roots");

    let mut visited = MTLock::new(FxHashSet::default());
    let mut inlining_map = MTLock::new(InliningMap::new());

    {
        let visited: MTRef<'_, _> = &mut visited;
        let inlining_map: MTRef<'_, _> = &mut inlining_map;

        time(tcx.sess, "collecting mono items", || {
            par_iter(roots).for_each(|root| {
                let mut recursion_depths = DefIdMap::default();
                collect_items_rec(tcx,
                                root,
                                visited,
                                &mut recursion_depths,
                                inlining_map);
            });
        });
    }

    (visited.into_inner(), inlining_map.into_inner())
}

// Find all non-generic items by walking the HIR. These items serve as roots to
// start monomorphizing from.
fn collect_roots(tcx: TyCtxt<'_>, mode: MonoItemCollectionMode) -> Vec<MonoItem<'_>> {
    debug!("Collecting roots");
    let mut roots = Vec::new();

    {
        let entry_fn = tcx.entry_fn(LOCAL_CRATE);

        debug!("collect_roots: entry_fn = {:?}", entry_fn);

        let mut visitor = RootCollector {
            tcx,
            mode,
            entry_fn,
            output: &mut roots,
        };

        tcx.hir().krate().visit_all_item_likes(&mut visitor);

        visitor.push_extra_entry_roots();
    }

    // We can only codegen items that are instantiable - items all of
    // whose predicates hold. Luckily, items that aren't instantiable
    // can't actually be used, so we can just skip codegenning them.
    roots.retain(|root| root.is_instantiable(tcx));

    roots
}

// Collect all monomorphized items reachable from `starting_point`
fn collect_items_rec<'tcx>(
    tcx: TyCtxt<'tcx>,
    starting_point: MonoItem<'tcx>,
    visited: MTRef<'_, MTLock<FxHashSet<MonoItem<'tcx>>>>,
    recursion_depths: &mut DefIdMap<usize>,
    inlining_map: MTRef<'_, MTLock<InliningMap<'tcx>>>,
) {
    if !visited.lock_mut().insert(starting_point.clone()) {
        // We've been here already, no need to search again.
        return;
    }
    debug!("BEGIN collect_items_rec({})", starting_point.to_string(tcx, true));

    let mut neighbors = Vec::new();
    let recursion_depth_reset;

    match starting_point {
        MonoItem::Static(def_id) => {
            let instance = Instance::mono(tcx, def_id);

            // Sanity check whether this ended up being collected accidentally
            debug_assert!(should_monomorphize_locally(tcx, &instance));

            let ty = instance.ty(tcx);
            visit_drop_use(tcx, ty, true, &mut neighbors);

            recursion_depth_reset = None;

            let cid = GlobalId {
                instance,
                promoted: None,
            };
            let param_env = ty::ParamEnv::reveal_all();

            if let Ok(val) = tcx.const_eval(param_env.and(cid)) {
                collect_const(tcx, val, InternalSubsts::empty(), &mut neighbors);
            }
        }
        MonoItem::Fn(instance) => {
            // Sanity check whether this ended up being collected accidentally
            debug_assert!(should_monomorphize_locally(tcx, &instance));

            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(tcx,
                                                               instance,
                                                               recursion_depths));
            check_type_length_limit(tcx, instance);

            collect_neighbours(tcx, instance, &mut neighbors);
        }
        MonoItem::GlobalAsm(..) => {
            recursion_depth_reset = None;
        }
    }

    record_accesses(tcx, starting_point, &neighbors[..], inlining_map);

    for neighbour in neighbors {
        collect_items_rec(tcx, neighbour, visited, recursion_depths, inlining_map);
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }

    debug!("END collect_items_rec({})", starting_point.to_string(tcx, true));
}

fn record_accesses<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: MonoItem<'tcx>,
    callees: &[MonoItem<'tcx>],
    inlining_map: MTRef<'_, MTLock<InliningMap<'tcx>>>,
) {
    let is_inlining_candidate = |mono_item: &MonoItem<'tcx>| {
        mono_item.instantiation_mode(tcx) == InstantiationMode::LocalCopy
    };

    let accesses = callees.into_iter()
                          .map(|mono_item| {
                             (*mono_item, is_inlining_candidate(mono_item))
                          });

    inlining_map.lock_mut().record_accesses(caller, accesses);
}

fn check_recursion_limit<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    recursion_depths: &mut DefIdMap<usize>,
) -> (DefId, usize) {
    let def_id = instance.def_id();
    let recursion_depth = recursion_depths.get(&def_id).cloned().unwrap_or(0);
    debug!(" => recursion depth={}", recursion_depth);

    let recursion_depth = if Some(def_id) == tcx.lang_items().drop_in_place_fn() {
        // HACK: drop_in_place creates tight monomorphization loops. Give
        // it more margin.
        recursion_depth / 4
    } else {
        recursion_depth
    };

    // Code that needs to instantiate the same function recursively
    // more than the recursion limit is assumed to be causing an
    // infinite expansion.
    if recursion_depth > *tcx.sess.recursion_limit.get() {
        let error = format!("reached the recursion limit while instantiating `{}`",
                            instance);
        if let Some(hir_id) = tcx.hir().as_local_hir_id(def_id) {
            tcx.sess.span_fatal(tcx.hir().span(hir_id), &error);
        } else {
            tcx.sess.fatal(&error);
        }
    }

    recursion_depths.insert(def_id, recursion_depth + 1);

    (def_id, recursion_depth)
}

fn check_type_length_limit<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let type_length = instance.substs.types().flat_map(|ty| ty.walk()).count();
    let const_length = instance.substs.consts().flat_map(|ct| ct.ty.walk()).count();
    debug!(" => type length={}, const length={}", type_length, const_length);

    // Rust code can easily create exponentially-long types using only a
    // polynomial recursion depth. Even with the default recursion
    // depth, you can easily get cases that take >2^60 steps to run,
    // which means that rustc basically hangs.
    //
    // Bail out in these cases to avoid that bad user experience.
    let type_length_limit = *tcx.sess.type_length_limit.get();
    // We include the const length in the type length, as it's better
    // to be overly conservative.
    // FIXME(const_generics): we should instead uniformly walk through `substs`,
    // ignoring lifetimes.
    if type_length + const_length > type_length_limit {
        // The instance name is already known to be too long for rustc.
        // Show only the first and last 32 characters to avoid blasting
        // the user's terminal with thousands of lines of type-name.
        let shrink = |s: String, before: usize, after: usize| {
            // An iterator of all byte positions including the end of the string.
            let positions = || s.char_indices().map(|(i, _)| i).chain(iter::once(s.len()));

            let shrunk = format!(
                "{before}...{after}",
                before = &s[..positions().nth(before).unwrap_or(s.len())],
                after = &s[positions().rev().nth(after).unwrap_or(0)..],
            );

            // Only use the shrunk version if it's really shorter.
            // This also avoids the case where before and after slices overlap.
            if shrunk.len() < s.len() {
                shrunk
            } else {
                s
            }
        };
        let msg = format!("reached the type-length limit while instantiating `{}`",
                          shrink(instance.to_string(), 32, 32));
        let mut diag = tcx.sess.struct_span_fatal(tcx.def_span(instance.def_id()), &msg);
        diag.note(&format!(
            "consider adding a `#![type_length_limit=\"{}\"]` attribute to your crate",
            type_length));
        diag.emit();
        tcx.sess.abort_if_errors();
    }
}

struct MirNeighborCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    output: &'a mut Vec<MonoItem<'tcx>>,
    param_substs: SubstsRef<'tcx>,
}

impl<'a, 'tcx> MirVisitor<'tcx> for MirNeighborCollector<'a, 'tcx> {

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        debug!("visiting rvalue {:?}", *rvalue);

        match *rvalue {
            // When doing an cast from a regular pointer to a fat pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::Unsize), ref operand, target_ty
            ) => {
                let target_ty = self.tcx.subst_and_normalize_erasing_regions(
                    self.param_substs,
                    ty::ParamEnv::reveal_all(),
                    &target_ty,
                );
                let source_ty = operand.ty(self.body, self.tcx);
                let source_ty = self.tcx.subst_and_normalize_erasing_regions(
                    self.param_substs,
                    ty::ParamEnv::reveal_all(),
                    &source_ty,
                );
                let (source_ty, target_ty) = find_vtable_types_for_unsizing(self.tcx,
                                                                            source_ty,
                                                                            target_ty);
                // This could also be a different Unsize instruction, like
                // from a fixed sized array to a slice. But we are only
                // interested in things that produce a vtable.
                if target_ty.is_trait() && !source_ty.is_trait() {
                    create_mono_items_for_vtable_methods(self.tcx,
                                                         target_ty,
                                                         source_ty,
                                                         self.output);
                }
            }
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::ReifyFnPointer), ref operand, _
            ) => {
                let fn_ty = operand.ty(self.body, self.tcx);
                let fn_ty = self.tcx.subst_and_normalize_erasing_regions(
                    self.param_substs,
                    ty::ParamEnv::reveal_all(),
                    &fn_ty,
                );
                visit_fn_use(self.tcx, fn_ty, false, &mut self.output);
            }
            mir::Rvalue::Cast(
                mir::CastKind::Pointer(PointerCast::ClosureFnPointer(_)), ref operand, _
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                let source_ty = self.tcx.subst_and_normalize_erasing_regions(
                    self.param_substs,
                    ty::ParamEnv::reveal_all(),
                    &source_ty,
                );
                match source_ty.sty {
                    ty::Closure(def_id, substs) => {
                        let instance = Instance::resolve_closure(
                            self.tcx, def_id, substs, ty::ClosureKind::FnOnce);
                        if should_monomorphize_locally(self.tcx, &instance) {
                            self.output.push(create_fn_mono_item(instance));
                        }
                    }
                    _ => bug!(),
                }
            }
            mir::Rvalue::NullaryOp(mir::NullOp::Box, _) => {
                let tcx = self.tcx;
                let exchange_malloc_fn_def_id = tcx
                    .lang_items()
                    .require(ExchangeMallocFnLangItem)
                    .unwrap_or_else(|e| tcx.sess.fatal(&e));
                let instance = Instance::mono(tcx, exchange_malloc_fn_def_id);
                if should_monomorphize_locally(tcx, &instance) {
                    self.output.push(create_fn_mono_item(instance));
                }
            }
            _ => { /* not interesting */ }
        }

        self.super_rvalue(rvalue, location);
    }

    fn visit_const(&mut self, constant: &&'tcx ty::Const<'tcx>, location: Location) {
        debug!("visiting const {:?} @ {:?}", *constant, location);

        collect_const(self.tcx, *constant, self.param_substs, self.output);

        self.super_const(constant);
    }

    fn visit_terminator_kind(&mut self,
                             kind: &mir::TerminatorKind<'tcx>,
                             location: Location) {
        debug!("visiting terminator {:?} @ {:?}", kind, location);

        let tcx = self.tcx;
        match *kind {
            mir::TerminatorKind::Call { ref func, .. } => {
                let callee_ty = func.ty(self.body, tcx);
                let callee_ty = tcx.subst_and_normalize_erasing_regions(
                    self.param_substs,
                    ty::ParamEnv::reveal_all(),
                    &callee_ty,
                );
                visit_fn_use(self.tcx, callee_ty, true, &mut self.output);
            }
            mir::TerminatorKind::Drop { ref location, .. } |
            mir::TerminatorKind::DropAndReplace { ref location, .. } => {
                let ty = location.ty(self.body, self.tcx).ty;
                let ty = tcx.subst_and_normalize_erasing_regions(
                    self.param_substs,
                    ty::ParamEnv::reveal_all(),
                    &ty,
                );
                visit_drop_use(self.tcx, ty, true, self.output);
            }
            mir::TerminatorKind::Goto { .. } |
            mir::TerminatorKind::SwitchInt { .. } |
            mir::TerminatorKind::Resume |
            mir::TerminatorKind::Abort |
            mir::TerminatorKind::Return |
            mir::TerminatorKind::Unreachable |
            mir::TerminatorKind::Assert { .. } => {}
            mir::TerminatorKind::GeneratorDrop |
            mir::TerminatorKind::Yield { .. } |
            mir::TerminatorKind::FalseEdges { .. } |
            mir::TerminatorKind::FalseUnwind { .. } => bug!(),
        }

        self.super_terminator_kind(kind, location);
    }

    fn visit_place_base(&mut self,
                        place_base: &mir::PlaceBase<'tcx>,
                        _context: mir::visit::PlaceContext,
                        location: Location) {
        match place_base {
            PlaceBase::Static(box Static { kind: StaticKind::Static(def_id), .. }) => {
                debug!("visiting static {:?} @ {:?}", def_id, location);

                let tcx = self.tcx;
                let instance = Instance::mono(tcx, *def_id);
                if should_monomorphize_locally(tcx, &instance) {
                    self.output.push(MonoItem::Static(*def_id));
                }
            }
            PlaceBase::Static(box Static { kind: StaticKind::Promoted(_), .. }) => {
                // FIXME: should we handle promoteds here instead of eagerly in collect_neighbours?
            }
            PlaceBase::Local(_) => {
                // Locals have no relevance for collector
            }
        }
    }
}

fn visit_drop_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    is_direct_call: bool,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    let instance = Instance::resolve_drop_in_place(tcx, ty);
    visit_instance_use(tcx, instance, is_direct_call, output);
}

fn visit_fn_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    is_direct_call: bool,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    if let ty::FnDef(def_id, substs) = ty.sty {
        let instance = ty::Instance::resolve(tcx,
                                             ty::ParamEnv::reveal_all(),
                                             def_id,
                                             substs).unwrap();
        visit_instance_use(tcx, instance, is_direct_call, output);
    }
}

fn visit_instance_use<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    is_direct_call: bool,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    debug!("visit_item_use({:?}, is_direct_call={:?})", instance, is_direct_call);
    if !should_monomorphize_locally(tcx, &instance) {
        return
    }

    match instance.def {
        ty::InstanceDef::Intrinsic(def_id) => {
            if !is_direct_call {
                bug!("intrinsic {:?} being reified", def_id);
            }
        }
        ty::InstanceDef::VtableShim(..) |
        ty::InstanceDef::Virtual(..) |
        ty::InstanceDef::DropGlue(_, None) => {
            // don't need to emit shim if we are calling directly.
            if !is_direct_call {
                output.push(create_fn_mono_item(instance));
            }
        }
        ty::InstanceDef::DropGlue(_, Some(_)) => {
            output.push(create_fn_mono_item(instance));
        }
        ty::InstanceDef::ClosureOnceShim { .. } |
        ty::InstanceDef::Item(..) |
        ty::InstanceDef::FnPtrShim(..) |
        ty::InstanceDef::CloneShim(..) => {
            output.push(create_fn_mono_item(instance));
        }
    }
}

// Returns true if we should codegen an instance in the local crate.
// Returns false if we can just link to the upstream crate and therefore don't
// need a mono item.
fn should_monomorphize_locally<'tcx>(tcx: TyCtxt<'tcx>, instance: &Instance<'tcx>) -> bool {
    let def_id = match instance.def {
        ty::InstanceDef::Item(def_id) => def_id,
        ty::InstanceDef::VtableShim(..) |
        ty::InstanceDef::ClosureOnceShim { .. } |
        ty::InstanceDef::Virtual(..) |
        ty::InstanceDef::FnPtrShim(..) |
        ty::InstanceDef::DropGlue(..) |
        ty::InstanceDef::Intrinsic(_) |
        ty::InstanceDef::CloneShim(..) => return true
    };

    if tcx.is_foreign_item(def_id) {
        // We can always link to foreign items
        return false;
    }

    if def_id.is_local() {
        // local items cannot be referred to locally without monomorphizing them locally
        return true;
    }

    if tcx.is_reachable_non_generic(def_id) ||
       is_available_upstream_generic(tcx, def_id, instance.substs) {
        // We can link to the item in question, no instance needed
        // in this crate
        return false;
    }

    if !tcx.is_mir_available(def_id) {
        bug!("Cannot create local mono-item for {:?}", def_id)
    }
    return true;

    fn is_available_upstream_generic<'tcx>(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> bool {
        debug_assert!(!def_id.is_local());

        // If we are not in share generics mode, we don't link to upstream
        // monomorphizations but always instantiate our own internal versions
        // instead.
        if !tcx.sess.opts.share_generics() {
            return false
        }

        // If this instance has non-erasable parameters, it cannot be a shared
        // monomorphization. Non-generic instances are already handled above
        // by `is_reachable_non_generic()`
        if substs.non_erasable_generics().next().is_none() {
            return false
        }

        // Take a look at the available monomorphizations listed in the metadata
        // of upstream crates.
        tcx.upstream_monomorphizations_for(def_id)
           .map(|set| set.contains_key(substs))
           .unwrap_or(false)
    }
}

/// For given pair of source and target type that occur in an unsizing coercion,
/// this function finds the pair of types that determines the vtable linking
/// them.
///
/// For example, the source type might be `&SomeStruct` and the target type\
/// might be `&SomeTrait` in a cast like:
///
/// let src: &SomeStruct = ...;
/// let target = src as &SomeTrait;
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
/// let src: &ComplexStruct<SomeStruct> = ...;
/// let target = src as &ComplexStruct<SomeTrait>;
///
/// Again, we want this `find_vtable_types_for_unsizing()` to provide the pair
/// `(SomeStruct, SomeTrait)`.
///
/// Finally, there is also the case of custom unsizing coercions, e.g., for
/// smart pointers such as `Rc` and `Arc`.
fn find_vtable_types_for_unsizing<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> (Ty<'tcx>, Ty<'tcx>) {
    let ptr_vtable = |inner_source: Ty<'tcx>, inner_target: Ty<'tcx>| {
        let type_has_metadata = |ty: Ty<'tcx>| -> bool {
            use syntax_pos::DUMMY_SP;
            if ty.is_sized(tcx.at(DUMMY_SP), ty::ParamEnv::reveal_all()) {
                return false;
            }
            let tail = tcx.struct_tail(ty);
            match tail.sty {
                ty::Foreign(..) => false,
                ty::Str | ty::Slice(..) | ty::Dynamic(..) => true,
                _ => bug!("unexpected unsized tail: {:?}", tail),
            }
        };
        if type_has_metadata(inner_source) {
            (inner_source, inner_target)
        } else {
            tcx.struct_lockstep_tails(inner_source, inner_target)
        }
    };

    match (&source_ty.sty, &target_ty.sty) {
        (&ty::Ref(_, a, _),
         &ty::Ref(_, b, _)) |
        (&ty::Ref(_, a, _),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            ptr_vtable(a, b)
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            ptr_vtable(source_ty.boxed_ty(), target_ty.boxed_ty())
        }

        (&ty::Adt(source_adt_def, source_substs),
         &ty::Adt(target_adt_def, target_substs)) => {
            assert_eq!(source_adt_def, target_adt_def);

            let kind =
                monomorphize::custom_coerce_unsize_info(tcx, source_ty, target_ty);

            let coerce_index = match kind {
                CustomCoerceUnsized::Struct(i) => i
            };

            let source_fields = &source_adt_def.non_enum_variant().fields;
            let target_fields = &target_adt_def.non_enum_variant().fields;

            assert!(coerce_index < source_fields.len() &&
                    source_fields.len() == target_fields.len());

            find_vtable_types_for_unsizing(tcx,
                                           source_fields[coerce_index].ty(tcx,
                                                                          source_substs),
                                           target_fields[coerce_index].ty(tcx,
                                                                          target_substs))
        }
        _ => bug!("find_vtable_types_for_unsizing: invalid coercion {:?} -> {:?}",
                  source_ty,
                  target_ty)
    }
}

fn create_fn_mono_item(instance: Instance<'_>) -> MonoItem<'_> {
    debug!("create_fn_mono_item(instance={})", instance);
    MonoItem::Fn(instance)
}

/// Creates a `MonoItem` for each method that is referenced by the vtable for
/// the given trait/impl pair.
fn create_mono_items_for_vtable_methods<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ty: Ty<'tcx>,
    impl_ty: Ty<'tcx>,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    assert!(!trait_ty.needs_subst() && !trait_ty.has_escaping_bound_vars() &&
            !impl_ty.needs_subst() && !impl_ty.has_escaping_bound_vars());

    if let ty::Dynamic(ref trait_ty, ..) = trait_ty.sty {
        if let Some(principal) = trait_ty.principal() {
            let poly_trait_ref = principal.with_self_ty(tcx, impl_ty);
            assert!(!poly_trait_ref.has_escaping_bound_vars());

            // Walk all methods of the trait, including those of its supertraits
            let methods = tcx.vtable_methods(poly_trait_ref);
            let methods = methods.iter().cloned().filter_map(|method| method)
                .map(|(def_id, substs)| ty::Instance::resolve_for_vtable(
                    tcx,
                    ty::ParamEnv::reveal_all(),
                    def_id,
                    substs).unwrap())
                .filter(|&instance| should_monomorphize_locally(tcx, &instance))
                .map(|instance| create_fn_mono_item(instance));
            output.extend(methods);
        }

        // Also add the destructor
        visit_drop_use(tcx, impl_ty, false, output);
    }
}

//=-----------------------------------------------------------------------------
// Root Collection
//=-----------------------------------------------------------------------------

struct RootCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    mode: MonoItemCollectionMode,
    output: &'a mut Vec<MonoItem<'tcx>>,
    entry_fn: Option<(DefId, EntryFnType)>,
}

impl ItemLikeVisitor<'v> for RootCollector<'_, 'v> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemKind::ExternCrate(..) |
            hir::ItemKind::Use(..)         |
            hir::ItemKind::ForeignMod(..)  |
            hir::ItemKind::Ty(..)          |
            hir::ItemKind::Trait(..)       |
            hir::ItemKind::TraitAlias(..)  |
            hir::ItemKind::Existential(..) |
            hir::ItemKind::Mod(..)         => {
                // Nothing to do, just keep recursing...
            }

            hir::ItemKind::Impl(..) => {
                if self.mode == MonoItemCollectionMode::Eager {
                    create_mono_items_for_default_impls(self.tcx,
                                                        item,
                                                        self.output);
                }
            }

            hir::ItemKind::Enum(_, ref generics) |
            hir::ItemKind::Struct(_, ref generics) |
            hir::ItemKind::Union(_, ref generics) => {
                if generics.params.is_empty() {
                    if self.mode == MonoItemCollectionMode::Eager {
                        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                        debug!("RootCollector: ADT drop-glue for {}",
                               def_id_to_string(self.tcx, def_id));

                        let ty = Instance::new(def_id, InternalSubsts::empty()).ty(self.tcx);
                        visit_drop_use(self.tcx, ty, true, self.output);
                    }
                }
            }
            hir::ItemKind::GlobalAsm(..) => {
                debug!("RootCollector: ItemKind::GlobalAsm({})",
                       def_id_to_string(self.tcx,
                                        self.tcx.hir().local_def_id_from_hir_id(item.hir_id)));
                self.output.push(MonoItem::GlobalAsm(item.hir_id));
            }
            hir::ItemKind::Static(..) => {
                let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                debug!("RootCollector: ItemKind::Static({})",
                       def_id_to_string(self.tcx, def_id));
                self.output.push(MonoItem::Static(def_id));
            }
            hir::ItemKind::Const(..) => {
                // const items only generate mono items if they are
                // actually used somewhere. Just declaring them is insufficient.

                // but even just declaring them must collect the items they refer to
                let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);

                let instance = Instance::mono(self.tcx, def_id);
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                let param_env = ty::ParamEnv::reveal_all();

                if let Ok(val) = self.tcx.const_eval(param_env.and(cid)) {
                    collect_const(self.tcx, val, InternalSubsts::empty(), &mut self.output);
                }
            }
            hir::ItemKind::Fn(..) => {
                let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
                self.push_if_root(def_id);
            }
        }
    }

    fn visit_trait_item(&mut self, _: &'v hir::TraitItem) {
        // Even if there's a default body with no explicit generics,
        // it's still generic over some `Self: Trait`, so not a root.
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem) {
        match ii.node {
            hir::ImplItemKind::Method(hir::MethodSig { .. }, _) => {
                let def_id = self.tcx.hir().local_def_id_from_hir_id(ii.hir_id);
                self.push_if_root(def_id);
            }
            _ => { /* Nothing to do here */ }
        }
    }
}

impl RootCollector<'_, 'v> {
    fn is_root(&self, def_id: DefId) -> bool {
        !item_requires_monomorphization(self.tcx, def_id) && match self.mode {
            MonoItemCollectionMode::Eager => {
                true
            }
            MonoItemCollectionMode::Lazy => {
                self.entry_fn.map(|(id, _)| id) == Some(def_id) ||
                self.tcx.is_reachable_non_generic(def_id) ||
                self.tcx.codegen_fn_attrs(def_id).flags.contains(
                    CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
            }
        }
    }

    /// If `def_id` represents a root, then push it onto the list of
    /// outputs. (Note that all roots must be monomorphic.)
    fn push_if_root(&mut self, def_id: DefId) {
        if self.is_root(def_id) {
            debug!("RootCollector::push_if_root: found root def_id={:?}", def_id);

            let instance = Instance::mono(self.tcx, def_id);
            self.output.push(create_fn_mono_item(instance));
        }
    }

    /// As a special case, when/if we encounter the
    /// `main()` function, we also have to generate a
    /// monomorphized copy of the start lang item based on
    /// the return type of `main`. This is not needed when
    /// the user writes their own `start` manually.
    fn push_extra_entry_roots(&mut self) {
        let main_def_id = match self.entry_fn {
            Some((def_id, EntryFnType::Main)) => def_id,
            _ => return,
        };

        let start_def_id = match self.tcx.lang_items().require(StartFnLangItem) {
            Ok(s) => s,
            Err(err) => self.tcx.sess.fatal(&err),
        };
        let main_ret_ty = self.tcx.fn_sig(main_def_id).output();

        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = self.tcx.erase_regions(
            &main_ret_ty.no_bound_vars().unwrap(),
        );

        let start_instance = Instance::resolve(
            self.tcx,
            ty::ParamEnv::reveal_all(),
            start_def_id,
            self.tcx.intern_substs(&[main_ret_ty.into()])
        ).unwrap();

        self.output.push(create_fn_mono_item(start_instance));
    }
}

fn item_requires_monomorphization(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let generics = tcx.generics_of(def_id);
    generics.requires_monomorphization(tcx)
}

fn create_mono_items_for_default_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: &'tcx hir::Item,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    match item.node {
        hir::ItemKind::Impl(_, _, _, ref generics, .., ref impl_item_refs) => {
            for param in &generics.params {
                match param.kind {
                    hir::GenericParamKind::Lifetime { .. } => {}
                    hir::GenericParamKind::Type { .. } |
                    hir::GenericParamKind::Const { .. } => {
                        return
                    }
                }
            }

            let impl_def_id = tcx.hir().local_def_id_from_hir_id(item.hir_id);

            debug!("create_mono_items_for_default_impls(item={})",
                   def_id_to_string(tcx, impl_def_id));

            if let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                let overridden_methods: FxHashSet<_> =
                    impl_item_refs.iter()
                                  .map(|iiref| iiref.ident.modern())
                                  .collect();
                for method in tcx.provided_trait_methods(trait_ref.def_id) {
                    if overridden_methods.contains(&method.ident.modern()) {
                        continue;
                    }

                    if tcx.generics_of(method.def_id).own_requires_monomorphization() {
                        continue;
                    }

                    let substs = InternalSubsts::for_item(tcx, method.def_id, |param, _| {
                        match param.kind {
                            GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
                            GenericParamDefKind::Type { .. } |
                            GenericParamDefKind::Const => {
                                trait_ref.substs[param.index as usize]
                            }
                        }
                    });

                    let instance = ty::Instance::resolve(tcx,
                                                         ty::ParamEnv::reveal_all(),
                                                         method.def_id,
                                                         substs).unwrap();

                    let mono_item = create_fn_mono_item(instance);
                    if mono_item.is_instantiable(tcx)
                        && should_monomorphize_locally(tcx, &instance) {
                        output.push(mono_item);
                    }
                }
            }
        }
        _ => {
            bug!()
        }
    }
}

/// Scan the miri alloc in order to find function calls, closures, and drop-glue
fn collect_miri<'tcx>(tcx: TyCtxt<'tcx>, alloc_id: AllocId, output: &mut Vec<MonoItem<'tcx>>) {
    let alloc_kind = tcx.alloc_map.lock().get(alloc_id);
    match alloc_kind {
        Some(GlobalAlloc::Static(def_id)) => {
            let instance = Instance::mono(tcx, def_id);
            if should_monomorphize_locally(tcx, &instance) {
                trace!("collecting static {:?}", def_id);
                output.push(MonoItem::Static(def_id));
            }
        }
        Some(GlobalAlloc::Memory(alloc)) => {
            trace!("collecting {:?} with {:#?}", alloc_id, alloc);
            for &((), inner) in alloc.relocations.values() {
                collect_miri(tcx, inner, output);
            }
        },
        Some(GlobalAlloc::Function(fn_instance)) => {
            if should_monomorphize_locally(tcx, &fn_instance) {
                trace!("collecting {:?} with {:#?}", alloc_id, fn_instance);
                output.push(create_fn_mono_item(fn_instance));
            }
        }
        None => bug!("alloc id without corresponding allocation: {}", alloc_id),
    }
}

/// Scan the MIR in order to find function calls, closures, and drop-glue
fn collect_neighbours<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    let body = tcx.instance_mir(instance.def);

    MirNeighborCollector {
        tcx,
        body: &body,
        output,
        param_substs: instance.substs,
    }.visit_body(&body);
    let param_env = ty::ParamEnv::reveal_all();
    for i in 0..body.promoted.len() {
        use rustc_data_structures::indexed_vec::Idx;
        let i = Promoted::new(i);
        let cid = GlobalId {
            instance,
            promoted: Some(i),
        };
        match tcx.const_eval(param_env.and(cid)) {
            Ok(val) => collect_const(tcx, val, instance.substs, output),
            Err(ErrorHandled::Reported) => {},
            Err(ErrorHandled::TooGeneric) => span_bug!(
                body.promoted[i].span, "collection encountered polymorphic constant",
            ),
        }
    }
}

fn def_id_to_string(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    let mut output = String::new();
    let printer = DefPathBasedNames::new(tcx, false, false);
    printer.push_def_path(def_id, &mut output);
    output
}

fn collect_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: &'tcx ty::Const<'tcx>,
    param_substs: SubstsRef<'tcx>,
    output: &mut Vec<MonoItem<'tcx>>,
) {
    debug!("visiting const {:?}", constant);

    match constant.val {
        ConstValue::Scalar(Scalar::Ptr(ptr)) =>
            collect_miri(tcx, ptr.alloc_id, output),
        ConstValue::Slice { data: alloc, start: _, end: _ } |
        ConstValue::ByRef { alloc, .. } => {
            for &((), id) in alloc.relocations.values() {
                collect_miri(tcx, id, output);
            }
        }
        ConstValue::Unevaluated(def_id, substs) => {
            let param_env = ty::ParamEnv::reveal_all();
            let substs = tcx.subst_and_normalize_erasing_regions(
                param_substs,
                param_env,
                &substs,
            );
            let instance = ty::Instance::resolve(tcx,
                                                param_env,
                                                def_id,
                                                substs).unwrap();

            let cid = GlobalId {
                instance,
                promoted: None,
            };
            match tcx.const_eval(param_env.and(cid)) {
                Ok(val) => collect_const(tcx, val, param_substs, output),
                Err(ErrorHandled::Reported) => {},
                Err(ErrorHandled::TooGeneric) => span_bug!(
                    tcx.def_span(def_id), "collection encountered polymorphic constant",
                ),
            }
        }
        _ => {},
    }
}
