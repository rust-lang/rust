// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Translation Item Collection
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
//! The following kinds of "translation items" are handled here:
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
//! - A "translation item" is something that results in a function or global in
//!   the LLVM IR of a codegen unit. Translation items do not stand on their
//!   own, they can reference other translation items. For example, if function
//!   `foo()` calls function `bar()` then the translation item for `foo()`
//!   references the translation item for function `bar()`. In general, the
//!   definition for translation item A referencing a translation item B is that
//!   the LLVM artifact produced for A references the LLVM artifact produced
//!   for B.
//!
//! - Translation items and the references between them for a directed graph,
//!   where the translation items are the nodes and references form the edges.
//!   Let's call this graph the "translation item graph".
//!
//! - The translation item graph for a program contains all translation items
//!   that are needed in order to produce the complete LLVM IR of the program.
//!
//! The purpose of the algorithm implemented in this module is to build the
//! translation item graph for the current crate. It runs in two phases:
//!
//! 1. Discover the roots of the graph by traversing the HIR of the crate.
//! 2. Starting from the roots, find neighboring nodes by inspecting the MIR
//!    representation of the item corresponding to a given node, until no more
//!    new nodes are found.
//!
//! ### Discovering roots
//!
//! The roots of the translation item graph correspond to the non-generic
//! syntactic items in the source code. We find them by walking the HIR of the
//! crate, and whenever we hit upon a function, method, or static item, we
//! create a translation item consisting of the items DefId and, since we only
//! consider non-generic items, an empty type-substitution set.
//!
//! ### Finding neighbor nodes
//! Given a translation item node, we can discover neighbors by inspecting its
//! MIR. We walk the MIR and any time we hit upon something that signifies a
//! reference to another translation item, we have found a neighbor. Since the
//! translation item we are currently at is always monomorphic, we also know the
//! concrete type arguments of its neighbors, and so all neighbors again will be
//! monomorphic. The specific forms a reference to a neighboring node can take
//! in MIR are quite diverse. Here is an overview:
//!
//! #### Calling Functions/Methods
//! The most obvious form of one translation item referencing another is a
//! function or method call (represented by a CALL terminator in MIR). But
//! calls are not the only thing that might introduce a reference between two
//! function translation items, and as we will see below, they are just a
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
//! `print_val::<i32>`. Nonetheless, in order to translate this program, we need
//! an instance of this function. Thus, whenever we encounter a function or
//! method in operand position, we treat it as a neighbor of the current
//! translation item. Calls are just a special case of that.
//!
//! #### Closures
//! In a way, closures are a simple case. Since every closure object needs to be
//! constructed somewhere, we can reliably discover them by observing
//! `RValue::Aggregate` expressions with `AggregateKind::Closure`. This is also
//! true for closures inlined from other crates.
//!
//! #### Drop glue
//! Drop glue translation items are introduced by MIR drop-statements. The
//! generated translation item will again have drop-glue item neighbors if the
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
//! `exchange_malloc()` and `exchange_free()` may show up in MIR, even if the
//! compiler will generate them. We have to observe `Rvalue::Box` expressions
//! and Box-typed drop-statements for that purpose.
//!
//!
//! Interaction with Cross-Crate Inlining
//! -------------------------------------
//! The binary of a crate will not only contain machine code for the items
//! defined in the source code of that crate. It will also contain monomorphic
//! instantiations of any extern generic functions and of functions marked with
//! #[inline].
//! The collection algorithm handles this more or less transparently. If it is
//! about to create a translation item for something with an external `DefId`,
//! it will take a look if the MIR for that item is available, and if so just
//! proceed normally. If the MIR is not available, it assumes that the item is
//! just linked to and no node is created; which is exactly what we want, since
//! no machine code should be generated in the current crate for such an item.
//!
//! Eager and Lazy Collection Mode
//! ------------------------------
//! Translation item collection can be performed in one of two modes:
//!
//! - Lazy mode means that items will only be instantiated when actually
//!   referenced. The goal is to produce the least amount of machine code
//!   possible.
//!
//! - Eager mode is meant to be used in conjunction with incremental compilation
//!   where a stable set of translation items is more important than a minimal
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
//! ### Initializers of Constants and Statics
//! Since no MIR is constructed yet for initializer expressions of constants and
//! statics we cannot inspect these properly.
//!
//! ### Const Fns
//! Ideally, no translation item should be generated for const fns unless there
//! is a call to them that cannot be evaluated at compile time. At the moment
//! this is not implemented however: a translation item will be produced
//! regardless of whether it is actually needed or not.

use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;

use rustc::hir::map as hir_map;
use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::{BoxFreeFnLangItem, ExchangeMallocFnLangItem};
use rustc::traits;
use rustc::ty::subst::{Kind, Substs, Subst};
use rustc::ty::{self, TypeFoldable, TyCtxt};
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::mir::{self, Location};
use rustc::mir::visit as mir_visit;
use rustc::mir::visit::Visitor as MirVisitor;

use syntax::abi::Abi;
use syntax_pos::DUMMY_SP;
use base::custom_coerce_unsize_info;
use callee::needs_fn_once_adapter_shim;
use context::SharedCrateContext;
use common::fulfill_obligation;
use glue::{self, DropGlueKind};
use monomorphize::{self, Instance};
use util::nodemap::{FxHashSet, FxHashMap, DefIdMap};

use trans_item::{TransItem, DefPathBasedNames, InstantiationMode};

use std::iter;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum TransItemCollectionMode {
    Eager,
    Lazy
}

/// Maps every translation item to all translation items it references in its
/// body.
pub struct InliningMap<'tcx> {
    // Maps a source translation item to a range of target translation items
    // that are potentially inlined by LLVM into the source.
    // The two numbers in the tuple are the start (inclusive) and
    // end index (exclusive) within the `targets` vecs.
    index: FxHashMap<TransItem<'tcx>, (usize, usize)>,
    targets: Vec<TransItem<'tcx>>,
}

impl<'tcx> InliningMap<'tcx> {

    fn new() -> InliningMap<'tcx> {
        InliningMap {
            index: FxHashMap(),
            targets: Vec::new(),
        }
    }

    fn record_inlining_canditates<I>(&mut self,
                                     source: TransItem<'tcx>,
                                     targets: I)
        where I: Iterator<Item=TransItem<'tcx>>
    {
        assert!(!self.index.contains_key(&source));

        let start_index = self.targets.len();
        self.targets.extend(targets);
        let end_index = self.targets.len();
        self.index.insert(source, (start_index, end_index));
    }

    // Internally iterate over all items referenced by `source` which will be
    // made available for inlining.
    pub fn with_inlining_candidates<F>(&self, source: TransItem<'tcx>, mut f: F)
        where F: FnMut(TransItem<'tcx>) {
        if let Some(&(start_index, end_index)) = self.index.get(&source)
        {
            for candidate in &self.targets[start_index .. end_index] {
                f(*candidate)
            }
        }
    }
}

pub fn collect_crate_translation_items<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                                 mode: TransItemCollectionMode)
                                                 -> (FxHashSet<TransItem<'tcx>>,
                                                     InliningMap<'tcx>) {
    // We are not tracking dependencies of this pass as it has to be re-executed
    // every time no matter what.
    scx.tcx().dep_graph.with_ignore(|| {
        let roots = collect_roots(scx, mode);

        debug!("Building translation item graph, beginning at roots");
        let mut visited = FxHashSet();
        let mut recursion_depths = DefIdMap();
        let mut inlining_map = InliningMap::new();

        for root in roots {
            collect_items_rec(scx,
                              root,
                              &mut visited,
                              &mut recursion_depths,
                              &mut inlining_map);
        }

        (visited, inlining_map)
    })
}

// Find all non-generic items by walking the HIR. These items serve as roots to
// start monomorphizing from.
fn collect_roots<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                           mode: TransItemCollectionMode)
                           -> Vec<TransItem<'tcx>> {
    debug!("Collecting roots");
    let mut roots = Vec::new();

    {
        let mut visitor = RootCollector {
            scx: scx,
            mode: mode,
            output: &mut roots,
        };

        scx.tcx().map.krate().visit_all_item_likes(&mut visitor);
    }

    roots
}

// Collect all monomorphized translation items reachable from `starting_point`
fn collect_items_rec<'a, 'tcx: 'a>(scx: &SharedCrateContext<'a, 'tcx>,
                                   starting_point: TransItem<'tcx>,
                                   visited: &mut FxHashSet<TransItem<'tcx>>,
                                   recursion_depths: &mut DefIdMap<usize>,
                                   inlining_map: &mut InliningMap<'tcx>) {
    if !visited.insert(starting_point.clone()) {
        // We've been here already, no need to search again.
        return;
    }
    debug!("BEGIN collect_items_rec({})", starting_point.to_string(scx.tcx()));

    let mut neighbors = Vec::new();
    let recursion_depth_reset;

    match starting_point {
        TransItem::DropGlue(t) => {
            find_drop_glue_neighbors(scx, t, &mut neighbors);
            recursion_depth_reset = None;
        }
        TransItem::Static(node_id) => {
            let def_id = scx.tcx().map.local_def_id(node_id);

            // Sanity check whether this ended up being collected accidentally
            debug_assert!(should_trans_locally(scx.tcx(), def_id));

            let ty = scx.tcx().item_type(def_id);
            let ty = glue::get_drop_glue_type(scx, ty);
            neighbors.push(TransItem::DropGlue(DropGlueKind::Ty(ty)));

            recursion_depth_reset = None;

            collect_neighbours(scx, Instance::mono(scx, def_id), &mut neighbors);
        }
        TransItem::Fn(instance) => {
            // Sanity check whether this ended up being collected accidentally
            debug_assert!(should_trans_locally(scx.tcx(), instance.def));

            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(scx.tcx(),
                                                               instance,
                                                               recursion_depths));
            check_type_length_limit(scx.tcx(), instance);

            collect_neighbours(scx, instance, &mut neighbors);
        }
    }

    record_inlining_canditates(scx.tcx(), starting_point, &neighbors[..], inlining_map);

    for neighbour in neighbors {
        collect_items_rec(scx, neighbour, visited, recursion_depths, inlining_map);
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }

    debug!("END collect_items_rec({})", starting_point.to_string(scx.tcx()));
}

fn record_inlining_canditates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        caller: TransItem<'tcx>,
                                        callees: &[TransItem<'tcx>],
                                        inlining_map: &mut InliningMap<'tcx>) {
    let is_inlining_candidate = |trans_item: &TransItem<'tcx>| {
        trans_item.instantiation_mode(tcx) == InstantiationMode::LocalCopy
    };

    let inlining_candidates = callees.into_iter()
                                     .map(|x| *x)
                                     .filter(is_inlining_candidate);

    inlining_map.record_inlining_canditates(caller, inlining_candidates);
}

fn check_recursion_limit<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                   instance: Instance<'tcx>,
                                   recursion_depths: &mut DefIdMap<usize>)
                                   -> (DefId, usize) {
    let recursion_depth = recursion_depths.get(&instance.def)
                                          .map(|x| *x)
                                          .unwrap_or(0);
    debug!(" => recursion depth={}", recursion_depth);

    // Code that needs to instantiate the same function recursively
    // more than the recursion limit is assumed to be causing an
    // infinite expansion.
    if recursion_depth > tcx.sess.recursion_limit.get() {
        let error = format!("reached the recursion limit while instantiating `{}`",
                            instance);
        if let Some(node_id) = tcx.map.as_local_node_id(instance.def) {
            tcx.sess.span_fatal(tcx.map.span(node_id), &error);
        } else {
            tcx.sess.fatal(&error);
        }
    }

    recursion_depths.insert(instance.def, recursion_depth + 1);

    (instance.def, recursion_depth)
}

fn check_type_length_limit<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     instance: Instance<'tcx>)
{
    let type_length = instance.substs.types().flat_map(|ty| ty.walk()).count();
    debug!(" => type length={}", type_length);

    // Rust code can easily create exponentially-long types using only a
    // polynomial recursion depth. Even with the default recursion
    // depth, you can easily get cases that take >2^60 steps to run,
    // which means that rustc basically hangs.
    //
    // Bail out in these cases to avoid that bad user experience.
    let type_length_limit = tcx.sess.type_length_limit.get();
    if type_length > type_length_limit {
        // The instance name is already known to be too long for rustc. Use
        // `{:.64}` to avoid blasting the user's terminal with thousands of
        // lines of type-name.
        let instance_name = instance.to_string();
        let msg = format!("reached the type-length limit while instantiating `{:.64}...`",
                          instance_name);
        let mut diag = if let Some(node_id) = tcx.map.as_local_node_id(instance.def) {
            tcx.sess.struct_span_fatal(tcx.map.span(node_id), &msg)
        } else {
            tcx.sess.struct_fatal(&msg)
        };

        diag.note(&format!(
            "consider adding a `#![type_length_limit=\"{}\"]` attribute to your crate",
            type_length_limit*2));
        diag.emit();
        tcx.sess.abort_if_errors();
    }
}

struct MirNeighborCollector<'a, 'tcx: 'a> {
    scx: &'a SharedCrateContext<'a, 'tcx>,
    mir: &'a mir::Mir<'tcx>,
    output: &'a mut Vec<TransItem<'tcx>>,
    param_substs: &'tcx Substs<'tcx>
}

impl<'a, 'tcx> MirVisitor<'tcx> for MirNeighborCollector<'a, 'tcx> {

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        debug!("visiting rvalue {:?}", *rvalue);

        match *rvalue {
            // When doing an cast from a regular pointer to a fat pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(mir::CastKind::Unsize, ref operand, target_ty) => {
                let target_ty = monomorphize::apply_param_substs(self.scx,
                                                                 self.param_substs,
                                                                 &target_ty);
                let source_ty = operand.ty(self.mir, self.scx.tcx());
                let source_ty = monomorphize::apply_param_substs(self.scx,
                                                                 self.param_substs,
                                                                 &source_ty);
                let (source_ty, target_ty) = find_vtable_types_for_unsizing(self.scx,
                                                                            source_ty,
                                                                            target_ty);
                // This could also be a different Unsize instruction, like
                // from a fixed sized array to a slice. But we are only
                // interested in things that produce a vtable.
                if target_ty.is_trait() && !source_ty.is_trait() {
                    create_trans_items_for_vtable_methods(self.scx,
                                                          target_ty,
                                                          source_ty,
                                                          self.output);
                }
            }
            mir::Rvalue::Box(..) => {
                let exchange_malloc_fn_def_id =
                    self.scx
                        .tcx()
                        .lang_items
                        .require(ExchangeMallocFnLangItem)
                        .unwrap_or_else(|e| self.scx.sess().fatal(&e));

                if should_trans_locally(self.scx.tcx(), exchange_malloc_fn_def_id) {
                    let empty_substs = self.scx.empty_substs_for_def_id(exchange_malloc_fn_def_id);
                    let exchange_malloc_fn_trans_item =
                        create_fn_trans_item(self.scx,
                                             exchange_malloc_fn_def_id,
                                             empty_substs,
                                             self.param_substs);

                    self.output.push(exchange_malloc_fn_trans_item);
                }
            }
            _ => { /* not interesting */ }
        }

        self.super_rvalue(rvalue, location);
    }

    fn visit_lvalue(&mut self,
                    lvalue: &mir::Lvalue<'tcx>,
                    context: mir_visit::LvalueContext<'tcx>,
                    location: Location) {
        debug!("visiting lvalue {:?}", *lvalue);

        if let mir_visit::LvalueContext::Drop = context {
            let ty = lvalue.ty(self.mir, self.scx.tcx())
                           .to_ty(self.scx.tcx());

            let ty = monomorphize::apply_param_substs(self.scx,
                                                      self.param_substs,
                                                      &ty);
            assert!(ty.is_normalized_for_trans());
            let ty = glue::get_drop_glue_type(self.scx, ty);
            self.output.push(TransItem::DropGlue(DropGlueKind::Ty(ty)));
        }

        self.super_lvalue(lvalue, context, location);
    }

    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: Location) {
        debug!("visiting operand {:?}", *operand);

        let callee = match *operand {
            mir::Operand::Constant(ref constant) => {
                if let ty::TyFnDef(def_id, substs, _) = constant.ty.sty {
                    // This is something that can act as a callee, proceed
                    Some((def_id, substs))
                } else {
                    // This is not a callee, but we still have to look for
                    // references to `const` items
                    if let mir::Literal::Item { def_id, substs } = constant.literal {
                        let substs = monomorphize::apply_param_substs(self.scx,
                                                                      self.param_substs,
                                                                      &substs);

                        let instance = Instance::new(def_id, substs).resolve_const(self.scx);
                        collect_neighbours(self.scx, instance, self.output);
                    }

                    None
                }
            }
            _ => None
        };

        if let Some((callee_def_id, callee_substs)) = callee {
            debug!(" => operand is callable");

            // `callee_def_id` might refer to a trait method instead of a
            // concrete implementation, so we have to find the actual
            // implementation. For example, the call might look like
            //
            // std::cmp::partial_cmp(0i32, 1i32)
            //
            // Calling do_static_dispatch() here will map the def_id of
            // `std::cmp::partial_cmp` to the def_id of `i32::partial_cmp<i32>`
            let dispatched = do_static_dispatch(self.scx,
                                                callee_def_id,
                                                callee_substs,
                                                self.param_substs);

            if let StaticDispatchResult::Dispatched {
                    def_id: callee_def_id,
                    substs: callee_substs,
                    fn_once_adjustment,
                } = dispatched {
                // if we have a concrete impl (which we might not have
                // in the case of something compiler generated like an
                // object shim or a closure that is handled differently),
                // we check if the callee is something that will actually
                // result in a translation item ...
                if can_result_in_trans_item(self.scx.tcx(), callee_def_id) {
                    // ... and create one if it does.
                    let trans_item = create_fn_trans_item(self.scx,
                                                          callee_def_id,
                                                          callee_substs,
                                                          self.param_substs);
                    self.output.push(trans_item);

                    // This call will instantiate an FnOnce adapter, which drops
                    // the closure environment. Therefore we need to make sure
                    // that we collect the drop-glue for the environment type.
                    if let Some(env_ty) = fn_once_adjustment {
                        let env_ty = glue::get_drop_glue_type(self.scx, env_ty);
                        if self.scx.type_needs_drop(env_ty) {
                            let dg = DropGlueKind::Ty(env_ty);
                            self.output.push(TransItem::DropGlue(dg));
                        }
                    }
                }
            }
        }

        self.super_operand(operand, location);

        fn can_result_in_trans_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                              def_id: DefId)
                                              -> bool {
            match tcx.item_type(def_id).sty {
                ty::TyFnDef(def_id, _, f) => {
                    // Some constructors also have type TyFnDef but they are
                    // always instantiated inline and don't result in a
                    // translation item. Same for FFI functions.
                    if let Some(hir_map::NodeForeignItem(_)) = tcx.map.get_if_local(def_id) {
                        return false;
                    }

                    if let Some(adt_def) = f.sig.output().skip_binder().ty_adt_def() {
                        if adt_def.variants.iter().any(|v| def_id == v.did) {
                            return false;
                        }
                    }
                }
                ty::TyClosure(..) => {}
                _ => return false
            }

            should_trans_locally(tcx, def_id)
        }
    }

    // This takes care of the "drop_in_place" intrinsic for which we otherwise
    // we would not register drop-glues.
    fn visit_terminator_kind(&mut self,
                             block: mir::BasicBlock,
                             kind: &mir::TerminatorKind<'tcx>,
                             location: Location) {
        let tcx = self.scx.tcx();
        match *kind {
            mir::TerminatorKind::Call {
                func: mir::Operand::Constant(ref constant),
                ref args,
                ..
            } => {
                match constant.ty.sty {
                    ty::TyFnDef(def_id, _, bare_fn_ty)
                        if is_drop_in_place_intrinsic(tcx, def_id, bare_fn_ty) => {
                        let operand_ty = args[0].ty(self.mir, tcx);
                        if let ty::TyRawPtr(mt) = operand_ty.sty {
                            let operand_ty = monomorphize::apply_param_substs(self.scx,
                                                                              self.param_substs,
                                                                              &mt.ty);
                            let ty = glue::get_drop_glue_type(self.scx, operand_ty);
                            self.output.push(TransItem::DropGlue(DropGlueKind::Ty(ty)));
                        } else {
                            bug!("Has the drop_in_place() intrinsic's signature changed?")
                        }
                    }
                    _ => { /* Nothing to do. */ }
                }
            }
            _ => { /* Nothing to do. */ }
        }

        self.super_terminator_kind(block, kind, location);

        fn is_drop_in_place_intrinsic<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                def_id: DefId,
                                                bare_fn_ty: &ty::BareFnTy<'tcx>)
                                                -> bool {
            (bare_fn_ty.abi == Abi::RustIntrinsic ||
             bare_fn_ty.abi == Abi::PlatformIntrinsic) &&
            tcx.item_name(def_id) == "drop_in_place"
        }
    }
}

// Returns true if we should translate an instance in the local crate.
// Returns false if we can just link to the upstream crate and therefore don't
// need a translation item.
fn should_trans_locally<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  def_id: DefId)
                                  -> bool {
    if def_id.is_local() {
        true
    } else {
        if tcx.sess.cstore.is_exported_symbol(def_id) ||
           tcx.sess.cstore.is_foreign_item(def_id) {
            // We can link to the item in question, no instance needed in this
            // crate
            false
        } else {
            if !tcx.sess.cstore.is_item_mir_available(def_id) {
                bug!("Cannot create local trans-item for {:?}", def_id)
            }
            true
        }
    }
}

fn find_drop_glue_neighbors<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                      dg: DropGlueKind<'tcx>,
                                      output: &mut Vec<TransItem<'tcx>>) {
    let ty = match dg {
        DropGlueKind::Ty(ty) => ty,
        DropGlueKind::TyContents(_) => {
            // We already collected the neighbors of this item via the
            // DropGlueKind::Ty variant.
            return
        }
    };

    debug!("find_drop_glue_neighbors: {}", type_to_string(scx.tcx(), ty));

    // Make sure the BoxFreeFn lang-item gets translated if there is a boxed value.
    if let ty::TyBox(content_type) = ty.sty {
        let def_id = scx.tcx().require_lang_item(BoxFreeFnLangItem);

        if should_trans_locally(scx.tcx(), def_id) {
            let box_free_fn_trans_item =
                create_fn_trans_item(scx,
                                     def_id,
                                     scx.tcx().mk_substs(iter::once(Kind::from(content_type))),
                                     scx.tcx().intern_substs(&[]));
            output.push(box_free_fn_trans_item);
        }
    }

    // If the type implements Drop, also add a translation item for the
    // monomorphized Drop::drop() implementation.
    let destructor_did = match ty.sty {
        ty::TyAdt(def, _) => def.destructor(),
        _ => None
    };

    if let Some(destructor_did) = destructor_did {
        use rustc::ty::ToPolyTraitRef;

        let drop_trait_def_id = scx.tcx()
                                   .lang_items
                                   .drop_trait()
                                   .unwrap();

        let self_type_substs = scx.tcx().mk_substs_trait(ty, &[]);

        let trait_ref = ty::TraitRef {
            def_id: drop_trait_def_id,
            substs: self_type_substs,
        }.to_poly_trait_ref();

        let substs = match fulfill_obligation(scx, DUMMY_SP, trait_ref) {
            traits::VtableImpl(data) => data.substs,
            _ => bug!()
        };

        if should_trans_locally(scx.tcx(), destructor_did) {
            let trans_item = create_fn_trans_item(scx,
                                                  destructor_did,
                                                  substs,
                                                  scx.tcx().intern_substs(&[]));
            output.push(trans_item);
        }

        // This type has a Drop implementation, we'll need the contents-only
        // version of the glue too.
        output.push(TransItem::DropGlue(DropGlueKind::TyContents(ty)));
    }

    // Finally add the types of nested values
    match ty.sty {
        ty::TyBool      |
        ty::TyChar      |
        ty::TyInt(_)    |
        ty::TyUint(_)   |
        ty::TyStr       |
        ty::TyFloat(_)  |
        ty::TyRawPtr(_) |
        ty::TyRef(..)   |
        ty::TyFnDef(..) |
        ty::TyFnPtr(_)  |
        ty::TyNever     |
        ty::TyDynamic(..)  => {
            /* nothing to do */
        }
        ty::TyAdt(adt_def, substs) => {
            for field in adt_def.all_fields() {
                let field_type = scx.tcx().item_type(field.did);
                let field_type = monomorphize::apply_param_substs(scx,
                                                                  substs,
                                                                  &field_type);
                let field_type = glue::get_drop_glue_type(scx, field_type);

                if scx.type_needs_drop(field_type) {
                    output.push(TransItem::DropGlue(DropGlueKind::Ty(field_type)));
                }
            }
        }
        ty::TyClosure(def_id, substs) => {
            for upvar_ty in substs.upvar_tys(def_id, scx.tcx()) {
                let upvar_ty = glue::get_drop_glue_type(scx, upvar_ty);
                if scx.type_needs_drop(upvar_ty) {
                    output.push(TransItem::DropGlue(DropGlueKind::Ty(upvar_ty)));
                }
            }
        }
        ty::TyBox(inner_type)      |
        ty::TySlice(inner_type)    |
        ty::TyArray(inner_type, _) => {
            let inner_type = glue::get_drop_glue_type(scx, inner_type);
            if scx.type_needs_drop(inner_type) {
                output.push(TransItem::DropGlue(DropGlueKind::Ty(inner_type)));
            }
        }
        ty::TyTuple(args) => {
            for arg in args {
                let arg = glue::get_drop_glue_type(scx, arg);
                if scx.type_needs_drop(arg) {
                    output.push(TransItem::DropGlue(DropGlueKind::Ty(arg)));
                }
            }
        }
        ty::TyProjection(_) |
        ty::TyParam(_)      |
        ty::TyInfer(_)      |
        ty::TyAnon(..)      |
        ty::TyError         => {
            bug!("encountered unexpected type");
        }
    }
}

fn do_static_dispatch<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                fn_def_id: DefId,
                                fn_substs: &'tcx Substs<'tcx>,
                                param_substs: &'tcx Substs<'tcx>)
                                -> StaticDispatchResult<'tcx> {
    debug!("do_static_dispatch(fn_def_id={}, fn_substs={:?}, param_substs={:?})",
           def_id_to_string(scx.tcx(), fn_def_id),
           fn_substs,
           param_substs);

    if let Some(trait_def_id) = scx.tcx().trait_of_item(fn_def_id) {
        debug!(" => trait method, attempting to find impl");
        do_static_trait_method_dispatch(scx,
                                        &scx.tcx().associated_item(fn_def_id),
                                        trait_def_id,
                                        fn_substs,
                                        param_substs)
    } else {
        debug!(" => regular function");
        // The function is not part of an impl or trait, no dispatching
        // to be done
        StaticDispatchResult::Dispatched {
            def_id: fn_def_id,
            substs: fn_substs,
            fn_once_adjustment: None,
        }
    }
}

enum StaticDispatchResult<'tcx> {
    // The call could be resolved statically as going to the method with
    // `def_id` and `substs`.
    Dispatched {
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,

        // If this is a call to a closure that needs an FnOnce adjustment,
        // this contains the new self type of the call (= type of the closure
        // environment)
        fn_once_adjustment: Option<ty::Ty<'tcx>>,
    },
    // This goes to somewhere that we don't know at compile-time
    Unknown
}

// Given a trait-method and substitution information, find out the actual
// implementation of the trait method.
fn do_static_trait_method_dispatch<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                             trait_method: &ty::AssociatedItem,
                                             trait_id: DefId,
                                             callee_substs: &'tcx Substs<'tcx>,
                                             param_substs: &'tcx Substs<'tcx>)
                                             -> StaticDispatchResult<'tcx> {
    let tcx = scx.tcx();
    debug!("do_static_trait_method_dispatch(trait_method={}, \
                                            trait_id={}, \
                                            callee_substs={:?}, \
                                            param_substs={:?}",
           def_id_to_string(scx.tcx(), trait_method.def_id),
           def_id_to_string(scx.tcx(), trait_id),
           callee_substs,
           param_substs);

    let rcvr_substs = monomorphize::apply_param_substs(scx,
                                                       param_substs,
                                                       &callee_substs);
    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);
    let vtbl = fulfill_obligation(scx, DUMMY_SP, ty::Binder(trait_ref));

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        traits::VtableImpl(impl_data) => {
            let (def_id, substs) = traits::find_method(tcx,
                                                       trait_method.name,
                                                       rcvr_substs,
                                                       &impl_data);
            StaticDispatchResult::Dispatched {
                def_id: def_id,
                substs: substs,
                fn_once_adjustment: None,
            }
        }
        traits::VtableClosure(closure_data) => {
            let closure_def_id = closure_data.closure_def_id;
            let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
            let actual_closure_kind = tcx.closure_kind(closure_def_id);

            let needs_fn_once_adapter_shim =
                match needs_fn_once_adapter_shim(actual_closure_kind,
                                                 trait_closure_kind) {
                Ok(true) => true,
                _ => false,
            };

            let fn_once_adjustment = if needs_fn_once_adapter_shim {
                Some(tcx.mk_closure_from_closure_substs(closure_def_id,
                                                        closure_data.substs))
            } else {
                None
            };

            StaticDispatchResult::Dispatched {
                def_id: closure_def_id,
                substs: closure_data.substs.substs,
                fn_once_adjustment: fn_once_adjustment,
            }
        }
        traits::VtableFnPointer(ref data) => {
            // If we know the destination of this fn-pointer, we'll have to make
            // sure that this destination actually gets instantiated.
            if let ty::TyFnDef(def_id, substs, _) = data.fn_ty.sty {
                // The destination of the pointer might be something that needs
                // further dispatching, such as a trait method, so we do that.
                do_static_dispatch(scx, def_id, substs, param_substs)
            } else {
                StaticDispatchResult::Unknown
            }
        }
        // Trait object shims are always instantiated in-place, and as they are
        // just an ABI-adjusting indirect call they do not have any dependencies.
        traits::VtableObject(..) => {
            StaticDispatchResult::Unknown
        }
        _ => {
            bug!("static call to invalid vtable: {:?}", vtbl)
        }
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
/// Finally, there is also the case of custom unsizing coercions, e.g. for
/// smart pointers such as `Rc` and `Arc`.
fn find_vtable_types_for_unsizing<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                            source_ty: ty::Ty<'tcx>,
                                            target_ty: ty::Ty<'tcx>)
                                            -> (ty::Ty<'tcx>, ty::Ty<'tcx>) {
    match (&source_ty.sty, &target_ty.sty) {
        (&ty::TyBox(a), &ty::TyBox(b)) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRef(_, ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) => {
            let (inner_source, inner_target) = (a, b);

            if !scx.type_is_sized(inner_source) {
                (inner_source, inner_target)
            } else {
                scx.tcx().struct_lockstep_tails(inner_source, inner_target)
            }
        }

        (&ty::TyAdt(source_adt_def, source_substs),
         &ty::TyAdt(target_adt_def, target_substs)) => {
            assert_eq!(source_adt_def, target_adt_def);

            let kind = custom_coerce_unsize_info(scx, source_ty, target_ty);

            let coerce_index = match kind {
                CustomCoerceUnsized::Struct(i) => i
            };

            let source_fields = &source_adt_def.struct_variant().fields;
            let target_fields = &target_adt_def.struct_variant().fields;

            assert!(coerce_index < source_fields.len() &&
                    source_fields.len() == target_fields.len());

            find_vtable_types_for_unsizing(scx,
                                           source_fields[coerce_index].ty(scx.tcx(),
                                                                          source_substs),
                                           target_fields[coerce_index].ty(scx.tcx(),
                                                                          target_substs))
        }
        _ => bug!("find_vtable_types_for_unsizing: invalid coercion {:?} -> {:?}",
                  source_ty,
                  target_ty)
    }
}

fn create_fn_trans_item<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                  def_id: DefId,
                                  fn_substs: &'tcx Substs<'tcx>,
                                  param_substs: &'tcx Substs<'tcx>)
                                  -> TransItem<'tcx> {
    let tcx = scx.tcx();

    debug!("create_fn_trans_item(def_id={}, fn_substs={:?}, param_substs={:?})",
            def_id_to_string(tcx, def_id),
            fn_substs,
            param_substs);

    // We only get here, if fn_def_id either designates a local item or
    // an inlineable external item. Non-inlineable external items are
    // ignored because we don't want to generate any code for them.
    let concrete_substs = monomorphize::apply_param_substs(scx,
                                                           param_substs,
                                                           &fn_substs);
    assert!(concrete_substs.is_normalized_for_trans(),
            "concrete_substs not normalized for trans: {:?}",
            concrete_substs);
    TransItem::Fn(Instance::new(def_id, concrete_substs))
}

/// Creates a `TransItem` for each method that is referenced by the vtable for
/// the given trait/impl pair.
fn create_trans_items_for_vtable_methods<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                                   trait_ty: ty::Ty<'tcx>,
                                                   impl_ty: ty::Ty<'tcx>,
                                                   output: &mut Vec<TransItem<'tcx>>) {
    assert!(!trait_ty.needs_subst() && !impl_ty.needs_subst());

    if let ty::TyDynamic(ref trait_ty, ..) = trait_ty.sty {
        if let Some(principal) = trait_ty.principal() {
            let poly_trait_ref = principal.with_self_ty(scx.tcx(), impl_ty);
            let param_substs = scx.tcx().intern_substs(&[]);

            // Walk all methods of the trait, including those of its supertraits
            let methods = traits::get_vtable_methods(scx.tcx(), poly_trait_ref);
            let methods = methods.filter_map(|method| method)
                .filter_map(|(def_id, substs)| {
                    if let StaticDispatchResult::Dispatched {
                        def_id,
                        substs,
                        // We already add the drop-glue for the closure env
                        // unconditionally below.
                        fn_once_adjustment: _ ,
                    } = do_static_dispatch(scx, def_id, substs, param_substs) {
                        Some((def_id, substs))
                    } else {
                        None
                    }
                })
                .filter(|&(def_id, _)| should_trans_locally(scx.tcx(), def_id))
                .map(|(def_id, substs)| create_fn_trans_item(scx, def_id, substs, param_substs));
            output.extend(methods);
        }
        // Also add the destructor
        let dg_type = glue::get_drop_glue_type(scx, impl_ty);
        output.push(TransItem::DropGlue(DropGlueKind::Ty(dg_type)));
    }
}

//=-----------------------------------------------------------------------------
// Root Collection
//=-----------------------------------------------------------------------------

struct RootCollector<'b, 'a: 'b, 'tcx: 'a + 'b> {
    scx: &'b SharedCrateContext<'a, 'tcx>,
    mode: TransItemCollectionMode,
    output: &'b mut Vec<TransItem<'tcx>>,
}

impl<'b, 'a, 'v> ItemLikeVisitor<'v> for RootCollector<'b, 'a, 'v> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemExternCrate(..) |
            hir::ItemUse(..)         |
            hir::ItemForeignMod(..)  |
            hir::ItemTy(..)          |
            hir::ItemDefaultImpl(..) |
            hir::ItemTrait(..)       |
            hir::ItemMod(..)         => {
                // Nothing to do, just keep recursing...
            }

            hir::ItemImpl(..) => {
                if self.mode == TransItemCollectionMode::Eager {
                    create_trans_items_for_default_impls(self.scx,
                                                         item,
                                                         self.output);
                }
            }

            hir::ItemEnum(_, ref generics) |
            hir::ItemStruct(_, ref generics) |
            hir::ItemUnion(_, ref generics) => {
                if !generics.is_parameterized() {
                    if self.mode == TransItemCollectionMode::Eager {
                        let def_id = self.scx.tcx().map.local_def_id(item.id);
                        debug!("RootCollector: ADT drop-glue for {}",
                               def_id_to_string(self.scx.tcx(), def_id));

                        let ty = self.scx.tcx().item_type(def_id);
                        let ty = glue::get_drop_glue_type(self.scx, ty);
                        self.output.push(TransItem::DropGlue(DropGlueKind::Ty(ty)));
                    }
                }
            }
            hir::ItemStatic(..) => {
                debug!("RootCollector: ItemStatic({})",
                       def_id_to_string(self.scx.tcx(),
                                        self.scx.tcx().map.local_def_id(item.id)));
                self.output.push(TransItem::Static(item.id));
            }
            hir::ItemConst(..) => {
                // const items only generate translation items if they are
                // actually used somewhere. Just declaring them is insufficient.
            }
            hir::ItemFn(.., ref generics, _) => {
                if !generics.is_type_parameterized() {
                    let def_id = self.scx.tcx().map.local_def_id(item.id);

                    debug!("RootCollector: ItemFn({})",
                           def_id_to_string(self.scx.tcx(), def_id));

                    let instance = Instance::mono(self.scx, def_id);
                    self.output.push(TransItem::Fn(instance));
                }
            }
        }
    }

    fn visit_trait_item(&mut self, _: &'v hir::TraitItem) {
        // Even if there's a default body with no explicit generics,
        // it's still generic over some `Self: Trait`, so not a root.
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem) {
        match ii.node {
            hir::ImplItemKind::Method(hir::MethodSig {
                ref generics,
                ..
            }, _) => {
                let hir_map = &self.scx.tcx().map;
                let parent_node_id = hir_map.get_parent_node(ii.id);
                let is_impl_generic = match hir_map.expect_item(parent_node_id) {
                    &hir::Item {
                        node: hir::ItemImpl(_, _, ref generics, ..),
                        ..
                    } => {
                        generics.is_type_parameterized()
                    }
                    _ => {
                        bug!()
                    }
                };

                if !generics.is_type_parameterized() && !is_impl_generic {
                    let def_id = self.scx.tcx().map.local_def_id(ii.id);

                    debug!("RootCollector: MethodImplItem({})",
                           def_id_to_string(self.scx.tcx(), def_id));

                    let instance = Instance::mono(self.scx, def_id);
                    self.output.push(TransItem::Fn(instance));
                }
            }
            _ => { /* Nothing to do here */ }
        }
    }
}

fn create_trans_items_for_default_impls<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                                  item: &'tcx hir::Item,
                                                  output: &mut Vec<TransItem<'tcx>>) {
    let tcx = scx.tcx();
    match item.node {
        hir::ItemImpl(_,
                      _,
                      ref generics,
                      ..,
                      ref impl_item_refs) => {
            if generics.is_type_parameterized() {
                return
            }

            let impl_def_id = tcx.map.local_def_id(item.id);

            debug!("create_trans_items_for_default_impls(item={})",
                   def_id_to_string(tcx, impl_def_id));

            if let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                let callee_substs = tcx.erase_regions(&trait_ref.substs);
                let overridden_methods: FxHashSet<_> =
                    impl_item_refs.iter()
                                  .map(|iiref| iiref.name)
                                  .collect();
                for method in tcx.provided_trait_methods(trait_ref.def_id) {
                    if overridden_methods.contains(&method.name) {
                        continue;
                    }

                    if !tcx.item_generics(method.def_id).types.is_empty() {
                        continue;
                    }

                    // The substitutions we have are on the impl, so we grab
                    // the method type from the impl to substitute into.
                    let impl_substs = Substs::for_item(tcx, impl_def_id,
                                                       |_, _| tcx.mk_region(ty::ReErased),
                                                       |_, _| tcx.types.err);
                    let impl_data = traits::VtableImplData {
                        impl_def_id: impl_def_id,
                        substs: impl_substs,
                        nested: vec![]
                    };
                    let (def_id, substs) = traits::find_method(tcx,
                                                               method.name,
                                                               callee_substs,
                                                               &impl_data);

                    let predicates = tcx.item_predicates(def_id).predicates
                                        .subst(tcx, substs);
                    if !traits::normalize_and_test_predicates(tcx, predicates) {
                        continue;
                    }

                    if should_trans_locally(tcx, method.def_id) {
                        let item = create_fn_trans_item(scx,
                                                        method.def_id,
                                                        callee_substs,
                                                        tcx.erase_regions(&substs));
                        output.push(item);
                    }
                }
            }
        }
        _ => {
            bug!()
        }
    }
}

/// Scan the MIR in order to find function calls, closures, and drop-glue
fn collect_neighbours<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                instance: Instance<'tcx>,
                                output: &mut Vec<TransItem<'tcx>>)
{
    let mir = scx.tcx().item_mir(instance.def);

    let mut visitor = MirNeighborCollector {
        scx: scx,
        mir: &mir,
        output: output,
        param_substs: instance.substs
    };

    visitor.visit_mir(&mir);
    for promoted in &mir.promoted {
        visitor.mir = promoted;
        visitor.visit_mir(promoted);
    }
}

fn def_id_to_string<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              def_id: DefId)
                              -> String {
    let mut output = String::new();
    let printer = DefPathBasedNames::new(tcx, false, false);
    printer.push_def_path(def_id, &mut output);
    output
}

fn type_to_string<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            ty: ty::Ty<'tcx>)
                            -> String {
    let mut output = String::new();
    let printer = DefPathBasedNames::new(tcx, false, false);
    printer.push_type_name(ty, &mut output);
    output
}
