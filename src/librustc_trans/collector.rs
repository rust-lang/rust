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
//! proceed normally. If the MIR is not available, it assumes that that item is
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

use rustc_data_structures::bitvec::BitVector;

use rustc::hir;
use rustc::hir::intravisit as hir_visit;

use rustc::hir::map as hir_map;
use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::{ExchangeFreeFnLangItem, ExchangeMallocFnLangItem};
use rustc::traits;
use rustc::ty::subst::{self, Substs, Subst};
use rustc::ty::{self, TypeFoldable, TyCtxt};
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::mir::repr as mir;
use rustc::mir::visit as mir_visit;
use rustc::mir::visit::Visitor as MirVisitor;

use syntax::codemap::DUMMY_SP;
use syntax::errors;

use base::custom_coerce_unsize_info;
use context::SharedCrateContext;
use common::{fulfill_obligation, normalize_and_test_predicates, type_is_sized};
use glue::{self, DropGlueKind};
use meth;
use monomorphize::{self, Instance};
use util::nodemap::{FnvHashSet, FnvHashMap, DefIdMap};

use std::hash::{Hash, Hasher};
use trans_item::{TransItem, type_to_string, def_id_to_string};

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum TransItemCollectionMode {
    Eager,
    Lazy
}

/// Maps every translation item to all translation items it references in its
/// body.
pub struct ReferenceMap<'tcx> {
    // Maps a source translation item to a range of target translation items.
    // The two numbers in the tuple are the start (inclusive) and
    // end index (exclusive) within the `targets` and the `inlined` vecs.
    index: FnvHashMap<TransItem<'tcx>, (usize, usize)>,
    targets: Vec<TransItem<'tcx>>,
    inlined: BitVector
}

impl<'tcx> ReferenceMap<'tcx> {

    fn new() -> ReferenceMap<'tcx> {
        ReferenceMap {
            index: FnvHashMap(),
            targets: Vec::new(),
            inlined: BitVector::new(64 * 256),
        }
    }

    fn record_references<I>(&mut self, source: TransItem<'tcx>, targets: I)
        where I: Iterator<Item=(TransItem<'tcx>, bool)>
    {
        assert!(!self.index.contains_key(&source));

        let start_index = self.targets.len();

        for (target, inlined) in targets {
            let index = self.targets.len();
            self.targets.push(target);
            self.inlined.grow(index + 1);

            if inlined {
                self.inlined.insert(index);
            }
        }

        let end_index = self.targets.len();
        self.index.insert(source, (start_index, end_index));
    }

    // Internally iterate over all items referenced by `source` which will be
    // made available for inlining.
    pub fn with_inlining_candidates<F>(&self, source: TransItem<'tcx>, mut f: F)
        where F: FnMut(TransItem<'tcx>) {
        if let Some(&(start_index, end_index)) = self.index.get(&source)
        {
            for index in start_index .. end_index {
                if self.inlined.contains(index) {
                    f(self.targets[index])
                }
            }
        }
    }

    pub fn get_direct_references_from(&self, source: TransItem<'tcx>) -> &[TransItem<'tcx>]
    {
        if let Some(&(start_index, end_index)) = self.index.get(&source) {
            &self.targets[start_index .. end_index]
        } else {
            &self.targets[0 .. 0]
        }
    }
}

pub fn collect_crate_translation_items<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                                 mode: TransItemCollectionMode)
                                                 -> (FnvHashSet<TransItem<'tcx>>,
                                                     ReferenceMap<'tcx>) {
    // We are not tracking dependencies of this pass as it has to be re-executed
    // every time no matter what.
    scx.tcx().dep_graph.with_ignore(|| {
        let roots = collect_roots(scx, mode);

        debug!("Building translation item graph, beginning at roots");
        let mut visited = FnvHashSet();
        let mut recursion_depths = DefIdMap();
        let mut reference_map = ReferenceMap::new();

        for root in roots {
            collect_items_rec(scx,
                              root,
                              &mut visited,
                              &mut recursion_depths,
                              &mut reference_map);
        }

        (visited, reference_map)
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
            enclosing_item: None,
        };

        scx.tcx().map.krate().visit_all_items(&mut visitor);
    }

    roots
}

// Collect all monomorphized translation items reachable from `starting_point`
fn collect_items_rec<'a, 'tcx: 'a>(scx: &SharedCrateContext<'a, 'tcx>,
                                   starting_point: TransItem<'tcx>,
                                   visited: &mut FnvHashSet<TransItem<'tcx>>,
                                   recursion_depths: &mut DefIdMap<usize>,
                                   reference_map: &mut ReferenceMap<'tcx>) {
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
            let ty = scx.tcx().lookup_item_type(def_id).ty;
            let ty = glue::get_drop_glue_type(scx.tcx(), ty);
            neighbors.push(TransItem::DropGlue(DropGlueKind::Ty(ty)));
            recursion_depth_reset = None;
        }
        TransItem::Fn(instance) => {
            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(scx.tcx(),
                                                               instance,
                                                               recursion_depths));

            // Scan the MIR in order to find function calls, closures, and
            // drop-glue
            let mir = errors::expect(scx.sess().diagnostic(), scx.get_mir(instance.def),
                || format!("Could not find MIR for function: {}", instance));

            let mut visitor = MirNeighborCollector {
                scx: scx,
                mir: &mir,
                output: &mut neighbors,
                param_substs: instance.substs
            };

            visitor.visit_mir(&mir);
            for promoted in &mir.promoted {
                visitor.visit_mir(promoted);
            }
        }
    }

    record_references(scx.tcx(), starting_point, &neighbors[..], reference_map);

    for neighbour in neighbors {
        collect_items_rec(scx, neighbour, visited, recursion_depths, reference_map);
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }

    debug!("END collect_items_rec({})", starting_point.to_string(scx.tcx()));
}

fn record_references<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               caller: TransItem<'tcx>,
                               callees: &[TransItem<'tcx>],
                               reference_map: &mut ReferenceMap<'tcx>) {
    let iter = callees.into_iter()
                      .map(|callee| {
                        let is_inlining_candidate = callee.is_from_extern_crate() ||
                                                    callee.requests_inline(tcx);
                        (*callee, is_inlining_candidate)
                      });
    reference_map.record_references(caller, iter);
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

struct MirNeighborCollector<'a, 'tcx: 'a> {
    scx: &'a SharedCrateContext<'a, 'tcx>,
    mir: &'a mir::Mir<'tcx>,
    output: &'a mut Vec<TransItem<'tcx>>,
    param_substs: &'tcx Substs<'tcx>
}

impl<'a, 'tcx> MirVisitor<'tcx> for MirNeighborCollector<'a, 'tcx> {

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>) {
        debug!("visiting rvalue {:?}", *rvalue);

        match *rvalue {
            mir::Rvalue::Aggregate(mir::AggregateKind::Closure(def_id,
                                                               ref substs), _) => {
                assert!(can_have_local_instance(self.scx.tcx(), def_id));
                let trans_item = create_fn_trans_item(self.scx.tcx(),
                                                      def_id,
                                                      substs.func_substs,
                                                      self.param_substs);
                self.output.push(trans_item);
            }
            // When doing an cast from a regular pointer to a fat pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(mir::CastKind::Unsize, ref operand, target_ty) => {
                let target_ty = monomorphize::apply_param_substs(self.scx.tcx(),
                                                                 self.param_substs,
                                                                 &target_ty);
                let source_ty = self.mir.operand_ty(self.scx.tcx(), operand);
                let source_ty = monomorphize::apply_param_substs(self.scx.tcx(),
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
            mir::Rvalue::Box(_) => {
                let exchange_malloc_fn_def_id =
                    self.scx
                        .tcx()
                        .lang_items
                        .require(ExchangeMallocFnLangItem)
                        .unwrap_or_else(|e| self.scx.sess().fatal(&e));

                assert!(can_have_local_instance(self.scx.tcx(), exchange_malloc_fn_def_id));
                let exchange_malloc_fn_trans_item =
                    create_fn_trans_item(self.scx.tcx(),
                                         exchange_malloc_fn_def_id,
                                         self.scx.tcx().mk_substs(Substs::empty()),
                                         self.param_substs);

                self.output.push(exchange_malloc_fn_trans_item);
            }
            _ => { /* not interesting */ }
        }

        self.super_rvalue(rvalue);
    }

    fn visit_lvalue(&mut self,
                    lvalue: &mir::Lvalue<'tcx>,
                    context: mir_visit::LvalueContext) {
        debug!("visiting lvalue {:?}", *lvalue);

        if let mir_visit::LvalueContext::Drop = context {
            let ty = self.mir.lvalue_ty(self.scx.tcx(), lvalue)
                             .to_ty(self.scx.tcx());

            let ty = monomorphize::apply_param_substs(self.scx.tcx(),
                                                      self.param_substs,
                                                      &ty);
            let ty = self.scx.tcx().erase_regions(&ty);
            let ty = glue::get_drop_glue_type(self.scx.tcx(), ty);
            self.output.push(TransItem::DropGlue(DropGlueKind::Ty(ty)));
        }

        self.super_lvalue(lvalue, context);
    }

    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>) {
        debug!("visiting operand {:?}", *operand);

        let callee = match *operand {
            mir::Operand::Constant(mir::Constant { ty: &ty::TyS {
                sty: ty::TyFnDef(def_id, substs, _), ..
            }, .. }) => Some((def_id, substs)),
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

            if let Some((callee_def_id, callee_substs)) = dispatched {
                // if we have a concrete impl (which we might not have
                // in the case of something compiler generated like an
                // object shim or a closure that is handled differently),
                // we check if the callee is something that will actually
                // result in a translation item ...
                if can_result_in_trans_item(self.scx.tcx(), callee_def_id) {
                    // ... and create one if it does.
                    let trans_item = create_fn_trans_item(self.scx.tcx(),
                                                          callee_def_id,
                                                          callee_substs,
                                                          self.param_substs);
                    self.output.push(trans_item);
                }
            }
        }

        self.super_operand(operand);

        fn can_result_in_trans_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                              def_id: DefId)
                                              -> bool {
            if !match tcx.lookup_item_type(def_id).ty.sty {
                ty::TyFnDef(def_id, _, _) => {
                    // Some constructors also have type TyFnDef but they are
                    // always instantiated inline and don't result in
                    // translation item. Same for FFI functions.
                    match tcx.map.get_if_local(def_id) {
                        Some(hir_map::NodeVariant(_))    |
                        Some(hir_map::NodeStructCtor(_)) |
                        Some(hir_map::NodeForeignItem(_)) => false,
                        Some(_) => true,
                        None => {
                            tcx.sess.cstore.variant_kind(def_id).is_none()
                        }
                    }
                }
                ty::TyClosure(..) => true,
                _ => false
            } {
                return false;
            }

            can_have_local_instance(tcx, def_id)
        }
    }
}

fn can_have_local_instance<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     def_id: DefId)
                                     -> bool {
    // Take a look if we have the definition available. If not, we
    // will not emit code for this item in the local crate, and thus
    // don't create a translation item for it.
    def_id.is_local() || tcx.sess.cstore.is_item_mir_available(def_id)
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

    // Make sure the exchange_free_fn() lang-item gets translated if
    // there is a boxed value.
    if let ty::TyBox(_) = ty.sty {
        let exchange_free_fn_def_id = scx.tcx()
                                         .lang_items
                                         .require(ExchangeFreeFnLangItem)
                                         .unwrap_or_else(|e| scx.sess().fatal(&e));

        assert!(can_have_local_instance(scx.tcx(), exchange_free_fn_def_id));
        let exchange_free_fn_trans_item =
            create_fn_trans_item(scx.tcx(),
                                 exchange_free_fn_def_id,
                                 scx.tcx().mk_substs(Substs::empty()),
                                 scx.tcx().mk_substs(Substs::empty()));

        output.push(exchange_free_fn_trans_item);
    }

    // If the type implements Drop, also add a translation item for the
    // monomorphized Drop::drop() implementation.
    let destructor_did = match ty.sty {
        ty::TyStruct(def, _) |
        ty::TyEnum(def, _)   => def.destructor(),
        _ => None
    };

    if let Some(destructor_did) = destructor_did {
        use rustc::ty::ToPolyTraitRef;

        let drop_trait_def_id = scx.tcx()
                                   .lang_items
                                   .drop_trait()
                                   .unwrap();

        let self_type_substs = scx.tcx().mk_substs(
            Substs::empty().with_self_ty(ty));

        let trait_ref = ty::TraitRef {
            def_id: drop_trait_def_id,
            substs: self_type_substs,
        }.to_poly_trait_ref();

        let substs = match fulfill_obligation(scx, DUMMY_SP, trait_ref) {
            traits::VtableImpl(data) => data.substs,
            _ => bug!()
        };

        if can_have_local_instance(scx.tcx(), destructor_did) {
            let trans_item = create_fn_trans_item(scx.tcx(),
                                                  destructor_did,
                                                  substs,
                                                  scx.tcx().mk_substs(Substs::empty()));
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
        ty::TySlice(_)  |
        ty::TyTrait(_)  => {
            /* nothing to do */
        }
        ty::TyStruct(ref adt_def, substs) |
        ty::TyEnum(ref adt_def, substs) => {
            for field in adt_def.all_fields() {
                let field_type = monomorphize::apply_param_substs(scx.tcx(),
                                                                  substs,
                                                                  &field.unsubst_ty());
                let field_type = glue::get_drop_glue_type(scx.tcx(), field_type);

                if glue::type_needs_drop(scx.tcx(), field_type) {
                    output.push(TransItem::DropGlue(DropGlueKind::Ty(field_type)));
                }
            }
        }
        ty::TyClosure(_, substs) => {
            for upvar_ty in substs.upvar_tys {
                let upvar_ty = glue::get_drop_glue_type(scx.tcx(), upvar_ty);
                if glue::type_needs_drop(scx.tcx(), upvar_ty) {
                    output.push(TransItem::DropGlue(DropGlueKind::Ty(upvar_ty)));
                }
            }
        }
        ty::TyBox(inner_type)      |
        ty::TyArray(inner_type, _) => {
            let inner_type = glue::get_drop_glue_type(scx.tcx(), inner_type);
            if glue::type_needs_drop(scx.tcx(), inner_type) {
                output.push(TransItem::DropGlue(DropGlueKind::Ty(inner_type)));
            }
        }
        ty::TyTuple(args) => {
            for arg in args {
                let arg = glue::get_drop_glue_type(scx.tcx(), arg);
                if glue::type_needs_drop(scx.tcx(), arg) {
                    output.push(TransItem::DropGlue(DropGlueKind::Ty(arg)));
                }
            }
        }
        ty::TyProjection(_) |
        ty::TyParam(_)      |
        ty::TyInfer(_)      |
        ty::TyError         => {
            bug!("encountered unexpected type");
        }
    }
}

fn do_static_dispatch<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                fn_def_id: DefId,
                                fn_substs: &'tcx Substs<'tcx>,
                                param_substs: &'tcx Substs<'tcx>)
                                -> Option<(DefId, &'tcx Substs<'tcx>)> {
    debug!("do_static_dispatch(fn_def_id={}, fn_substs={:?}, param_substs={:?})",
           def_id_to_string(scx.tcx(), fn_def_id),
           fn_substs,
           param_substs);

    let is_trait_method = scx.tcx().trait_of_item(fn_def_id).is_some();

    if is_trait_method {
        match scx.tcx().impl_or_trait_item(fn_def_id) {
            ty::MethodTraitItem(ref method) => {
                match method.container {
                    ty::TraitContainer(trait_def_id) => {
                        debug!(" => trait method, attempting to find impl");
                        do_static_trait_method_dispatch(scx,
                                                        method,
                                                        trait_def_id,
                                                        fn_substs,
                                                        param_substs)
                    }
                    ty::ImplContainer(_) => {
                        // This is already a concrete implementation
                        debug!(" => impl method");
                        Some((fn_def_id, fn_substs))
                    }
                }
            }
            _ => bug!()
        }
    } else {
        debug!(" => regular function");
        // The function is not part of an impl or trait, no dispatching
        // to be done
        Some((fn_def_id, fn_substs))
    }
}

// Given a trait-method and substitution information, find out the actual
// implementation of the trait method.
fn do_static_trait_method_dispatch<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                             trait_method: &ty::Method,
                                             trait_id: DefId,
                                             callee_substs: &'tcx Substs<'tcx>,
                                             param_substs: &'tcx Substs<'tcx>)
                                             -> Option<(DefId, &'tcx Substs<'tcx>)> {
    let tcx = scx.tcx();
    debug!("do_static_trait_method_dispatch(trait_method={}, \
                                            trait_id={}, \
                                            callee_substs={:?}, \
                                            param_substs={:?}",
           def_id_to_string(scx.tcx(), trait_method.def_id),
           def_id_to_string(scx.tcx(), trait_id),
           callee_substs,
           param_substs);

    let rcvr_substs = monomorphize::apply_param_substs(tcx,
                                                       param_substs,
                                                       &callee_substs);

    let trait_ref = ty::Binder(rcvr_substs.to_trait_ref(tcx, trait_id));
    let vtbl = fulfill_obligation(scx, DUMMY_SP, trait_ref);

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        traits::VtableImpl(traits::VtableImplData {
            impl_def_id: impl_did,
            substs: impl_substs,
            nested: _ }) =>
        {
            let callee_substs = impl_substs.with_method_from(&rcvr_substs);
            let impl_method = meth::get_impl_method(tcx,
                                                    impl_did,
                                                    tcx.mk_substs(callee_substs),
                                                    trait_method.name);
            Some((impl_method.method.def_id, &impl_method.substs))
        }
        // If we have a closure or a function pointer, we will also encounter
        // the concrete closure/function somewhere else (during closure or fn
        // pointer construction). That's where we track those things.
        traits::VtableClosure(..) |
        traits::VtableFnPointer(..) |
        traits::VtableObject(..) => {
            None
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

            if !type_is_sized(scx.tcx(), inner_source) {
                (inner_source, inner_target)
            } else {
                scx.tcx().struct_lockstep_tails(inner_source, inner_target)
            }
        }

        (&ty::TyStruct(source_adt_def, source_substs),
         &ty::TyStruct(target_adt_def, target_substs)) => {
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

fn create_fn_trans_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  def_id: DefId,
                                  fn_substs: &'tcx Substs<'tcx>,
                                  param_substs: &'tcx Substs<'tcx>)
                                  -> TransItem<'tcx> {
    debug!("create_fn_trans_item(def_id={}, fn_substs={:?}, param_substs={:?})",
            def_id_to_string(tcx, def_id),
            fn_substs,
            param_substs);

    // We only get here, if fn_def_id either designates a local item or
    // an inlineable external item. Non-inlineable external items are
    // ignored because we don't want to generate any code for them.
    let concrete_substs = monomorphize::apply_param_substs(tcx,
                                                           param_substs,
                                                           &fn_substs);
    let concrete_substs = tcx.erase_regions(&concrete_substs);

    let trans_item =
        TransItem::Fn(Instance::new(def_id, concrete_substs));
    return trans_item;
}

/// Creates a `TransItem` for each method that is referenced by the vtable for
/// the given trait/impl pair.
fn create_trans_items_for_vtable_methods<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                                   trait_ty: ty::Ty<'tcx>,
                                                   impl_ty: ty::Ty<'tcx>,
                                                   output: &mut Vec<TransItem<'tcx>>) {
    assert!(!trait_ty.needs_subst() && !impl_ty.needs_subst());

    if let ty::TyTrait(ref trait_ty) = trait_ty.sty {
        let poly_trait_ref = trait_ty.principal_trait_ref_with_self_ty(scx.tcx(),
                                                                       impl_ty);

        // Walk all methods of the trait, including those of its supertraits
        for trait_ref in traits::supertraits(scx.tcx(), poly_trait_ref) {
            let vtable = fulfill_obligation(scx, DUMMY_SP, trait_ref);
            match vtable {
                traits::VtableImpl(
                    traits::VtableImplData {
                        impl_def_id,
                        substs,
                        nested: _ }) => {
                    let items = meth::get_vtable_methods(scx.tcx(), impl_def_id, substs)
                        .into_iter()
                        // filter out None values
                        .filter_map(|opt_impl_method| opt_impl_method)
                        // create translation items
                        .filter_map(|impl_method| {
                            if can_have_local_instance(scx.tcx(), impl_method.method.def_id) {
                                Some(create_fn_trans_item(scx.tcx(),
                                    impl_method.method.def_id,
                                    impl_method.substs,
                                    scx.tcx().mk_substs(Substs::empty())))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    output.extend(items.into_iter());
                }
                _ => { /* */ }
            }
        }
    }
}

//=-----------------------------------------------------------------------------
// Root Collection
//=-----------------------------------------------------------------------------

struct RootCollector<'b, 'a: 'b, 'tcx: 'a + 'b> {
    scx: &'b SharedCrateContext<'a, 'tcx>,
    mode: TransItemCollectionMode,
    output: &'b mut Vec<TransItem<'tcx>>,
    enclosing_item: Option<&'tcx hir::Item>,
}

impl<'b, 'a, 'v> hir_visit::Visitor<'v> for RootCollector<'b, 'a, 'v> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        let old_enclosing_item = self.enclosing_item;
        self.enclosing_item = Some(item);

        match item.node {
            hir::ItemExternCrate(..) |
            hir::ItemUse(..)         |
            hir::ItemForeignMod(..)  |
            hir::ItemTy(..)          |
            hir::ItemDefaultImpl(..) |
            hir::ItemTrait(..)       |
            hir::ItemConst(..)       |
            hir::ItemMod(..)         => {
                // Nothing to do, just keep recursing...
            }

            hir::ItemImpl(..) => {
                if self.mode == TransItemCollectionMode::Eager {
                    create_trans_items_for_default_impls(self.scx.tcx(),
                                                         item,
                                                         self.output);
                }
            }

            hir::ItemEnum(_, ref generics)        |
            hir::ItemStruct(_, ref generics)      => {
                if !generics.is_parameterized() {
                    let ty = {
                        let tables = self.scx.tcx().tables.borrow();
                        tables.node_types[&item.id]
                    };

                    if self.mode == TransItemCollectionMode::Eager {
                        debug!("RootCollector: ADT drop-glue for {}",
                               def_id_to_string(self.scx.tcx(),
                                                self.scx.tcx().map.local_def_id(item.id)));

                        let ty = glue::get_drop_glue_type(self.scx.tcx(), ty);
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
            hir::ItemFn(_, _, constness, _, ref generics, _) => {
                if !generics.is_type_parameterized() &&
                   constness == hir::Constness::NotConst {
                    let def_id = self.scx.tcx().map.local_def_id(item.id);

                    debug!("RootCollector: ItemFn({})",
                           def_id_to_string(self.scx.tcx(), def_id));

                    let instance = Instance::mono(self.scx.tcx(), def_id);
                    self.output.push(TransItem::Fn(instance));
                }
            }
        }

        hir_visit::walk_item(self, item);
        self.enclosing_item = old_enclosing_item;
    }

    fn visit_impl_item(&mut self, ii: &'v hir::ImplItem) {
        match ii.node {
            hir::ImplItemKind::Method(hir::MethodSig {
                ref generics,
                constness,
                ..
            }, _) if constness == hir::Constness::NotConst => {
                let hir_map = &self.scx.tcx().map;
                let parent_node_id = hir_map.get_parent_node(ii.id);
                let is_impl_generic = match hir_map.expect_item(parent_node_id) {
                    &hir::Item {
                        node: hir::ItemImpl(_, _, ref generics, _, _, _),
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

                    let instance = Instance::mono(self.scx.tcx(), def_id);
                    self.output.push(TransItem::Fn(instance));
                }
            }
            _ => { /* Nothing to do here */ }
        }

        hir_visit::walk_impl_item(self, ii)
    }
}

fn create_trans_items_for_default_impls<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                  item: &'tcx hir::Item,
                                                  output: &mut Vec<TransItem<'tcx>>) {
    match item.node {
        hir::ItemImpl(_,
                      _,
                      ref generics,
                      _,
                      _,
                      ref items) => {
            if generics.is_type_parameterized() {
                return
            }

            let impl_def_id = tcx.map.local_def_id(item.id);

            debug!("create_trans_items_for_default_impls(item={})",
                   def_id_to_string(tcx, impl_def_id));

            if let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                let default_impls = tcx.provided_trait_methods(trait_ref.def_id);
                let callee_substs = tcx.erase_regions(&trait_ref.substs);
                let overridden_methods: FnvHashSet<_> = items.iter()
                                                             .map(|item| item.name)
                                                             .collect();
                for default_impl in default_impls {
                    if overridden_methods.contains(&default_impl.name) {
                        continue;
                    }

                    if default_impl.generics.has_type_params(subst::FnSpace) {
                        continue;
                    }

                    // The substitutions we have are on the impl, so we grab
                    // the method type from the impl to substitute into.
                    let mth = meth::get_impl_method(tcx,
                                                    impl_def_id,
                                                    callee_substs,
                                                    default_impl.name);

                    assert!(mth.is_provided);

                    let predicates = mth.method.predicates.predicates.subst(tcx, &mth.substs);
                    if !normalize_and_test_predicates(tcx, predicates.into_vec()) {
                        continue;
                    }

                    if can_have_local_instance(tcx, default_impl.def_id) {
                        let empty_substs = tcx.erase_regions(&mth.substs);
                        let item = create_fn_trans_item(tcx,
                                                        default_impl.def_id,
                                                        callee_substs,
                                                        empty_substs);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransItemState {
    PredictedAndGenerated,
    PredictedButNotGenerated,
    NotPredictedButGenerated,
}

pub fn collecting_debug_information(scx: &SharedCrateContext) -> bool {
    return cfg!(debug_assertions) &&
           scx.sess().opts.debugging_opts.print_trans_items.is_some();
}

pub fn print_collection_results<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>) {
    use std::hash::{Hash, SipHasher, Hasher};

    if !collecting_debug_information(scx) {
        return;
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = SipHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    let trans_items = scx.translation_items().borrow();

    {
        // Check for duplicate item keys
        let mut item_keys = FnvHashMap();

        for (item, item_state) in trans_items.iter() {
            let k = item.to_string(scx.tcx());

            if item_keys.contains_key(&k) {
                let prev: (TransItem, TransItemState) = item_keys[&k];
                debug!("DUPLICATE KEY: {}", k);
                debug!(" (1) {:?}, {:?}, hash: {}, raw: {}",
                       prev.0,
                       prev.1,
                       hash(&prev.0),
                       prev.0.to_raw_string());

                debug!(" (2) {:?}, {:?}, hash: {}, raw: {}",
                       *item,
                       *item_state,
                       hash(item),
                       item.to_raw_string());
            } else {
                item_keys.insert(k, (*item, *item_state));
            }
        }
    }

    let mut predicted_but_not_generated = FnvHashSet();
    let mut not_predicted_but_generated = FnvHashSet();
    let mut predicted = FnvHashSet();
    let mut generated = FnvHashSet();

    for (item, item_state) in trans_items.iter() {
        let item_key = item.to_string(scx.tcx());

        match *item_state {
            TransItemState::PredictedAndGenerated => {
                predicted.insert(item_key.clone());
                generated.insert(item_key);
            }
            TransItemState::PredictedButNotGenerated => {
                predicted_but_not_generated.insert(item_key.clone());
                predicted.insert(item_key);
            }
            TransItemState::NotPredictedButGenerated => {
                not_predicted_but_generated.insert(item_key.clone());
                generated.insert(item_key);
            }
        }
    }

    debug!("Total number of translation items predicted: {}", predicted.len());
    debug!("Total number of translation items generated: {}", generated.len());
    debug!("Total number of translation items predicted but not generated: {}",
           predicted_but_not_generated.len());
    debug!("Total number of translation items not predicted but generated: {}",
           not_predicted_but_generated.len());

    if generated.len() > 0 {
        debug!("Failed to predict {}% of translation items",
               (100 * not_predicted_but_generated.len()) / generated.len());
    }
    if generated.len() > 0 {
        debug!("Predict {}% too many translation items",
               (100 * predicted_but_not_generated.len()) / generated.len());
    }

    debug!("");
    debug!("Not predicted but generated:");
    debug!("============================");
    for item in not_predicted_but_generated {
        debug!(" - {}", item);
    }

    debug!("");
    debug!("Predicted but not generated:");
    debug!("============================");
    for item in predicted_but_not_generated {
        debug!(" - {}", item);
    }
}
