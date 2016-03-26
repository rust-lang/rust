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

use rustc_front::hir;
use rustc_front::intravisit as hir_visit;

use rustc::front::map as hir_map;
use rustc::middle::def_id::DefId;
use rustc::middle::lang_items::{ExchangeFreeFnLangItem, ExchangeMallocFnLangItem};
use rustc::middle::traits;
use rustc::middle::subst::{self, Substs, Subst};
use rustc::middle::ty::{self, Ty, TypeFoldable};
use rustc::middle::ty::adjustment::CustomCoerceUnsized;
use rustc::mir::repr as mir;
use rustc::mir::visit as mir_visit;
use rustc::mir::visit::Visitor as MirVisitor;

use syntax::ast::{self, NodeId};
use syntax::codemap::DUMMY_SP;
use syntax::errors;
use syntax::parse::token;

use trans::base::custom_coerce_unsize_info;
use trans::context::CrateContext;
use trans::common::{fulfill_obligation, normalize_and_test_predicates,
                    type_is_sized};
use trans::glue;
use trans::meth;
use trans::monomorphize::{self, Instance};
use util::nodemap::{FnvHashSet, FnvHashMap, DefIdMap};

use std::hash::{Hash, Hasher};

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum TransItemCollectionMode {
    Eager,
    Lazy
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TransItem<'tcx> {
    DropGlue(Ty<'tcx>),
    Fn(Instance<'tcx>),
    Static(NodeId)
}

impl<'tcx> Hash for TransItem<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        match *self {
            TransItem::DropGlue(t) => {
                0u8.hash(s);
                t.hash(s);
            },
            TransItem::Fn(instance) => {
                1u8.hash(s);
                instance.def.hash(s);
                (instance.substs as *const _ as usize).hash(s);
            }
            TransItem::Static(node_id) => {
                2u8.hash(s);
                node_id.hash(s);
            }
        };
    }
}

pub fn collect_crate_translation_items<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                 mode: TransItemCollectionMode)
                                                 -> FnvHashSet<TransItem<'tcx>> {
    // We are not tracking dependencies of this pass as it has to be re-executed
    // every time no matter what.
    ccx.tcx().dep_graph.with_ignore(|| {
        let roots = collect_roots(ccx, mode);

        debug!("Building translation item graph, beginning at roots");
        let mut visited = FnvHashSet();
        let mut recursion_depths = DefIdMap();

        for root in roots {
            collect_items_rec(ccx, root, &mut visited, &mut recursion_depths);
        }

        visited
    })
}

// Find all non-generic items by walking the HIR. These items serve as roots to
// start monomorphizing from.
fn collect_roots<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                           mode: TransItemCollectionMode)
                           -> Vec<TransItem<'tcx>> {
    debug!("Collecting roots");
    let mut roots = Vec::new();

    {
        let mut visitor = RootCollector {
            ccx: ccx,
            mode: mode,
            output: &mut roots,
            enclosing_item: None,
        };

        ccx.tcx().map.krate().visit_all_items(&mut visitor);
    }

    roots
}

// Collect all monomorphized translation items reachable from `starting_point`
fn collect_items_rec<'a, 'tcx: 'a>(ccx: &CrateContext<'a, 'tcx>,
                                   starting_point: TransItem<'tcx>,
                                   visited: &mut FnvHashSet<TransItem<'tcx>>,
                                   recursion_depths: &mut DefIdMap<usize>) {
    if !visited.insert(starting_point.clone()) {
        // We've been here already, no need to search again.
        return;
    }
    debug!("BEGIN collect_items_rec({})", starting_point.to_string(ccx));

    let mut neighbors = Vec::new();
    let recursion_depth_reset;

    match starting_point {
        TransItem::DropGlue(t) => {
            find_drop_glue_neighbors(ccx, t, &mut neighbors);
            recursion_depth_reset = None;
        }
        TransItem::Static(_) => {
            recursion_depth_reset = None;
        }
        TransItem::Fn(instance) => {
            // Keep track of the monomorphization recursion depth
            recursion_depth_reset = Some(check_recursion_limit(ccx,
                                                               instance,
                                                               recursion_depths));

            // Scan the MIR in order to find function calls, closures, and
            // drop-glue
            let mir = errors::expect(ccx.sess().diagnostic(), ccx.get_mir(instance.def),
                || format!("Could not find MIR for function: {}", instance));

            let mut visitor = MirNeighborCollector {
                ccx: ccx,
                mir: &mir,
                output: &mut neighbors,
                param_substs: instance.substs
            };

            visitor.visit_mir(&mir);
        }
    }

    for neighbour in neighbors {
        collect_items_rec(ccx, neighbour, visited, recursion_depths);
    }

    if let Some((def_id, depth)) = recursion_depth_reset {
        recursion_depths.insert(def_id, depth);
    }

    debug!("END collect_items_rec({})", starting_point.to_string(ccx));
}

fn check_recursion_limit<'a, 'tcx: 'a>(ccx: &CrateContext<'a, 'tcx>,
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
    if recursion_depth > ccx.sess().recursion_limit.get() {
        let error = format!("reached the recursion limit while instantiating `{}`",
                            instance);
        if let Some(node_id) = ccx.tcx().map.as_local_node_id(instance.def) {
            ccx.sess().span_fatal(ccx.tcx().map.span(node_id), &error);
        } else {
            ccx.sess().fatal(&error);
        }
    }

    recursion_depths.insert(instance.def, recursion_depth + 1);

    (instance.def, recursion_depth)
}

struct MirNeighborCollector<'a, 'tcx: 'a> {
    ccx: &'a CrateContext<'a, 'tcx>,
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
                assert!(can_have_local_instance(self.ccx, def_id));
                let trans_item = create_fn_trans_item(self.ccx,
                                                      def_id,
                                                      substs.func_substs,
                                                      self.param_substs);
                self.output.push(trans_item);
            }
            // When doing an cast from a regular pointer to a fat pointer, we
            // have to instantiate all methods of the trait being cast to, so we
            // can build the appropriate vtable.
            mir::Rvalue::Cast(mir::CastKind::Unsize, ref operand, target_ty) => {
                let target_ty = monomorphize::apply_param_substs(self.ccx.tcx(),
                                                                 self.param_substs,
                                                                 &target_ty);
                let source_ty = self.mir.operand_ty(self.ccx.tcx(), operand);
                let source_ty = monomorphize::apply_param_substs(self.ccx.tcx(),
                                                                 self.param_substs,
                                                                 &source_ty);
                let (source_ty, target_ty) = find_vtable_types_for_unsizing(self.ccx,
                                                                            source_ty,
                                                                            target_ty);
                // This could also be a different Unsize instruction, like
                // from a fixed sized array to a slice. But we are only
                // interested in things that produce a vtable.
                if target_ty.is_trait() && !source_ty.is_trait() {
                    create_trans_items_for_vtable_methods(self.ccx,
                                                          target_ty,
                                                          source_ty,
                                                          self.output);
                }
            }
            mir::Rvalue::Box(_) => {
                let exchange_malloc_fn_def_id =
                    self.ccx
                        .tcx()
                        .lang_items
                        .require(ExchangeMallocFnLangItem)
                        .unwrap_or_else(|e| self.ccx.sess().fatal(&e));

                assert!(can_have_local_instance(self.ccx, exchange_malloc_fn_def_id));
                let exchange_malloc_fn_trans_item =
                    create_fn_trans_item(self.ccx,
                                         exchange_malloc_fn_def_id,
                                         &Substs::empty(),
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
            let ty = self.mir.lvalue_ty(self.ccx.tcx(), lvalue)
                             .to_ty(self.ccx.tcx());

            let ty = monomorphize::apply_param_substs(self.ccx.tcx(),
                                                      self.param_substs,
                                                      &ty);
            let ty = self.ccx.tcx().erase_regions(&ty);
            let ty = glue::get_drop_glue_type(self.ccx, ty);
            self.output.push(TransItem::DropGlue(ty));
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
            let dispatched = do_static_dispatch(self.ccx,
                                                callee_def_id,
                                                callee_substs,
                                                self.param_substs);

            if let Some((callee_def_id, callee_substs)) = dispatched {
                // if we have a concrete impl (which we might not have
                // in the case of something compiler generated like an
                // object shim or a closure that is handled differently),
                // we check if the callee is something that will actually
                // result in a translation item ...
                if can_result_in_trans_item(self.ccx, callee_def_id) {
                    // ... and create one if it does.
                    let trans_item = create_fn_trans_item(self.ccx,
                                                          callee_def_id,
                                                          callee_substs,
                                                          self.param_substs);
                    self.output.push(trans_item);
                }
            }
        }

        self.super_operand(operand);

        fn can_result_in_trans_item<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                              def_id: DefId)
                                              -> bool {
            if !match ccx.tcx().lookup_item_type(def_id).ty.sty {
                ty::TyFnDef(def_id, _, _) => {
                    // Some constructors also have type TyFnDef but they are
                    // always instantiated inline and don't result in
                    // translation item. Same for FFI functions.
                    match ccx.tcx().map.get_if_local(def_id) {
                        Some(hir_map::NodeVariant(_))    |
                        Some(hir_map::NodeStructCtor(_)) |
                        Some(hir_map::NodeForeignItem(_)) => false,
                        Some(_) => true,
                        None => {
                            ccx.sess().cstore.variant_kind(def_id).is_none()
                        }
                    }
                }
                ty::TyClosure(..) => true,
                _ => false
            } {
                return false;
            }

            can_have_local_instance(ccx, def_id)
        }
    }
}

fn can_have_local_instance<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                     def_id: DefId)
                                     -> bool {
    // Take a look if we have the definition available. If not, we
    // will not emit code for this item in the local crate, and thus
    // don't create a translation item for it.
    def_id.is_local() || ccx.sess().cstore.is_item_mir_available(def_id)
}

fn find_drop_glue_neighbors<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                      ty: ty::Ty<'tcx>,
                                      output: &mut Vec<TransItem<'tcx>>)
{
    debug!("find_drop_glue_neighbors: {}", type_to_string(ccx, ty));

    // Make sure the exchange_free_fn() lang-item gets translated if
    // there is a boxed value.
    if let ty::TyBox(_) = ty.sty {
        let exchange_free_fn_def_id = ccx.tcx()
                                         .lang_items
                                         .require(ExchangeFreeFnLangItem)
                                         .unwrap_or_else(|e| ccx.sess().fatal(&e));

        assert!(can_have_local_instance(ccx, exchange_free_fn_def_id));
        let exchange_free_fn_trans_item =
            create_fn_trans_item(ccx,
                                 exchange_free_fn_def_id,
                                 &Substs::empty(),
                                 &Substs::empty());

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
        use rustc::middle::ty::ToPolyTraitRef;

        let drop_trait_def_id = ccx.tcx()
                                   .lang_items
                                   .drop_trait()
                                   .unwrap();

        let self_type_substs = ccx.tcx().mk_substs(
            Substs::empty().with_self_ty(ty));

        let trait_ref = ty::TraitRef {
            def_id: drop_trait_def_id,
            substs: self_type_substs,
        }.to_poly_trait_ref();

        let substs = match fulfill_obligation(ccx, DUMMY_SP, trait_ref) {
            traits::VtableImpl(data) => data.substs,
            _ => unreachable!()
        };

        if can_have_local_instance(ccx, destructor_did) {
            let trans_item = create_fn_trans_item(ccx,
                                                  destructor_did,
                                                  substs,
                                                  &Substs::empty());
            output.push(trans_item);
        }
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
                let field_type = monomorphize::apply_param_substs(ccx.tcx(),
                                                                  substs,
                                                                  &field.unsubst_ty());
                let field_type = glue::get_drop_glue_type(ccx, field_type);

                if glue::type_needs_drop(ccx.tcx(), field_type) {
                    output.push(TransItem::DropGlue(field_type));
                }
            }
        }
        ty::TyClosure(_, ref substs) => {
            for upvar_ty in &substs.upvar_tys {
                let upvar_ty = glue::get_drop_glue_type(ccx, upvar_ty);
                if glue::type_needs_drop(ccx.tcx(), upvar_ty) {
                    output.push(TransItem::DropGlue(upvar_ty));
                }
            }
        }
        ty::TyBox(inner_type)      |
        ty::TyArray(inner_type, _) => {
            let inner_type = glue::get_drop_glue_type(ccx, inner_type);
            if glue::type_needs_drop(ccx.tcx(), inner_type) {
                output.push(TransItem::DropGlue(inner_type));
            }
        }
        ty::TyTuple(ref args) => {
            for arg in args {
                let arg = glue::get_drop_glue_type(ccx, arg);
                if glue::type_needs_drop(ccx.tcx(), arg) {
                    output.push(TransItem::DropGlue(arg));
                }
            }
        }
        ty::TyProjection(_) |
        ty::TyParam(_)      |
        ty::TyInfer(_)      |
        ty::TyError         => {
            ccx.sess().bug("encountered unexpected type");
        }
    }
}

fn do_static_dispatch<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                fn_def_id: DefId,
                                fn_substs: &'tcx Substs<'tcx>,
                                param_substs: &'tcx Substs<'tcx>)
                                -> Option<(DefId, &'tcx Substs<'tcx>)> {
    debug!("do_static_dispatch(fn_def_id={}, fn_substs={:?}, param_substs={:?})",
           def_id_to_string(ccx, fn_def_id),
           fn_substs,
           param_substs);

    let is_trait_method = ccx.tcx().trait_of_item(fn_def_id).is_some();

    if is_trait_method {
        match ccx.tcx().impl_or_trait_item(fn_def_id) {
            ty::MethodTraitItem(ref method) => {
                match method.container {
                    ty::TraitContainer(trait_def_id) => {
                        debug!(" => trait method, attempting to find impl");
                        do_static_trait_method_dispatch(ccx,
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
            _ => unreachable!()
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
fn do_static_trait_method_dispatch<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                             trait_method: &ty::Method,
                                             trait_id: DefId,
                                             callee_substs: &'tcx Substs<'tcx>,
                                             param_substs: &'tcx Substs<'tcx>)
                                             -> Option<(DefId, &'tcx Substs<'tcx>)> {
    let tcx = ccx.tcx();
    debug!("do_static_trait_method_dispatch(trait_method={}, \
                                            trait_id={}, \
                                            callee_substs={:?}, \
                                            param_substs={:?}",
           def_id_to_string(ccx, trait_method.def_id),
           def_id_to_string(ccx, trait_id),
           callee_substs,
           param_substs);

    let rcvr_substs = monomorphize::apply_param_substs(tcx,
                                                       param_substs,
                                                       callee_substs);

    let trait_ref = ty::Binder(rcvr_substs.to_trait_ref(tcx, trait_id));
    let vtbl = fulfill_obligation(ccx, DUMMY_SP, trait_ref);

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
            tcx.sess.bug(&format!("static call to invalid vtable: {:?}", vtbl))
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
fn find_vtable_types_for_unsizing<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
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

            if !type_is_sized(ccx.tcx(), inner_source) {
                (inner_source, inner_target)
            } else {
                ccx.tcx().struct_lockstep_tails(inner_source, inner_target)
            }
        }

        (&ty::TyStruct(source_adt_def, source_substs),
         &ty::TyStruct(target_adt_def, target_substs)) => {
            assert_eq!(source_adt_def, target_adt_def);

            let kind = custom_coerce_unsize_info(ccx, source_ty, target_ty);

            let coerce_index = match kind {
                CustomCoerceUnsized::Struct(i) => i
            };

            let source_fields = &source_adt_def.struct_variant().fields;
            let target_fields = &target_adt_def.struct_variant().fields;

            assert!(coerce_index < source_fields.len() &&
                    source_fields.len() == target_fields.len());

            find_vtable_types_for_unsizing(ccx,
                                           source_fields[coerce_index].ty(ccx.tcx(),
                                                                          source_substs),
                                           target_fields[coerce_index].ty(ccx.tcx(),
                                                                          target_substs))
        }
        _ => ccx.sess()
                .bug(&format!("find_vtable_types_for_unsizing: invalid coercion {:?} -> {:?}",
                               source_ty,
                               target_ty))
    }
}

fn create_fn_trans_item<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                  def_id: DefId,
                                  fn_substs: &Substs<'tcx>,
                                  param_substs: &Substs<'tcx>)
                                  -> TransItem<'tcx>
{
    debug!("create_fn_trans_item(def_id={}, fn_substs={:?}, param_substs={:?})",
            def_id_to_string(ccx, def_id),
            fn_substs,
            param_substs);

    // We only get here, if fn_def_id either designates a local item or
    // an inlineable external item. Non-inlineable external items are
    // ignored because we don't want to generate any code for them.
    let concrete_substs = monomorphize::apply_param_substs(ccx.tcx(),
                                                           param_substs,
                                                           fn_substs);
    let concrete_substs = ccx.tcx().erase_regions(&concrete_substs);

    let trans_item =
        TransItem::Fn(Instance::new(def_id,
                                    &ccx.tcx().mk_substs(concrete_substs)));

    return trans_item;
}

/// Creates a `TransItem` for each method that is referenced by the vtable for
/// the given trait/impl pair.
fn create_trans_items_for_vtable_methods<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                   trait_ty: ty::Ty<'tcx>,
                                                   impl_ty: ty::Ty<'tcx>,
                                                   output: &mut Vec<TransItem<'tcx>>) {
    assert!(!trait_ty.needs_subst() && !impl_ty.needs_subst());

    if let ty::TyTrait(ref trait_ty) = trait_ty.sty {
        let poly_trait_ref = trait_ty.principal_trait_ref_with_self_ty(ccx.tcx(),
                                                                       impl_ty);

        // Walk all methods of the trait, including those of its supertraits
        for trait_ref in traits::supertraits(ccx.tcx(), poly_trait_ref) {
            let vtable = fulfill_obligation(ccx, DUMMY_SP, trait_ref);
            match vtable {
                traits::VtableImpl(
                    traits::VtableImplData {
                        impl_def_id,
                        substs,
                        nested: _ }) => {
                    let items = meth::get_vtable_methods(ccx, impl_def_id, substs)
                        .into_iter()
                        // filter out None values
                        .filter_map(|opt_impl_method| opt_impl_method)
                        // create translation items
                        .filter_map(|impl_method| {
                            if can_have_local_instance(ccx, impl_method.method.def_id) {
                                Some(create_fn_trans_item(ccx,
                                                          impl_method.method.def_id,
                                                          &impl_method.substs,
                                                          &Substs::empty()))
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
    ccx: &'b CrateContext<'a, 'tcx>,
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
                    create_trans_items_for_default_impls(self.ccx,
                                                         item,
                                                         self.output);
                }
            }

            hir::ItemEnum(_, ref generics)        |
            hir::ItemStruct(_, ref generics)      => {
                if !generics.is_parameterized() {
                    let ty = {
                        let tables = self.ccx.tcx().tables.borrow();
                        tables.node_types[&item.id]
                    };

                    if self.mode == TransItemCollectionMode::Eager {
                        debug!("RootCollector: ADT drop-glue for {}",
                               def_id_to_string(self.ccx,
                                                self.ccx.tcx().map.local_def_id(item.id)));

                        let ty = glue::get_drop_glue_type(self.ccx, ty);
                        self.output.push(TransItem::DropGlue(ty));
                    }
                }
            }
            hir::ItemStatic(..) => {
                debug!("RootCollector: ItemStatic({})",
                       def_id_to_string(self.ccx,
                                        self.ccx.tcx().map.local_def_id(item.id)));
                self.output.push(TransItem::Static(item.id));
            }
            hir::ItemFn(_, _, constness, _, ref generics, _) => {
                if !generics.is_type_parameterized() &&
                   constness == hir::Constness::NotConst {
                    let def_id = self.ccx.tcx().map.local_def_id(item.id);

                    debug!("RootCollector: ItemFn({})",
                           def_id_to_string(self.ccx, def_id));

                    let instance = Instance::mono(self.ccx.tcx(), def_id);
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
                let hir_map = &self.ccx.tcx().map;
                let parent_node_id = hir_map.get_parent_node(ii.id);
                let is_impl_generic = match hir_map.expect_item(parent_node_id) {
                    &hir::Item {
                        node: hir::ItemImpl(_, _, ref generics, _, _, _),
                        ..
                    } => {
                        generics.is_type_parameterized()
                    }
                    _ => {
                        unreachable!()
                    }
                };

                if !generics.is_type_parameterized() && !is_impl_generic {
                    let def_id = self.ccx.tcx().map.local_def_id(ii.id);

                    debug!("RootCollector: MethodImplItem({})",
                           def_id_to_string(self.ccx, def_id));

                    let instance = Instance::mono(self.ccx.tcx(), def_id);
                    self.output.push(TransItem::Fn(instance));
                }
            }
            _ => { /* Nothing to do here */ }
        }

        hir_visit::walk_impl_item(self, ii)
    }
}

fn create_trans_items_for_default_impls<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
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

            let tcx = ccx.tcx();
            let impl_def_id = tcx.map.local_def_id(item.id);

            debug!("create_trans_items_for_default_impls(item={})",
                   def_id_to_string(ccx, impl_def_id));

            if let Some(trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                let default_impls = tcx.provided_trait_methods(trait_ref.def_id);
                let callee_substs = tcx.mk_substs(tcx.erase_regions(trait_ref.substs));
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
                    if !normalize_and_test_predicates(ccx, predicates.into_vec()) {
                        continue;
                    }

                    if can_have_local_instance(ccx, default_impl.def_id) {
                        let empty_substs = ccx.tcx().mk_substs(ccx.tcx().erase_regions(mth.substs));
                        let item = create_fn_trans_item(ccx,
                                                        default_impl.def_id,
                                                        callee_substs,
                                                        empty_substs);
                        output.push(item);
                    }
                }
            }
        }
        _ => {
            unreachable!()
        }
    }
}

//=-----------------------------------------------------------------------------
// TransItem String Keys
//=-----------------------------------------------------------------------------

// The code below allows for producing a unique string key for a trans item.
// These keys are used by the handwritten auto-tests, so they need to be
// predictable and human-readable.
//
// Note: A lot of this could looks very similar to what's already in the
//       ppaux module. It would be good to refactor things so we only have one
//       parameterizable implementation for printing types.

/// Same as `unique_type_name()` but with the result pushed onto the given
/// `output` parameter.
pub fn push_unique_type_name<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                       t: ty::Ty<'tcx>,
                                       output: &mut String) {
    match t.sty {
        ty::TyBool              => output.push_str("bool"),
        ty::TyChar              => output.push_str("char"),
        ty::TyStr               => output.push_str("str"),
        ty::TyInt(ast::IntTy::Is)    => output.push_str("isize"),
        ty::TyInt(ast::IntTy::I8)    => output.push_str("i8"),
        ty::TyInt(ast::IntTy::I16)   => output.push_str("i16"),
        ty::TyInt(ast::IntTy::I32)   => output.push_str("i32"),
        ty::TyInt(ast::IntTy::I64)   => output.push_str("i64"),
        ty::TyUint(ast::UintTy::Us)   => output.push_str("usize"),
        ty::TyUint(ast::UintTy::U8)   => output.push_str("u8"),
        ty::TyUint(ast::UintTy::U16)  => output.push_str("u16"),
        ty::TyUint(ast::UintTy::U32)  => output.push_str("u32"),
        ty::TyUint(ast::UintTy::U64)  => output.push_str("u64"),
        ty::TyFloat(ast::FloatTy::F32) => output.push_str("f32"),
        ty::TyFloat(ast::FloatTy::F64) => output.push_str("f64"),
        ty::TyStruct(adt_def, substs) |
        ty::TyEnum(adt_def, substs) => {
            push_item_name(cx, adt_def.did, output);
            push_type_params(cx, &substs.types, &[], output);
        },
        ty::TyTuple(ref component_types) => {
            output.push('(');
            for &component_type in component_types {
                push_unique_type_name(cx, component_type, output);
                output.push_str(", ");
            }
            if !component_types.is_empty() {
                output.pop();
                output.pop();
            }
            output.push(')');
        },
        ty::TyBox(inner_type) => {
            output.push_str("Box<");
            push_unique_type_name(cx, inner_type, output);
            output.push('>');
        },
        ty::TyRawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
            output.push('*');
            match mutbl {
                hir::MutImmutable => output.push_str("const "),
                hir::MutMutable => output.push_str("mut "),
            }

            push_unique_type_name(cx, inner_type, output);
        },
        ty::TyRef(_, ty::TypeAndMut { ty: inner_type, mutbl }) => {
            output.push('&');
            if mutbl == hir::MutMutable {
                output.push_str("mut ");
            }

            push_unique_type_name(cx, inner_type, output);
        },
        ty::TyArray(inner_type, len) => {
            output.push('[');
            push_unique_type_name(cx, inner_type, output);
            output.push_str(&format!("; {}", len));
            output.push(']');
        },
        ty::TySlice(inner_type) => {
            output.push('[');
            push_unique_type_name(cx, inner_type, output);
            output.push(']');
        },
        ty::TyTrait(ref trait_data) => {
            push_item_name(cx, trait_data.principal.skip_binder().def_id, output);
            push_type_params(cx,
                             &trait_data.principal.skip_binder().substs.types,
                             &trait_data.bounds.projection_bounds,
                             output);
        },
        ty::TyFnDef(_, _, &ty::BareFnTy{ unsafety, abi, ref sig } ) |
        ty::TyFnPtr(&ty::BareFnTy{ unsafety, abi, ref sig } ) => {
            if unsafety == hir::Unsafety::Unsafe {
                output.push_str("unsafe ");
            }

            if abi != ::trans::abi::Abi::Rust {
                output.push_str("extern \"");
                output.push_str(abi.name());
                output.push_str("\" ");
            }

            output.push_str("fn(");

            let sig = cx.tcx().erase_late_bound_regions(sig);
            if !sig.inputs.is_empty() {
                for &parameter_type in &sig.inputs {
                    push_unique_type_name(cx, parameter_type, output);
                    output.push_str(", ");
                }
                output.pop();
                output.pop();
            }

            if sig.variadic {
                if !sig.inputs.is_empty() {
                    output.push_str(", ...");
                } else {
                    output.push_str("...");
                }
            }

            output.push(')');

            match sig.output {
                ty::FnConverging(result_type) if result_type.is_nil() => {}
                ty::FnConverging(result_type) => {
                    output.push_str(" -> ");
                    push_unique_type_name(cx, result_type, output);
                }
                ty::FnDiverging => {
                    output.push_str(" -> !");
                }
            }
        },
        ty::TyClosure(def_id, ref closure_substs) => {
            push_item_name(cx, def_id, output);
            output.push_str("{");
            output.push_str(&format!("{}:{}", def_id.krate, def_id.index.as_usize()));
            output.push_str("}");
            push_type_params(cx, &closure_substs.func_substs.types, &[], output);
        }
        ty::TyError |
        ty::TyInfer(_) |
        ty::TyProjection(..) |
        ty::TyParam(_) => {
            cx.sess().bug(&format!("debuginfo: Trying to create type name for \
                unexpected type: {:?}", t));
        }
    }
}

fn push_item_name(ccx: &CrateContext,
                  def_id: DefId,
                  output: &mut String) {
    let def_path = ccx.tcx().def_path(def_id);

    // some_crate::
    output.push_str(&ccx.tcx().crate_name(def_path.krate));
    output.push_str("::");

    // foo::bar::ItemName::
    for part in ccx.tcx().def_path(def_id).data {
        output.push_str(&format!("{}[{}]::",
                        part.data.as_interned_str(),
                        part.disambiguator));
    }

    // remove final "::"
    output.pop();
    output.pop();
}

fn push_type_params<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                              types: &'tcx subst::VecPerParamSpace<Ty<'tcx>>,
                              projections: &[ty::PolyProjectionPredicate<'tcx>],
                              output: &mut String) {
    if types.is_empty() && projections.is_empty() {
        return;
    }

    output.push('<');

    for &type_parameter in types {
        push_unique_type_name(cx, type_parameter, output);
        output.push_str(", ");
    }

    for projection in projections {
        let projection = projection.skip_binder();
        let name = token::get_ident_interner().get(projection.projection_ty.item_name);
        output.push_str(&name[..]);
        output.push_str("=");
        push_unique_type_name(cx, projection.ty, output);
        output.push_str(", ");
    }

    output.pop();
    output.pop();

    output.push('>');
}

fn push_instance_as_string<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                     instance: Instance<'tcx>,
                                     output: &mut String) {
    push_item_name(ccx, instance.def, output);
    push_type_params(ccx, &instance.substs.types, &[], output);
}

fn def_id_to_string(ccx: &CrateContext, def_id: DefId) -> String {
    let mut output = String::new();
    push_item_name(ccx, def_id, &mut output);
    output
}

fn type_to_string<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                            ty: ty::Ty<'tcx>)
                            -> String {
    let mut output = String::new();
    push_unique_type_name(ccx, ty, &mut output);
    output
}

impl<'tcx> TransItem<'tcx> {

    pub fn to_string<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> String {
        let hir_map = &ccx.tcx().map;

        return match *self {
            TransItem::DropGlue(t) => {
                let mut s = String::with_capacity(32);
                s.push_str("drop-glue ");
                push_unique_type_name(ccx, t, &mut s);
                s
            }
            TransItem::Fn(instance) => {
                to_string_internal(ccx, "fn ", instance)
            },
            TransItem::Static(node_id) => {
                let def_id = hir_map.local_def_id(node_id);
                let instance = Instance::mono(ccx.tcx(), def_id);
                to_string_internal(ccx, "static ", instance)
            },
        };

        fn to_string_internal<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                        prefix: &str,
                                        instance: Instance<'tcx>)
                                        -> String {
            let mut result = String::with_capacity(32);
            result.push_str(prefix);
            push_instance_as_string(ccx, instance, &mut result);
            result
        }
    }

    fn to_raw_string(&self) -> String {
        match *self {
            TransItem::DropGlue(t) => {
                format!("DropGlue({})", t as *const _ as usize)
            }
            TransItem::Fn(instance) => {
                format!("Fn({:?}, {})",
                         instance.def,
                         instance.substs as *const _ as usize)
            }
            TransItem::Static(id) => {
                format!("Static({:?})", id)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransItemState {
    PredictedAndGenerated,
    PredictedButNotGenerated,
    NotPredictedButGenerated,
}

pub fn collecting_debug_information(ccx: &CrateContext) -> bool {
    return cfg!(debug_assertions) &&
           ccx.sess().opts.debugging_opts.print_trans_items.is_some();
}

pub fn print_collection_results<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>) {
    use std::hash::{Hash, SipHasher, Hasher};

    if !collecting_debug_information(ccx) {
        return;
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = SipHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    let trans_items = ccx.translation_items().borrow();

    {
        // Check for duplicate item keys
        let mut item_keys = FnvHashMap();

        for (item, item_state) in trans_items.iter() {
            let k = item.to_string(&ccx);

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
        let item_key = item.to_string(&ccx);

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
