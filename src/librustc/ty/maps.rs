// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepGraph, DepNode, DepTrackingMap, DepTrackingMapConfig};
use hir::def_id::{CrateNum, DefId};
use middle::const_val::ConstVal;
use mir;
use ty::{self, Ty, TyCtxt};
use util::common::MemoizationMap;

use rustc_data_structures::indexed_vec::IndexVec;
use std::cell::RefCell;
use std::rc::Rc;
use syntax_pos::{Span, DUMMY_SP};

trait Key {
    fn map_crate(&self) -> CrateNum;
    fn default_span(&self, tcx: TyCtxt) -> Span;
}

impl Key for DefId {
    fn map_crate(&self) -> CrateNum {
        self.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(*self)
    }
}

impl Key for (DefId, DefId) {
    fn map_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.1.default_span(tcx)
    }
}

trait Value<'tcx>: Sized {
    fn from_cycle_error<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self;
}

impl<'tcx, T> Value<'tcx> for T {
    default fn from_cycle_error<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> T {
        tcx.sess.abort_if_errors();
        bug!("Value::from_cycle_error called without errors");
    }
}

impl<'tcx, T: Default> Value<'tcx> for T {
    default fn from_cycle_error<'a>(_: TyCtxt<'a, 'tcx, 'tcx>) -> T {
        T::default()
    }
}

impl<'tcx> Value<'tcx> for Ty<'tcx> {
    fn from_cycle_error<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        tcx.types.err
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    fn report_cycle(self, span: Span, cycle: &[(Span, Query)]) {
        assert!(!cycle.is_empty());

        let mut err = struct_span_err!(self.sess, span, E0391,
            "unsupported cyclic reference between types/traits detected");
        err.span_label(span, &format!("cyclic reference"));

        err.span_note(cycle[0].0, &format!("the cycle begins when {}...",
                                           cycle[0].1.describe(self)));

        for &(span, ref query) in &cycle[1..] {
            err.span_note(span, &format!("...which then requires {}...",
                                         query.describe(self)));
        }

        err.note(&format!("...which then again requires {}, completing the cycle.",
                          cycle[0].1.describe(self)));

        err.emit();
    }

    fn cycle_check<F, R>(self, span: Span, query: Query, compute: F) -> R
        where F: FnOnce() -> R
    {
        {
            let mut stack = self.maps.query_stack.borrow_mut();
            if let Some((i, _)) = stack.iter().enumerate().rev()
                                       .find(|&(_, &(_, ref q))| *q == query) {
                let cycle = &stack[i..];
                self.report_cycle(span, cycle);
                return R::from_cycle_error(self.global_tcx());
            }
            stack.push((span, query));
        }

        let result = compute();

        self.maps.query_stack.borrow_mut().pop();
        result
    }
}

trait QueryDescription: DepTrackingMapConfig {
    fn describe(tcx: TyCtxt, key: Self::Key) -> String;
}

impl<M: DepTrackingMapConfig<Key=DefId>> QueryDescription for M {
    default fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("processing `{}`", tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::super_predicates<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("computing the supertraits of `{}`",
                tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::type_param_predicates<'tcx> {
    fn describe(tcx: TyCtxt, (_, def_id): (DefId, DefId)) -> String {
        let id = tcx.hir.as_local_node_id(def_id).unwrap();
        format!("computing the bounds for type parameter `{}`",
                tcx.hir.ty_param_name(id))
    }
}

macro_rules! define_maps {
    (<$tcx:tt>
     $($(#[$attr:meta])*
       pub $name:ident: $node:ident($K:ty) -> $V:ty),*) => {
        pub struct Maps<$tcx> {
            providers: IndexVec<CrateNum, Providers<$tcx>>,
            pub query_stack: RefCell<Vec<(Span, Query)>>,
            $($(#[$attr])* pub $name: RefCell<DepTrackingMap<queries::$name<$tcx>>>),*
        }

        impl<$tcx> Maps<$tcx> {
            pub fn new(dep_graph: DepGraph,
                       providers: IndexVec<CrateNum, Providers<$tcx>>)
                       -> Self {
                Maps {
                    providers,
                    query_stack: RefCell::new(vec![]),
                    $($name: RefCell::new(DepTrackingMap::new(dep_graph.clone()))),*
                }
            }
        }

        #[allow(bad_style)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum Query {
            $($(#[$attr])* $name($K)),*
        }

        impl Query {
            pub fn describe(&self, tcx: TyCtxt) -> String {
                match *self {
                    $(Query::$name(key) => queries::$name::describe(tcx, key)),*
                }
            }
        }

        pub mod queries {
            use std::marker::PhantomData;

            $(#[allow(bad_style)]
            pub struct $name<$tcx> {
                data: PhantomData<&$tcx ()>
            })*
        }

        $(impl<$tcx> DepTrackingMapConfig for queries::$name<$tcx> {
            type Key = $K;
            type Value = $V;
            fn to_dep_node(key: &$K) -> DepNode<DefId> { DepNode::$node(*key) }
        })*

        pub struct Providers<$tcx> {
            $(pub $name: for<'a> fn(TyCtxt<'a, $tcx, $tcx>, $K) -> $V),*
        }

        impl<$tcx> Copy for Providers<$tcx> {}
        impl<$tcx> Clone for Providers<$tcx> {
            fn clone(&self) -> Self { *self }
        }

        impl<$tcx> Default for Providers<$tcx> {
            fn default() -> Self {
                $(fn $name<'a, $tcx>(_: TyCtxt<'a, $tcx, $tcx>, key: $K) -> $V {
                    bug!("tcx.maps.{}({:?}) unsupported by its crate",
                         stringify!($name), key);
                })*
                Providers { $($name),* }
            }
        }

        impl<'a, $tcx, 'lcx> Maps<$tcx> {
            $($(#[$attr])*
              pub fn $name(&self,
                           tcx: TyCtxt<'a, $tcx, 'lcx>,
                           mut span: Span,
                           key: $K) -> $V {
                self.$name.memoize(key, || {
                    // FIXME(eddyb) Get more valid Span's on queries.
                    if span == DUMMY_SP {
                        span = key.default_span(tcx);
                    }

                    tcx.cycle_check(span, Query::$name(key), || {
                        let provider = self.providers[key.map_crate()].$name;
                        provider(tcx.global_tcx(), key)
                    })
                })
            })*
        }
    }
}

// Each of these maps also corresponds to a method on a
// `Provider` trait for requesting a value of that type,
// and a method on `Maps` itself for doing that in a
// a way that memoizes and does dep-graph tracking,
// wrapping around the actual chain of providers that
// the driver creates (using several `rustc_*` crates).
define_maps! { <'tcx>
    /// Records the type of every item.
    pub ty: ItemSignature(DefId) -> Ty<'tcx>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated generics and predicates.
    pub generics: ItemSignature(DefId) -> &'tcx ty::Generics,
    pub predicates: ItemSignature(DefId) -> ty::GenericPredicates<'tcx>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    pub super_predicates: ItemSignature(DefId) -> ty::GenericPredicates<'tcx>,

    /// To avoid cycles within the predicates of a single item we compute
    /// per-type-parameter predicates for resolving `T::AssocTy`.
    pub type_param_predicates: TypeParamPredicates((DefId, DefId))
        -> ty::GenericPredicates<'tcx>,

    pub trait_def: ItemSignature(DefId) -> &'tcx ty::TraitDef,
    pub adt_def: ItemSignature(DefId) -> &'tcx ty::AdtDef,
    pub adt_sized_constraint: SizedConstraint(DefId) -> Ty<'tcx>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub variances: ItemSignature(DefId) -> Rc<Vec<ty::Variance>>,

    /// Maps from an impl/trait def-id to a list of the def-ids of its items
    pub associated_item_def_ids: AssociatedItemDefIds(DefId) -> Rc<Vec<DefId>>,

    /// Maps from a trait item to the trait item "descriptor"
    pub associated_item: AssociatedItems(DefId) -> ty::AssociatedItem,

    pub impl_trait_ref: ItemSignature(DefId) -> Option<ty::TraitRef<'tcx>>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: InherentImpls(DefId) -> Vec<DefId>,

    /// Maps from the def-id of a function/method or const/static
    /// to its MIR. Mutation is done at an item granularity to
    /// allow MIR optimization passes to function and still
    /// access cross-crate MIR (e.g. inlining or const eval).
    ///
    /// Note that cross-crate MIR appears to be always borrowed
    /// (in the `RefCell` sense) to prevent accidental mutation.
    pub mir: Mir(DefId) -> &'tcx RefCell<mir::Mir<'tcx>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_kind: ItemSignature(DefId) -> ty::ClosureKind,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_type: ItemSignature(DefId) -> ty::PolyFnSig<'tcx>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    pub custom_coerce_unsized_kind: ItemSignature(DefId)
        -> ty::adjustment::CustomCoerceUnsized,

    pub typeck_tables: TypeckTables(DefId) -> &'tcx ty::TypeckTables<'tcx>,

    /// Results of evaluating monomorphic constants embedded in
    /// other items, such as enum variant explicit discriminants.
    pub monomorphic_const_eval: MonomorphicConstEval(DefId) -> Result<ConstVal, ()>
}
