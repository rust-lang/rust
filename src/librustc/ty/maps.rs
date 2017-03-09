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

use rustc_data_structures::indexed_vec::IndexVec;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;
use syntax_pos::{Span, DUMMY_SP};

trait Key {
    fn map_crate(&self) -> CrateNum;
    fn default_span(&self, tcx: TyCtxt) -> Span;
}

impl Key for CrateNum {
    fn map_crate(&self) -> CrateNum {
        *self
    }
    fn default_span(&self, _: TyCtxt) -> Span {
        DUMMY_SP
    }
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

impl Key for (CrateNum, DefId) {
    fn map_crate(&self) -> CrateNum {
        self.0
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

pub struct CycleError<'a> {
    span: Span,
    cycle: RefMut<'a, [(Span, Query)]>
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn report_cycle(self, CycleError { span, cycle }: CycleError) {
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

    fn cycle_check<F, R>(self, span: Span, query: Query, compute: F)
                         -> Result<R, CycleError<'a>>
        where F: FnOnce() -> R
    {
        {
            let mut stack = self.maps.query_stack.borrow_mut();
            if let Some((i, _)) = stack.iter().enumerate().rev()
                                       .find(|&(_, &(_, ref q))| *q == query) {
                return Err(CycleError {
                    span: span,
                    cycle: RefMut::map(stack, |stack| &mut stack[i..])
                });
            }
            stack.push((span, query));
        }

        let result = compute();

        self.maps.query_stack.borrow_mut().pop();

        Ok(result)
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

impl<'tcx> QueryDescription for queries::coherent_trait<'tcx> {
    fn describe(tcx: TyCtxt, (_, def_id): (CrateNum, DefId)) -> String {
        format!("coherence checking all impls of trait `{}`",
                tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::coherent_inherent_impls<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        format!("coherence checking all inherent impls")
    }
}

macro_rules! define_maps {
    (<$tcx:tt>
     $($(#[$attr:meta])*
       pub $name:ident: $node:ident($K:ty) -> $V:ty),*) => {
        pub struct Maps<$tcx> {
            providers: IndexVec<CrateNum, Providers<$tcx>>,
            query_stack: RefCell<Vec<(Span, Query)>>,
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

            #[allow(unused)]
            fn to_dep_node(key: &$K) -> DepNode<DefId> {
                use dep_graph::DepNode::*;

                $node(*key)
            }
        }
        impl<'a, $tcx, 'lcx> queries::$name<$tcx> {
            fn try_get_with<F, R>(tcx: TyCtxt<'a, $tcx, 'lcx>,
                                  mut span: Span,
                                  key: $K,
                                  f: F)
                                  -> Result<R, CycleError<'a>>
                where F: FnOnce(&$V) -> R
            {
                if let Some(result) = tcx.maps.$name.borrow().get(&key) {
                    return Ok(f(result));
                }

                // FIXME(eddyb) Get more valid Span's on queries.
                if span == DUMMY_SP {
                    span = key.default_span(tcx);
                }

                let _task = tcx.dep_graph.in_task(Self::to_dep_node(&key));

                let result = tcx.cycle_check(span, Query::$name(key), || {
                    let provider = tcx.maps.providers[key.map_crate()].$name;
                    provider(tcx.global_tcx(), key)
                })?;

                Ok(f(&tcx.maps.$name.borrow_mut().entry(key).or_insert(result)))
            }

            pub fn try_get(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K)
                           -> Result<$V, CycleError<'a>> {
                Self::try_get_with(tcx, span, key, Clone::clone)
            }

            $(#[$attr])*
            pub fn get(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K) -> $V {
                Self::try_get(tcx, span, key).unwrap_or_else(|e| {
                    tcx.report_cycle(e);
                    Value::from_cycle_error(tcx.global_tcx())
                })
            }

            pub fn force(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K) {
                // FIXME(eddyb) Move away from using `DepTrackingMap`
                // so we don't have to explicitly ignore a false edge:
                // we can't observe a value dependency, only side-effects,
                // through `force`, and once everything has been updated,
                // perhaps only diagnostics, if those, will remain.
                let _ignore = tcx.dep_graph.in_ignore();
                match Self::try_get_with(tcx, span, key, |_| ()) {
                    Ok(()) => {}
                    Err(e) => tcx.report_cycle(e)
                }
            }
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
    pub adt_destructor: AdtDestructor(DefId) -> Option<ty::Destructor>,
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

    /// Maps DefId's that have an associated Mir to the result
    /// of the MIR qualify_consts pass. The actual meaning of
    /// the value isn't known except to the pass itself.
    pub mir_const_qualif: Mir(DefId) -> u8,

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

    pub coherent_trait: coherent_trait_dep_node((CrateNum, DefId)) -> (),

    pub coherent_inherent_impls: coherent_inherent_impls_dep_node(CrateNum) -> (),

    /// Results of evaluating monomorphic constants embedded in
    /// other items, such as enum variant explicit discriminants.
    pub monomorphic_const_eval: MonomorphicConstEval(DefId) -> Result<ConstVal<'tcx>, ()>
}

fn coherent_trait_dep_node((_, def_id): (CrateNum, DefId)) -> DepNode<DefId> {
    DepNode::CoherenceCheckTrait(def_id)
}

fn coherent_inherent_impls_dep_node(_: CrateNum) -> DepNode<DefId> {
    DepNode::Coherence
}
