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
use hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId, LOCAL_CRATE};
use hir::def::Def;
use hir;
use middle::const_val;
use middle::privacy::AccessLevels;
use middle::region::RegionMaps;
use mir;
use mir::transform::{MirSuite, MirPassIndex};
use session::CompileResult;
use traits::specialization_graph;
use ty::{self, CrateInherentImpls, Ty, TyCtxt};
use ty::item_path;
use ty::steal::Steal;
use ty::subst::Substs;
use ty::fast_reject::SimplifiedType;
use util::nodemap::{DefIdSet, NodeSet};

use rustc_data_structures::indexed_vec::IndexVec;
use std::cell::{RefCell, RefMut};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::rc::Rc;
use syntax_pos::{Span, DUMMY_SP};
use syntax::attr;
use syntax::ast;
use syntax::symbol::Symbol;

pub trait Key: Clone + Hash + Eq + Debug {
    fn map_crate(&self) -> CrateNum;
    fn default_span(&self, tcx: TyCtxt) -> Span;
}

impl<'tcx> Key for ty::InstanceDef<'tcx> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(self.def_id())
    }
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

impl Key for (DefId, SimplifiedType) {
    fn map_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (DefId, &'tcx Substs<'tcx>) {
    fn map_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (MirSuite, DefId) {
    fn map_crate(&self) -> CrateNum {
        self.1.map_crate()
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (MirSuite, MirPassIndex, DefId) {
    fn map_crate(&self) -> CrateNum {
        self.2.map_crate()
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.2.default_span(tcx)
    }
}

impl<'tcx, T: Clone + Hash + Eq + Debug> Key for ty::ParamEnvAnd<'tcx, T> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt) -> Span {
        DUMMY_SP
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

impl<'tcx> Value<'tcx> for ty::DtorckConstraint<'tcx> {
    fn from_cycle_error<'a>(_: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        Self::empty()
    }
}

impl<'tcx> Value<'tcx> for ty::SymbolName {
    fn from_cycle_error<'a>(_: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        ty::SymbolName { name: Symbol::intern("<error>").as_str() }
    }
}

pub struct CycleError<'a, 'tcx: 'a> {
    span: Span,
    cycle: RefMut<'a, [(Span, Query<'tcx>)]>,
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn report_cycle(self, CycleError { span, cycle }: CycleError) {
        // Subtle: release the refcell lock before invoking `describe()`
        // below by dropping `cycle`.
        let stack = cycle.to_vec();
        mem::drop(cycle);

        assert!(!stack.is_empty());

        // Disable naming impls with types in this path, since that
        // sometimes cycles itself, leading to extra cycle errors.
        // (And cycle errors around impls tend to occur during the
        // collect/coherence phases anyhow.)
        item_path::with_forced_impl_filename_line(|| {
            let mut err =
                struct_span_err!(self.sess, span, E0391,
                                 "unsupported cyclic reference between types/traits detected");
            err.span_label(span, "cyclic reference");

            err.span_note(stack[0].0, &format!("the cycle begins when {}...",
                                               stack[0].1.describe(self)));

            for &(span, ref query) in &stack[1..] {
                err.span_note(span, &format!("...which then requires {}...",
                                             query.describe(self)));
            }

            err.note(&format!("...which then again requires {}, completing the cycle.",
                              stack[0].1.describe(self)));

            err.emit();
        });
    }

    fn cycle_check<F, R>(self, span: Span, query: Query<'gcx>, compute: F)
                         -> Result<R, CycleError<'a, 'gcx>>
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

impl<'tcx> QueryDescription for queries::is_copy_raw<'tcx> {
    fn describe(_tcx: TyCtxt, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> String {
        format!("computing whether `{}` is `Copy`", env.value)
    }
}

impl<'tcx> QueryDescription for queries::is_sized_raw<'tcx> {
    fn describe(_tcx: TyCtxt, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> String {
        format!("computing whether `{}` is `Sized`", env.value)
    }
}

impl<'tcx> QueryDescription for queries::is_freeze_raw<'tcx> {
    fn describe(_tcx: TyCtxt, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> String {
        format!("computing whether `{}` is freeze", env.value)
    }
}

impl<'tcx> QueryDescription for queries::needs_drop_raw<'tcx> {
    fn describe(_tcx: TyCtxt, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> String {
        format!("computing whether `{}` needs drop", env.value)
    }
}

impl<'tcx> QueryDescription for queries::super_predicates_of<'tcx> {
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

impl<'tcx> QueryDescription for queries::crate_inherent_impls<'tcx> {
    fn describe(_: TyCtxt, k: CrateNum) -> String {
        format!("all inherent impls defined in crate `{:?}`", k)
    }
}

impl<'tcx> QueryDescription for queries::crate_inherent_impls_overlap_check<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        format!("check for overlap between inherent impls defined in this crate")
    }
}

impl<'tcx> QueryDescription for queries::crate_variances<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("computing the variances for items in this crate")
    }
}

impl<'tcx> QueryDescription for queries::mir_shims<'tcx> {
    fn describe(tcx: TyCtxt, def: ty::InstanceDef<'tcx>) -> String {
        format!("generating MIR shim for `{}`",
                tcx.item_path_str(def.def_id()))
    }
}

impl<'tcx> QueryDescription for queries::privacy_access_levels<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        format!("privacy access levels")
    }
}

impl<'tcx> QueryDescription for queries::typeck_item_bodies<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        format!("type-checking all item bodies")
    }
}

impl<'tcx> QueryDescription for queries::reachable_set<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        format!("reachability")
    }
}

impl<'tcx> QueryDescription for queries::const_eval<'tcx> {
    fn describe(tcx: TyCtxt, (def_id, _): (DefId, &'tcx Substs<'tcx>)) -> String {
        format!("const-evaluating `{}`", tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::mir_keys<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        format!("getting a list of all mir_keys")
    }
}

impl<'tcx> QueryDescription for queries::symbol_name<'tcx> {
    fn describe(_tcx: TyCtxt, instance: ty::Instance<'tcx>) -> String {
        format!("computing the symbol for `{}`", instance)
    }
}

impl<'tcx> QueryDescription for queries::describe_def<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("describe_def")
    }
}

impl<'tcx> QueryDescription for queries::def_span<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("def_span")
    }
}


impl<'tcx> QueryDescription for queries::stability<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("stability")
    }
}

impl<'tcx> QueryDescription for queries::deprecation<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("deprecation")
    }
}

impl<'tcx> QueryDescription for queries::item_attrs<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("item_attrs")
    }
}

impl<'tcx> QueryDescription for queries::is_exported_symbol<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("is_exported_symbol")
    }
}

impl<'tcx> QueryDescription for queries::fn_arg_names<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("fn_arg_names")
    }
}

impl<'tcx> QueryDescription for queries::impl_parent<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("impl_parent")
    }
}

impl<'tcx> QueryDescription for queries::trait_of_item<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("trait_of_item")
    }
}

impl<'tcx> QueryDescription for queries::item_body_nested_bodies<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("nested item bodies of `{}`", tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::const_is_rvalue_promotable_to_static<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("const checking if rvalue is promotable to static `{}`",
            tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::is_mir_available<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("checking if item is mir available: `{}`",
            tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::trait_impls_of<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("trait impls of `{}`", tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::relevant_trait_impls_for<'tcx> {
    fn describe(tcx: TyCtxt, (def_id, ty): (DefId, SimplifiedType)) -> String {
        format!("relevant impls for: `({}, {:?})`", tcx.item_path_str(def_id), ty)
    }
}

impl<'tcx> QueryDescription for queries::is_object_safe<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("determine object safety of trait `{}`", tcx.item_path_str(def_id))
    }
}

macro_rules! define_maps {
    (<$tcx:tt>
     $($(#[$attr:meta])*
       [$($modifiers:tt)*] $name:ident: $node:ident($K:ty) -> $V:ty,)*) => {
        define_map_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$($attr)*] [$name]))*)
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
        pub enum Query<$tcx> {
            $($(#[$attr])* $name($K)),*
        }

        impl<$tcx> Query<$tcx> {
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
                                  -> Result<R, CycleError<'a, $tcx>>
                where F: FnOnce(&$V) -> R
            {
                debug!("ty::queries::{}::try_get_with(key={:?}, span={:?})",
                       stringify!($name),
                       key,
                       span);

                if let Some(result) = tcx.maps.$name.borrow().get(&key) {
                    return Ok(f(result));
                }

                // FIXME(eddyb) Get more valid Span's on queries.
                // def_span guard is necesary to prevent a recursive loop,
                // default_span calls def_span query internally.
                if span == DUMMY_SP && stringify!($name) != "def_span" {
                    span = key.default_span(tcx)
                }

                let _task = tcx.dep_graph.in_task(Self::to_dep_node(&key));

                let result = tcx.cycle_check(span, Query::$name(key), || {
                    let provider = tcx.maps.providers[key.map_crate()].$name;
                    provider(tcx.global_tcx(), key)
                })?;

                Ok(f(tcx.maps.$name.borrow_mut().entry(key).or_insert(result)))
            }

            pub fn try_get(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K)
                           -> Result<$V, CycleError<'a, $tcx>> {
                Self::try_get_with(tcx, span, key, Clone::clone)
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

        #[derive(Copy, Clone)]
        pub struct TyCtxtAt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
            pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
            pub span: Span,
        }

        impl<'a, 'gcx, 'tcx> Deref for TyCtxtAt<'a, 'gcx, 'tcx> {
            type Target = TyCtxt<'a, 'gcx, 'tcx>;
            fn deref(&self) -> &Self::Target {
                &self.tcx
            }
        }

        impl<'a, $tcx, 'lcx> TyCtxt<'a, $tcx, 'lcx> {
            /// Return a transparent wrapper for `TyCtxt` which uses
            /// `span` as the location of queries performed through it.
            pub fn at(self, span: Span) -> TyCtxtAt<'a, $tcx, 'lcx> {
                TyCtxtAt {
                    tcx: self,
                    span
                }
            }

            $($(#[$attr])*
            pub fn $name(self, key: $K) -> $V {
                self.at(DUMMY_SP).$name(key)
            })*
        }

        impl<'a, $tcx, 'lcx> TyCtxtAt<'a, $tcx, 'lcx> {
            $($(#[$attr])*
            pub fn $name(self, key: $K) -> $V {
                queries::$name::try_get(self.tcx, self.span, key).unwrap_or_else(|e| {
                    self.report_cycle(e);
                    Value::from_cycle_error(self.global_tcx())
                })
            })*
        }

        define_provider_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$name] [$K] [$V]))*),
            output: ()
        }

        impl<$tcx> Copy for Providers<$tcx> {}
        impl<$tcx> Clone for Providers<$tcx> {
            fn clone(&self) -> Self { *self }
        }
    }
}

macro_rules! define_map_struct {
    // Initial state
    (tcx: $tcx:tt,
     input: $input:tt) => {
        define_map_struct! {
            tcx: $tcx,
            input: $input,
            output: ()
        }
    };

    // Final output
    (tcx: $tcx:tt,
     input: (),
     output: ($($output:tt)*)) => {
        pub struct Maps<$tcx> {
            providers: IndexVec<CrateNum, Providers<$tcx>>,
            query_stack: RefCell<Vec<(Span, Query<$tcx>)>>,
            $($output)*
        }
    };

    // Field recognized and ready to shift into the output
    (tcx: $tcx:tt,
     ready: ([$($pub:tt)*] [$($attr:tt)*] [$name:ident]),
     input: $input:tt,
     output: ($($output:tt)*)) => {
        define_map_struct! {
            tcx: $tcx,
            input: $input,
            output: ($($output)*
                     $(#[$attr])* $($pub)* $name: RefCell<DepTrackingMap<queries::$name<$tcx>>>,)
        }
    };

    // No modifiers left? This is a private item.
    (tcx: $tcx:tt,
     input: (([] $attrs:tt $name:tt) $($input:tt)*),
     output: $output:tt) => {
        define_map_struct! {
            tcx: $tcx,
            ready: ([] $attrs $name),
            input: ($($input)*),
            output: $output
        }
    };

    // Skip other modifiers
    (tcx: $tcx:tt,
     input: (([$other_modifier:tt $($modifiers:tt)*] $($fields:tt)*) $($input:tt)*),
     output: $output:tt) => {
        define_map_struct! {
            tcx: $tcx,
            input: (([$($modifiers)*] $($fields)*) $($input)*),
            output: $output
        }
    };
}

macro_rules! define_provider_struct {
    // Initial state:
    (tcx: $tcx:tt, input: $input:tt) => {
        define_provider_struct! {
            tcx: $tcx,
            input: $input,
            output: ()
        }
    };

    // Final state:
    (tcx: $tcx:tt,
     input: (),
     output: ($(([$name:ident] [$K:ty] [$R:ty]))*)) => {
        pub struct Providers<$tcx> {
            $(pub $name: for<'a> fn(TyCtxt<'a, $tcx, $tcx>, $K) -> $R,)*
        }

        impl<$tcx> Default for Providers<$tcx> {
            fn default() -> Self {
                $(fn $name<'a, $tcx>(_: TyCtxt<'a, $tcx, $tcx>, key: $K) -> $R {
                    bug!("tcx.maps.{}({:?}) unsupported by its crate",
                         stringify!($name), key);
                })*
                Providers { $($name),* }
            }
        }
    };

    // Something ready to shift:
    (tcx: $tcx:tt,
     ready: ($name:tt $K:tt $V:tt),
     input: $input:tt,
     output: ($($output:tt)*)) => {
        define_provider_struct! {
            tcx: $tcx,
            input: $input,
            output: ($($output)* ($name $K $V))
        }
    };

    // Regular queries produce a `V` only.
    (tcx: $tcx:tt,
     input: (([] $name:tt $K:tt $V:tt) $($input:tt)*),
     output: $output:tt) => {
        define_provider_struct! {
            tcx: $tcx,
            ready: ($name $K $V),
            input: ($($input)*),
            output: $output
        }
    };

    // Skip modifiers.
    (tcx: $tcx:tt,
     input: (([$other_modifier:tt $($modifiers:tt)*] $($fields:tt)*) $($input:tt)*),
     output: $output:tt) => {
        define_provider_struct! {
            tcx: $tcx,
            input: (([$($modifiers)*] $($fields)*) $($input)*),
            output: $output
        }
    };
}

// Each of these maps also corresponds to a method on a
// `Provider` trait for requesting a value of that type,
// and a method on `Maps` itself for doing that in a
// a way that memoizes and does dep-graph tracking,
// wrapping around the actual chain of providers that
// the driver creates (using several `rustc_*` crates).
define_maps! { <'tcx>
    /// Records the type of every item.
    [] type_of: ItemSignature(DefId) -> Ty<'tcx>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated generics and predicates.
    [] generics_of: ItemSignature(DefId) -> &'tcx ty::Generics,
    [] predicates_of: ItemSignature(DefId) -> ty::GenericPredicates<'tcx>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    [] super_predicates_of: ItemSignature(DefId) -> ty::GenericPredicates<'tcx>,

    /// To avoid cycles within the predicates of a single item we compute
    /// per-type-parameter predicates for resolving `T::AssocTy`.
    [] type_param_predicates: TypeParamPredicates((DefId, DefId))
        -> ty::GenericPredicates<'tcx>,

    [] trait_def: ItemSignature(DefId) -> &'tcx ty::TraitDef,
    [] adt_def: ItemSignature(DefId) -> &'tcx ty::AdtDef,
    [] adt_destructor: AdtDestructor(DefId) -> Option<ty::Destructor>,
    [] adt_sized_constraint: SizedConstraint(DefId) -> &'tcx [Ty<'tcx>],
    [] adt_dtorck_constraint: DtorckConstraint(DefId) -> ty::DtorckConstraint<'tcx>,

    /// True if this is a foreign item (i.e., linked via `extern { ... }`).
    [] is_foreign_item: IsForeignItem(DefId) -> bool,

    /// True if this is a default impl (aka impl Foo for ..)
    [] is_default_impl: ItemSignature(DefId) -> bool,

    /// Get a map with the variance of every item; use `item_variance`
    /// instead.
    [] crate_variances: crate_variances(CrateNum) -> Rc<ty::CrateVariancesMap>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    [] variances_of: ItemVariances(DefId) -> Rc<Vec<ty::Variance>>,

    /// Maps from an impl/trait def-id to a list of the def-ids of its items
    [] associated_item_def_ids: AssociatedItemDefIds(DefId) -> Rc<Vec<DefId>>,

    /// Maps from a trait item to the trait item "descriptor"
    [] associated_item: AssociatedItems(DefId) -> ty::AssociatedItem,

    [] impl_trait_ref: ItemSignature(DefId) -> Option<ty::TraitRef<'tcx>>,
    [] impl_polarity: ItemSignature(DefId) -> hir::ImplPolarity,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    [] inherent_impls: InherentImpls(DefId) -> Rc<Vec<DefId>>,

    /// Set of all the def-ids in this crate that have MIR associated with
    /// them. This includes all the body owners, but also things like struct
    /// constructors.
    [] mir_keys: mir_keys(CrateNum) -> Rc<DefIdSet>,

    /// Maps DefId's that have an associated Mir to the result
    /// of the MIR qualify_consts pass. The actual meaning of
    /// the value isn't known except to the pass itself.
    [] mir_const_qualif: Mir(DefId) -> u8,

    /// Fetch the MIR for a given def-id up till the point where it is
    /// ready for const evaluation.
    ///
    /// See the README for the `mir` module for details.
    [] mir_const: Mir(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    [] mir_validated: Mir(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    /// MIR after our optimization passes have run. This is MIR that is ready
    /// for trans. This is also the only query that can fetch non-local MIR, at present.
    [] optimized_mir: Mir(DefId) -> &'tcx mir::Mir<'tcx>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    [] closure_kind: ItemSignature(DefId) -> ty::ClosureKind,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    [] closure_type: ItemSignature(DefId) -> ty::PolyFnSig<'tcx>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    [] coerce_unsized_info: ItemSignature(DefId)
        -> ty::adjustment::CoerceUnsizedInfo,

    [] typeck_item_bodies: typeck_item_bodies_dep_node(CrateNum) -> CompileResult,

    [] typeck_tables_of: TypeckTables(DefId) -> &'tcx ty::TypeckTables<'tcx>,

    [] has_typeck_tables: TypeckTables(DefId) -> bool,

    [] coherent_trait: coherent_trait_dep_node((CrateNum, DefId)) -> (),

    [] borrowck: BorrowCheck(DefId) -> (),

    /// Gets a complete map from all types to their inherent impls.
    /// Not meant to be used directly outside of coherence.
    /// (Defined only for LOCAL_CRATE)
    [] crate_inherent_impls: crate_inherent_impls_dep_node(CrateNum) -> CrateInherentImpls,

    /// Checks all types in the krate for overlap in their inherent impls. Reports errors.
    /// Not meant to be used directly outside of coherence.
    /// (Defined only for LOCAL_CRATE)
    [] crate_inherent_impls_overlap_check: crate_inherent_impls_dep_node(CrateNum) -> (),

    /// Results of evaluating const items or constants embedded in
    /// other items (such as enum variant explicit discriminants).
    [] const_eval: const_eval_dep_node((DefId, &'tcx Substs<'tcx>))
        -> const_val::EvalResult<'tcx>,

    /// Performs the privacy check and computes "access levels".
    [] privacy_access_levels: PrivacyAccessLevels(CrateNum) -> Rc<AccessLevels>,

    [] reachable_set: reachability_dep_node(CrateNum) -> Rc<NodeSet>,

    /// Per-function `RegionMaps`. The `DefId` should be the owner-def-id for the fn body;
    /// in the case of closures or "inline" expressions, this will be redirected to the enclosing
    /// fn item.
    [] region_maps: RegionMaps(DefId) -> Rc<RegionMaps>,

    [] mir_shims: mir_shim_dep_node(ty::InstanceDef<'tcx>) -> &'tcx mir::Mir<'tcx>,

    [] def_symbol_name: SymbolName(DefId) -> ty::SymbolName,
    [] symbol_name: symbol_name_dep_node(ty::Instance<'tcx>) -> ty::SymbolName,

    [] describe_def: DescribeDef(DefId) -> Option<Def>,
    [] def_span: DefSpan(DefId) -> Span,
    [] stability: Stability(DefId) -> Option<attr::Stability>,
    [] deprecation: Deprecation(DefId) -> Option<attr::Deprecation>,
    [] item_attrs: ItemAttrs(DefId) -> Rc<[ast::Attribute]>,
    [] fn_arg_names: FnArgNames(DefId) -> Vec<ast::Name>,
    [] impl_parent: ImplParent(DefId) -> Option<DefId>,
    [] trait_of_item: TraitOfItem(DefId) -> Option<DefId>,
    [] is_exported_symbol: IsExportedSymbol(DefId) -> bool,
    [] item_body_nested_bodies: ItemBodyNestedBodies(DefId) -> Rc<BTreeMap<hir::BodyId, hir::Body>>,
    [] const_is_rvalue_promotable_to_static: ConstIsRvaluePromotableToStatic(DefId) -> bool,
    [] is_mir_available: IsMirAvailable(DefId) -> bool,

    [] trait_impls_of: TraitImpls(DefId) -> ty::trait_def::TraitImpls,
    // Note that TraitDef::for_each_relevant_impl() will do type simplication for you.
    [] relevant_trait_impls_for: relevant_trait_impls_for((DefId, SimplifiedType))
        -> ty::trait_def::TraitImpls,
    [] specialization_graph_of: SpecializationGraph(DefId) -> Rc<specialization_graph::Graph>,
    [] is_object_safe: ObjectSafety(DefId) -> bool,

    [] param_env: ParamEnv(DefId) -> ty::ParamEnv<'tcx>,

    // Trait selection queries. These are best used by invoking `ty.moves_by_default()`,
    // `ty.is_copy()`, etc, since that will prune the environment where possible.
    [] is_copy_raw: is_copy_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] is_sized_raw: is_sized_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] is_freeze_raw: is_freeze_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] needs_drop_raw: needs_drop_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
}

fn coherent_trait_dep_node((_, def_id): (CrateNum, DefId)) -> DepNode<DefId> {
    DepNode::CoherenceCheckTrait(def_id)
}

fn crate_inherent_impls_dep_node(_: CrateNum) -> DepNode<DefId> {
    DepNode::Coherence
}

fn reachability_dep_node(_: CrateNum) -> DepNode<DefId> {
    DepNode::Reachability
}

fn mir_shim_dep_node(instance: ty::InstanceDef) -> DepNode<DefId> {
    instance.dep_node()
}

fn symbol_name_dep_node(instance: ty::Instance) -> DepNode<DefId> {
    // symbol_name uses the substs only to traverse them to find the
    // hash, and that does not create any new dep-nodes.
    DepNode::SymbolName(instance.def.def_id())
}

fn typeck_item_bodies_dep_node(_: CrateNum) -> DepNode<DefId> {
    DepNode::TypeckBodiesKrate
}

fn const_eval_dep_node((def_id, _): (DefId, &Substs)) -> DepNode<DefId> {
    DepNode::ConstEval(def_id)
}

fn mir_keys(_: CrateNum) -> DepNode<DefId> {
    DepNode::MirKeys
}

fn crate_variances(_: CrateNum) -> DepNode<DefId> {
    DepNode::CrateVariances
}

fn relevant_trait_impls_for((def_id, _): (DefId, SimplifiedType)) -> DepNode<DefId> {
    DepNode::TraitImpls(def_id)
}

fn is_copy_dep_node<'tcx>(key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepNode<DefId> {
    let def_id = ty::item_path::characteristic_def_id_of_type(key.value)
        .unwrap_or(DefId::local(CRATE_DEF_INDEX));
    DepNode::IsCopy(def_id)
}

fn is_sized_dep_node<'tcx>(key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepNode<DefId> {
    let def_id = ty::item_path::characteristic_def_id_of_type(key.value)
        .unwrap_or(DefId::local(CRATE_DEF_INDEX));
    DepNode::IsSized(def_id)
}

fn is_freeze_dep_node<'tcx>(key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepNode<DefId> {
    let def_id = ty::item_path::characteristic_def_id_of_type(key.value)
        .unwrap_or(DefId::local(CRATE_DEF_INDEX));
    DepNode::IsFreeze(def_id)
}

fn needs_drop_dep_node<'tcx>(key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepNode<DefId> {
    let def_id = ty::item_path::characteristic_def_id_of_type(key.value)
        .unwrap_or(DefId::local(CRATE_DEF_INDEX));
    DepNode::NeedsDrop(def_id)
}
