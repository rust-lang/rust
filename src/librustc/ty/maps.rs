// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepConstructor, DepNode, DepNodeIndex};
use errors::{Diagnostic, DiagnosticBuilder};
use hir::def_id::{CrateNum, DefId, LOCAL_CRATE, DefIndex};
use hir::def::{Def, Export};
use hir::{self, TraitCandidate, ItemLocalId};
use hir::svh::Svh;
use lint;
use middle::const_val;
use middle::cstore::{ExternCrate, LinkagePreference, NativeLibrary,
                     ExternBodyNestedBodies};
use middle::cstore::{NativeLibraryKind, DepKind, CrateSource, ExternConstBody};
use middle::privacy::AccessLevels;
use middle::reachable::ReachableSet;
use middle::region;
use middle::resolve_lifetime::{Region, ObjectLifetimeDefault};
use middle::stability::{self, DeprecationEntry};
use middle::lang_items::{LanguageItems, LangItem};
use middle::exported_symbols::SymbolExportLevel;
use middle::trans::{CodegenUnit, Stats};
use mir;
use mir::transform::{MirSuite, MirPassIndex};
use session::CompileResult;
use session::config::OutputFilenames;
use traits::specialization_graph;
use ty::{self, CrateInherentImpls, Ty, TyCtxt};
use ty::layout::{Layout, LayoutError};
use ty::item_path;
use ty::steal::Steal;
use ty::subst::Substs;
use ty::fast_reject::SimplifiedType;
use util::nodemap::{DefIdSet, DefIdMap};
use util::common::{profq_msg, ProfileQueriesMsg};

use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_back::PanicStrategy;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use std::cell::{RefCell, RefMut, Cell};

use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::symbol::InternedString;
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

impl Key for DefIndex {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _tcx: TyCtxt) -> Span {
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

impl<'tcx> Key for Ty<'tcx> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::ParamEnvAnd<'tcx, T> {
    fn map_crate(&self) -> CrateNum {
        self.value.map_crate()
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.value.default_span(tcx)
    }
}

impl Key for InternedString {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _tcx: TyCtxt) -> Span {
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

struct QueryMap<D: QueryDescription> {
    phantom: PhantomData<D>,
    map: FxHashMap<D::Key, QueryValue<D::Value>>,
}

struct QueryValue<T> {
    value: T,
    index: DepNodeIndex,
    diagnostics: Option<Box<QueryDiagnostics>>,
}

struct QueryDiagnostics {
    diagnostics: Vec<Diagnostic>,
    emitted_diagnostics: Cell<bool>,
}

impl<M: QueryDescription> QueryMap<M> {
    fn new() -> QueryMap<M> {
        QueryMap {
            phantom: PhantomData,
            map: FxHashMap(),
        }
    }
}

struct CycleError<'a, 'tcx: 'a> {
    span: Span,
    cycle: RefMut<'a, [(Span, Query<'tcx>)]>,
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    fn report_cycle(self, CycleError { span, cycle }: CycleError)
        -> DiagnosticBuilder<'a>
    {
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

            return err
        })
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
                    span,
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

pub trait QueryConfig {
    type Key: Eq + Hash + Clone;
    type Value;
}

trait QueryDescription: QueryConfig {
    fn describe(tcx: TyCtxt, key: Self::Key) -> String;
}

impl<M: QueryConfig<Key=DefId>> QueryDescription for M {
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

impl<'tcx> QueryDescription for queries::layout_raw<'tcx> {
    fn describe(_tcx: TyCtxt, env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> String {
        format!("computing layout of `{}`", env.value)
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
    fn describe(tcx: TyCtxt, key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>) -> String {
        format!("const-evaluating `{}`", tcx.item_path_str(key.value.0))
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


impl<'tcx> QueryDescription for queries::lookup_stability<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        bug!("stability")
    }
}

impl<'tcx> QueryDescription for queries::lookup_deprecation_entry<'tcx> {
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

impl<'tcx> QueryDescription for queries::is_object_safe<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("determine object safety of trait `{}`", tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::is_const_fn<'tcx> {
    fn describe(tcx: TyCtxt, def_id: DefId) -> String {
        format!("checking if item is const fn: `{}`", tcx.item_path_str(def_id))
    }
}

impl<'tcx> QueryDescription for queries::dylib_dependency_formats<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        "dylib dependency formats of crate".to_string()
    }
}

impl<'tcx> QueryDescription for queries::is_panic_runtime<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        "checking if the crate is_panic_runtime".to_string()
    }
}

impl<'tcx> QueryDescription for queries::is_compiler_builtins<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        "checking if the crate is_compiler_builtins".to_string()
    }
}

impl<'tcx> QueryDescription for queries::has_global_allocator<'tcx> {
    fn describe(_: TyCtxt, _: CrateNum) -> String {
        "checking if the crate has_global_allocator".to_string()
    }
}

impl<'tcx> QueryDescription for queries::extern_crate<'tcx> {
    fn describe(_: TyCtxt, _: DefId) -> String {
        "getting crate's ExternCrateData".to_string()
    }
}

impl<'tcx> QueryDescription for queries::lint_levels<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("computing the lint levels for items in this crate")
    }
}

impl<'tcx> QueryDescription for queries::specializes<'tcx> {
    fn describe(_tcx: TyCtxt, _: (DefId, DefId)) -> String {
        format!("computing whether impls specialize one another")
    }
}

impl<'tcx> QueryDescription for queries::in_scope_traits_map<'tcx> {
    fn describe(_tcx: TyCtxt, _: DefIndex) -> String {
        format!("traits in scope at a block")
    }
}

impl<'tcx> QueryDescription for queries::is_no_builtins<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("test whether a crate has #![no_builtins]")
    }
}

impl<'tcx> QueryDescription for queries::panic_strategy<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("query a crate's configured panic strategy")
    }
}

impl<'tcx> QueryDescription for queries::is_profiler_runtime<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("query a crate is #![profiler_runtime]")
    }
}

impl<'tcx> QueryDescription for queries::is_sanitizer_runtime<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("query a crate is #![sanitizer_runtime]")
    }
}

impl<'tcx> QueryDescription for queries::exported_symbol_ids<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the exported symbols of a crate")
    }
}

impl<'tcx> QueryDescription for queries::native_libraries<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the native libraries of a linked crate")
    }
}

impl<'tcx> QueryDescription for queries::plugin_registrar_fn<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the plugin registrar for a crate")
    }
}

impl<'tcx> QueryDescription for queries::derive_registrar_fn<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the derive registrar for a crate")
    }
}

impl<'tcx> QueryDescription for queries::crate_disambiguator<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the disambiguator a crate")
    }
}

impl<'tcx> QueryDescription for queries::crate_hash<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the hash a crate")
    }
}

impl<'tcx> QueryDescription for queries::original_crate_name<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up the original name a crate")
    }
}

impl<'tcx> QueryDescription for queries::implementations_of_trait<'tcx> {
    fn describe(_tcx: TyCtxt, _: (CrateNum, DefId)) -> String {
        format!("looking up implementations of a trait in a crate")
    }
}

impl<'tcx> QueryDescription for queries::all_trait_implementations<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up all (?) trait implementations")
    }
}

impl<'tcx> QueryDescription for queries::link_args<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up link arguments for a crate")
    }
}

impl<'tcx> QueryDescription for queries::named_region_map<'tcx> {
    fn describe(_tcx: TyCtxt, _: DefIndex) -> String {
        format!("looking up a named region")
    }
}

impl<'tcx> QueryDescription for queries::is_late_bound_map<'tcx> {
    fn describe(_tcx: TyCtxt, _: DefIndex) -> String {
        format!("testing if a region is late boudn")
    }
}

impl<'tcx> QueryDescription for queries::object_lifetime_defaults_map<'tcx> {
    fn describe(_tcx: TyCtxt, _: DefIndex) -> String {
        format!("looking up lifetime defaults for a region")
    }
}

impl<'tcx> QueryDescription for queries::dep_kind<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("fetching what a dependency looks like")
    }
}

impl<'tcx> QueryDescription for queries::crate_name<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("fetching what a crate is named")
    }
}

impl<'tcx> QueryDescription for queries::get_lang_items<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("calculating the lang items map")
    }
}

impl<'tcx> QueryDescription for queries::defined_lang_items<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("calculating the lang items defined in a crate")
    }
}

impl<'tcx> QueryDescription for queries::missing_lang_items<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("calculating the missing lang items in a crate")
    }
}

impl<'tcx> QueryDescription for queries::visible_parent_map<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("calculating the visible parent map")
    }
}

impl<'tcx> QueryDescription for queries::missing_extern_crate_item<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("seeing if we're missing an `extern crate` item for this crate")
    }
}

impl<'tcx> QueryDescription for queries::used_crate_source<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking at the source for a crate")
    }
}

impl<'tcx> QueryDescription for queries::postorder_cnums<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("generating a postorder list of CrateNums")
    }
}

impl<'tcx> QueryDescription for queries::maybe_unused_extern_crates<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("looking up all possibly unused extern crates")
    }
}

impl<'tcx> QueryDescription for queries::stability_index<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("calculating the stability index for the local crate")
    }
}

impl<'tcx> QueryDescription for queries::all_crate_nums<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("fetching all foreign CrateNum instances")
    }
}

impl<'tcx> QueryDescription for queries::exported_symbols<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("exported_symbols")
    }
}

impl<'tcx> QueryDescription for queries::collect_and_partition_translation_items<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("collect_and_partition_translation_items")
    }
}

impl<'tcx> QueryDescription for queries::codegen_unit<'tcx> {
    fn describe(_tcx: TyCtxt, _: InternedString) -> String {
        format!("codegen_unit")
    }
}

impl<'tcx> QueryDescription for queries::compile_codegen_unit<'tcx> {
    fn describe(_tcx: TyCtxt, _: InternedString) -> String {
        format!("compile_codegen_unit")
    }
}

impl<'tcx> QueryDescription for queries::output_filenames<'tcx> {
    fn describe(_tcx: TyCtxt, _: CrateNum) -> String {
        format!("output_filenames")
    }
}

// If enabled, send a message to the profile-queries thread
macro_rules! profq_msg {
    ($tcx:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            if  $tcx.sess.profile_queries() {
                profq_msg($msg)
            }
        }
    }
}

// If enabled, format a key using its debug string, which can be
// expensive to compute (in terms of time).
macro_rules! profq_key {
    ($tcx:expr, $key:expr) => {
        if cfg!(debug_assertions) {
            if $tcx.sess.profile_queries_and_keys() {
                Some(format!("{:?}", $key))
            } else { None }
        } else { None }
    }
}

macro_rules! define_maps {
    (<$tcx:tt>
     $($(#[$attr:meta])*
       [$($modifiers:tt)*] fn $name:ident: $node:ident($K:ty) -> $V:ty,)*) => {
        define_map_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$($attr)*] [$name]))*)
        }

        impl<$tcx> Maps<$tcx> {
            pub fn new(providers: IndexVec<CrateNum, Providers<$tcx>>)
                       -> Self {
                Maps {
                    providers,
                    query_stack: RefCell::new(vec![]),
                    $($name: RefCell::new(QueryMap::new())),*
                }
            }
        }

        #[allow(bad_style)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum Query<$tcx> {
            $($(#[$attr])* $name($K)),*
        }

        #[allow(bad_style)]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum QueryMsg {
            $($name(Option<String>)),*
        }

        impl<$tcx> Query<$tcx> {
            pub fn describe(&self, tcx: TyCtxt) -> String {
                let (r, name) = match *self {
                    $(Query::$name(key) => {
                        (queries::$name::describe(tcx, key), stringify!($name))
                    })*
                };
                if tcx.sess.verbose() {
                    format!("{} [{}]", r, name)
                } else {
                    r
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

        $(impl<$tcx> QueryConfig for queries::$name<$tcx> {
            type Key = $K;
            type Value = $V;
        }

        impl<'a, $tcx, 'lcx> queries::$name<$tcx> {
            #[allow(unused)]
            fn to_dep_node(tcx: TyCtxt<'a, $tcx, 'lcx>, key: &$K) -> DepNode {
                use dep_graph::DepConstructor::*;

                DepNode::new(tcx, $node(*key))
            }

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

                profq_msg!(tcx,
                    ProfileQueriesMsg::QueryBegin(
                        span.clone(),
                        QueryMsg::$name(profq_key!(tcx, key))
                    )
                );

                if let Some(value) = tcx.maps.$name.borrow().map.get(&key) {
                    if let Some(ref d) = value.diagnostics {
                        if !d.emitted_diagnostics.get() {
                            d.emitted_diagnostics.set(true);
                            let handle = tcx.sess.diagnostic();
                            for diagnostic in d.diagnostics.iter() {
                                DiagnosticBuilder::new_diagnostic(handle, diagnostic.clone())
                                    .emit();
                            }
                        }
                    }
                    profq_msg!(tcx, ProfileQueriesMsg::CacheHit);
                    tcx.dep_graph.read_index(value.index);
                    return Ok(f(&value.value));
                }
                // else, we are going to run the provider:
                profq_msg!(tcx, ProfileQueriesMsg::ProviderBegin);

                // FIXME(eddyb) Get more valid Span's on queries.
                // def_span guard is necessary to prevent a recursive loop,
                // default_span calls def_span query internally.
                if span == DUMMY_SP && stringify!($name) != "def_span" {
                    span = key.default_span(tcx)
                }

                let dep_node = Self::to_dep_node(tcx, &key);
                let res = tcx.cycle_check(span, Query::$name(key), || {
                    tcx.sess.diagnostic().track_diagnostics(|| {
                        if dep_node.kind.is_anon() {
                            tcx.dep_graph.with_anon_task(dep_node.kind, || {
                                let provider = tcx.maps.providers[key.map_crate()].$name;
                                provider(tcx.global_tcx(), key)
                            })
                        } else {
                            fn run_provider<'a, 'tcx, 'lcx>(tcx: TyCtxt<'a, 'tcx, 'lcx>,
                                                            key: $K)
                                                            -> $V {
                                let provider = tcx.maps.providers[key.map_crate()].$name;
                                provider(tcx.global_tcx(), key)
                            }

                            tcx.dep_graph.with_task(dep_node, tcx, key, run_provider)
                        }
                    })
                })?;
                profq_msg!(tcx, ProfileQueriesMsg::ProviderEnd);
                let ((result, dep_node_index), diagnostics) = res;

                tcx.dep_graph.read_index(dep_node_index);

                let value = QueryValue {
                    value: result,
                    index: dep_node_index,
                    diagnostics: if diagnostics.len() == 0 {
                        None
                    } else {
                        Some(Box::new(QueryDiagnostics {
                            diagnostics,
                            emitted_diagnostics: Cell::new(true),
                        }))
                    },
                };

                Ok(f(&tcx.maps
                         .$name
                         .borrow_mut()
                         .map
                         .entry(key)
                         .or_insert(value)
                         .value))
            }

            pub fn try_get(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K)
                           -> Result<$V, DiagnosticBuilder<'a>> {
                match Self::try_get_with(tcx, span, key, Clone::clone) {
                    Ok(e) => Ok(e),
                    Err(e) => Err(tcx.report_cycle(e)),
                }
            }

            pub fn force(tcx: TyCtxt<'a, $tcx, 'lcx>, span: Span, key: $K) {
                // Ignore dependencies, since we not reading the computed value
                let _task = tcx.dep_graph.in_ignore();

                match Self::try_get_with(tcx, span, key, |_| ()) {
                    Ok(()) => {}
                    Err(e) => tcx.report_cycle(e).emit(),
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
                queries::$name::try_get(self.tcx, self.span, key).unwrap_or_else(|mut e| {
                    e.emit();
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
                     $(#[$attr])* $($pub)* $name: RefCell<QueryMap<queries::$name<$tcx>>>,)
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
    [] fn type_of: TypeOfItem(DefId) -> Ty<'tcx>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated generics and predicates.
    [] fn generics_of: GenericsOfItem(DefId) -> &'tcx ty::Generics,
    [] fn predicates_of: PredicatesOfItem(DefId) -> ty::GenericPredicates<'tcx>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    [] fn super_predicates_of: SuperPredicatesOfItem(DefId) -> ty::GenericPredicates<'tcx>,

    /// To avoid cycles within the predicates of a single item we compute
    /// per-type-parameter predicates for resolving `T::AssocTy`.
    [] fn type_param_predicates: type_param_predicates((DefId, DefId))
        -> ty::GenericPredicates<'tcx>,

    [] fn trait_def: TraitDefOfItem(DefId) -> &'tcx ty::TraitDef,
    [] fn adt_def: AdtDefOfItem(DefId) -> &'tcx ty::AdtDef,
    [] fn adt_destructor: AdtDestructor(DefId) -> Option<ty::Destructor>,
    [] fn adt_sized_constraint: SizedConstraint(DefId) -> &'tcx [Ty<'tcx>],
    [] fn adt_dtorck_constraint: DtorckConstraint(DefId) -> ty::DtorckConstraint<'tcx>,

    /// True if this is a const fn
    [] fn is_const_fn: IsConstFn(DefId) -> bool,

    /// True if this is a foreign item (i.e., linked via `extern { ... }`).
    [] fn is_foreign_item: IsForeignItem(DefId) -> bool,

    /// True if this is a default impl (aka impl Foo for ..)
    [] fn is_default_impl: IsDefaultImpl(DefId) -> bool,

    /// Get a map with the variance of every item; use `item_variance`
    /// instead.
    [] fn crate_variances: crate_variances(CrateNum) -> Rc<ty::CrateVariancesMap>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    [] fn variances_of: ItemVariances(DefId) -> Rc<Vec<ty::Variance>>,

    /// Maps from an impl/trait def-id to a list of the def-ids of its items
    [] fn associated_item_def_ids: AssociatedItemDefIds(DefId) -> Rc<Vec<DefId>>,

    /// Maps from a trait item to the trait item "descriptor"
    [] fn associated_item: AssociatedItems(DefId) -> ty::AssociatedItem,

    [] fn impl_trait_ref: ImplTraitRef(DefId) -> Option<ty::TraitRef<'tcx>>,
    [] fn impl_polarity: ImplPolarity(DefId) -> hir::ImplPolarity,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    [] fn inherent_impls: InherentImpls(DefId) -> Rc<Vec<DefId>>,

    /// Set of all the def-ids in this crate that have MIR associated with
    /// them. This includes all the body owners, but also things like struct
    /// constructors.
    [] fn mir_keys: mir_keys(CrateNum) -> Rc<DefIdSet>,

    /// Maps DefId's that have an associated Mir to the result
    /// of the MIR qualify_consts pass. The actual meaning of
    /// the value isn't known except to the pass itself.
    [] fn mir_const_qualif: MirConstQualif(DefId) -> (u8, Rc<IdxSetBuf<mir::Local>>),

    /// Fetch the MIR for a given def-id up till the point where it is
    /// ready for const evaluation.
    ///
    /// See the README for the `mir` module for details.
    [] fn mir_const: MirConst(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    [] fn mir_validated: MirValidated(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    /// MIR after our optimization passes have run. This is MIR that is ready
    /// for trans. This is also the only query that can fetch non-local MIR, at present.
    [] fn optimized_mir: MirOptimized(DefId) -> &'tcx mir::Mir<'tcx>,

    /// Type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    [] fn closure_kind: ClosureKind(DefId) -> ty::ClosureKind,

    /// The signature of functions and closures.
    [] fn fn_sig: FnSignature(DefId) -> ty::PolyFnSig<'tcx>,

    /// Records the signature of each generator. The def ID is the ID of the
    /// expression defining the closure.
    [] fn generator_sig: GenSignature(DefId) -> Option<ty::PolyGenSig<'tcx>>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    [] fn coerce_unsized_info: CoerceUnsizedInfo(DefId)
        -> ty::adjustment::CoerceUnsizedInfo,

    [] fn typeck_item_bodies: typeck_item_bodies_dep_node(CrateNum) -> CompileResult,

    [] fn typeck_tables_of: TypeckTables(DefId) -> &'tcx ty::TypeckTables<'tcx>,

    [] fn has_typeck_tables: HasTypeckTables(DefId) -> bool,

    [] fn coherent_trait: coherent_trait_dep_node((CrateNum, DefId)) -> (),

    [] fn borrowck: BorrowCheck(DefId) -> (),
    // FIXME: shouldn't this return a `Result<(), BorrowckErrors>` instead?
    [] fn mir_borrowck: MirBorrowCheck(DefId) -> (),

    /// Gets a complete map from all types to their inherent impls.
    /// Not meant to be used directly outside of coherence.
    /// (Defined only for LOCAL_CRATE)
    [] fn crate_inherent_impls: crate_inherent_impls_dep_node(CrateNum) -> CrateInherentImpls,

    /// Checks all types in the krate for overlap in their inherent impls. Reports errors.
    /// Not meant to be used directly outside of coherence.
    /// (Defined only for LOCAL_CRATE)
    [] fn crate_inherent_impls_overlap_check: inherent_impls_overlap_check_dep_node(CrateNum) -> (),

    /// Results of evaluating const items or constants embedded in
    /// other items (such as enum variant explicit discriminants).
    [] fn const_eval: const_eval_dep_node(ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
        -> const_val::EvalResult<'tcx>,

    /// Performs the privacy check and computes "access levels".
    [] fn privacy_access_levels: PrivacyAccessLevels(CrateNum) -> Rc<AccessLevels>,

    [] fn reachable_set: reachability_dep_node(CrateNum) -> ReachableSet,

    /// Per-body `region::ScopeTree`. The `DefId` should be the owner-def-id for the body;
    /// in the case of closures, this will be redirected to the enclosing function.
    [] fn region_scope_tree: RegionScopeTree(DefId) -> Rc<region::ScopeTree>,

    [] fn mir_shims: mir_shim_dep_node(ty::InstanceDef<'tcx>) -> &'tcx mir::Mir<'tcx>,

    [] fn def_symbol_name: SymbolName(DefId) -> ty::SymbolName,
    [] fn symbol_name: symbol_name_dep_node(ty::Instance<'tcx>) -> ty::SymbolName,

    [] fn describe_def: DescribeDef(DefId) -> Option<Def>,
    [] fn def_span: DefSpan(DefId) -> Span,
    [] fn lookup_stability: LookupStability(DefId) -> Option<&'tcx attr::Stability>,
    [] fn lookup_deprecation_entry: LookupDeprecationEntry(DefId) -> Option<DeprecationEntry>,
    [] fn item_attrs: ItemAttrs(DefId) -> Rc<[ast::Attribute]>,
    [] fn fn_arg_names: FnArgNames(DefId) -> Vec<ast::Name>,
    [] fn impl_parent: ImplParent(DefId) -> Option<DefId>,
    [] fn trait_of_item: TraitOfItem(DefId) -> Option<DefId>,
    [] fn is_exported_symbol: IsExportedSymbol(DefId) -> bool,
    [] fn item_body_nested_bodies: ItemBodyNestedBodies(DefId) -> ExternBodyNestedBodies,
    [] fn const_is_rvalue_promotable_to_static: ConstIsRvaluePromotableToStatic(DefId) -> bool,
    [] fn is_mir_available: IsMirAvailable(DefId) -> bool,

    [] fn trait_impls_of: TraitImpls(DefId) -> Rc<ty::trait_def::TraitImpls>,
    [] fn specialization_graph_of: SpecializationGraph(DefId) -> Rc<specialization_graph::Graph>,
    [] fn is_object_safe: ObjectSafety(DefId) -> bool,

    // Get the ParameterEnvironment for a given item; this environment
    // will be in "user-facing" mode, meaning that it is suitabe for
    // type-checking etc, and it does not normalize specializable
    // associated types. This is almost always what you want,
    // unless you are doing MIR optimizations, in which case you
    // might want to use `reveal_all()` method to change modes.
    [] fn param_env: ParamEnv(DefId) -> ty::ParamEnv<'tcx>,

    // Trait selection queries. These are best used by invoking `ty.moves_by_default()`,
    // `ty.is_copy()`, etc, since that will prune the environment where possible.
    [] fn is_copy_raw: is_copy_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn is_sized_raw: is_sized_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn is_freeze_raw: is_freeze_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn needs_drop_raw: needs_drop_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn layout_raw: layout_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                                  -> Result<&'tcx Layout, LayoutError<'tcx>>,

    [] fn dylib_dependency_formats: DylibDepFormats(CrateNum)
                                    -> Rc<Vec<(CrateNum, LinkagePreference)>>,

    [] fn is_panic_runtime: IsPanicRuntime(CrateNum) -> bool,
    [] fn is_compiler_builtins: IsCompilerBuiltins(CrateNum) -> bool,
    [] fn has_global_allocator: HasGlobalAllocator(CrateNum) -> bool,
    [] fn is_sanitizer_runtime: IsSanitizerRuntime(CrateNum) -> bool,
    [] fn is_profiler_runtime: IsProfilerRuntime(CrateNum) -> bool,
    [] fn panic_strategy: GetPanicStrategy(CrateNum) -> PanicStrategy,
    [] fn is_no_builtins: IsNoBuiltins(CrateNum) -> bool,

    [] fn extern_crate: ExternCrate(DefId) -> Rc<Option<ExternCrate>>,

    [] fn specializes: specializes_node((DefId, DefId)) -> bool,
    [] fn in_scope_traits_map: InScopeTraits(DefIndex)
        -> Option<Rc<FxHashMap<ItemLocalId, Rc<Vec<TraitCandidate>>>>>,
    [] fn module_exports: ModuleExports(DefId) -> Option<Rc<Vec<Export>>>,
    [] fn lint_levels: lint_levels_node(CrateNum) -> Rc<lint::LintLevelMap>,

    [] fn impl_defaultness: ImplDefaultness(DefId) -> hir::Defaultness,
    [] fn exported_symbol_ids: ExportedSymbolIds(CrateNum) -> Rc<DefIdSet>,
    [] fn native_libraries: NativeLibraries(CrateNum) -> Rc<Vec<NativeLibrary>>,
    [] fn plugin_registrar_fn: PluginRegistrarFn(CrateNum) -> Option<DefId>,
    [] fn derive_registrar_fn: DeriveRegistrarFn(CrateNum) -> Option<DefId>,
    [] fn crate_disambiguator: CrateDisambiguator(CrateNum) -> Symbol,
    [] fn crate_hash: CrateHash(CrateNum) -> Svh,
    [] fn original_crate_name: OriginalCrateName(CrateNum) -> Symbol,

    [] fn implementations_of_trait: implementations_of_trait_node((CrateNum, DefId))
        -> Rc<Vec<DefId>>,
    [] fn all_trait_implementations: AllTraitImplementations(CrateNum)
        -> Rc<Vec<DefId>>,

    [] fn is_dllimport_foreign_item: IsDllimportForeignItem(DefId) -> bool,
    [] fn is_statically_included_foreign_item: IsStaticallyIncludedForeignItem(DefId) -> bool,
    [] fn native_library_kind: NativeLibraryKind(DefId)
        -> Option<NativeLibraryKind>,
    [] fn link_args: link_args_node(CrateNum) -> Rc<Vec<String>>,

    [] fn named_region_map: NamedRegion(DefIndex) ->
        Option<Rc<FxHashMap<ItemLocalId, Region>>>,
    [] fn is_late_bound_map: IsLateBound(DefIndex) ->
        Option<Rc<FxHashSet<ItemLocalId>>>,
    [] fn object_lifetime_defaults_map: ObjectLifetimeDefaults(DefIndex)
        -> Option<Rc<FxHashMap<ItemLocalId, Rc<Vec<ObjectLifetimeDefault>>>>>,

    [] fn visibility: Visibility(DefId) -> ty::Visibility,
    [] fn dep_kind: DepKind(CrateNum) -> DepKind,
    [] fn crate_name: CrateName(CrateNum) -> Symbol,
    [] fn item_children: ItemChildren(DefId) -> Rc<Vec<Export>>,
    [] fn extern_mod_stmt_cnum: ExternModStmtCnum(DefId) -> Option<CrateNum>,

    [] fn get_lang_items: get_lang_items_node(CrateNum) -> Rc<LanguageItems>,
    [] fn defined_lang_items: DefinedLangItems(CrateNum) -> Rc<Vec<(DefId, usize)>>,
    [] fn missing_lang_items: MissingLangItems(CrateNum) -> Rc<Vec<LangItem>>,
    [] fn extern_const_body: ExternConstBody(DefId) -> ExternConstBody<'tcx>,
    [] fn visible_parent_map: visible_parent_map_node(CrateNum)
        -> Rc<DefIdMap<DefId>>,
    [] fn missing_extern_crate_item: MissingExternCrateItem(CrateNum) -> bool,
    [] fn used_crate_source: UsedCrateSource(CrateNum) -> Rc<CrateSource>,
    [] fn postorder_cnums: postorder_cnums_node(CrateNum) -> Rc<Vec<CrateNum>>,

    [] fn freevars: Freevars(DefId) -> Option<Rc<Vec<hir::Freevar>>>,
    [] fn maybe_unused_trait_import: MaybeUnusedTraitImport(DefId) -> bool,
    [] fn maybe_unused_extern_crates: maybe_unused_extern_crates_node(CrateNum)
        -> Rc<Vec<(DefId, Span)>>,

    [] fn stability_index: stability_index_node(CrateNum) -> Rc<stability::Index<'tcx>>,
    [] fn all_crate_nums: all_crate_nums_node(CrateNum) -> Rc<Vec<CrateNum>>,

    [] fn exported_symbols: ExportedSymbols(CrateNum)
        -> Arc<Vec<(String, Option<DefId>, SymbolExportLevel)>>,
    [] fn collect_and_partition_translation_items:
        collect_and_partition_translation_items_node(CrateNum)
        -> (Arc<DefIdSet>, Arc<Vec<Arc<CodegenUnit<'tcx>>>>),
    [] fn export_name: ExportName(DefId) -> Option<Symbol>,
    [] fn contains_extern_indicator: ContainsExternIndicator(DefId) -> bool,
    [] fn is_translated_function: IsTranslatedFunction(DefId) -> bool,
    [] fn codegen_unit: CodegenUnit(InternedString) -> Arc<CodegenUnit<'tcx>>,
    [] fn compile_codegen_unit: CompileCodegenUnit(InternedString) -> Stats,
    [] fn output_filenames: output_filenames_node(CrateNum)
        -> Arc<OutputFilenames>,
}

fn type_param_predicates<'tcx>((item_id, param_id): (DefId, DefId)) -> DepConstructor<'tcx> {
    DepConstructor::TypeParamPredicates {
        item_id,
        param_id
    }
}

fn coherent_trait_dep_node<'tcx>((_, def_id): (CrateNum, DefId)) -> DepConstructor<'tcx> {
    DepConstructor::CoherenceCheckTrait(def_id)
}

fn crate_inherent_impls_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::Coherence
}

fn inherent_impls_overlap_check_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CoherenceInherentImplOverlapCheck
}

fn reachability_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::Reachability
}

fn mir_shim_dep_node<'tcx>(instance_def: ty::InstanceDef<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::MirShim {
        instance_def
    }
}

fn symbol_name_dep_node<'tcx>(instance: ty::Instance<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::InstanceSymbolName { instance }
}

fn typeck_item_bodies_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::TypeckBodiesKrate
}

fn const_eval_dep_node<'tcx>(_: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                             -> DepConstructor<'tcx> {
    DepConstructor::ConstEval
}

fn mir_keys<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::MirKeys
}

fn crate_variances<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CrateVariances
}

fn is_copy_dep_node<'tcx>(_: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::IsCopy
}

fn is_sized_dep_node<'tcx>(_: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::IsSized
}

fn is_freeze_dep_node<'tcx>(_: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::IsFreeze
}

fn needs_drop_dep_node<'tcx>(_: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::NeedsDrop
}

fn layout_dep_node<'tcx>(_: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::Layout
}

fn lint_levels_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::LintLevels
}

fn specializes_node<'tcx>((a, b): (DefId, DefId)) -> DepConstructor<'tcx> {
    DepConstructor::Specializes { impl1: a, impl2: b }
}

fn implementations_of_trait_node<'tcx>((krate, trait_id): (CrateNum, DefId))
    -> DepConstructor<'tcx>
{
    DepConstructor::ImplementationsOfTrait { krate, trait_id }
}

fn link_args_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::LinkArgs
}

fn get_lang_items_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::GetLangItems
}

fn visible_parent_map_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::VisibleParentMap
}

fn postorder_cnums_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::PostorderCnums
}

fn maybe_unused_extern_crates_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::MaybeUnusedExternCrates
}

fn stability_index_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::StabilityIndex
}

fn all_crate_nums_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::AllCrateNums
}

fn collect_and_partition_translation_items_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CollectAndPartitionTranslationItems
}

fn output_filenames_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::OutputFilenames
}
