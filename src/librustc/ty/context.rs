// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! type context book-keeping

use dep_graph::{DepGraph, DepTrackingMap};
use session::Session;
use middle;
use hir::TraitMap;
use hir::def::Def;
use hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use hir::map as ast_map;
use hir::map::DisambiguatedDefPathData;
use middle::free_region::FreeRegionMap;
use middle::region::RegionMaps;
use middle::resolve_lifetime;
use middle::stability;
use mir::Mir;
use ty::subst::{Kind, Substs};
use traits;
use ty::{self, TraitRef, Ty, TypeAndMut};
use ty::{TyS, TypeVariants, Slice};
use ty::{AdtKind, AdtDef, ClosureSubsts, Region};
use hir::FreevarMap;
use ty::{BareFnTy, InferTy, ParamTy, ProjectionTy, ExistentialPredicate};
use ty::{TyVar, TyVid, IntVar, IntVid, FloatVar, FloatVid};
use ty::TypeVariants::*;
use ty::layout::{Layout, TargetDataLayout};
use ty::inhabitedness::DefIdForest;
use ty::maps;
use util::common::MemoizationMap;
use util::nodemap::{NodeMap, NodeSet, DefIdMap, DefIdSet};
use util::nodemap::{FxHashMap, FxHashSet};
use rustc_data_structures::accumulate_vec::AccumulateVec;

use arena::{TypedArena, DroplessArena};
use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use std::iter;
use std::cmp::Ordering;
use syntax::ast::{self, Name, NodeId};
use syntax::attr;
use syntax::symbol::{Symbol, keywords};

use hir;

/// Internal storage
pub struct GlobalArenas<'tcx> {
    // internings
    layout: TypedArena<Layout>,

    // references
    generics: TypedArena<ty::Generics<'tcx>>,
    trait_def: TypedArena<ty::TraitDef>,
    adt_def: TypedArena<ty::AdtDef>,
    mir: TypedArena<RefCell<Mir<'tcx>>>,
    tables: TypedArena<ty::Tables<'tcx>>,
}

impl<'tcx> GlobalArenas<'tcx> {
    pub fn new() -> GlobalArenas<'tcx> {
        GlobalArenas {
            layout: TypedArena::new(),
            generics: TypedArena::new(),
            trait_def: TypedArena::new(),
            adt_def: TypedArena::new(),
            mir: TypedArena::new(),
            tables: TypedArena::new(),
        }
    }
}

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc are allocated from
    arena: &'tcx DroplessArena,

    /// Specifically use a speedy hash algorithm for these hash sets,
    /// they're accessed quite often.
    type_: RefCell<FxHashSet<Interned<'tcx, TyS<'tcx>>>>,
    type_list: RefCell<FxHashSet<Interned<'tcx, Slice<Ty<'tcx>>>>>,
    substs: RefCell<FxHashSet<Interned<'tcx, Substs<'tcx>>>>,
    bare_fn: RefCell<FxHashSet<Interned<'tcx, BareFnTy<'tcx>>>>,
    region: RefCell<FxHashSet<Interned<'tcx, Region>>>,
    existential_predicates: RefCell<FxHashSet<Interned<'tcx, Slice<ExistentialPredicate<'tcx>>>>>,
}

impl<'gcx: 'tcx, 'tcx> CtxtInterners<'tcx> {
    fn new(arena: &'tcx DroplessArena) -> CtxtInterners<'tcx> {
        CtxtInterners {
            arena: arena,
            type_: RefCell::new(FxHashSet()),
            type_list: RefCell::new(FxHashSet()),
            substs: RefCell::new(FxHashSet()),
            bare_fn: RefCell::new(FxHashSet()),
            region: RefCell::new(FxHashSet()),
            existential_predicates: RefCell::new(FxHashSet()),
        }
    }

    /// Intern a type. global_interners is Some only if this is
    /// a local interner and global_interners is its counterpart.
    fn intern_ty(&self, st: TypeVariants<'tcx>,
                 global_interners: Option<&CtxtInterners<'gcx>>)
                 -> Ty<'tcx> {
        let ty = {
            let mut interner = self.type_.borrow_mut();
            let global_interner = global_interners.map(|interners| {
                interners.type_.borrow_mut()
            });
            if let Some(&Interned(ty)) = interner.get(&st) {
                return ty;
            }
            if let Some(ref interner) = global_interner {
                if let Some(&Interned(ty)) = interner.get(&st) {
                    return ty;
                }
            }

            let flags = super::flags::FlagComputation::for_sty(&st);
            let ty_struct = TyS {
                sty: st,
                flags: Cell::new(flags.flags),
                region_depth: flags.depth,
            };

            // HACK(eddyb) Depend on flags being accurate to
            // determine that all contents are in the global tcx.
            // See comments on Lift for why we can't use that.
            if !flags.flags.intersects(ty::TypeFlags::KEEP_IN_LOCAL_TCX) {
                if let Some(interner) = global_interners {
                    let ty_struct: TyS<'gcx> = unsafe {
                        mem::transmute(ty_struct)
                    };
                    let ty: Ty<'gcx> = interner.arena.alloc(ty_struct);
                    global_interner.unwrap().insert(Interned(ty));
                    return ty;
                }
            } else {
                // Make sure we don't end up with inference
                // types/regions in the global tcx.
                if global_interners.is_none() {
                    drop(interner);
                    bug!("Attempted to intern `{:?}` which contains \
                          inference types/regions in the global type context",
                         &ty_struct);
                }
            }

            // Don't be &mut TyS.
            let ty: Ty<'tcx> = self.arena.alloc(ty_struct);
            interner.insert(Interned(ty));
            ty
        };

        debug!("Interned type: {:?} Pointer: {:?}",
            ty, ty as *const TyS);
        ty
    }

}

pub struct CommonTypes<'tcx> {
    pub bool: Ty<'tcx>,
    pub char: Ty<'tcx>,
    pub isize: Ty<'tcx>,
    pub i8: Ty<'tcx>,
    pub i16: Ty<'tcx>,
    pub i32: Ty<'tcx>,
    pub i64: Ty<'tcx>,
    pub i128: Ty<'tcx>,
    pub usize: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub u128: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub never: Ty<'tcx>,
    pub err: Ty<'tcx>,
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct Tables<'tcx> {
    /// Resolved definitions for `<T>::X` associated paths.
    pub type_relative_path_defs: NodeMap<Def>,

    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    pub node_types: NodeMap<Ty<'tcx>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    pub item_substs: NodeMap<ty::ItemSubsts<'tcx>>,

    pub adjustments: NodeMap<ty::adjustment::Adjustment<'tcx>>,

    pub method_map: ty::MethodMap<'tcx>,

    /// Borrows
    pub upvar_capture_map: ty::UpvarCaptureMap<'tcx>,

    /// Records the type of each closure.
    pub closure_tys: NodeMap<ty::ClosureTy<'tcx>>,

    /// Records the type of each closure.
    pub closure_kinds: NodeMap<ty::ClosureKind>,

    /// For each fn, records the "liberated" types of its arguments
    /// and return type. Liberated means that all bound regions
    /// (including late-bound regions) are replaced with free
    /// equivalents. This table is not used in trans (since regions
    /// are erased there) and hence is not serialized to metadata.
    pub liberated_fn_sigs: NodeMap<ty::FnSig<'tcx>>,

    /// For each FRU expression, record the normalized types of the fields
    /// of the struct - this is needed because it is non-trivial to
    /// normalize while preserving regions. This table is used only in
    /// MIR construction and hence is not serialized to metadata.
    pub fru_field_types: NodeMap<Vec<Ty<'tcx>>>
}

impl<'tcx> Tables<'tcx> {
    pub fn empty() -> Tables<'tcx> {
        Tables {
            type_relative_path_defs: NodeMap(),
            node_types: FxHashMap(),
            item_substs: NodeMap(),
            adjustments: NodeMap(),
            method_map: FxHashMap(),
            upvar_capture_map: FxHashMap(),
            closure_tys: NodeMap(),
            closure_kinds: NodeMap(),
            liberated_fn_sigs: NodeMap(),
            fru_field_types: NodeMap()
        }
    }

    /// Returns the final resolution of a `QPath` in an `Expr` or `Pat` node.
    pub fn qpath_def(&self, qpath: &hir::QPath, id: NodeId) -> Def {
        match *qpath {
            hir::QPath::Resolved(_, ref path) => path.def,
            hir::QPath::TypeRelative(..) => {
                self.type_relative_path_defs.get(&id).cloned().unwrap_or(Def::Err)
            }
        }
    }

    pub fn node_id_to_type(&self, id: NodeId) -> Ty<'tcx> {
        match self.node_id_to_type_opt(id) {
            Some(ty) => ty,
            None => {
                bug!("node_id_to_type: no type for node `{}`",
                     tls::with(|tcx| tcx.map.node_to_string(id)))
            }
        }
    }

    pub fn node_id_to_type_opt(&self, id: NodeId) -> Option<Ty<'tcx>> {
        self.node_types.get(&id).cloned()
    }

    pub fn node_id_item_substs(&self, id: NodeId) -> Option<&'tcx Substs<'tcx>> {
        self.item_substs.get(&id).map(|ts| ts.substs)
    }

    // Returns the type of a pattern as a monotype. Like @expr_ty, this function
    // doesn't provide type parameter substitutions.
    pub fn pat_ty(&self, pat: &hir::Pat) -> Ty<'tcx> {
        self.node_id_to_type(pat.id)
    }

    pub fn pat_ty_opt(&self, pat: &hir::Pat) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(pat.id)
    }

    // Returns the type of an expression as a monotype.
    //
    // NB (1): This is the PRE-ADJUSTMENT TYPE for the expression.  That is, in
    // some cases, we insert `Adjustment` annotations such as auto-deref or
    // auto-ref.  The type returned by this function does not consider such
    // adjustments.  See `expr_ty_adjusted()` instead.
    //
    // NB (2): This type doesn't provide type parameter substitutions; e.g. if you
    // ask for the type of "id" in "id(3)", it will return "fn(&isize) -> isize"
    // instead of "fn(ty) -> T with T = isize".
    pub fn expr_ty(&self, expr: &hir::Expr) -> Ty<'tcx> {
        self.node_id_to_type(expr.id)
    }

    pub fn expr_ty_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(expr.id)
    }

    /// Returns the type of `expr`, considering any `Adjustment`
    /// entry recorded for that expression.
    pub fn expr_ty_adjusted(&self, expr: &hir::Expr) -> Ty<'tcx> {
        self.adjustments.get(&expr.id)
            .map_or_else(|| self.expr_ty(expr), |adj| adj.target)
    }

    pub fn expr_ty_adjusted_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.adjustments.get(&expr.id)
            .map(|adj| adj.target).or_else(|| self.expr_ty_opt(expr))
    }

    pub fn is_method_call(&self, expr_id: NodeId) -> bool {
        self.method_map.contains_key(&ty::MethodCall::expr(expr_id))
    }

    pub fn is_overloaded_autoderef(&self, expr_id: NodeId, autoderefs: u32) -> bool {
        self.method_map.contains_key(&ty::MethodCall::autoderef(expr_id, autoderefs))
    }

    pub fn upvar_capture(&self, upvar_id: ty::UpvarId) -> Option<ty::UpvarCapture<'tcx>> {
        Some(self.upvar_capture_map.get(&upvar_id).unwrap().clone())
    }
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonTypes<'tcx> {
        let mk = |sty| interners.intern_ty(sty, None);
        CommonTypes {
            bool: mk(TyBool),
            char: mk(TyChar),
            never: mk(TyNever),
            err: mk(TyError),
            isize: mk(TyInt(ast::IntTy::Is)),
            i8: mk(TyInt(ast::IntTy::I8)),
            i16: mk(TyInt(ast::IntTy::I16)),
            i32: mk(TyInt(ast::IntTy::I32)),
            i64: mk(TyInt(ast::IntTy::I64)),
            i128: mk(TyInt(ast::IntTy::I128)),
            usize: mk(TyUint(ast::UintTy::Us)),
            u8: mk(TyUint(ast::UintTy::U8)),
            u16: mk(TyUint(ast::UintTy::U16)),
            u32: mk(TyUint(ast::UintTy::U32)),
            u64: mk(TyUint(ast::UintTy::U64)),
            u128: mk(TyUint(ast::UintTy::U128)),
            f32: mk(TyFloat(ast::FloatTy::F32)),
            f64: mk(TyFloat(ast::FloatTy::F64)),
        }
    }
}

/// The data structure to keep track of all the information that typechecker
/// generates so that so that it can be reused and doesn't have to be redone
/// later on.
#[derive(Copy, Clone)]
pub struct TyCtxt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    gcx: &'a GlobalCtxt<'gcx>,
    interners: &'a CtxtInterners<'tcx>
}

impl<'a, 'gcx, 'tcx> Deref for TyCtxt<'a, 'gcx, 'tcx> {
    type Target = &'a GlobalCtxt<'gcx>;
    fn deref(&self) -> &Self::Target {
        &self.gcx
    }
}

pub struct GlobalCtxt<'tcx> {
    global_arenas: &'tcx GlobalArenas<'tcx>,
    global_interners: CtxtInterners<'tcx>,

    pub specializes_cache: RefCell<traits::SpecializesCache>,

    pub dep_graph: DepGraph,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    pub sess: &'tcx Session,

    /// Map indicating what traits are in scope for places where this
    /// is relevant; generated by resolve.
    pub trait_map: TraitMap,

    pub named_region_map: resolve_lifetime::NamedRegionMap,

    pub region_maps: RegionMaps,

    // For each fn declared in the local crate, type check stores the
    // free-region relationships that were deduced from its where
    // clauses and parameter types. These are then read-again by
    // borrowck. (They are not used during trans, and hence are not
    // serialized or needed for cross-crate fns.)
    free_region_maps: RefCell<NodeMap<FreeRegionMap>>,
    // FIXME: jroesch make this a refcell

    pub tables: RefCell<DepTrackingMap<maps::Tables<'tcx>>>,

    /// Maps from a trait item to the trait item "descriptor"
    pub associated_items: RefCell<DepTrackingMap<maps::AssociatedItems<'tcx>>>,

    /// Maps from an impl/trait def-id to a list of the def-ids of its items
    pub associated_item_def_ids: RefCell<DepTrackingMap<maps::AssociatedItemDefIds<'tcx>>>,

    pub impl_trait_refs: RefCell<DepTrackingMap<maps::ImplTraitRefs<'tcx>>>,
    pub trait_defs: RefCell<DepTrackingMap<maps::TraitDefs<'tcx>>>,
    pub adt_defs: RefCell<DepTrackingMap<maps::AdtDefs<'tcx>>>,
    pub adt_sized_constraint: RefCell<DepTrackingMap<maps::AdtSizedConstraint<'tcx>>>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated generics and predicates.
    pub generics: RefCell<DepTrackingMap<maps::Generics<'tcx>>>,
    pub predicates: RefCell<DepTrackingMap<maps::Predicates<'tcx>>>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    pub super_predicates: RefCell<DepTrackingMap<maps::Predicates<'tcx>>>,

    pub map: ast_map::Map<'tcx>,

    /// Maps from the def-id of a function/method or const/static
    /// to its MIR. Mutation is done at an item granularity to
    /// allow MIR optimization passes to function and still
    /// access cross-crate MIR (e.g. inlining or const eval).
    ///
    /// Note that cross-crate MIR appears to be always borrowed
    /// (in the `RefCell` sense) to prevent accidental mutation.
    pub mir_map: RefCell<DepTrackingMap<maps::Mir<'tcx>>>,

    // Records the free variables refrenced by every closure
    // expression. Do not track deps for this, just recompute it from
    // scratch every time.
    pub freevars: RefCell<FreevarMap>,

    pub maybe_unused_trait_imports: NodeSet,

    // Records the type of every item.
    pub item_types: RefCell<DepTrackingMap<maps::Types<'tcx>>>,

    // Internal cache for metadata decoding. No need to track deps on this.
    pub rcache: RefCell<FxHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,

    // Cache for the type-contents routine. FIXME -- track deps?
    pub tc_cache: RefCell<FxHashMap<Ty<'tcx>, ty::contents::TypeContents>>,

    // FIXME no dep tracking, but we should be able to remove this
    pub ty_param_defs: RefCell<NodeMap<ty::TypeParameterDef<'tcx>>>,

    // FIXME dep tracking -- should be harmless enough
    pub normalized_cache: RefCell<FxHashMap<Ty<'tcx>, Ty<'tcx>>>,

    pub inhabitedness_cache: RefCell<FxHashMap<Ty<'tcx>, DefIdForest>>,

    pub lang_items: middle::lang_items::LanguageItems,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub item_variance_map: RefCell<DepTrackingMap<maps::ItemVariances<'tcx>>>,

    /// True if the variance has been computed yet; false otherwise.
    pub variance_computed: Cell<bool>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: RefCell<DepTrackingMap<maps::InherentImpls<'tcx>>>,

    /// Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    /// present in this set can be warned about.
    pub used_unsafe: RefCell<NodeSet>,

    /// Set of nodes which mark locals as mutable which end up getting used at
    /// some point. Local variable definitions not in this set can be warned
    /// about.
    pub used_mut_nodes: RefCell<NodeSet>,

    /// Set of trait imports actually used in the method resolution.
    /// This is used for warning unused imports.
    pub used_trait_imports: RefCell<NodeSet>,

    /// The set of external nominal types whose implementations have been read.
    /// This is used for lazy resolution of methods.
    pub populated_external_types: RefCell<DefIdSet>,

    /// The set of external primitive types whose implementations have been read.
    /// FIXME(arielb1): why is this separate from populated_external_types?
    pub populated_external_primitive_impls: RefCell<DefIdSet>,

    /// Maps any item's def-id to its stability index.
    pub stability: RefCell<stability::Index<'tcx>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that do not have to do with the parameters in scope.
    /// Merge this with `selection_cache`?
    pub evaluation_cache: traits::EvaluationCache<'tcx>,

    /// A set of predicates that have been fulfilled *somewhere*.
    /// This is used to avoid duplicate work. Predicates are only
    /// added to this set when they mention only "global" names
    /// (i.e., no type or lifetime parameters).
    pub fulfilled_predicates: RefCell<traits::GlobalFulfilledPredicates<'tcx>>,

    /// Caches the representation hints for struct definitions.
    repr_hint_cache: RefCell<DepTrackingMap<maps::ReprHints<'tcx>>>,

    /// Maps Expr NodeId's to `true` iff `&expr` can have 'static lifetime.
    pub rvalue_promotable_to_static: RefCell<NodeMap<bool>>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    pub custom_coerce_unsized_kinds: RefCell<DefIdMap<ty::adjustment::CustomCoerceUnsized>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_tys: RefCell<DepTrackingMap<maps::ClosureTypes<'tcx>>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_kinds: RefCell<DepTrackingMap<maps::ClosureKinds<'tcx>>>,

    /// Maps a cast expression to its kind. This is keyed on the
    /// *from* expression of the cast, not the cast itself.
    pub cast_kinds: RefCell<NodeMap<ty::cast::CastKind>>,

    /// Maps Fn items to a collection of fragment infos.
    ///
    /// The main goal is to identify data (each of which may be moved
    /// or assigned) whose subparts are not moved nor assigned
    /// (i.e. their state is *unfragmented*) and corresponding ast
    /// nodes where the path to that data is moved or assigned.
    ///
    /// In the long term, unfragmented values will have their
    /// destructor entirely driven by a single stack-local drop-flag,
    /// and their parents, the collections of the unfragmented values
    /// (or more simply, "fragmented values"), are mapped to the
    /// corresponding collections of stack-local drop-flags.
    ///
    /// (However, in the short term that is not the case; e.g. some
    /// unfragmented paths still need to be zeroed, namely when they
    /// reference parent data from an outer scope that was not
    /// entirely moved, and therefore that needs to be zeroed so that
    /// we do not get double-drop when we hit the end of the parent
    /// scope.)
    ///
    /// Also: currently the table solely holds keys for node-ids of
    /// unfragmented values (see `FragmentInfo` enum definition), but
    /// longer-term we will need to also store mappings from
    /// fragmented data to the set of unfragmented pieces that
    /// constitute it.
    pub fragment_infos: RefCell<DefIdMap<Vec<ty::FragmentInfo>>>,

    /// The definite name of the current crate after taking into account
    /// attributes, commandline parameters, etc.
    pub crate_name: Symbol,

    /// Data layout specification for the current target.
    pub data_layout: TargetDataLayout,

    /// Cache for layouts computed from types.
    pub layout_cache: RefCell<FxHashMap<Ty<'tcx>, &'tcx Layout>>,

    /// Used to prevent layout from recursing too deeply.
    pub layout_depth: Cell<usize>,

    /// Map from function to the `#[derive]` mode that it's defining. Only used
    /// by `proc-macro` crates.
    pub derive_macros: RefCell<NodeMap<Symbol>>,

    stability_interner: RefCell<FxHashSet<&'tcx attr::Stability>>,

    layout_interner: RefCell<FxHashSet<&'tcx Layout>>,
}

impl<'tcx> GlobalCtxt<'tcx> {
    /// Get the global TyCtxt.
    pub fn global_tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        TyCtxt {
            gcx: self,
            interners: &self.global_interners
        }
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn crate_name(self, cnum: CrateNum) -> Symbol {
        if cnum == LOCAL_CRATE {
            self.crate_name
        } else {
            self.sess.cstore.crate_name(cnum)
        }
    }

    pub fn original_crate_name(self, cnum: CrateNum) -> Symbol {
        if cnum == LOCAL_CRATE {
            self.crate_name.clone()
        } else {
            self.sess.cstore.original_crate_name(cnum)
        }
    }

    pub fn crate_disambiguator(self, cnum: CrateNum) -> Symbol {
        if cnum == LOCAL_CRATE {
            self.sess.local_crate_disambiguator()
        } else {
            self.sess.cstore.crate_disambiguator(cnum)
        }
    }

    pub fn retrace_path(self,
                        krate: CrateNum,
                        path_data: &[DisambiguatedDefPathData])
                        -> Option<DefId> {
        debug!("retrace_path(path={:?}, krate={:?})", path_data, self.crate_name(krate));

        if krate == LOCAL_CRATE {
            self.map
                .definitions()
                .def_path_table()
                .retrace_path(path_data)
                .map(|def_index| DefId { krate: krate, index: def_index })
        } else {
            self.sess.cstore.retrace_path(krate, path_data)
        }
    }

    pub fn type_parameter_def(self,
                              node_id: NodeId)
                              -> ty::TypeParameterDef<'tcx>
    {
        self.ty_param_defs.borrow().get(&node_id).unwrap().clone()
    }

    pub fn alloc_generics(self, generics: ty::Generics<'gcx>)
                          -> &'gcx ty::Generics<'gcx> {
        self.global_arenas.generics.alloc(generics)
    }

    pub fn alloc_mir(self, mir: Mir<'gcx>) -> &'gcx RefCell<Mir<'gcx>> {
        self.global_arenas.mir.alloc(RefCell::new(mir))
    }

    pub fn alloc_tables(self, tables: ty::Tables<'gcx>) -> &'gcx ty::Tables<'gcx> {
        self.global_arenas.tables.alloc(tables)
    }

    pub fn alloc_trait_def(self, def: ty::TraitDef) -> &'gcx ty::TraitDef {
        self.global_arenas.trait_def.alloc(def)
    }

    pub fn alloc_adt_def(self,
                         did: DefId,
                         kind: AdtKind,
                         variants: Vec<ty::VariantDef>)
                         -> &'gcx ty::AdtDef {
        let def = ty::AdtDef::new(self, did, kind, variants);
        self.global_arenas.adt_def.alloc(def)
    }

    pub fn intern_stability(self, stab: attr::Stability) -> &'gcx attr::Stability {
        if let Some(st) = self.stability_interner.borrow().get(&stab) {
            return st;
        }

        let interned = self.global_interners.arena.alloc(stab);
        if let Some(prev) = self.stability_interner.borrow_mut().replace(interned) {
            bug!("Tried to overwrite interned Stability: {:?}", prev)
        }
        interned
    }

    pub fn intern_layout(self, layout: Layout) -> &'gcx Layout {
        if let Some(layout) = self.layout_interner.borrow().get(&layout) {
            return layout;
        }

        let interned = self.global_arenas.layout.alloc(layout);
        if let Some(prev) = self.layout_interner.borrow_mut().replace(interned) {
            bug!("Tried to overwrite interned Layout: {:?}", prev)
        }
        interned
    }

    pub fn store_free_region_map(self, id: NodeId, map: FreeRegionMap) {
        if self.free_region_maps.borrow_mut().insert(id, map).is_some() {
            bug!("Tried to overwrite interned FreeRegionMap for NodeId {:?}", id)
        }
    }

    pub fn free_region_map(self, id: NodeId) -> FreeRegionMap {
        self.free_region_maps.borrow()[&id].clone()
    }

    pub fn lift<T: ?Sized + Lift<'tcx>>(self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }

    /// Like lift, but only tries in the global tcx.
    pub fn lift_to_global<T: ?Sized + Lift<'gcx>>(self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self.global_tcx())
    }

    /// Returns true if self is the same as self.global_tcx().
    fn is_global(self) -> bool {
        let local = self.interners as *const _;
        let global = &self.global_interners as *const _;
        local as usize == global as usize
    }

    /// Create a type context and call the closure with a `TyCtxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_and_enter<F, R>(s: &'tcx Session,
                                  arenas: &'tcx GlobalArenas<'tcx>,
                                  arena: &'tcx DroplessArena,
                                  resolutions: ty::Resolutions,
                                  named_region_map: resolve_lifetime::NamedRegionMap,
                                  map: ast_map::Map<'tcx>,
                                  region_maps: RegionMaps,
                                  lang_items: middle::lang_items::LanguageItems,
                                  stability: stability::Index<'tcx>,
                                  crate_name: &str,
                                  f: F) -> R
                                  where F: for<'b> FnOnce(TyCtxt<'b, 'tcx, 'tcx>) -> R
    {
        let data_layout = TargetDataLayout::parse(s);
        let interners = CtxtInterners::new(arena);
        let common_types = CommonTypes::new(&interners);
        let dep_graph = map.dep_graph.clone();
        let fulfilled_predicates = traits::GlobalFulfilledPredicates::new(dep_graph.clone());
        tls::enter_global(GlobalCtxt {
            specializes_cache: RefCell::new(traits::SpecializesCache::new()),
            global_arenas: arenas,
            global_interners: interners,
            dep_graph: dep_graph.clone(),
            types: common_types,
            named_region_map: named_region_map,
            region_maps: region_maps,
            free_region_maps: RefCell::new(FxHashMap()),
            item_variance_map: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            variance_computed: Cell::new(false),
            sess: s,
            trait_map: resolutions.trait_map,
            tables: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            impl_trait_refs: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            trait_defs: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            adt_defs: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            adt_sized_constraint: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            generics: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            predicates: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            super_predicates: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            fulfilled_predicates: RefCell::new(fulfilled_predicates),
            map: map,
            mir_map: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            freevars: RefCell::new(resolutions.freevars),
            maybe_unused_trait_imports: resolutions.maybe_unused_trait_imports,
            item_types: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            rcache: RefCell::new(FxHashMap()),
            tc_cache: RefCell::new(FxHashMap()),
            associated_items: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            associated_item_def_ids: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            ty_param_defs: RefCell::new(NodeMap()),
            normalized_cache: RefCell::new(FxHashMap()),
            inhabitedness_cache: RefCell::new(FxHashMap()),
            lang_items: lang_items,
            inherent_impls: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            used_unsafe: RefCell::new(NodeSet()),
            used_mut_nodes: RefCell::new(NodeSet()),
            used_trait_imports: RefCell::new(NodeSet()),
            populated_external_types: RefCell::new(DefIdSet()),
            populated_external_primitive_impls: RefCell::new(DefIdSet()),
            stability: RefCell::new(stability),
            selection_cache: traits::SelectionCache::new(),
            evaluation_cache: traits::EvaluationCache::new(),
            repr_hint_cache: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            rvalue_promotable_to_static: RefCell::new(NodeMap()),
            custom_coerce_unsized_kinds: RefCell::new(DefIdMap()),
            closure_tys: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            closure_kinds: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            cast_kinds: RefCell::new(NodeMap()),
            fragment_infos: RefCell::new(DefIdMap()),
            crate_name: Symbol::intern(crate_name),
            data_layout: data_layout,
            layout_cache: RefCell::new(FxHashMap()),
            layout_interner: RefCell::new(FxHashSet()),
            layout_depth: Cell::new(0),
            derive_macros: RefCell::new(NodeMap()),
            stability_interner: RefCell::new(FxHashSet()),
       }, f)
    }
}

impl<'gcx: 'tcx, 'tcx> GlobalCtxt<'gcx> {
    /// Call the closure with a local `TyCtxt` using the given arena.
    pub fn enter_local<F, R>(&self, arena: &'tcx DroplessArena, f: F) -> R
        where F: for<'a> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        let interners = CtxtInterners::new(arena);
        tls::enter(self, &interners, f)
    }
}

/// A trait implemented for all X<'a> types which can be safely and
/// efficiently converted to X<'tcx> as long as they are part of the
/// provided TyCtxt<'tcx>.
/// This can be done, for example, for Ty<'tcx> or &'tcx Substs<'tcx>
/// by looking them up in their respective interners.
///
/// However, this is still not the best implementation as it does
/// need to compare the components, even for interned values.
/// It would be more efficient if TypedArena provided a way to
/// determine whether the address is in the allocated range.
///
/// None is returned if the value or one of the components is not part
/// of the provided context.
/// For Ty, None can be returned if either the type interner doesn't
/// contain the TypeVariants key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g. `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx> {
    type Lifted;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted>;
}

impl<'a, 'tcx> Lift<'tcx> for Ty<'a> {
    type Lifted = Ty<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Ty<'tcx>> {
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Substs<'a> {
    type Lifted = &'tcx Substs<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<&'tcx Substs<'tcx>> {
        if self.len() == 0 {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(&self[..] as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Region {
    type Lifted = &'tcx Region;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<&'tcx Region> {
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Slice<Ty<'a>> {
    type Lifted = &'tcx Slice<Ty<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<&'tcx Slice<Ty<'tcx>>> {
        if self.len() == 0 {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Slice<ExistentialPredicate<'a>> {
    type Lifted = &'tcx Slice<ExistentialPredicate<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
        -> Option<&'tcx Slice<ExistentialPredicate<'tcx>>> {
        if self.is_empty() {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a BareFnTy<'a> {
    type Lifted = &'tcx BareFnTy<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<&'tcx BareFnTy<'tcx>> {
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}


pub mod tls {
    use super::{CtxtInterners, GlobalCtxt, TyCtxt};

    use std::cell::Cell;
    use std::fmt;
    use syntax_pos;

    /// Marker types used for the scoped TLS slot.
    /// The type context cannot be used directly because the scoped TLS
    /// in libstd doesn't allow types generic over lifetimes.
    enum ThreadLocalGlobalCtxt {}
    enum ThreadLocalInterners {}

    thread_local! {
        static TLS_TCX: Cell<Option<(*const ThreadLocalGlobalCtxt,
                                     *const ThreadLocalInterners)>> = Cell::new(None)
    }

    fn span_debug(span: syntax_pos::Span, f: &mut fmt::Formatter) -> fmt::Result {
        with(|tcx| {
            write!(f, "{}", tcx.sess.codemap().span_to_string(span))
        })
    }

    pub fn enter_global<'gcx, F, R>(gcx: GlobalCtxt<'gcx>, f: F) -> R
        where F: for<'a> FnOnce(TyCtxt<'a, 'gcx, 'gcx>) -> R
    {
        syntax_pos::SPAN_DEBUG.with(|span_dbg| {
            let original_span_debug = span_dbg.get();
            span_dbg.set(span_debug);
            let result = enter(&gcx, &gcx.global_interners, f);
            span_dbg.set(original_span_debug);
            result
        })
    }

    pub fn enter<'a, 'gcx: 'tcx, 'tcx, F, R>(gcx: &'a GlobalCtxt<'gcx>,
                                             interners: &'a CtxtInterners<'tcx>,
                                             f: F) -> R
        where F: FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        let gcx_ptr = gcx as *const _ as *const ThreadLocalGlobalCtxt;
        let interners_ptr = interners as *const _ as *const ThreadLocalInterners;
        TLS_TCX.with(|tls| {
            let prev = tls.get();
            tls.set(Some((gcx_ptr, interners_ptr)));
            let ret = f(TyCtxt {
                gcx: gcx,
                interners: interners
            });
            tls.set(prev);
            ret
        })
    }

    pub fn with<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        TLS_TCX.with(|tcx| {
            let (gcx, interners) = tcx.get().unwrap();
            let gcx = unsafe { &*(gcx as *const GlobalCtxt) };
            let interners = unsafe { &*(interners as *const CtxtInterners) };
            f(TyCtxt {
                gcx: gcx,
                interners: interners
            })
        })
    }

    pub fn with_opt<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(Option<TyCtxt<'a, 'gcx, 'tcx>>) -> R
    {
        if TLS_TCX.with(|tcx| tcx.get().is_some()) {
            with(|v| f(Some(v)))
        } else {
            f(None)
        }
    }
}

macro_rules! sty_debug_print {
    ($ctxt: expr, $($variant: ident),*) => {{
        // curious inner module to allow variant names to be used as
        // variable names.
        #[allow(non_snake_case)]
        mod inner {
            use ty::{self, TyCtxt};
            use ty::context::Interned;

            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                region_infer: usize,
                ty_infer: usize,
                both_infer: usize,
            }

            pub fn go(tcx: TyCtxt) {
                let mut total = DebugStat {
                    total: 0,
                    region_infer: 0, ty_infer: 0, both_infer: 0,
                };
                $(let mut $variant = total;)*


                for &Interned(t) in tcx.interners.type_.borrow().iter() {
                    let variant = match t.sty {
                        ty::TyBool | ty::TyChar | ty::TyInt(..) | ty::TyUint(..) |
                            ty::TyFloat(..) | ty::TyStr | ty::TyNever => continue,
                        ty::TyError => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let region = t.flags.get().intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags.get().intersects(ty::TypeFlags::HAS_TY_INFER);

                    variant.total += 1;
                    total.total += 1;
                    if region { total.region_infer += 1; variant.region_infer += 1 }
                    if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                    if region && ty { total.both_infer += 1; variant.both_infer += 1 }
                }
                println!("Ty interner             total           ty region  both");
                $(println!("    {:18}: {uses:6} {usespc:4.1}%, \
{ty:4.1}% {region:5.1}% {both:4.1}%",
                           stringify!($variant),
                           uses = $variant.total,
                           usespc = $variant.total as f64 * 100.0 / total.total as f64,
                           ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                           region = $variant.region_infer as f64 * 100.0  / total.total as f64,
                           both = $variant.both_infer as f64 * 100.0  / total.total as f64);
                  )*
                println!("                  total {uses:6}        \
{ty:4.1}% {region:5.1}% {both:4.1}%",
                         uses = total.total,
                         ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                         region = total.region_infer as f64 * 100.0  / total.total as f64,
                         both = total.both_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($ctxt)
    }}
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    pub fn print_debug_stats(self) {
        sty_debug_print!(
            self,
            TyAdt, TyBox, TyArray, TySlice, TyRawPtr, TyRef, TyFnDef, TyFnPtr,
            TyDynamic, TyClosure, TyTuple, TyParam, TyInfer, TyProjection, TyAnon);

        println!("Substs interner: #{}", self.interners.substs.borrow().len());
        println!("BareFnTy interner: #{}", self.interners.bare_fn.borrow().len());
        println!("Region interner: #{}", self.interners.region.borrow().len());
        println!("Stability interner: #{}", self.stability_interner.borrow().len());
        println!("Layout interner: #{}", self.layout_interner.borrow().len());
    }
}


/// An entry in an interner.
struct Interned<'tcx, T: 'tcx+?Sized>(&'tcx T);

// NB: An Interned<Ty> compares and hashes as a sty.
impl<'tcx> PartialEq for Interned<'tcx, TyS<'tcx>> {
    fn eq(&self, other: &Interned<'tcx, TyS<'tcx>>) -> bool {
        self.0.sty == other.0.sty
    }
}

impl<'tcx> Eq for Interned<'tcx, TyS<'tcx>> {}

impl<'tcx> Hash for Interned<'tcx, TyS<'tcx>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0.sty.hash(s)
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<TypeVariants<'lcx>> for Interned<'tcx, TyS<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a TypeVariants<'lcx> {
        &self.0.sty
    }
}

// NB: An Interned<Slice<T>> compares and hashes as its elements.
impl<'tcx, T: PartialEq> PartialEq for Interned<'tcx, Slice<T>> {
    fn eq(&self, other: &Interned<'tcx, Slice<T>>) -> bool {
        self.0[..] == other.0[..]
    }
}

impl<'tcx, T: Eq> Eq for Interned<'tcx, Slice<T>> {}

impl<'tcx, T: Hash> Hash for Interned<'tcx, Slice<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0[..].hash(s)
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Ty<'lcx>]> for Interned<'tcx, Slice<Ty<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Ty<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Kind<'lcx>]> for Interned<'tcx, Substs<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a [Kind<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<BareFnTy<'lcx>> for Interned<'tcx, BareFnTy<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a BareFnTy<'lcx> {
        self.0
    }
}

impl<'tcx> Borrow<Region> for Interned<'tcx, Region> {
    fn borrow<'a>(&'a self) -> &'a Region {
        self.0
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[ExistentialPredicate<'lcx>]>
    for Interned<'tcx, Slice<ExistentialPredicate<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [ExistentialPredicate<'lcx>] {
        &self.0[..]
    }
}

macro_rules! intern_method {
    ($lt_tcx:tt, $name:ident: $method:ident($alloc:ty,
                                            $alloc_method:ident,
                                            $alloc_to_key:expr,
                                            $alloc_to_ret:expr,
                                            $needs_infer:expr) -> $ty:ty) => {
        impl<'a, 'gcx, $lt_tcx> TyCtxt<'a, 'gcx, $lt_tcx> {
            pub fn $method(self, v: $alloc) -> &$lt_tcx $ty {
                {
                    let key = ($alloc_to_key)(&v);
                    if let Some(i) = self.interners.$name.borrow().get(key) {
                        return i.0;
                    }
                    if !self.is_global() {
                        if let Some(i) = self.global_interners.$name.borrow().get(key) {
                            return i.0;
                        }
                    }
                }

                // HACK(eddyb) Depend on flags being accurate to
                // determine that all contents are in the global tcx.
                // See comments on Lift for why we can't use that.
                if !($needs_infer)(&v) {
                    if !self.is_global() {
                        let v = unsafe {
                            mem::transmute(v)
                        };
                        let i = ($alloc_to_ret)(self.global_interners.arena.$alloc_method(v));
                        self.global_interners.$name.borrow_mut().insert(Interned(i));
                        return i;
                    }
                } else {
                    // Make sure we don't end up with inference
                    // types/regions in the global tcx.
                    if self.is_global() {
                        bug!("Attempted to intern `{:?}` which contains \
                              inference types/regions in the global type context",
                             v);
                    }
                }

                let i = ($alloc_to_ret)(self.interners.arena.$alloc_method(v));
                self.interners.$name.borrow_mut().insert(Interned(i));
                i
            }
        }
    }
}

macro_rules! direct_interners {
    ($lt_tcx:tt, $($name:ident: $method:ident($needs_infer:expr) -> $ty:ty),+) => {
        $(impl<$lt_tcx> PartialEq for Interned<$lt_tcx, $ty> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<$lt_tcx> Eq for Interned<$lt_tcx, $ty> {}

        impl<$lt_tcx> Hash for Interned<$lt_tcx, $ty> {
            fn hash<H: Hasher>(&self, s: &mut H) {
                self.0.hash(s)
            }
        }

        intern_method!($lt_tcx, $name: $method($ty, alloc, |x| x, |x| x, $needs_infer) -> $ty);)+
    }
}

fn keep_local<'tcx, T: ty::TypeFoldable<'tcx>>(x: &T) -> bool {
    x.has_type_flags(ty::TypeFlags::KEEP_IN_LOCAL_TCX)
}

direct_interners!('tcx,
    bare_fn: mk_bare_fn(|fty: &BareFnTy| {
        keep_local(&fty.sig)
    }) -> BareFnTy<'tcx>,
    region: mk_region(|r| {
        match r {
            &ty::ReVar(_) | &ty::ReSkolemized(..) => true,
            _ => false
        }
    }) -> Region
);

macro_rules! slice_interners {
    ($($field:ident: $method:ident($ty:ident)),+) => (
        $(intern_method!('tcx, $field: $method(&[$ty<'tcx>], alloc_slice, Deref::deref,
                                               |xs: &[$ty]| -> &Slice<$ty> {
            unsafe { mem::transmute(xs) }
        }, |xs: &[$ty]| xs.iter().any(keep_local)) -> Slice<$ty<'tcx>>);)+
    )
}

slice_interners!(
    existential_predicates: _intern_existential_predicates(ExistentialPredicate),
    type_list: _intern_type_list(Ty),
    substs: _intern_substs(Kind)
);

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Create an unsafe fn ty based on a safe fn ty.
    pub fn safe_to_unsafe_fn_ty(self, bare_fn: &BareFnTy<'tcx>) -> Ty<'tcx> {
        assert_eq!(bare_fn.unsafety, hir::Unsafety::Normal);
        self.mk_fn_ptr(self.mk_bare_fn(ty::BareFnTy {
            unsafety: hir::Unsafety::Unsafe,
            abi: bare_fn.abi,
            sig: bare_fn.sig.clone()
        }))
    }

    // Interns a type/name combination, stores the resulting box in cx.interners,
    // and returns the box as cast to an unsafe ptr (see comments for Ty above).
    pub fn mk_ty(self, st: TypeVariants<'tcx>) -> Ty<'tcx> {
        let global_interners = if !self.is_global() {
            Some(&self.global_interners)
        } else {
            None
        };
        self.interners.intern_ty(st, global_interners)
    }

    pub fn mk_mach_int(self, tm: ast::IntTy) -> Ty<'tcx> {
        match tm {
            ast::IntTy::Is   => self.types.isize,
            ast::IntTy::I8   => self.types.i8,
            ast::IntTy::I16  => self.types.i16,
            ast::IntTy::I32  => self.types.i32,
            ast::IntTy::I64  => self.types.i64,
            ast::IntTy::I128  => self.types.i128,
        }
    }

    pub fn mk_mach_uint(self, tm: ast::UintTy) -> Ty<'tcx> {
        match tm {
            ast::UintTy::Us   => self.types.usize,
            ast::UintTy::U8   => self.types.u8,
            ast::UintTy::U16  => self.types.u16,
            ast::UintTy::U32  => self.types.u32,
            ast::UintTy::U64  => self.types.u64,
            ast::UintTy::U128  => self.types.u128,
        }
    }

    pub fn mk_mach_float(self, tm: ast::FloatTy) -> Ty<'tcx> {
        match tm {
            ast::FloatTy::F32  => self.types.f32,
            ast::FloatTy::F64  => self.types.f64,
        }
    }

    pub fn mk_str(self) -> Ty<'tcx> {
        self.mk_ty(TyStr)
    }

    pub fn mk_static_str(self) -> Ty<'tcx> {
        self.mk_imm_ref(self.mk_region(ty::ReStatic), self.mk_str())
    }

    pub fn mk_adt(self, def: &'tcx AdtDef, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(TyAdt(def, substs))
    }

    pub fn mk_box(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyBox(ty))
    }

    pub fn mk_ptr(self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRawPtr(tm))
    }

    pub fn mk_ref(self, r: &'tcx Region, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRef(r, tm))
    }

    pub fn mk_mut_ref(self, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ref(self, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_nil_ptr(self) -> Ty<'tcx> {
        self.mk_imm_ptr(self.mk_nil())
    }

    pub fn mk_array(self, ty: Ty<'tcx>, n: usize) -> Ty<'tcx> {
        self.mk_ty(TyArray(ty, n))
    }

    pub fn mk_slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TySlice(ty))
    }

    pub fn intern_tup(self, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        self.mk_ty(TyTuple(self.intern_type_list(ts)))
    }

    pub fn mk_tup<I: InternAs<[Ty<'tcx>], Ty<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|ts| self.mk_ty(TyTuple(self.intern_type_list(ts))))
    }

    pub fn mk_nil(self) -> Ty<'tcx> {
        self.intern_tup(&[])
    }

    pub fn mk_diverging_default(self) -> Ty<'tcx> {
        if self.sess.features.borrow().never_type {
            self.types.never
        } else {
            self.mk_nil()
        }
    }

    pub fn mk_bool(self) -> Ty<'tcx> {
        self.mk_ty(TyBool)
    }

    pub fn mk_fn_def(self, def_id: DefId,
                     substs: &'tcx Substs<'tcx>,
                     fty: &'tcx BareFnTy<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyFnDef(def_id, substs, fty))
    }

    pub fn mk_fn_ptr(self, fty: &'tcx BareFnTy<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyFnPtr(fty))
    }

    pub fn mk_dynamic(
        self,
        obj: ty::Binder<&'tcx Slice<ExistentialPredicate<'tcx>>>,
        reg: &'tcx ty::Region
    ) -> Ty<'tcx> {
        self.mk_ty(TyDynamic(obj, reg))
    }

    pub fn mk_projection(self,
                         trait_ref: TraitRef<'tcx>,
                         item_name: Name)
        -> Ty<'tcx> {
            // take a copy of substs so that we own the vectors inside
            let inner = ProjectionTy { trait_ref: trait_ref, item_name: item_name };
            self.mk_ty(TyProjection(inner))
        }

    pub fn mk_closure(self,
                      closure_id: DefId,
                      substs: &'tcx Substs<'tcx>)
        -> Ty<'tcx> {
        self.mk_closure_from_closure_substs(closure_id, ClosureSubsts {
            substs: substs
        })
    }

    pub fn mk_closure_from_closure_substs(self,
                                          closure_id: DefId,
                                          closure_substs: ClosureSubsts<'tcx>)
                                          -> Ty<'tcx> {
        self.mk_ty(TyClosure(closure_id, closure_substs))
    }

    pub fn mk_var(self, v: TyVid) -> Ty<'tcx> {
        self.mk_infer(TyVar(v))
    }

    pub fn mk_int_var(self, v: IntVid) -> Ty<'tcx> {
        self.mk_infer(IntVar(v))
    }

    pub fn mk_float_var(self, v: FloatVid) -> Ty<'tcx> {
        self.mk_infer(FloatVar(v))
    }

    pub fn mk_infer(self, it: InferTy) -> Ty<'tcx> {
        self.mk_ty(TyInfer(it))
    }

    pub fn mk_param(self,
                    index: u32,
                    name: Name) -> Ty<'tcx> {
        self.mk_ty(TyParam(ParamTy { idx: index, name: name }))
    }

    pub fn mk_self_type(self) -> Ty<'tcx> {
        self.mk_param(0, keywords::SelfType.name())
    }

    pub fn mk_param_from_def(self, def: &ty::TypeParameterDef) -> Ty<'tcx> {
        self.mk_param(def.index, def.name)
    }

    pub fn mk_anon(self, def_id: DefId, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyAnon(def_id, substs))
    }

    pub fn intern_existential_predicates(self, eps: &[ExistentialPredicate<'tcx>])
        -> &'tcx Slice<ExistentialPredicate<'tcx>> {
        assert!(!eps.is_empty());
        assert!(eps.windows(2).all(|w| w[0].cmp(self, &w[1]) != Ordering::Greater));
        self._intern_existential_predicates(eps)
    }

    pub fn intern_type_list(self, ts: &[Ty<'tcx>]) -> &'tcx Slice<Ty<'tcx>> {
        if ts.len() == 0 {
            Slice::empty()
        } else {
            self._intern_type_list(ts)
        }
    }

    pub fn intern_substs(self, ts: &[Kind<'tcx>]) -> &'tcx Slice<Kind<'tcx>> {
        if ts.len() == 0 {
            Slice::empty()
        } else {
            self._intern_substs(ts)
        }
    }

    pub fn mk_fn_sig<I>(self, inputs: I, output: I::Item, variadic: bool)
        -> <I::Item as InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>::Output
        where I: Iterator,
              I::Item: InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>
    {
        inputs.chain(iter::once(output)).intern_with(|xs| ty::FnSig {
            inputs_and_output: self.intern_type_list(xs),
            variadic: variadic
        })
    }

    pub fn mk_existential_predicates<I: InternAs<[ExistentialPredicate<'tcx>],
                                     &'tcx Slice<ExistentialPredicate<'tcx>>>>(self, iter: I)
                                     -> I::Output {
        iter.intern_with(|xs| self.intern_existential_predicates(xs))
    }

    pub fn mk_type_list<I: InternAs<[Ty<'tcx>],
                        &'tcx Slice<Ty<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_type_list(xs))
    }

    pub fn mk_substs<I: InternAs<[Kind<'tcx>],
                     &'tcx Slice<Kind<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_substs(xs))
    }

    pub fn mk_substs_trait(self,
                     s: Ty<'tcx>,
                     t: &[Ty<'tcx>])
                    -> &'tcx Substs<'tcx>
    {
        self.mk_substs(iter::once(s).chain(t.into_iter().cloned()).map(Kind::from))
    }

    /// Obtain the representation annotation for a struct definition.
    pub fn lookup_repr_hints(self, did: DefId) -> Rc<Vec<attr::ReprAttr>> {
        self.repr_hint_cache.memoize(did, || {
            Rc::new(self.get_attrs(did).iter().flat_map(|meta| {
                attr::find_repr_attrs(self.sess.diagnostic(), meta).into_iter()
            }).collect())
        })
    }
}

pub trait InternAs<T: ?Sized, R> {
    type Output;
    fn intern_with<F>(self, F) -> Self::Output
        where F: FnOnce(&T) -> R;
}

impl<I, T, R, E> InternAs<[T], R> for I
    where E: InternIteratorElement<T, R>,
          I: Iterator<Item=E> {
    type Output = E::Output;
    fn intern_with<F>(self, f: F) -> Self::Output
        where F: FnOnce(&[T]) -> R {
        E::intern_with(self, f)
    }
}

pub trait InternIteratorElement<T, R>: Sized {
    type Output;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output;
}

impl<T, R> InternIteratorElement<T, R> for T {
    type Output = R;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        f(&iter.collect::<AccumulateVec<[_; 8]>>())
    }
}

impl<T, R, E> InternIteratorElement<T, R> for Result<T, E> {
    type Output = Result<R, E>;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        Ok(f(&iter.collect::<Result<AccumulateVec<[_; 8]>, _>>()?))
    }
}
