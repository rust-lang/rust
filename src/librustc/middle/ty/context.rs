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

// FIXME: (@jroesch) @eddyb should remove this when he renames ctxt
#![allow(non_camel_case_types)]

use dep_graph::{DepGraph, DepTrackingMap};
use front::map as ast_map;
use session::Session;
use lint;
use middle;
use middle::cstore::CrateStore;
use middle::def::DefMap;
use middle::def_id::DefId;
use middle::free_region::FreeRegionMap;
use middle::region::RegionMaps;
use middle::resolve_lifetime;
use middle::stability;
use middle::subst::{self, Subst, Substs};
use middle::traits;
use middle::ty::{self, TraitRef, Ty, TypeAndMut};
use middle::ty::{TyS, TypeVariants};
use middle::ty::{AdtDef, ClosureSubsts, ExistentialBounds, Region};
use middle::ty::{FreevarMap, GenericPredicates};
use middle::ty::{BareFnTy, InferTy, ParamTy, ProjectionTy, TraitTy};
use middle::ty::{TyVar, TyVid, IntVar, IntVid, FloatVar, FloatVid};
use middle::ty::TypeVariants::*;
use middle::ty::maps;
use util::nodemap::{NodeMap, NodeSet, DefIdMap, DefIdSet};
use util::nodemap::FnvHashMap;

use arena::TypedArena;
use std::borrow::Borrow;
use std::cell::{Cell, RefCell, Ref};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use syntax::abi;
use syntax::ast::{self, Name, NodeId};
use syntax::attr;
use syntax::parse::token::special_idents;

use rustc_front::hir;

/// Internal storage
pub struct CtxtArenas<'tcx> {
    // internings
    type_: TypedArena<TyS<'tcx>>,
    substs: TypedArena<Substs<'tcx>>,
    bare_fn: TypedArena<BareFnTy<'tcx>>,
    region: TypedArena<Region>,
    stability: TypedArena<attr::Stability>,

    // references
    trait_defs: TypedArena<ty::TraitDef<'tcx>>,
    adt_defs: TypedArena<ty::AdtDefData<'tcx, 'tcx>>,
}

impl<'tcx> CtxtArenas<'tcx> {
    pub fn new() -> CtxtArenas<'tcx> {
        CtxtArenas {
            type_: TypedArena::new(),
            substs: TypedArena::new(),
            bare_fn: TypedArena::new(),
            region: TypedArena::new(),
            stability: TypedArena::new(),

            trait_defs: TypedArena::new(),
            adt_defs: TypedArena::new()
        }
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
    pub usize: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub err: Ty<'tcx>,
}

pub struct Tables<'tcx> {
    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    pub node_types: NodeMap<Ty<'tcx>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    pub item_substs: NodeMap<ty::ItemSubsts<'tcx>>,

    pub adjustments: NodeMap<ty::adjustment::AutoAdjustment<'tcx>>,

    pub method_map: ty::MethodMap<'tcx>,

    /// Borrows
    pub upvar_capture_map: ty::UpvarCaptureMap,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_tys: DefIdMap<ty::ClosureTy<'tcx>>,

    /// Records the type of each closure. The def ID is the ID of the
    /// expression defining the closure.
    pub closure_kinds: DefIdMap<ty::ClosureKind>,

    /// For each fn, records the "liberated" types of its arguments
    /// and return type. Liberated means that all bound regions
    /// (including late-bound regions) are replaced with free
    /// equivalents. This table is not used in trans (since regions
    /// are erased there) and hence is not serialized to metadata.
    pub liberated_fn_sigs: NodeMap<ty::FnSig<'tcx>>,
}

impl<'tcx> Tables<'tcx> {
    pub fn empty() -> Tables<'tcx> {
        Tables {
            node_types: FnvHashMap(),
            item_substs: NodeMap(),
            adjustments: NodeMap(),
            method_map: FnvHashMap(),
            upvar_capture_map: FnvHashMap(),
            closure_tys: DefIdMap(),
            closure_kinds: DefIdMap(),
            liberated_fn_sigs: NodeMap(),
        }
    }

    pub fn closure_kind(this: &RefCell<Self>,
                        tcx: &ty::ctxt<'tcx>,
                        def_id: DefId)
                        -> ty::ClosureKind {
        // If this is a local def-id, it should be inserted into the
        // tables by typeck; else, it will be retreived from
        // the external crate metadata.
        if let Some(&kind) = this.borrow().closure_kinds.get(&def_id) {
            return kind;
        }

        let kind = tcx.sess.cstore.closure_kind(tcx, def_id);
        this.borrow_mut().closure_kinds.insert(def_id, kind);
        kind
    }

    pub fn closure_type(this: &RefCell<Self>,
                        tcx: &ty::ctxt<'tcx>,
                        def_id: DefId,
                        substs: &ClosureSubsts<'tcx>)
                        -> ty::ClosureTy<'tcx>
    {
        // If this is a local def-id, it should be inserted into the
        // tables by typeck; else, it will be retreived from
        // the external crate metadata.
        if let Some(ty) = this.borrow().closure_tys.get(&def_id) {
            return ty.subst(tcx, &substs.func_substs);
        }

        let ty = tcx.sess.cstore.closure_ty(tcx, def_id);
        this.borrow_mut().closure_tys.insert(def_id, ty.clone());
        ty.subst(tcx, &substs.func_substs)
    }
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(arena: &'tcx TypedArena<TyS<'tcx>>,
           interner: &RefCell<FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>>)
           -> CommonTypes<'tcx>
    {
        let mk = |sty| ctxt::intern_ty(arena, interner, sty);
        CommonTypes {
            bool: mk(TyBool),
            char: mk(TyChar),
            err: mk(TyError),
            isize: mk(TyInt(ast::TyIs)),
            i8: mk(TyInt(ast::TyI8)),
            i16: mk(TyInt(ast::TyI16)),
            i32: mk(TyInt(ast::TyI32)),
            i64: mk(TyInt(ast::TyI64)),
            usize: mk(TyUint(ast::TyUs)),
            u8: mk(TyUint(ast::TyU8)),
            u16: mk(TyUint(ast::TyU16)),
            u32: mk(TyUint(ast::TyU32)),
            u64: mk(TyUint(ast::TyU64)),
            f32: mk(TyFloat(ast::TyF32)),
            f64: mk(TyFloat(ast::TyF64)),
        }
    }
}

/// The data structure to keep track of all the information that typechecker
/// generates so that so that it can be reused and doesn't have to be redone
/// later on.
pub struct ctxt<'tcx> {
    /// The arenas that types etc are allocated from.
    arenas: &'tcx CtxtArenas<'tcx>,

    /// Specifically use a speedy hash algorithm for this hash map, it's used
    /// quite often.
    // FIXME(eddyb) use a FnvHashSet<InternedTy<'tcx>> when equivalent keys can
    // queried from a HashSet.
    interner: RefCell<FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>>,

    // FIXME as above, use a hashset if equivalent elements can be queried.
    substs_interner: RefCell<FnvHashMap<&'tcx Substs<'tcx>, &'tcx Substs<'tcx>>>,
    bare_fn_interner: RefCell<FnvHashMap<&'tcx BareFnTy<'tcx>, &'tcx BareFnTy<'tcx>>>,
    region_interner: RefCell<FnvHashMap<&'tcx Region, &'tcx Region>>,
    stability_interner: RefCell<FnvHashMap<&'tcx attr::Stability, &'tcx attr::Stability>>,

    pub dep_graph: Rc<DepGraph>,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    pub sess: &'tcx Session,
    pub def_map: RefCell<DefMap>,

    pub named_region_map: resolve_lifetime::NamedRegionMap,

    pub region_maps: RegionMaps,

    // For each fn declared in the local crate, type check stores the
    // free-region relationships that were deduced from its where
    // clauses and parameter types. These are then read-again by
    // borrowck. (They are not used during trans, and hence are not
    // serialized or needed for cross-crate fns.)
    free_region_maps: RefCell<NodeMap<FreeRegionMap>>,
    // FIXME: jroesch make this a refcell

    pub tables: RefCell<Tables<'tcx>>,

    /// Maps from a trait item to the trait item "descriptor"
    pub impl_or_trait_items: RefCell<DepTrackingMap<maps::ImplOrTraitItems<'tcx>>>,

    /// Maps from a trait def-id to a list of the def-ids of its trait items
    pub trait_item_def_ids: RefCell<DefIdMap<Rc<Vec<ty::ImplOrTraitItemId>>>>,

    /// A cache for the trait_items() routine
    pub trait_items_cache: RefCell<DefIdMap<Rc<Vec<ty::ImplOrTraitItem<'tcx>>>>>,

    pub impl_trait_refs: RefCell<DefIdMap<Option<TraitRef<'tcx>>>>,
    pub trait_defs: RefCell<DefIdMap<&'tcx ty::TraitDef<'tcx>>>,
    pub adt_defs: RefCell<DefIdMap<ty::AdtDefMaster<'tcx>>>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated predicates.
    pub predicates: RefCell<DefIdMap<GenericPredicates<'tcx>>>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    pub super_predicates: RefCell<DefIdMap<GenericPredicates<'tcx>>>,

    pub map: ast_map::Map<'tcx>,
    pub freevars: RefCell<FreevarMap>,
    pub tcache: RefCell<DepTrackingMap<maps::Tcache<'tcx>>>,
    pub rcache: RefCell<FnvHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,
    pub tc_cache: RefCell<FnvHashMap<Ty<'tcx>, ty::contents::TypeContents>>,
    pub ast_ty_to_ty_cache: RefCell<NodeMap<Ty<'tcx>>>,
    pub ty_param_defs: RefCell<NodeMap<ty::TypeParameterDef<'tcx>>>,
    pub normalized_cache: RefCell<FnvHashMap<Ty<'tcx>, Ty<'tcx>>>,
    pub lang_items: middle::lang_items::LanguageItems,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub item_variance_map: RefCell<DefIdMap<Rc<ty::ItemVariances>>>,

    /// True if the variance has been computed yet; false otherwise.
    pub variance_computed: Cell<bool>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: RefCell<DefIdMap<Rc<Vec<DefId>>>>,

    /// Maps a DefId of an impl to a list of its items.
    /// Note that this contains all of the impls that we know about,
    /// including ones in other crates. It's not clear that this is the best
    /// way to do it.
    pub impl_items: RefCell<DefIdMap<Vec<ty::ImplOrTraitItemId>>>,

    /// Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    /// present in this set can be warned about.
    pub used_unsafe: RefCell<NodeSet>,

    /// Set of nodes which mark locals as mutable which end up getting used at
    /// some point. Local variable definitions not in this set can be warned
    /// about.
    pub used_mut_nodes: RefCell<NodeSet>,

    /// The set of external nominal types whose implementations have been read.
    /// This is used for lazy resolution of methods.
    pub populated_external_types: RefCell<DefIdSet>,
    /// The set of external primitive types whose implementations have been read.
    /// FIXME(arielb1): why is this separate from populated_external_types?
    pub populated_external_primitive_impls: RefCell<DefIdSet>,

    /// These caches are used by const_eval when decoding external constants.
    pub extern_const_statics: RefCell<DefIdMap<NodeId>>,
    pub extern_const_fns: RefCell<DefIdMap<NodeId>>,

    pub node_lint_levels: RefCell<FnvHashMap<(NodeId, lint::LintId),
                                              lint::LevelSource>>,

    /// The types that must be asserted to be the same size for `transmute`
    /// to be valid. We gather up these restrictions in the intrinsicck pass
    /// and check them in trans.
    pub transmute_restrictions: RefCell<Vec<ty::TransmuteRestriction<'tcx>>>,

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
    pub fulfilled_predicates: RefCell<traits::FulfilledPredicates<'tcx>>,

    /// Caches the representation hints for struct definitions.
    pub repr_hint_cache: RefCell<DefIdMap<Rc<Vec<attr::ReprAttr>>>>,

    /// Maps Expr NodeId's to their constant qualification.
    pub const_qualif_map: RefCell<NodeMap<middle::check_const::ConstQualif>>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    pub custom_coerce_unsized_kinds: RefCell<DefIdMap<ty::adjustment::CustomCoerceUnsized>>,

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
}

impl<'tcx> ctxt<'tcx> {
    pub fn type_parameter_def(&self,
                              node_id: NodeId)
                              -> ty::TypeParameterDef<'tcx>
    {
        self.ty_param_defs.borrow().get(&node_id).unwrap().clone()
    }

    pub fn node_types(&self) -> Ref<NodeMap<Ty<'tcx>>> {
        fn projection<'a, 'tcx>(tables: &'a Tables<'tcx>) -> &'a NodeMap<Ty<'tcx>> {
            &tables.node_types
        }

        Ref::map(self.tables.borrow(), projection)
    }

    pub fn node_type_insert(&self, id: NodeId, ty: Ty<'tcx>) {
        self.tables.borrow_mut().node_types.insert(id, ty);
    }

    pub fn intern_trait_def(&self, def: ty::TraitDef<'tcx>)
                            -> &'tcx ty::TraitDef<'tcx> {
        let did = def.trait_ref.def_id;
        let interned = self.arenas.trait_defs.alloc(def);
        if let Some(prev) = self.trait_defs.borrow_mut().insert(did, interned) {
            self.sess.bug(&format!("Tried to overwrite interned TraitDef: {:?}",
                                   prev))
        }
        interned
    }

    pub fn alloc_trait_def(&self, def: ty::TraitDef<'tcx>)
                           -> &'tcx ty::TraitDef<'tcx> {
        self.arenas.trait_defs.alloc(def)
    }

    pub fn intern_adt_def(&self,
                          did: DefId,
                          kind: ty::AdtKind,
                          variants: Vec<ty::VariantDefData<'tcx, 'tcx>>)
                          -> ty::AdtDefMaster<'tcx> {
        let def = ty::AdtDefData::new(self, did, kind, variants);
        let interned = self.arenas.adt_defs.alloc(def);
        // this will need a transmute when reverse-variance is removed
        if let Some(prev) = self.adt_defs.borrow_mut().insert(did, interned) {
            self.sess.bug(&format!("Tried to overwrite interned AdtDef: {:?}",
                                   prev))
        }
        interned
    }

    pub fn intern_stability(&self, stab: attr::Stability) -> &'tcx attr::Stability {
        if let Some(st) = self.stability_interner.borrow().get(&stab) {
            return st;
        }

        let interned = self.arenas.stability.alloc(stab);
        if let Some(prev) = self.stability_interner
                                .borrow_mut()
                                .insert(interned, interned) {
            self.sess.bug(&format!("Tried to overwrite interned Stability: {:?}",
                                   prev))
        }
        interned
    }

    pub fn store_free_region_map(&self, id: NodeId, map: FreeRegionMap) {
        if self.free_region_maps.borrow_mut().insert(id, map).is_some() {
            self.sess.bug(&format!("Tried to overwrite interned FreeRegionMap for NodeId {:?}",
                                   id))
        }
    }

    pub fn free_region_map(&self, id: NodeId) -> FreeRegionMap {
        self.free_region_maps.borrow()[&id].clone()
    }

    pub fn lift<T: ?Sized + Lift<'tcx>>(&self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }

    /// Create a type context and call the closure with a `&ty::ctxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_and_enter<F, R>(s: &'tcx Session,
                                 arenas: &'tcx CtxtArenas<'tcx>,
                                 def_map: RefCell<DefMap>,
                                 named_region_map: resolve_lifetime::NamedRegionMap,
                                 map: ast_map::Map<'tcx>,
                                 freevars: FreevarMap,
                                 region_maps: RegionMaps,
                                 lang_items: middle::lang_items::LanguageItems,
                                 stability: stability::Index<'tcx>,
                                 f: F) -> R
                                 where F: FnOnce(&ctxt<'tcx>) -> R
    {
        let interner = RefCell::new(FnvHashMap());
        let common_types = CommonTypes::new(&arenas.type_, &interner);
        let dep_graph = Rc::new(DepGraph::new());
        tls::enter(ctxt {
            arenas: arenas,
            interner: interner,
            substs_interner: RefCell::new(FnvHashMap()),
            bare_fn_interner: RefCell::new(FnvHashMap()),
            region_interner: RefCell::new(FnvHashMap()),
            stability_interner: RefCell::new(FnvHashMap()),
            dep_graph: dep_graph.clone(),
            types: common_types,
            named_region_map: named_region_map,
            region_maps: region_maps,
            free_region_maps: RefCell::new(FnvHashMap()),
            item_variance_map: RefCell::new(DefIdMap()),
            variance_computed: Cell::new(false),
            sess: s,
            def_map: def_map,
            tables: RefCell::new(Tables::empty()),
            impl_trait_refs: RefCell::new(DefIdMap()),
            trait_defs: RefCell::new(DefIdMap()),
            adt_defs: RefCell::new(DefIdMap()),
            predicates: RefCell::new(DefIdMap()),
            super_predicates: RefCell::new(DefIdMap()),
            fulfilled_predicates: RefCell::new(traits::FulfilledPredicates::new()),
            map: map,
            freevars: RefCell::new(freevars),
            tcache: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            rcache: RefCell::new(FnvHashMap()),
            tc_cache: RefCell::new(FnvHashMap()),
            ast_ty_to_ty_cache: RefCell::new(NodeMap()),
            impl_or_trait_items: RefCell::new(DepTrackingMap::new(dep_graph.clone())),
            trait_item_def_ids: RefCell::new(DefIdMap()),
            trait_items_cache: RefCell::new(DefIdMap()),
            ty_param_defs: RefCell::new(NodeMap()),
            normalized_cache: RefCell::new(FnvHashMap()),
            lang_items: lang_items,
            inherent_impls: RefCell::new(DefIdMap()),
            impl_items: RefCell::new(DefIdMap()),
            used_unsafe: RefCell::new(NodeSet()),
            used_mut_nodes: RefCell::new(NodeSet()),
            populated_external_types: RefCell::new(DefIdSet()),
            populated_external_primitive_impls: RefCell::new(DefIdSet()),
            extern_const_statics: RefCell::new(DefIdMap()),
            extern_const_fns: RefCell::new(DefIdMap()),
            node_lint_levels: RefCell::new(FnvHashMap()),
            transmute_restrictions: RefCell::new(Vec::new()),
            stability: RefCell::new(stability),
            selection_cache: traits::SelectionCache::new(),
            evaluation_cache: traits::EvaluationCache::new(),
            repr_hint_cache: RefCell::new(DefIdMap()),
            const_qualif_map: RefCell::new(NodeMap()),
            custom_coerce_unsized_kinds: RefCell::new(DefIdMap()),
            cast_kinds: RefCell::new(NodeMap()),
            fragment_infos: RefCell::new(DefIdMap()),
       }, f)
    }
}

/// A trait implemented for all X<'a> types which can be safely and
/// efficiently converted to X<'tcx> as long as they are part of the
/// provided ty::ctxt<'tcx>.
/// This can be done, for example, for Ty<'tcx> or &'tcx Substs<'tcx>
/// by looking them up in their respective interners.
/// None is returned if the value or one of the components is not part
/// of the provided context.
/// For Ty, None can be returned if either the type interner doesn't
/// contain the TypeVariants key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g. `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx> {
    type Lifted;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Self::Lifted>;
}

impl<'a, 'tcx> Lift<'tcx> for Ty<'a> {
    type Lifted = Ty<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<Ty<'tcx>> {
        if let Some(&ty) = tcx.interner.borrow().get(&self.sty) {
            if *self as *const _ == ty as *const _ {
                return Some(ty);
            }
        }
        None
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Substs<'a> {
    type Lifted = &'tcx Substs<'tcx>;
    fn lift_to_tcx(&self, tcx: &ctxt<'tcx>) -> Option<&'tcx Substs<'tcx>> {
        if let Some(&substs) = tcx.substs_interner.borrow().get(*self) {
            if *self as *const _ == substs as *const _ {
                return Some(substs);
            }
        }
        None
    }
}


pub mod tls {
    use middle::ty;

    use std::fmt;
    use syntax::codemap;

    /// Marker type used for the scoped TLS slot.
    /// The type context cannot be used directly because the scoped TLS
    /// in libstd doesn't allow types generic over lifetimes.
    struct ThreadLocalTyCx;

    scoped_thread_local!(static TLS_TCX: ThreadLocalTyCx);

    fn span_debug(span: codemap::Span, f: &mut fmt::Formatter) -> fmt::Result {
        with(|tcx| {
            write!(f, "{}", tcx.sess.codemap().span_to_string(span))
        })
    }

    pub fn enter<'tcx, F: FnOnce(&ty::ctxt<'tcx>) -> R, R>(tcx: ty::ctxt<'tcx>, f: F) -> R {
        codemap::SPAN_DEBUG.with(|span_dbg| {
            let original_span_debug = span_dbg.get();
            span_dbg.set(span_debug);
            let tls_ptr = &tcx as *const _ as *const ThreadLocalTyCx;
            let result = TLS_TCX.set(unsafe { &*tls_ptr }, || f(&tcx));
            span_dbg.set(original_span_debug);
            result
        })
    }

    pub fn with<F: FnOnce(&ty::ctxt) -> R, R>(f: F) -> R {
        TLS_TCX.with(|tcx| f(unsafe { &*(tcx as *const _ as *const ty::ctxt) }))
    }

    pub fn with_opt<F: FnOnce(Option<&ty::ctxt>) -> R, R>(f: F) -> R {
        if TLS_TCX.is_set() {
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
            use middle::ty;
            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                region_infer: usize,
                ty_infer: usize,
                both_infer: usize,
            }

            pub fn go(tcx: &ty::ctxt) {
                let mut total = DebugStat {
                    total: 0,
                    region_infer: 0, ty_infer: 0, both_infer: 0,
                };
                $(let mut $variant = total;)*


                for (_, t) in tcx.interner.borrow().iter() {
                    let variant = match t.sty {
                        ty::TyBool | ty::TyChar | ty::TyInt(..) | ty::TyUint(..) |
                            ty::TyFloat(..) | ty::TyStr => continue,
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

impl<'tcx> ctxt<'tcx> {
    pub fn print_debug_stats(&self) {
        sty_debug_print!(
            self,
            TyEnum, TyBox, TyArray, TySlice, TyRawPtr, TyRef, TyBareFn, TyTrait,
            TyStruct, TyClosure, TyTuple, TyParam, TyInfer, TyProjection);

        println!("Substs interner: #{}", self.substs_interner.borrow().len());
        println!("BareFnTy interner: #{}", self.bare_fn_interner.borrow().len());
        println!("Region interner: #{}", self.region_interner.borrow().len());
        println!("Stability interner: #{}", self.stability_interner.borrow().len());
    }
}


/// An entry in the type interner.
pub struct InternedTy<'tcx> {
    ty: Ty<'tcx>
}

// NB: An InternedTy compares and hashes as a sty.
impl<'tcx> PartialEq for InternedTy<'tcx> {
    fn eq(&self, other: &InternedTy<'tcx>) -> bool {
        self.ty.sty == other.ty.sty
    }
}

impl<'tcx> Eq for InternedTy<'tcx> {}

impl<'tcx> Hash for InternedTy<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.ty.sty.hash(s)
    }
}

impl<'tcx> Borrow<TypeVariants<'tcx>> for InternedTy<'tcx> {
    fn borrow<'a>(&'a self) -> &'a TypeVariants<'tcx> {
        &self.ty.sty
    }
}

fn bound_list_is_sorted(bounds: &[ty::PolyProjectionPredicate]) -> bool {
    bounds.is_empty() ||
        bounds[1..].iter().enumerate().all(
            |(index, bound)| bounds[index].sort_key() <= bound.sort_key())
}

impl<'tcx> ctxt<'tcx> {
    // Type constructors
    pub fn mk_substs(&self, substs: Substs<'tcx>) -> &'tcx Substs<'tcx> {
        if let Some(substs) = self.substs_interner.borrow().get(&substs) {
            return *substs;
        }

        let substs = self.arenas.substs.alloc(substs);
        self.substs_interner.borrow_mut().insert(substs, substs);
        substs
    }

    /// Create an unsafe fn ty based on a safe fn ty.
    pub fn safe_to_unsafe_fn_ty(&self, bare_fn: &BareFnTy<'tcx>) -> Ty<'tcx> {
        assert_eq!(bare_fn.unsafety, hir::Unsafety::Normal);
        let unsafe_fn_ty_a = self.mk_bare_fn(ty::BareFnTy {
            unsafety: hir::Unsafety::Unsafe,
            abi: bare_fn.abi,
            sig: bare_fn.sig.clone()
        });
        self.mk_fn(None, unsafe_fn_ty_a)
    }

    pub fn mk_bare_fn(&self, bare_fn: BareFnTy<'tcx>) -> &'tcx BareFnTy<'tcx> {
        if let Some(bare_fn) = self.bare_fn_interner.borrow().get(&bare_fn) {
            return *bare_fn;
        }

        let bare_fn = self.arenas.bare_fn.alloc(bare_fn);
        self.bare_fn_interner.borrow_mut().insert(bare_fn, bare_fn);
        bare_fn
    }

    pub fn mk_region(&self, region: Region) -> &'tcx Region {
        if let Some(region) = self.region_interner.borrow().get(&region) {
            return *region;
        }

        let region = self.arenas.region.alloc(region);
        self.region_interner.borrow_mut().insert(region, region);
        region
    }

    fn intern_ty(type_arena: &'tcx TypedArena<TyS<'tcx>>,
                 interner: &RefCell<FnvHashMap<InternedTy<'tcx>, Ty<'tcx>>>,
                 st: TypeVariants<'tcx>)
                 -> Ty<'tcx> {
        let ty: Ty /* don't be &mut TyS */ = {
            let mut interner = interner.borrow_mut();
            match interner.get(&st) {
                Some(ty) => return *ty,
                _ => ()
            }

            let flags = super::flags::FlagComputation::for_sty(&st);

            let ty = match () {
                () => type_arena.alloc(TyS { sty: st,
                                             flags: Cell::new(flags.flags),
                                             region_depth: flags.depth, }),
            };

            interner.insert(InternedTy { ty: ty }, ty);
            ty
        };

        debug!("Interned type: {:?} Pointer: {:?}",
            ty, ty as *const TyS);
        ty
    }

    // Interns a type/name combination, stores the resulting box in cx.interner,
    // and returns the box as cast to an unsafe ptr (see comments for Ty above).
    pub fn mk_ty(&self, st: TypeVariants<'tcx>) -> Ty<'tcx> {
        ctxt::intern_ty(&self.arenas.type_, &self.interner, st)
    }

    pub fn mk_mach_int(&self, tm: ast::IntTy) -> Ty<'tcx> {
        match tm {
            ast::TyIs   => self.types.isize,
            ast::TyI8   => self.types.i8,
            ast::TyI16  => self.types.i16,
            ast::TyI32  => self.types.i32,
            ast::TyI64  => self.types.i64,
        }
    }

    pub fn mk_mach_uint(&self, tm: ast::UintTy) -> Ty<'tcx> {
        match tm {
            ast::TyUs   => self.types.usize,
            ast::TyU8   => self.types.u8,
            ast::TyU16  => self.types.u16,
            ast::TyU32  => self.types.u32,
            ast::TyU64  => self.types.u64,
        }
    }

    pub fn mk_mach_float(&self, tm: ast::FloatTy) -> Ty<'tcx> {
        match tm {
            ast::TyF32  => self.types.f32,
            ast::TyF64  => self.types.f64,
        }
    }

    pub fn mk_str(&self) -> Ty<'tcx> {
        self.mk_ty(TyStr)
    }

    pub fn mk_static_str(&self) -> Ty<'tcx> {
        self.mk_imm_ref(self.mk_region(ty::ReStatic), self.mk_str())
    }

    pub fn mk_enum(&self, def: AdtDef<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(TyEnum(def, substs))
    }

    pub fn mk_box(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyBox(ty))
    }

    pub fn mk_ptr(&self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRawPtr(tm))
    }

    pub fn mk_ref(&self, r: &'tcx Region, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRef(r, tm))
    }

    pub fn mk_mut_ref(&self, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ref(&self, r: &'tcx Region, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_mut_ptr(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ptr(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_nil_ptr(&self) -> Ty<'tcx> {
        self.mk_imm_ptr(self.mk_nil())
    }

    pub fn mk_array(&self, ty: Ty<'tcx>, n: usize) -> Ty<'tcx> {
        self.mk_ty(TyArray(ty, n))
    }

    pub fn mk_slice(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TySlice(ty))
    }

    pub fn mk_tup(&self, ts: Vec<Ty<'tcx>>) -> Ty<'tcx> {
        self.mk_ty(TyTuple(ts))
    }

    pub fn mk_nil(&self) -> Ty<'tcx> {
        self.mk_tup(Vec::new())
    }

    pub fn mk_bool(&self) -> Ty<'tcx> {
        self.mk_ty(TyBool)
    }

    pub fn mk_fn(&self,
                 opt_def_id: Option<DefId>,
                 fty: &'tcx BareFnTy<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyBareFn(opt_def_id, fty))
    }

    pub fn mk_ctor_fn(&self,
                      def_id: DefId,
                      input_tys: &[Ty<'tcx>],
                      output: Ty<'tcx>) -> Ty<'tcx> {
        let input_args = input_tys.iter().cloned().collect();
        self.mk_fn(Some(def_id), self.mk_bare_fn(BareFnTy {
            unsafety: hir::Unsafety::Normal,
            abi: abi::Rust,
            sig: ty::Binder(ty::FnSig {
                inputs: input_args,
                output: ty::FnConverging(output),
                variadic: false
            })
        }))
    }

    pub fn mk_trait(&self,
                    principal: ty::PolyTraitRef<'tcx>,
                    bounds: ExistentialBounds<'tcx>)
                    -> Ty<'tcx>
    {
        assert!(bound_list_is_sorted(&bounds.projection_bounds));

        let inner = box TraitTy {
            principal: principal,
            bounds: bounds
        };
        self.mk_ty(TyTrait(inner))
    }

    pub fn mk_projection(&self,
                         trait_ref: TraitRef<'tcx>,
                         item_name: Name)
                         -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        let inner = ProjectionTy { trait_ref: trait_ref, item_name: item_name };
        self.mk_ty(TyProjection(inner))
    }

    pub fn mk_struct(&self, def: AdtDef<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(TyStruct(def, substs))
    }

    pub fn mk_closure(&self,
                      closure_id: DefId,
                      substs: &'tcx Substs<'tcx>,
                      tys: Vec<Ty<'tcx>>)
                      -> Ty<'tcx> {
        self.mk_closure_from_closure_substs(closure_id, Box::new(ClosureSubsts {
            func_substs: substs,
            upvar_tys: tys
        }))
    }

    pub fn mk_closure_from_closure_substs(&self,
                                          closure_id: DefId,
                                          closure_substs: Box<ClosureSubsts<'tcx>>)
                                          -> Ty<'tcx> {
        self.mk_ty(TyClosure(closure_id, closure_substs))
    }

    pub fn mk_var(&self, v: TyVid) -> Ty<'tcx> {
        self.mk_infer(TyVar(v))
    }

    pub fn mk_int_var(&self, v: IntVid) -> Ty<'tcx> {
        self.mk_infer(IntVar(v))
    }

    pub fn mk_float_var(&self, v: FloatVid) -> Ty<'tcx> {
        self.mk_infer(FloatVar(v))
    }

    pub fn mk_infer(&self, it: InferTy) -> Ty<'tcx> {
        self.mk_ty(TyInfer(it))
    }

    pub fn mk_param(&self,
                    space: subst::ParamSpace,
                    index: u32,
                    name: Name) -> Ty<'tcx> {
        self.mk_ty(TyParam(ParamTy { space: space, idx: index, name: name }))
    }

    pub fn mk_self_type(&self) -> Ty<'tcx> {
        self.mk_param(subst::SelfSpace, 0, special_idents::type_self.name)
    }

    pub fn mk_param_from_def(&self, def: &ty::TypeParameterDef) -> Ty<'tcx> {
        self.mk_param(def.space, def.index, def.name)
    }
}
