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

use dep_graph::DepGraph;
use dep_graph::{DepNode, DepConstructor};
use errors::DiagnosticBuilder;
use session::Session;
use session::config::{BorrowckMode, OutputFilenames};
use session::config::CrateType;
use middle;
use hir::{TraitCandidate, HirId, ItemKind, ItemLocalId, Node};
use hir::def::{Def, Export};
use hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use hir::map as hir_map;
use hir::map::DefPathHash;
use lint::{self, Lint};
use ich::{StableHashingContext, NodeIdHashingMode};
use infer::canonical::{CanonicalVarInfo, CanonicalVarInfos};
use infer::outlives::free_region_map::FreeRegionMap;
use middle::cstore::CrateStoreDyn;
use middle::cstore::EncodedMetadata;
use middle::lang_items;
use middle::resolve_lifetime::{self, ObjectLifetimeDefault};
use middle::stability;
use mir::{self, Mir, interpret, ProjectionKind};
use mir::interpret::Allocation;
use ty::subst::{CanonicalUserSubsts, Kind, Substs, Subst};
use ty::ReprOptions;
use traits;
use traits::{Clause, Clauses, GoalKind, Goal, Goals};
use ty::{self, Ty, TypeAndMut};
use ty::{TyS, TyKind, List};
use ty::{AdtKind, AdtDef, ClosureSubsts, GeneratorSubsts, Region, Const};
use ty::{PolyFnSig, InferTy, ParamTy, ProjectionTy, ExistentialPredicate, Predicate};
use ty::RegionKind;
use ty::{TyVar, TyVid, IntVar, IntVid, FloatVar, FloatVid};
use ty::TyKind::*;
use ty::GenericParamDefKind;
use ty::layout::{LayoutDetails, TargetDataLayout, VariantIdx};
use ty::query;
use ty::steal::Steal;
use ty::BindingMode;
use ty::CanonicalTy;
use ty::CanonicalPolyFnSig;
use util::nodemap::{DefIdMap, DefIdSet, ItemLocalMap};
use util::nodemap::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use rustc_data_structures::stable_hasher::{HashStable, hash_stable_hashmap,
                                           StableHasher, StableHasherResult,
                                           StableVec};
use arena::{TypedArena, SyncDroplessArena};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::sync::{self, Lrc, Lock, WorkerLocal};
use std::any::Any;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::{self, Entry};
use std::hash::{Hash, Hasher};
use std::fmt;
use std::mem;
use std::ops::{Deref, Bound};
use std::iter;
use std::sync::mpsc;
use std::sync::Arc;
use rustc_target::spec::abi;
use syntax::ast::{self, NodeId};
use syntax::attr;
use syntax::source_map::MultiSpan;
use syntax::edition::Edition;
use syntax::feature_gate;
use syntax::symbol::{Symbol, keywords, InternedString};
use syntax_pos::Span;

use hir;

pub struct AllArenas<'tcx> {
    pub global: WorkerLocal<GlobalArenas<'tcx>>,
    pub interner: SyncDroplessArena,
}

impl<'tcx> AllArenas<'tcx> {
    pub fn new() -> Self {
        AllArenas {
            global: WorkerLocal::new(|_| GlobalArenas::default()),
            interner: SyncDroplessArena::default(),
        }
    }
}

/// Internal storage
#[derive(Default)]
pub struct GlobalArenas<'tcx> {
    // internings
    layout: TypedArena<LayoutDetails>,

    // references
    generics: TypedArena<ty::Generics>,
    trait_def: TypedArena<ty::TraitDef>,
    adt_def: TypedArena<ty::AdtDef>,
    steal_mir: TypedArena<Steal<Mir<'tcx>>>,
    mir: TypedArena<Mir<'tcx>>,
    tables: TypedArena<ty::TypeckTables<'tcx>>,
    /// miri allocations
    const_allocs: TypedArena<interpret::Allocation>,
}

type InternedSet<'tcx, T> = Lock<FxHashSet<Interned<'tcx, T>>>;

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc are allocated from
    arena: &'tcx SyncDroplessArena,

    /// Specifically use a speedy hash algorithm for these hash sets,
    /// they're accessed quite often.
    type_: InternedSet<'tcx, TyS<'tcx>>,
    type_list: InternedSet<'tcx, List<Ty<'tcx>>>,
    substs: InternedSet<'tcx, Substs<'tcx>>,
    canonical_var_infos: InternedSet<'tcx, List<CanonicalVarInfo>>,
    region: InternedSet<'tcx, RegionKind>,
    existential_predicates: InternedSet<'tcx, List<ExistentialPredicate<'tcx>>>,
    predicates: InternedSet<'tcx, List<Predicate<'tcx>>>,
    const_: InternedSet<'tcx, Const<'tcx>>,
    clauses: InternedSet<'tcx, List<Clause<'tcx>>>,
    goal: InternedSet<'tcx, GoalKind<'tcx>>,
    goal_list: InternedSet<'tcx, List<Goal<'tcx>>>,
    projs: InternedSet<'tcx, List<ProjectionKind<'tcx>>>,
}

impl<'gcx: 'tcx, 'tcx> CtxtInterners<'tcx> {
    fn new(arena: &'tcx SyncDroplessArena) -> CtxtInterners<'tcx> {
        CtxtInterners {
            arena,
            type_: Default::default(),
            type_list: Default::default(),
            substs: Default::default(),
            region: Default::default(),
            existential_predicates: Default::default(),
            canonical_var_infos: Default::default(),
            predicates: Default::default(),
            const_: Default::default(),
            clauses: Default::default(),
            goal: Default::default(),
            goal_list: Default::default(),
            projs: Default::default(),
        }
    }

    /// Intern a type
    fn intern_ty(
        local: &CtxtInterners<'tcx>,
        global: &CtxtInterners<'gcx>,
        st: TyKind<'tcx>
    ) -> Ty<'tcx> {
        let flags = super::flags::FlagComputation::for_sty(&st);

        // HACK(eddyb) Depend on flags being accurate to
        // determine that all contents are in the global tcx.
        // See comments on Lift for why we can't use that.
        if flags.flags.intersects(ty::TypeFlags::KEEP_IN_LOCAL_TCX) {
            let mut interner = local.type_.borrow_mut();
            if let Some(&Interned(ty)) = interner.get(&st) {
                return ty;
            }

            let ty_struct = TyS {
                sty: st,
                flags: flags.flags,
                outer_exclusive_binder: flags.outer_exclusive_binder,
            };

            // Make sure we don't end up with inference
            // types/regions in the global interner
            if local as *const _ as usize == global as *const _ as usize {
                bug!("Attempted to intern `{:?}` which contains \
                      inference types/regions in the global type context",
                     &ty_struct);
            }

            // Don't be &mut TyS.
            let ty: Ty<'tcx> = local.arena.alloc(ty_struct);
            interner.insert(Interned(ty));
            ty
        } else {
            let mut interner = global.type_.borrow_mut();
            if let Some(&Interned(ty)) = interner.get(&st) {
                return ty;
            }

            let ty_struct = TyS {
                sty: st,
                flags: flags.flags,
                outer_exclusive_binder: flags.outer_exclusive_binder,
            };

            // This is safe because all the types the ty_struct can point to
            // already is in the global arena
            let ty_struct: TyS<'gcx> = unsafe {
                mem::transmute(ty_struct)
            };

            // Don't be &mut TyS.
            let ty: Ty<'gcx> = global.arena.alloc(ty_struct);
            interner.insert(Interned(ty));
            ty
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

    pub re_empty: Region<'tcx>,
    pub re_static: Region<'tcx>,
    pub re_erased: Region<'tcx>,
}

pub struct LocalTableInContext<'a, V: 'a> {
    local_id_root: Option<DefId>,
    data: &'a ItemLocalMap<V>
}

/// Validate that the given HirId (respectively its `local_id` part) can be
/// safely used as a key in the tables of a TypeckTable. For that to be
/// the case, the HirId must have the same `owner` as all the other IDs in
/// this table (signified by `local_id_root`). Otherwise the HirId
/// would be in a different frame of reference and using its `local_id`
/// would result in lookup errors, or worse, in silently wrong data being
/// stored/returned.
fn validate_hir_id_for_typeck_tables(local_id_root: Option<DefId>,
                                     hir_id: hir::HirId,
                                     mut_access: bool) {
    if cfg!(debug_assertions) {
        if let Some(local_id_root) = local_id_root {
            if hir_id.owner != local_id_root.index {
                ty::tls::with(|tcx| {
                    let node_id = tcx.hir.hir_to_node_id(hir_id);

                    bug!("node {} with HirId::owner {:?} cannot be placed in \
                          TypeckTables with local_id_root {:?}",
                         tcx.hir.node_to_string(node_id),
                         DefId::local(hir_id.owner),
                         local_id_root)
                });
            }
        } else {
            // We use "Null Object" TypeckTables in some of the analysis passes.
            // These are just expected to be empty and their `local_id_root` is
            // `None`. Therefore we cannot verify whether a given `HirId` would
            // be a valid key for the given table. Instead we make sure that
            // nobody tries to write to such a Null Object table.
            if mut_access {
                bug!("access to invalid TypeckTables")
            }
        }
    }
}

impl<'a, V> LocalTableInContext<'a, V> {
    pub fn contains_key(&self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.data.contains_key(&id.local_id)
    }

    pub fn get(&self, id: hir::HirId) -> Option<&V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.data.get(&id.local_id)
    }

    pub fn iter(&self) -> hash_map::Iter<'_, hir::ItemLocalId, V> {
        self.data.iter()
    }
}

impl<'a, V> ::std::ops::Index<hir::HirId> for LocalTableInContext<'a, V> {
    type Output = V;

    fn index(&self, key: hir::HirId) -> &V {
        self.get(key).expect("LocalTableInContext: key not found")
    }
}

pub struct LocalTableInContextMut<'a, V: 'a> {
    local_id_root: Option<DefId>,
    data: &'a mut ItemLocalMap<V>
}

impl<'a, V> LocalTableInContextMut<'a, V> {
    pub fn get_mut(&mut self, id: hir::HirId) -> Option<&mut V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.get_mut(&id.local_id)
    }

    pub fn entry(&mut self, id: hir::HirId) -> Entry<'_, hir::ItemLocalId, V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.entry(id.local_id)
    }

    pub fn insert(&mut self, id: hir::HirId, val: V) -> Option<V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.insert(id.local_id, val)
    }

    pub fn remove(&mut self, id: hir::HirId) -> Option<V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.remove(&id.local_id)
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct TypeckTables<'tcx> {
    /// The HirId::owner all ItemLocalIds in this table are relative to.
    pub local_id_root: Option<DefId>,

    /// Resolved definitions for `<T>::X` associated paths and
    /// method calls, including those of overloaded operators.
    type_dependent_defs: ItemLocalMap<Def>,

    /// Resolved field indices for field accesses in expressions (`S { field }`, `obj.field`)
    /// or patterns (`S { field }`). The index is often useful by itself, but to learn more
    /// about the field you also need definition of the variant to which the field
    /// belongs, but it may not exist if it's a tuple field (`tuple.0`).
    field_indices: ItemLocalMap<usize>,

    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    node_types: ItemLocalMap<Ty<'tcx>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    node_substs: ItemLocalMap<&'tcx Substs<'tcx>>,

    /// Stores the canonicalized types provided by the user. See also
    /// `AscribeUserType` statement in MIR.
    user_provided_tys: ItemLocalMap<CanonicalTy<'tcx>>,

    /// Stores the canonicalized types provided by the user. See also
    /// `AscribeUserType` statement in MIR.
    pub user_provided_sigs: DefIdMap<CanonicalPolyFnSig<'tcx>>,

    /// Stores the substitutions that the user explicitly gave (if any)
    /// attached to `id`. These will not include any inferred
    /// values. The canonical form is used to capture things like `_`
    /// or other unspecified values.
    ///
    /// Example:
    ///
    /// If the user wrote `foo.collect::<Vec<_>>()`, then the
    /// canonical substitutions would include only `for<X> { Vec<X>
    /// }`.
    user_substs: ItemLocalMap<CanonicalUserSubsts<'tcx>>,

    adjustments: ItemLocalMap<Vec<ty::adjustment::Adjustment<'tcx>>>,

    /// Stores the actual binding mode for all instances of hir::BindingAnnotation.
    pat_binding_modes: ItemLocalMap<BindingMode>,

    /// Stores the types which were implicitly dereferenced in pattern binding modes
    /// for later usage in HAIR lowering. For example,
    ///
    /// ```
    /// match &&Some(5i32) {
    ///     Some(n) => {},
    ///     _ => {},
    /// }
    /// ```
    /// leads to a `vec![&&Option<i32>, &Option<i32>]`. Empty vectors are not stored.
    ///
    /// See:
    /// https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md#definitions
    pat_adjustments: ItemLocalMap<Vec<Ty<'tcx>>>,

    /// Borrows
    pub upvar_capture_map: ty::UpvarCaptureMap<'tcx>,

    /// Records the reasons that we picked the kind of each closure;
    /// not all closures are present in the map.
    closure_kind_origins: ItemLocalMap<(Span, ast::Name)>,

    /// For each fn, records the "liberated" types of its arguments
    /// and return type. Liberated means that all bound regions
    /// (including late-bound regions) are replaced with free
    /// equivalents. This table is not used in codegen (since regions
    /// are erased there) and hence is not serialized to metadata.
    liberated_fn_sigs: ItemLocalMap<ty::FnSig<'tcx>>,

    /// For each FRU expression, record the normalized types of the fields
    /// of the struct - this is needed because it is non-trivial to
    /// normalize while preserving regions. This table is used only in
    /// MIR construction and hence is not serialized to metadata.
    fru_field_types: ItemLocalMap<Vec<Ty<'tcx>>>,

    /// Maps a cast expression to its kind. This is keyed on the
    /// *from* expression of the cast, not the cast itself.
    cast_kinds: ItemLocalMap<ty::cast::CastKind>,

    /// Set of trait imports actually used in the method resolution.
    /// This is used for warning unused imports. During type
    /// checking, this `Lrc` should not be cloned: it must have a ref-count
    /// of 1 so that we can insert things into the set mutably.
    pub used_trait_imports: Lrc<DefIdSet>,

    /// If any errors occurred while type-checking this body,
    /// this field will be set to `true`.
    pub tainted_by_errors: bool,

    /// Stores the free-region relationships that were deduced from
    /// its where clauses and parameter types. These are then
    /// read-again by borrowck.
    pub free_region_map: FreeRegionMap<'tcx>,

    /// All the existential types that are restricted to concrete types
    /// by this function
    pub concrete_existential_types: FxHashMap<DefId, Ty<'tcx>>,
}

impl<'tcx> TypeckTables<'tcx> {
    pub fn empty(local_id_root: Option<DefId>) -> TypeckTables<'tcx> {
        TypeckTables {
            local_id_root,
            type_dependent_defs: Default::default(),
            field_indices: Default::default(),
            user_provided_tys: Default::default(),
            user_provided_sigs: Default::default(),
            node_types: Default::default(),
            node_substs: Default::default(),
            user_substs: Default::default(),
            adjustments: Default::default(),
            pat_binding_modes: Default::default(),
            pat_adjustments: Default::default(),
            upvar_capture_map: Default::default(),
            closure_kind_origins: Default::default(),
            liberated_fn_sigs: Default::default(),
            fru_field_types: Default::default(),
            cast_kinds: Default::default(),
            used_trait_imports: Lrc::new(Default::default()),
            tainted_by_errors: false,
            free_region_map: Default::default(),
            concrete_existential_types: Default::default(),
        }
    }

    /// Returns the final resolution of a `QPath` in an `Expr` or `Pat` node.
    pub fn qpath_def(&self, qpath: &hir::QPath, id: hir::HirId) -> Def {
        match *qpath {
            hir::QPath::Resolved(_, ref path) => path.def,
            hir::QPath::TypeRelative(..) => {
                validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
                self.type_dependent_defs.get(&id.local_id).cloned().unwrap_or(Def::Err)
            }
        }
    }

    pub fn type_dependent_defs(&self) -> LocalTableInContext<'_, Def> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.type_dependent_defs
        }
    }

    pub fn type_dependent_defs_mut(&mut self) -> LocalTableInContextMut<'_, Def> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.type_dependent_defs
        }
    }

    pub fn field_indices(&self) -> LocalTableInContext<'_, usize> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.field_indices
        }
    }

    pub fn field_indices_mut(&mut self) -> LocalTableInContextMut<'_, usize> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.field_indices
        }
    }

    pub fn user_provided_tys(&self) -> LocalTableInContext<'_, CanonicalTy<'tcx>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.user_provided_tys
        }
    }

    pub fn user_provided_tys_mut(&mut self) -> LocalTableInContextMut<'_, CanonicalTy<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.user_provided_tys
        }
    }

    pub fn node_types(&self) -> LocalTableInContext<'_, Ty<'tcx>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.node_types
        }
    }

    pub fn node_types_mut(&mut self) -> LocalTableInContextMut<'_, Ty<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.node_types
        }
    }

    pub fn node_id_to_type(&self, id: hir::HirId) -> Ty<'tcx> {
        self.node_id_to_type_opt(id).unwrap_or_else(||
            bug!("node_id_to_type: no type for node `{}`",
                 tls::with(|tcx| {
                     let id = tcx.hir.hir_to_node_id(id);
                     tcx.hir.node_to_string(id)
                 }))
        )
    }

    pub fn node_id_to_type_opt(&self, id: hir::HirId) -> Option<Ty<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_types.get(&id.local_id).cloned()
    }

    pub fn node_substs_mut(&mut self) -> LocalTableInContextMut<'_, &'tcx Substs<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.node_substs
        }
    }

    pub fn node_substs(&self, id: hir::HirId) -> &'tcx Substs<'tcx> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_substs.get(&id.local_id).cloned().unwrap_or_else(|| Substs::empty())
    }

    pub fn node_substs_opt(&self, id: hir::HirId) -> Option<&'tcx Substs<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_substs.get(&id.local_id).cloned()
    }

    pub fn user_substs_mut(&mut self) -> LocalTableInContextMut<'_, CanonicalUserSubsts<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.user_substs
        }
    }

    pub fn user_substs(&self, id: hir::HirId) -> Option<CanonicalUserSubsts<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.user_substs.get(&id.local_id).cloned()
    }

    // Returns the type of a pattern as a monotype. Like @expr_ty, this function
    // doesn't provide type parameter substitutions.
    pub fn pat_ty(&self, pat: &hir::Pat) -> Ty<'tcx> {
        self.node_id_to_type(pat.hir_id)
    }

    pub fn pat_ty_opt(&self, pat: &hir::Pat) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(pat.hir_id)
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
        self.node_id_to_type(expr.hir_id)
    }

    pub fn expr_ty_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(expr.hir_id)
    }

    pub fn adjustments(&self) -> LocalTableInContext<'_, Vec<ty::adjustment::Adjustment<'tcx>>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.adjustments
        }
    }

    pub fn adjustments_mut(&mut self)
                           -> LocalTableInContextMut<'_, Vec<ty::adjustment::Adjustment<'tcx>>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.adjustments
        }
    }

    pub fn expr_adjustments(&self, expr: &hir::Expr)
                            -> &[ty::adjustment::Adjustment<'tcx>] {
        validate_hir_id_for_typeck_tables(self.local_id_root, expr.hir_id, false);
        self.adjustments.get(&expr.hir_id.local_id).map_or(&[], |a| &a[..])
    }

    /// Returns the type of `expr`, considering any `Adjustment`
    /// entry recorded for that expression.
    pub fn expr_ty_adjusted(&self, expr: &hir::Expr) -> Ty<'tcx> {
        self.expr_adjustments(expr)
            .last()
            .map_or_else(|| self.expr_ty(expr), |adj| adj.target)
    }

    pub fn expr_ty_adjusted_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.expr_adjustments(expr)
            .last()
            .map(|adj| adj.target)
            .or_else(|| self.expr_ty_opt(expr))
    }

    pub fn is_method_call(&self, expr: &hir::Expr) -> bool {
        // Only paths and method calls/overloaded operators have
        // entries in type_dependent_defs, ignore the former here.
        if let hir::ExprKind::Path(_) = expr.node {
            return false;
        }

        match self.type_dependent_defs().get(expr.hir_id) {
            Some(&Def::Method(_)) => true,
            _ => false
        }
    }

    pub fn pat_binding_modes(&self) -> LocalTableInContext<'_, BindingMode> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.pat_binding_modes
        }
    }

    pub fn pat_binding_modes_mut(&mut self)
                           -> LocalTableInContextMut<'_, BindingMode> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.pat_binding_modes
        }
    }

    pub fn pat_adjustments(&self) -> LocalTableInContext<'_, Vec<Ty<'tcx>>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.pat_adjustments,
        }
    }

    pub fn pat_adjustments_mut(&mut self)
                               -> LocalTableInContextMut<'_, Vec<Ty<'tcx>>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.pat_adjustments,
        }
    }

    pub fn upvar_capture(&self, upvar_id: ty::UpvarId) -> ty::UpvarCapture<'tcx> {
        self.upvar_capture_map[&upvar_id]
    }

    pub fn closure_kind_origins(&self) -> LocalTableInContext<'_, (Span, ast::Name)> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.closure_kind_origins
        }
    }

    pub fn closure_kind_origins_mut(&mut self) -> LocalTableInContextMut<'_, (Span, ast::Name)> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.closure_kind_origins
        }
    }

    pub fn liberated_fn_sigs(&self) -> LocalTableInContext<'_, ty::FnSig<'tcx>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.liberated_fn_sigs
        }
    }

    pub fn liberated_fn_sigs_mut(&mut self) -> LocalTableInContextMut<'_, ty::FnSig<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.liberated_fn_sigs
        }
    }

    pub fn fru_field_types(&self) -> LocalTableInContext<'_, Vec<Ty<'tcx>>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.fru_field_types
        }
    }

    pub fn fru_field_types_mut(&mut self) -> LocalTableInContextMut<'_, Vec<Ty<'tcx>>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.fru_field_types
        }
    }

    pub fn cast_kinds(&self) -> LocalTableInContext<'_, ty::cast::CastKind> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.cast_kinds
        }
    }

    pub fn cast_kinds_mut(&mut self) -> LocalTableInContextMut<'_, ty::cast::CastKind> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.cast_kinds
        }
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for TypeckTables<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::TypeckTables {
            local_id_root,
            ref type_dependent_defs,
            ref field_indices,
            ref user_provided_tys,
            ref user_provided_sigs,
            ref node_types,
            ref node_substs,
            ref user_substs,
            ref adjustments,
            ref pat_binding_modes,
            ref pat_adjustments,
            ref upvar_capture_map,
            ref closure_kind_origins,
            ref liberated_fn_sigs,
            ref fru_field_types,

            ref cast_kinds,

            ref used_trait_imports,
            tainted_by_errors,
            ref free_region_map,
            ref concrete_existential_types,
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            type_dependent_defs.hash_stable(hcx, hasher);
            field_indices.hash_stable(hcx, hasher);
            user_provided_tys.hash_stable(hcx, hasher);
            user_provided_sigs.hash_stable(hcx, hasher);
            node_types.hash_stable(hcx, hasher);
            node_substs.hash_stable(hcx, hasher);
            user_substs.hash_stable(hcx, hasher);
            adjustments.hash_stable(hcx, hasher);
            pat_binding_modes.hash_stable(hcx, hasher);
            pat_adjustments.hash_stable(hcx, hasher);
            hash_stable_hashmap(hcx, hasher, upvar_capture_map, |up_var_id, hcx| {
                let ty::UpvarId {
                    var_path,
                    closure_expr_id
                } = *up_var_id;

                let local_id_root =
                    local_id_root.expect("trying to hash invalid TypeckTables");

                let var_owner_def_id = DefId {
                    krate: local_id_root.krate,
                    index: var_path.hir_id.owner,
                };
                let closure_def_id = DefId {
                    krate: local_id_root.krate,
                    index: closure_expr_id.to_def_id().index,
                };
                (hcx.def_path_hash(var_owner_def_id),
                 var_path.hir_id.local_id,
                 hcx.def_path_hash(closure_def_id))
            });

            closure_kind_origins.hash_stable(hcx, hasher);
            liberated_fn_sigs.hash_stable(hcx, hasher);
            fru_field_types.hash_stable(hcx, hasher);
            cast_kinds.hash_stable(hcx, hasher);
            used_trait_imports.hash_stable(hcx, hasher);
            tainted_by_errors.hash_stable(hcx, hasher);
            free_region_map.hash_stable(hcx, hasher);
            concrete_existential_types.hash_stable(hcx, hasher);
        })
    }
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonTypes<'tcx> {
        let mk = |sty| CtxtInterners::intern_ty(interners, interners, sty);
        let mk_region = |r| {
            if let Some(r) = interners.region.borrow().get(&r) {
                return r.0;
            }
            let r = interners.arena.alloc(r);
            interners.region.borrow_mut().insert(Interned(r));
            &*r
        };
        CommonTypes {
            bool: mk(Bool),
            char: mk(Char),
            never: mk(Never),
            err: mk(Error),
            isize: mk(Int(ast::IntTy::Isize)),
            i8: mk(Int(ast::IntTy::I8)),
            i16: mk(Int(ast::IntTy::I16)),
            i32: mk(Int(ast::IntTy::I32)),
            i64: mk(Int(ast::IntTy::I64)),
            i128: mk(Int(ast::IntTy::I128)),
            usize: mk(Uint(ast::UintTy::Usize)),
            u8: mk(Uint(ast::UintTy::U8)),
            u16: mk(Uint(ast::UintTy::U16)),
            u32: mk(Uint(ast::UintTy::U32)),
            u64: mk(Uint(ast::UintTy::U64)),
            u128: mk(Uint(ast::UintTy::U128)),
            f32: mk(Float(ast::FloatTy::F32)),
            f64: mk(Float(ast::FloatTy::F64)),

            re_empty: mk_region(RegionKind::ReEmpty),
            re_static: mk_region(RegionKind::ReStatic),
            re_erased: mk_region(RegionKind::ReErased),
        }
    }
}

// This struct contains information regarding the `ReFree(FreeRegion)` corresponding to a lifetime
// conflict.
#[derive(Debug)]
pub struct FreeRegionInfo {
    // def id corresponding to FreeRegion
    pub def_id: DefId,
    // the bound region corresponding to FreeRegion
    pub boundregion: ty::BoundRegion,
    // checks if bound region is in Impl Item
    pub is_impl_item: bool,
}

/// The central data structure of the compiler. It stores references
/// to the various **arenas** and also houses the results of the
/// various **compiler queries** that have been performed. See the
/// [rustc guide] for more details.
///
/// [rustc guide]: https://rust-lang-nursery.github.io/rustc-guide/ty.html
#[derive(Copy, Clone)]
pub struct TyCtxt<'a, 'gcx: 'tcx, 'tcx: 'a> {
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
    global_arenas: &'tcx WorkerLocal<GlobalArenas<'tcx>>,
    global_interners: CtxtInterners<'tcx>,

    cstore: &'tcx CrateStoreDyn,

    pub sess: &'tcx Session,

    pub dep_graph: DepGraph,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    /// Map indicating what traits are in scope for places where this
    /// is relevant; generated by resolve.
    trait_map: FxHashMap<DefIndex,
                         Lrc<FxHashMap<ItemLocalId,
                                       Lrc<StableVec<TraitCandidate>>>>>,

    /// Export map produced by name resolution.
    export_map: FxHashMap<DefId, Lrc<Vec<Export>>>,

    pub hir: hir_map::Map<'tcx>,

    /// A map from DefPathHash -> DefId. Includes DefIds from the local crate
    /// as well as all upstream crates. Only populated in incremental mode.
    pub def_path_hash_to_def_id: Option<FxHashMap<DefPathHash, DefId>>,

    pub(crate) queries: query::Queries<'tcx>,

    // Records the free variables referenced by every closure
    // expression. Do not track deps for this, just recompute it from
    // scratch every time.
    freevars: FxHashMap<DefId, Lrc<Vec<hir::Freevar>>>,

    maybe_unused_trait_imports: FxHashSet<DefId>,
    maybe_unused_extern_crates: Vec<(DefId, Span)>,
    /// Extern prelude entries. The value is `true` if the entry was introduced
    /// via `extern crate` item and not `--extern` option or compiler built-in.
    pub extern_prelude: FxHashMap<ast::Name, bool>,

    // Internal cache for metadata decoding. No need to track deps on this.
    pub rcache: Lock<FxHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that do not have to do with the parameters in scope.
    /// Merge this with `selection_cache`?
    pub evaluation_cache: traits::EvaluationCache<'tcx>,

    /// The definite name of the current crate after taking into account
    /// attributes, commandline parameters, etc.
    pub crate_name: Symbol,

    /// Data layout specification for the current target.
    pub data_layout: TargetDataLayout,

    stability_interner: Lock<FxHashSet<&'tcx attr::Stability>>,

    /// Stores the value of constants (and deduplicates the actual memory)
    allocation_interner: Lock<FxHashSet<&'tcx Allocation>>,

    pub alloc_map: Lock<interpret::AllocMap<'tcx, &'tcx Allocation>>,

    layout_interner: Lock<FxHashSet<&'tcx LayoutDetails>>,

    /// A general purpose channel to throw data out the back towards LLVM worker
    /// threads.
    ///
    /// This is intended to only get used during the codegen phase of the compiler
    /// when satisfying the query for a particular codegen unit. Internally in
    /// the query it'll send data along this channel to get processed later.
    pub tx_to_llvm_workers: Lock<mpsc::Sender<Box<dyn Any + Send>>>,

    output_filenames: Arc<OutputFilenames>,
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Get the global TyCtxt.
    #[inline]
    pub fn global_tcx(self) -> TyCtxt<'a, 'gcx, 'gcx> {
        TyCtxt {
            gcx: self.gcx,
            interners: &self.gcx.global_interners,
        }
    }

    pub fn alloc_generics(self, generics: ty::Generics) -> &'gcx ty::Generics {
        self.global_arenas.generics.alloc(generics)
    }

    pub fn alloc_steal_mir(self, mir: Mir<'gcx>) -> &'gcx Steal<Mir<'gcx>> {
        self.global_arenas.steal_mir.alloc(Steal::new(mir))
    }

    pub fn alloc_mir(self, mir: Mir<'gcx>) -> &'gcx Mir<'gcx> {
        self.global_arenas.mir.alloc(mir)
    }

    pub fn alloc_tables(self, tables: ty::TypeckTables<'gcx>) -> &'gcx ty::TypeckTables<'gcx> {
        self.global_arenas.tables.alloc(tables)
    }

    pub fn alloc_trait_def(self, def: ty::TraitDef) -> &'gcx ty::TraitDef {
        self.global_arenas.trait_def.alloc(def)
    }

    pub fn alloc_adt_def(self,
                         did: DefId,
                         kind: AdtKind,
                         variants: IndexVec<VariantIdx, ty::VariantDef>,
                         repr: ReprOptions)
                         -> &'gcx ty::AdtDef {
        let def = ty::AdtDef::new(self, did, kind, variants, repr);
        self.global_arenas.adt_def.alloc(def)
    }

    pub fn alloc_byte_array(self, bytes: &[u8]) -> &'gcx [u8] {
        if bytes.is_empty() {
            &[]
        } else {
            self.global_interners.arena.alloc_slice(bytes)
        }
    }

    pub fn alloc_const_slice(self, values: &[&'tcx ty::Const<'tcx>])
                             -> &'tcx [&'tcx ty::Const<'tcx>] {
        if values.is_empty() {
            &[]
        } else {
            self.interners.arena.alloc_slice(values)
        }
    }

    pub fn alloc_name_const_slice(self, values: &[(ast::Name, &'tcx ty::Const<'tcx>)])
                                  -> &'tcx [(ast::Name, &'tcx ty::Const<'tcx>)] {
        if values.is_empty() {
            &[]
        } else {
            self.interners.arena.alloc_slice(values)
        }
    }

    pub fn intern_const_alloc(
        self,
        alloc: Allocation,
    ) -> &'gcx Allocation {
        let allocs = &mut self.allocation_interner.borrow_mut();
        if let Some(alloc) = allocs.get(&alloc) {
            return alloc;
        }

        let interned = self.global_arenas.const_allocs.alloc(alloc);
        if let Some(prev) = allocs.replace(interned) { // insert into interner
            bug!("Tried to overwrite interned Allocation: {:#?}", prev)
        }
        interned
    }

    /// Allocates a byte or string literal for `mir::interpret`, read-only
    pub fn allocate_bytes(self, bytes: &[u8]) -> interpret::AllocId {
        // create an allocation that just contains these bytes
        let alloc = interpret::Allocation::from_byte_aligned_bytes(bytes);
        let alloc = self.intern_const_alloc(alloc);
        self.alloc_map.lock().allocate(alloc)
    }

    pub fn intern_stability(self, stab: attr::Stability) -> &'gcx attr::Stability {
        let mut stability_interner = self.stability_interner.borrow_mut();
        if let Some(st) = stability_interner.get(&stab) {
            return st;
        }

        let interned = self.global_interners.arena.alloc(stab);
        if let Some(prev) = stability_interner.replace(interned) {
            bug!("Tried to overwrite interned Stability: {:?}", prev)
        }
        interned
    }

    pub fn intern_layout(self, layout: LayoutDetails) -> &'gcx LayoutDetails {
        let mut layout_interner = self.layout_interner.borrow_mut();
        if let Some(layout) = layout_interner.get(&layout) {
            return layout;
        }

        let interned = self.global_arenas.layout.alloc(layout);
        if let Some(prev) = layout_interner.replace(interned) {
            bug!("Tried to overwrite interned Layout: {:?}", prev)
        }
        interned
    }

    /// Returns a range of the start/end indices specified with the
    /// `rustc_layout_scalar_valid_range` attribute.
    pub fn layout_scalar_valid_range(self, def_id: DefId) -> (Bound<u128>, Bound<u128>) {
        let attrs = self.get_attrs(def_id);
        let get = |name| {
            let attr = match attrs.iter().find(|a| a.check_name(name)) {
                Some(attr) => attr,
                None => return Bound::Unbounded,
            };
            for meta in attr.meta_item_list().expect("rustc_layout_scalar_valid_range takes args") {
                match meta.literal().expect("attribute takes lit").node {
                    ast::LitKind::Int(a, _) => return Bound::Included(a),
                    _ => span_bug!(attr.span, "rustc_layout_scalar_valid_range expects int arg"),
                }
            }
            span_bug!(attr.span, "no arguments to `rustc_layout_scalar_valid_range` attribute");
        };
        (get("rustc_layout_scalar_valid_range_start"), get("rustc_layout_scalar_valid_range_end"))
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
                                  cstore: &'tcx CrateStoreDyn,
                                  local_providers: ty::query::Providers<'tcx>,
                                  extern_providers: ty::query::Providers<'tcx>,
                                  arenas: &'tcx AllArenas<'tcx>,
                                  resolutions: ty::Resolutions,
                                  hir: hir_map::Map<'tcx>,
                                  on_disk_query_result_cache: query::OnDiskCache<'tcx>,
                                  crate_name: &str,
                                  tx: mpsc::Sender<Box<dyn Any + Send>>,
                                  output_filenames: &OutputFilenames,
                                  f: F) -> R
                                  where F: for<'b> FnOnce(TyCtxt<'b, 'tcx, 'tcx>) -> R
    {
        let data_layout = TargetDataLayout::parse(&s.target.target).unwrap_or_else(|err| {
            s.fatal(&err);
        });
        let interners = CtxtInterners::new(&arenas.interner);
        let common_types = CommonTypes::new(&interners);
        let dep_graph = hir.dep_graph.clone();
        let max_cnum = cstore.crates_untracked().iter().map(|c| c.as_usize()).max().unwrap_or(0);
        let mut providers = IndexVec::from_elem_n(extern_providers, max_cnum + 1);
        providers[LOCAL_CRATE] = local_providers;

        let def_path_hash_to_def_id = if s.opts.build_dep_graph() {
            let upstream_def_path_tables: Vec<(CrateNum, Lrc<_>)> = cstore
                .crates_untracked()
                .iter()
                .map(|&cnum| (cnum, cstore.def_path_table(cnum)))
                .collect();

            let def_path_tables = || {
                upstream_def_path_tables
                    .iter()
                    .map(|&(cnum, ref rc)| (cnum, &**rc))
                    .chain(iter::once((LOCAL_CRATE, hir.definitions().def_path_table())))
            };

            // Precompute the capacity of the hashmap so we don't have to
            // re-allocate when populating it.
            let capacity = def_path_tables().map(|(_, t)| t.size()).sum::<usize>();

            let mut map: FxHashMap<_, _> = FxHashMap::with_capacity_and_hasher(
                capacity,
                ::std::default::Default::default()
            );

            for (cnum, def_path_table) in def_path_tables() {
                def_path_table.add_def_path_hashes_to(cnum, &mut map);
            }

            Some(map)
        } else {
            None
        };

        let mut trait_map: FxHashMap<_, Lrc<FxHashMap<_, _>>> = FxHashMap::default();
        for (k, v) in resolutions.trait_map {
            let hir_id = hir.node_to_hir_id(k);
            let map = trait_map.entry(hir_id.owner).or_default();
            Lrc::get_mut(map).unwrap()
                             .insert(hir_id.local_id,
                                     Lrc::new(StableVec::new(v)));
        }

        let gcx = &GlobalCtxt {
            sess: s,
            cstore,
            global_arenas: &arenas.global,
            global_interners: interners,
            dep_graph,
            types: common_types,
            trait_map,
            export_map: resolutions.export_map.into_iter().map(|(k, v)| {
                (k, Lrc::new(v))
            }).collect(),
            freevars: resolutions.freevars.into_iter().map(|(k, v)| {
                (hir.local_def_id(k), Lrc::new(v))
            }).collect(),
            maybe_unused_trait_imports:
                resolutions.maybe_unused_trait_imports
                    .into_iter()
                    .map(|id| hir.local_def_id(id))
                    .collect(),
            maybe_unused_extern_crates:
                resolutions.maybe_unused_extern_crates
                    .into_iter()
                    .map(|(id, sp)| (hir.local_def_id(id), sp))
                    .collect(),
            extern_prelude: resolutions.extern_prelude,
            hir,
            def_path_hash_to_def_id,
            queries: query::Queries::new(
                providers,
                extern_providers,
                on_disk_query_result_cache,
            ),
            rcache: Default::default(),
            selection_cache: Default::default(),
            evaluation_cache: Default::default(),
            crate_name: Symbol::intern(crate_name),
            data_layout,
            layout_interner: Default::default(),
            stability_interner: Default::default(),
            allocation_interner: Default::default(),
            alloc_map: Lock::new(interpret::AllocMap::new()),
            tx_to_llvm_workers: Lock::new(tx),
            output_filenames: Arc::new(output_filenames.clone()),
        };

        sync::assert_send_val(&gcx);

        tls::enter_global(gcx, f)
    }

    pub fn consider_optimizing<T: Fn() -> String>(&self, msg: T) -> bool {
        let cname = self.crate_name(LOCAL_CRATE).as_str();
        self.sess.consider_optimizing(&cname, msg)
    }

    pub fn lib_features(self) -> Lrc<middle::lib_features::LibFeatures> {
        self.get_lib_features(LOCAL_CRATE)
    }

    pub fn lang_items(self) -> Lrc<middle::lang_items::LanguageItems> {
        self.get_lang_items(LOCAL_CRATE)
    }

    /// Due to missing llvm support for lowering 128 bit math to software emulation
    /// (on some targets), the lowering can be done in MIR.
    ///
    /// This function only exists until said support is implemented.
    pub fn is_binop_lang_item(&self, def_id: DefId) -> Option<(mir::BinOp, bool)> {
        let items = self.lang_items();
        let def_id = Some(def_id);
        if items.i128_add_fn() == def_id { Some((mir::BinOp::Add, false)) }
        else if items.u128_add_fn() == def_id { Some((mir::BinOp::Add, false)) }
        else if items.i128_sub_fn() == def_id { Some((mir::BinOp::Sub, false)) }
        else if items.u128_sub_fn() == def_id { Some((mir::BinOp::Sub, false)) }
        else if items.i128_mul_fn() == def_id { Some((mir::BinOp::Mul, false)) }
        else if items.u128_mul_fn() == def_id { Some((mir::BinOp::Mul, false)) }
        else if items.i128_div_fn() == def_id { Some((mir::BinOp::Div, false)) }
        else if items.u128_div_fn() == def_id { Some((mir::BinOp::Div, false)) }
        else if items.i128_rem_fn() == def_id { Some((mir::BinOp::Rem, false)) }
        else if items.u128_rem_fn() == def_id { Some((mir::BinOp::Rem, false)) }
        else if items.i128_shl_fn() == def_id { Some((mir::BinOp::Shl, false)) }
        else if items.u128_shl_fn() == def_id { Some((mir::BinOp::Shl, false)) }
        else if items.i128_shr_fn() == def_id { Some((mir::BinOp::Shr, false)) }
        else if items.u128_shr_fn() == def_id { Some((mir::BinOp::Shr, false)) }
        else if items.i128_addo_fn() == def_id { Some((mir::BinOp::Add, true)) }
        else if items.u128_addo_fn() == def_id { Some((mir::BinOp::Add, true)) }
        else if items.i128_subo_fn() == def_id { Some((mir::BinOp::Sub, true)) }
        else if items.u128_subo_fn() == def_id { Some((mir::BinOp::Sub, true)) }
        else if items.i128_mulo_fn() == def_id { Some((mir::BinOp::Mul, true)) }
        else if items.u128_mulo_fn() == def_id { Some((mir::BinOp::Mul, true)) }
        else if items.i128_shlo_fn() == def_id { Some((mir::BinOp::Shl, true)) }
        else if items.u128_shlo_fn() == def_id { Some((mir::BinOp::Shl, true)) }
        else if items.i128_shro_fn() == def_id { Some((mir::BinOp::Shr, true)) }
        else if items.u128_shro_fn() == def_id { Some((mir::BinOp::Shr, true)) }
        else { None }
    }

    pub fn stability(self) -> Lrc<stability::Index<'tcx>> {
        self.stability_index(LOCAL_CRATE)
    }

    pub fn crates(self) -> Lrc<Vec<CrateNum>> {
        self.all_crate_nums(LOCAL_CRATE)
    }

    pub fn features(self) -> Lrc<feature_gate::Features> {
        self.features_query(LOCAL_CRATE)
    }

    pub fn def_key(self, id: DefId) -> hir_map::DefKey {
        if id.is_local() {
            self.hir.def_key(id)
        } else {
            self.cstore.def_key(id)
        }
    }

    /// Convert a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned def-path).
    ///
    /// Note that if `id` is not local to this crate, the result will
    ///  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> hir_map::DefPath {
        if id.is_local() {
            self.hir.def_path(id)
        } else {
            self.cstore.def_path(id)
        }
    }

    #[inline]
    pub fn def_path_hash(self, def_id: DefId) -> hir_map::DefPathHash {
        if def_id.is_local() {
            self.hir.definitions().def_path_hash(def_id.index)
        } else {
            self.cstore.def_path_hash(def_id)
        }
    }

    pub fn def_path_debug_str(self, def_id: DefId) -> String {
        // We are explicitly not going through queries here in order to get
        // crate name and disambiguator since this code is called from debug!()
        // statements within the query system and we'd run into endless
        // recursion otherwise.
        let (crate_name, crate_disambiguator) = if def_id.is_local() {
            (self.crate_name.clone(),
             self.sess.local_crate_disambiguator())
        } else {
            (self.cstore.crate_name_untracked(def_id.krate),
             self.cstore.crate_disambiguator_untracked(def_id.krate))
        };

        format!("{}[{}]{}",
                crate_name,
                // Don't print the whole crate disambiguator. That's just
                // annoying in debug output.
                &(crate_disambiguator.to_fingerprint().to_hex())[..4],
                self.def_path(def_id).to_string_no_crate())
    }

    pub fn metadata_encoding_version(self) -> Vec<u8> {
        self.cstore.metadata_encoding_version().to_vec()
    }

    // Note that this is *untracked* and should only be used within the query
    // system if the result is otherwise tracked through queries
    pub fn crate_data_as_rc_any(self, cnum: CrateNum) -> Lrc<dyn Any> {
        self.cstore.crate_data_as_rc_any(cnum)
    }

    pub fn create_stable_hashing_context(self) -> StableHashingContext<'a> {
        let krate = self.dep_graph.with_ignore(|| self.gcx.hir.krate());

        StableHashingContext::new(self.sess,
                                  krate,
                                  self.hir.definitions(),
                                  self.cstore)
    }

    // This method makes sure that we have a DepNode and a Fingerprint for
    // every upstream crate. It needs to be called once right after the tcx is
    // created.
    // With full-fledged red/green, the method will probably become unnecessary
    // as this will be done on-demand.
    pub fn allocate_metadata_dep_nodes(self) {
        // We cannot use the query versions of crates() and crate_hash(), since
        // those would need the DepNodes that we are allocating here.
        for cnum in self.cstore.crates_untracked() {
            let dep_node = DepNode::new(self, DepConstructor::CrateMetadata(cnum));
            let crate_hash = self.cstore.crate_hash_untracked(cnum);
            self.dep_graph.with_task(dep_node,
                                     self,
                                     crate_hash,
                                     |_, x| x // No transformation needed
            );
        }
    }

    // This method exercises the `in_scope_traits_map` query for all possible
    // values so that we have their fingerprints available in the DepGraph.
    // This is only required as long as we still use the old dependency tracking
    // which needs to have the fingerprints of all input nodes beforehand.
    pub fn precompute_in_scope_traits_hashes(self) {
        for &def_index in self.trait_map.keys() {
            self.in_scope_traits_map(def_index);
        }
    }

    pub fn serialize_query_result_cache<E>(self,
                                           encoder: &mut E)
                                           -> Result<(), E::Error>
        where E: ty::codec::TyEncoder
    {
        self.queries.on_disk_cache.serialize(self.global_tcx(), encoder)
    }

    /// This checks whether one is allowed to have pattern bindings
    /// that bind-by-move on a match arm that has a guard, e.g.:
    ///
    /// ```rust
    /// match foo { A(inner) if { /* something */ } => ..., ... }
    /// ```
    ///
    /// It is separate from check_for_mutation_in_guard_via_ast_walk,
    /// because that method has a narrower effect that can be toggled
    /// off via a separate `-Z` flag, at least for the short term.
    pub fn allow_bind_by_move_patterns_with_guards(self) -> bool {
        self.features().bind_by_move_pattern_guards && self.use_mir_borrowck()
    }

    /// If true, we should use a naive AST walk to determine if match
    /// guard could perform bad mutations (or mutable-borrows).
    pub fn check_for_mutation_in_guard_via_ast_walk(self) -> bool {
        // If someone requests the feature, then be a little more
        // careful and ensure that MIR-borrowck is enabled (which can
        // happen via edition selection, via `feature(nll)`, or via an
        // appropriate `-Z` flag) before disabling the mutation check.
        if self.allow_bind_by_move_patterns_with_guards() {
            return false;
        }

        return true;
    }

    /// If true, we should use the AST-based borrowck (we may *also* use
    /// the MIR-based borrowck).
    pub fn use_ast_borrowck(self) -> bool {
        self.borrowck_mode().use_ast()
    }

    /// If true, we should use the MIR-based borrowck (we may *also* use
    /// the AST-based borrowck).
    pub fn use_mir_borrowck(self) -> bool {
        self.borrowck_mode().use_mir()
    }

    /// If true, we should use the MIR-based borrow check, but also
    /// fall back on the AST borrow check if the MIR-based one errors.
    pub fn migrate_borrowck(self) -> bool {
        self.borrowck_mode().migrate()
    }

    /// If true, make MIR codegen for `match` emit a temp that holds a
    /// borrow of the input to the match expression.
    pub fn generate_borrow_of_any_match_input(&self) -> bool {
        self.emit_read_for_match()
    }

    /// If true, make MIR codegen for `match` emit FakeRead
    /// statements (which simulate the maximal effect of executing the
    /// patterns in a match arm).
    pub fn emit_read_for_match(&self) -> bool {
        self.use_mir_borrowck() && !self.sess.opts.debugging_opts.nll_dont_emit_read_for_match
    }

    /// If true, pattern variables for use in guards on match arms
    /// will be bound as references to the data, and occurrences of
    /// those variables in the guard expression will implicitly
    /// dereference those bindings. (See rust-lang/rust#27282.)
    pub fn all_pat_vars_are_implicit_refs_within_guards(self) -> bool {
        self.borrowck_mode().use_mir()
    }

    /// If true, we should enable two-phase borrows checks. This is
    /// done with either: `-Ztwo-phase-borrows`, `#![feature(nll)]`,
    /// or by opting into an edition after 2015.
    pub fn two_phase_borrows(self) -> bool {
        if self.features().nll || self.sess.opts.debugging_opts.two_phase_borrows {
            return true;
        }

        match self.sess.edition() {
            Edition::Edition2015 => false,
            Edition::Edition2018 => true,
            _ => true,
        }
    }

    /// What mode(s) of borrowck should we run? AST? MIR? both?
    /// (Also considers the `#![feature(nll)]` setting.)
    pub fn borrowck_mode(&self) -> BorrowckMode {
        // Here are the main constraints we need to deal with:
        //
        // 1. An opts.borrowck_mode of `BorrowckMode::Ast` is
        //    synonymous with no `-Z borrowck=...` flag at all.
        //    (This is arguably a historical accident.)
        //
        // 2. `BorrowckMode::Migrate` is the limited migration to
        //    NLL that we are deploying with the 2018 edition.
        //
        // 3. We want to allow developers on the Nightly channel
        //    to opt back into the "hard error" mode for NLL,
        //    (which they can do via specifying `#![feature(nll)]`
        //    explicitly in their crate).
        //
        // So, this precedence list is how pnkfelix chose to work with
        // the above constraints:
        //
        // * `#![feature(nll)]` *always* means use NLL with hard
        //   errors. (To simplify the code here, it now even overrides
        //   a user's attempt to specify `-Z borrowck=compare`, which
        //   we arguably do not need anymore and should remove.)
        //
        // * Otherwise, if no `-Z borrowck=...` flag was given (or
        //   if `borrowck=ast` was specified), then use the default
        //   as required by the edition.
        //
        // * Otherwise, use the behavior requested via `-Z borrowck=...`

        if self.features().nll { return BorrowckMode::Mir; }

        match self.sess.opts.borrowck_mode {
            mode @ BorrowckMode::Mir |
            mode @ BorrowckMode::Compare |
            mode @ BorrowckMode::Migrate => mode,

            BorrowckMode::Ast => match self.sess.edition() {
                Edition::Edition2015 => BorrowckMode::Ast,
                Edition::Edition2018 => BorrowckMode::Migrate,

                // For now, future editions mean Migrate. (But it
                // would make a lot of sense for it to be changed to
                // `BorrowckMode::Mir`, depending on how we plan to
                // time the forcing of full migration to NLL.)
                _ => BorrowckMode::Migrate,
            },
        }
    }

    #[inline]
    pub fn local_crate_exports_generics(self) -> bool {
        debug_assert!(self.sess.opts.share_generics());

        self.sess.crate_types.borrow().iter().any(|crate_type| {
            match crate_type {
                CrateType::Executable |
                CrateType::Staticlib  |
                CrateType::ProcMacro  |
                CrateType::Cdylib     => false,
                CrateType::Rlib       |
                CrateType::Dylib      => true,
            }
        })
    }

    // This method returns the DefId and the BoundRegion corresponding to the given region.
    pub fn is_suitable_region(&self, region: Region<'tcx>) -> Option<FreeRegionInfo> {
        let (suitable_region_binding_scope, bound_region) = match *region {
            ty::ReFree(ref free_region) => (free_region.scope, free_region.bound_region),
            ty::ReEarlyBound(ref ebr) => (
                self.parent_def_id(ebr.def_id).unwrap(),
                ty::BoundRegion::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let node_id = self.hir
            .as_local_node_id(suitable_region_binding_scope)
            .unwrap();
        let is_impl_item = match self.hir.find(node_id) {
            Some(Node::Item(..)) | Some(Node::TraitItem(..)) => false,
            Some(Node::ImplItem(..)) => {
                self.is_bound_region_in_impl_item(suitable_region_binding_scope)
            }
            _ => return None,
        };

        return Some(FreeRegionInfo {
            def_id: suitable_region_binding_scope,
            boundregion: bound_region,
            is_impl_item: is_impl_item,
        });
    }

    pub fn return_type_impl_trait(
        &self,
        scope_def_id: DefId,
    ) -> Option<Ty<'tcx>> {
        // HACK: `type_of_def_id()` will fail on these (#55796), so return None
        let node_id = self.hir.as_local_node_id(scope_def_id).unwrap();
        match self.hir.get(node_id) {
            Node::Item(item) => {
                match item.node {
                    ItemKind::Fn(..) => { /* type_of_def_id() will work */ }
                    _ => {
                        return None;
                    }
                }
            }
            _ => { /* type_of_def_id() will work or panic */ }
        }

        let ret_ty = self.type_of(scope_def_id);
        match ret_ty.sty {
            ty::FnDef(_, _) => {
                let sig = ret_ty.fn_sig(*self);
                let output = self.erase_late_bound_regions(&sig.output());
                if output.is_impl_trait() {
                    Some(output)
                } else {
                    None
                }
            }
            _ => None
        }
    }

    // Here we check if the bound region is in Impl Item.
    pub fn is_bound_region_in_impl_item(
        &self,
        suitable_region_binding_scope: DefId,
    ) -> bool {
        let container_id = self.associated_item(suitable_region_binding_scope)
            .container
            .id();
        if self.impl_trait_ref(container_id).is_some() {
            // For now, we do not try to target impls of traits. This is
            // because this message is going to suggest that the user
            // change the fn signature, but they may not be free to do so,
            // since the signature must match the trait.
            //
            // FIXME(#42706) -- in some cases, we could do better here.
            return true;
        }
        false
    }
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    pub fn encode_metadata(self)
        -> EncodedMetadata
    {
        self.cstore.encode_metadata(self)
    }
}

impl<'gcx: 'tcx, 'tcx> GlobalCtxt<'gcx> {
    /// Call the closure with a local `TyCtxt` using the given arena.
    pub fn enter_local<F, R>(
        &self,
        arena: &'tcx SyncDroplessArena,
        f: F
    ) -> R
    where
        F: for<'a> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        let interners = CtxtInterners::new(arena);
        let tcx = TyCtxt {
            gcx: self,
            interners: &interners,
        };
        ty::tls::with_related_context(tcx.global_tcx(), |icx| {
            let new_icx = ty::tls::ImplicitCtxt {
                tcx,
                query: icx.query.clone(),
                layout_depth: icx.layout_depth,
                task: icx.task,
            };
            ty::tls::enter_context(&new_icx, |new_icx| {
                f(new_icx.tcx)
            })
        })
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
/// contain the TyKind key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g. `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx>: fmt::Debug {
    type Lifted: fmt::Debug + 'tcx;
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

impl<'a, 'tcx> Lift<'tcx> for Region<'a> {
    type Lifted = Region<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Region<'tcx>> {
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

impl<'a, 'tcx> Lift<'tcx> for Goal<'a> {
    type Lifted = Goal<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Goal<'tcx>> {
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<Goal<'a>> {
    type Lifted = &'tcx List<Goal<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(
        &self,
        tcx: TyCtxt<'b, 'gcx, 'tcx>,
    ) -> Option<&'tcx List<Goal<'tcx>>> {
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<Clause<'a>> {
    type Lifted = &'tcx List<Clause<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(
        &self,
        tcx: TyCtxt<'b, 'gcx, 'tcx>,
    ) -> Option<&'tcx List<Clause<'tcx>>> {
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

impl<'a, 'tcx> Lift<'tcx> for &'a Const<'a> {
    type Lifted = &'tcx Const<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<&'tcx Const<'tcx>> {
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
            return Some(List::empty());
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<Ty<'a>> {
    type Lifted = &'tcx List<Ty<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<&'tcx List<Ty<'tcx>>> {
        if self.len() == 0 {
            return Some(List::empty());
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<ExistentialPredicate<'a>> {
    type Lifted = &'tcx List<ExistentialPredicate<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
        -> Option<&'tcx List<ExistentialPredicate<'tcx>>> {
        if self.is_empty() {
            return Some(List::empty());
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<Predicate<'a>> {
    type Lifted = &'tcx List<Predicate<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
        -> Option<&'tcx List<Predicate<'tcx>>> {
        if self.is_empty() {
            return Some(List::empty());
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<CanonicalVarInfo> {
    type Lifted = &'tcx List<CanonicalVarInfo>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        if self.len() == 0 {
            return Some(List::empty());
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

impl<'a, 'tcx> Lift<'tcx> for &'a List<ProjectionKind<'a>> {
    type Lifted = &'tcx List<ProjectionKind<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        if self.len() == 0 {
            return Some(List::empty());
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

pub mod tls {
    use super::{GlobalCtxt, TyCtxt};

    use std::fmt;
    use std::mem;
    use syntax_pos;
    use ty::query;
    use errors::{Diagnostic, TRACK_DIAGNOSTICS};
    use rustc_data_structures::OnDrop;
    use rustc_data_structures::sync::{self, Lrc, Lock};
    use dep_graph::OpenTask;

    #[cfg(not(parallel_queries))]
    use std::cell::Cell;

    #[cfg(parallel_queries)]
    use rayon_core;

    /// This is the implicit state of rustc. It contains the current
    /// TyCtxt and query. It is updated when creating a local interner or
    /// executing a new query. Whenever there's a TyCtxt value available
    /// you should also have access to an ImplicitCtxt through the functions
    /// in this module.
    #[derive(Clone)]
    pub struct ImplicitCtxt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
        /// The current TyCtxt. Initially created by `enter_global` and updated
        /// by `enter_local` with a new local interner
        pub tcx: TyCtxt<'a, 'gcx, 'tcx>,

        /// The current query job, if any. This is updated by start_job in
        /// ty::query::plumbing when executing a query
        pub query: Option<Lrc<query::QueryJob<'gcx>>>,

        /// Used to prevent layout from recursing too deeply.
        pub layout_depth: usize,

        /// The current dep graph task. This is used to add dependencies to queries
        /// when executing them
        pub task: &'a OpenTask,
    }

    /// Sets Rayon's thread local variable which is preserved for Rayon jobs
    /// to `value` during the call to `f`. It is restored to its previous value after.
    /// This is used to set the pointer to the new ImplicitCtxt.
    #[cfg(parallel_queries)]
    fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
        rayon_core::tlv::with(value, f)
    }

    /// Gets Rayon's thread local variable which is preserved for Rayon jobs.
    /// This is used to get the pointer to the current ImplicitCtxt.
    #[cfg(parallel_queries)]
    fn get_tlv() -> usize {
        rayon_core::tlv::get()
    }

    /// A thread local variable which stores a pointer to the current ImplicitCtxt
    #[cfg(not(parallel_queries))]
    thread_local!(static TLV: Cell<usize> = Cell::new(0));

    /// Sets TLV to `value` during the call to `f`.
    /// It is restored to its previous value after.
    /// This is used to set the pointer to the new ImplicitCtxt.
    #[cfg(not(parallel_queries))]
    fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
        let old = get_tlv();
        let _reset = OnDrop(move || TLV.with(|tlv| tlv.set(old)));
        TLV.with(|tlv| tlv.set(value));
        f()
    }

    /// This is used to get the pointer to the current ImplicitCtxt.
    #[cfg(not(parallel_queries))]
    fn get_tlv() -> usize {
        TLV.with(|tlv| tlv.get())
    }

    /// This is a callback from libsyntax as it cannot access the implicit state
    /// in librustc otherwise
    fn span_debug(span: syntax_pos::Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with(|tcx| {
            write!(f, "{}", tcx.sess.source_map().span_to_string(span))
        })
    }

    /// This is a callback from libsyntax as it cannot access the implicit state
    /// in librustc otherwise. It is used to when diagnostic messages are
    /// emitted and stores them in the current query, if there is one.
    fn track_diagnostic(diagnostic: &Diagnostic) {
        with_context_opt(|icx| {
            if let Some(icx) = icx {
                if let Some(ref query) = icx.query {
                    query.diagnostics.lock().push(diagnostic.clone());
                }
            }
        })
    }

    /// Sets up the callbacks from libsyntax on the current thread
    pub fn with_thread_locals<F, R>(f: F) -> R
        where F: FnOnce() -> R
    {
        syntax_pos::SPAN_DEBUG.with(|span_dbg| {
            let original_span_debug = span_dbg.get();
            span_dbg.set(span_debug);

            let _on_drop = OnDrop(move || {
                span_dbg.set(original_span_debug);
            });

            TRACK_DIAGNOSTICS.with(|current| {
                let original = current.get();
                current.set(track_diagnostic);

                let _on_drop = OnDrop(move || {
                    current.set(original);
                });

                f()
            })
        })
    }

    /// Sets `context` as the new current ImplicitCtxt for the duration of the function `f`
    pub fn enter_context<'a, 'gcx: 'tcx, 'tcx, F, R>(context: &ImplicitCtxt<'a, 'gcx, 'tcx>,
                                                     f: F) -> R
        where F: FnOnce(&ImplicitCtxt<'a, 'gcx, 'tcx>) -> R
    {
        set_tlv(context as *const _ as usize, || {
            f(&context)
        })
    }

    /// Enters GlobalCtxt by setting up libsyntax callbacks and
    /// creating a initial TyCtxt and ImplicitCtxt.
    /// This happens once per rustc session and TyCtxts only exists
    /// inside the `f` function.
    pub fn enter_global<'gcx, F, R>(gcx: &GlobalCtxt<'gcx>, f: F) -> R
        where F: for<'a> FnOnce(TyCtxt<'a, 'gcx, 'gcx>) -> R
    {
        with_thread_locals(|| {
            // Update GCX_PTR to indicate there's a GlobalCtxt available
            GCX_PTR.with(|lock| {
                *lock.lock() = gcx as *const _ as usize;
            });
            // Set GCX_PTR back to 0 when we exit
            let _on_drop = OnDrop(move || {
                GCX_PTR.with(|lock| *lock.lock() = 0);
            });

            let tcx = TyCtxt {
                gcx,
                interners: &gcx.global_interners,
            };
            let icx = ImplicitCtxt {
                tcx,
                query: None,
                layout_depth: 0,
                task: &OpenTask::Ignore,
            };
            enter_context(&icx, |_| {
                f(tcx)
            })
        })
    }

    /// Stores a pointer to the GlobalCtxt if one is available.
    /// This is used to access the GlobalCtxt in the deadlock handler
    /// given to Rayon.
    scoped_thread_local!(pub static GCX_PTR: Lock<usize>);

    /// Creates a TyCtxt and ImplicitCtxt based on the GCX_PTR thread local.
    /// This is used in the deadlock handler.
    pub unsafe fn with_global<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        let gcx = GCX_PTR.with(|lock| *lock.lock());
        assert!(gcx != 0);
        let gcx = &*(gcx as *const GlobalCtxt<'_>);
        let tcx = TyCtxt {
            gcx,
            interners: &gcx.global_interners,
        };
        let icx = ImplicitCtxt {
            query: None,
            tcx,
            layout_depth: 0,
            task: &OpenTask::Ignore,
        };
        enter_context(&icx, |_| f(tcx))
    }

    /// Allows access to the current ImplicitCtxt in a closure if one is available
    pub fn with_context_opt<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(Option<&ImplicitCtxt<'a, 'gcx, 'tcx>>) -> R
    {
        let context = get_tlv();
        if context == 0 {
            f(None)
        } else {
            // We could get a ImplicitCtxt pointer from another thread.
            // Ensure that ImplicitCtxt is Sync
            sync::assert_sync::<ImplicitCtxt<'_, '_, '_>>();

            unsafe { f(Some(&*(context as *const ImplicitCtxt<'_, '_, '_>))) }
        }
    }

    /// Allows access to the current ImplicitCtxt.
    /// Panics if there is no ImplicitCtxt available
    pub fn with_context<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(&ImplicitCtxt<'a, 'gcx, 'tcx>) -> R
    {
        with_context_opt(|opt_context| f(opt_context.expect("no ImplicitCtxt stored in tls")))
    }

    /// Allows access to the current ImplicitCtxt whose tcx field has the same global
    /// interner as the tcx argument passed in. This means the closure is given an ImplicitCtxt
    /// with the same 'gcx lifetime as the TyCtxt passed in.
    /// This will panic if you pass it a TyCtxt which has a different global interner from
    /// the current ImplicitCtxt's tcx field.
    pub fn with_related_context<'a, 'gcx, 'tcx1, F, R>(tcx: TyCtxt<'a, 'gcx, 'tcx1>, f: F) -> R
        where F: for<'b, 'tcx2> FnOnce(&ImplicitCtxt<'b, 'gcx, 'tcx2>) -> R
    {
        with_context(|context| {
            unsafe {
                let gcx = tcx.gcx as *const _ as usize;
                assert!(context.tcx.gcx as *const _ as usize == gcx);
                let context: &ImplicitCtxt<'_, '_, '_> = mem::transmute(context);
                f(context)
            }
        })
    }

    /// Allows access to the current ImplicitCtxt whose tcx field has the same global
    /// interner and local interner as the tcx argument passed in. This means the closure
    /// is given an ImplicitCtxt with the same 'tcx and 'gcx lifetimes as the TyCtxt passed in.
    /// This will panic if you pass it a TyCtxt which has a different global interner or
    /// a different local interner from the current ImplicitCtxt's tcx field.
    pub fn with_fully_related_context<'a, 'gcx, 'tcx, F, R>(tcx: TyCtxt<'a, 'gcx, 'tcx>, f: F) -> R
        where F: for<'b> FnOnce(&ImplicitCtxt<'b, 'gcx, 'tcx>) -> R
    {
        with_context(|context| {
            unsafe {
                let gcx = tcx.gcx as *const _ as usize;
                let interners = tcx.interners as *const _ as usize;
                assert!(context.tcx.gcx as *const _ as usize == gcx);
                assert!(context.tcx.interners as *const _ as usize == interners);
                let context: &ImplicitCtxt<'_, '_, '_> = mem::transmute(context);
                f(context)
            }
        })
    }

    /// Allows access to the TyCtxt in the current ImplicitCtxt.
    /// Panics if there is no ImplicitCtxt available
    pub fn with<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        with_context(|context| f(context.tcx))
    }

    /// Allows access to the TyCtxt in the current ImplicitCtxt.
    /// The closure is passed None if there is no ImplicitCtxt available
    pub fn with_opt<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(Option<TyCtxt<'a, 'gcx, 'tcx>>) -> R
    {
        with_context_opt(|opt_context| f(opt_context.map(|context| context.tcx)))
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

            pub fn go(tcx: TyCtxt<'_, '_, '_>) {
                let mut total = DebugStat {
                    total: 0,
                    region_infer: 0, ty_infer: 0, both_infer: 0,
                };
                $(let mut $variant = total;)*

                for &Interned(t) in tcx.interners.type_.borrow().iter() {
                    let variant = match t.sty {
                        ty::Bool | ty::Char | ty::Int(..) | ty::Uint(..) |
                            ty::Float(..) | ty::Str | ty::Never => continue,
                        ty::Error => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let region = t.flags.intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags.intersects(ty::TypeFlags::HAS_TY_INFER);

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
            Adt, Array, Slice, RawPtr, Ref, FnDef, FnPtr,
            Generator, GeneratorWitness, Dynamic, Closure, Tuple, Bound,
            Param, Infer, UnnormalizedProjection, Projection, Opaque, Foreign);

        println!("Substs interner: #{}", self.interners.substs.borrow().len());
        println!("Region interner: #{}", self.interners.region.borrow().len());
        println!("Stability interner: #{}", self.stability_interner.borrow().len());
        println!("Allocation interner: #{}", self.allocation_interner.borrow().len());
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

impl<'tcx: 'lcx, 'lcx> Borrow<TyKind<'lcx>> for Interned<'tcx, TyS<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a TyKind<'lcx> {
        &self.0.sty
    }
}

// NB: An Interned<List<T>> compares and hashes as its elements.
impl<'tcx, T: PartialEq> PartialEq for Interned<'tcx, List<T>> {
    fn eq(&self, other: &Interned<'tcx, List<T>>) -> bool {
        self.0[..] == other.0[..]
    }
}

impl<'tcx, T: Eq> Eq for Interned<'tcx, List<T>> {}

impl<'tcx, T: Hash> Hash for Interned<'tcx, List<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0[..].hash(s)
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Ty<'lcx>]> for Interned<'tcx, List<Ty<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Ty<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[CanonicalVarInfo]> for Interned<'tcx, List<CanonicalVarInfo>> {
    fn borrow<'a>(&'a self) -> &'a [CanonicalVarInfo] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Kind<'lcx>]> for Interned<'tcx, Substs<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a [Kind<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[ProjectionKind<'lcx>]>
    for Interned<'tcx, List<ProjectionKind<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [ProjectionKind<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<RegionKind> for Interned<'tcx, RegionKind> {
    fn borrow<'a>(&'a self) -> &'a RegionKind {
        &self.0
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<GoalKind<'lcx>> for Interned<'tcx, GoalKind<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a GoalKind<'lcx> {
        &self.0
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[ExistentialPredicate<'lcx>]>
    for Interned<'tcx, List<ExistentialPredicate<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [ExistentialPredicate<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Predicate<'lcx>]>
    for Interned<'tcx, List<Predicate<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Predicate<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<Const<'lcx>> for Interned<'tcx, Const<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a Const<'lcx> {
        &self.0
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Clause<'lcx>]>
for Interned<'tcx, List<Clause<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Clause<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Goal<'lcx>]>
for Interned<'tcx, List<Goal<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Goal<'lcx>] {
        &self.0[..]
    }
}

macro_rules! intern_method {
    ($lt_tcx:tt, $name:ident: $method:ident($alloc:ty,
                                            $alloc_method:expr,
                                            $alloc_to_key:expr,
                                            $keep_in_local_tcx:expr) -> $ty:ty) => {
        impl<'a, 'gcx, $lt_tcx> TyCtxt<'a, 'gcx, $lt_tcx> {
            pub fn $method(self, v: $alloc) -> &$lt_tcx $ty {
                let key = ($alloc_to_key)(&v);

                // HACK(eddyb) Depend on flags being accurate to
                // determine that all contents are in the global tcx.
                // See comments on Lift for why we can't use that.
                if ($keep_in_local_tcx)(&v) {
                    let mut interner = self.interners.$name.borrow_mut();
                    if let Some(&Interned(v)) = interner.get(key) {
                        return v;
                    }

                    // Make sure we don't end up with inference
                    // types/regions in the global tcx.
                    if self.is_global() {
                        bug!("Attempted to intern `{:?}` which contains \
                              inference types/regions in the global type context",
                             v);
                    }

                    let i = $alloc_method(&self.interners.arena, v);
                    interner.insert(Interned(i));
                    i
                } else {
                    let mut interner = self.global_interners.$name.borrow_mut();
                    if let Some(&Interned(v)) = interner.get(key) {
                        return v;
                    }

                    // This transmutes $alloc<'tcx> to $alloc<'gcx>
                    let v = unsafe {
                        mem::transmute(v)
                    };
                    let i: &$lt_tcx $ty = $alloc_method(&self.global_interners.arena, v);
                    // Cast to 'gcx
                    let i = unsafe { mem::transmute(i) };
                    interner.insert(Interned(i));
                    i
                }
            }
        }
    }
}

macro_rules! direct_interners {
    ($lt_tcx:tt, $($name:ident: $method:ident($keep_in_local_tcx:expr) -> $ty:ty),+) => {
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

        intern_method!(
            $lt_tcx,
            $name: $method($ty,
                           |a: &$lt_tcx SyncDroplessArena, v| -> &$lt_tcx $ty { a.alloc(v) },
                           |x| x,
                           $keep_in_local_tcx) -> $ty);)+
    }
}

pub fn keep_local<'tcx, T: ty::TypeFoldable<'tcx>>(x: &T) -> bool {
    x.has_type_flags(ty::TypeFlags::KEEP_IN_LOCAL_TCX)
}

direct_interners!('tcx,
    region: mk_region(|r: &RegionKind| r.keep_in_local_tcx()) -> RegionKind,
    const_: mk_const(|c: &Const<'_>| keep_local(&c.ty) || keep_local(&c.val)) -> Const<'tcx>,
    goal: mk_goal(|c: &GoalKind<'_>| keep_local(c)) -> GoalKind<'tcx>
);

macro_rules! slice_interners {
    ($($field:ident: $method:ident($ty:ident)),+) => (
        $(intern_method!( 'tcx, $field: $method(
            &[$ty<'tcx>],
            |a, v| List::from_arena(a, v),
            Deref::deref,
            |xs: &[$ty<'_>]| xs.iter().any(keep_local)) -> List<$ty<'tcx>>);)+
    )
}

slice_interners!(
    existential_predicates: _intern_existential_predicates(ExistentialPredicate),
    predicates: _intern_predicates(Predicate),
    type_list: _intern_type_list(Ty),
    substs: _intern_substs(Kind),
    clauses: _intern_clauses(Clause),
    goal_list: _intern_goals(Goal),
    projs: _intern_projs(ProjectionKind)
);

// This isn't a perfect fit: CanonicalVarInfo slices are always
// allocated in the global arena, so this `intern_method!` macro is
// overly general.  But we just return false for the code that checks
// whether they belong in the thread-local arena, so no harm done, and
// seems better than open-coding the rest.
intern_method! {
    'tcx,
    canonical_var_infos: _intern_canonical_var_infos(
        &[CanonicalVarInfo],
        |a, v| List::from_arena(a, v),
        Deref::deref,
        |_xs: &[CanonicalVarInfo]| -> bool { false }
    ) -> List<CanonicalVarInfo>
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Given a `fn` type, returns an equivalent `unsafe fn` type;
    /// that is, a `fn` type that is equivalent in every way for being
    /// unsafe.
    pub fn safe_to_unsafe_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        assert_eq!(sig.unsafety(), hir::Unsafety::Normal);
        self.mk_fn_ptr(sig.map_bound(|sig| ty::FnSig {
            unsafety: hir::Unsafety::Unsafe,
            ..sig
        }))
    }

    /// Given a closure signature `sig`, returns an equivalent `fn`
    /// type with the same signature. Detuples and so forth -- so
    /// e.g. if we have a sig with `Fn<(u32, i32)>` then you would get
    /// a `fn(u32, i32)`.
    pub fn coerce_closure_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        let converted_sig = sig.map_bound(|s| {
            let params_iter = match s.inputs()[0].sty {
                ty::Tuple(params) => {
                    params.into_iter().cloned()
                }
                _ => bug!(),
            };
            self.mk_fn_sig(
                params_iter,
                s.output(),
                s.variadic,
                hir::Unsafety::Normal,
                abi::Abi::Rust,
            )
        });

        self.mk_fn_ptr(converted_sig)
    }

    pub fn mk_ty(&self, st: TyKind<'tcx>) -> Ty<'tcx> {
        CtxtInterners::intern_ty(&self.interners, &self.global_interners, st)
    }

    pub fn mk_mach_int(self, tm: ast::IntTy) -> Ty<'tcx> {
        match tm {
            ast::IntTy::Isize   => self.types.isize,
            ast::IntTy::I8   => self.types.i8,
            ast::IntTy::I16  => self.types.i16,
            ast::IntTy::I32  => self.types.i32,
            ast::IntTy::I64  => self.types.i64,
            ast::IntTy::I128  => self.types.i128,
        }
    }

    pub fn mk_mach_uint(self, tm: ast::UintTy) -> Ty<'tcx> {
        match tm {
            ast::UintTy::Usize   => self.types.usize,
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
        self.mk_ty(Str)
    }

    pub fn mk_static_str(self) -> Ty<'tcx> {
        self.mk_imm_ref(self.types.re_static, self.mk_str())
    }

    pub fn mk_adt(self, def: &'tcx AdtDef, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(Adt(def, substs))
    }

    pub fn mk_foreign(self, def_id: DefId) -> Ty<'tcx> {
        self.mk_ty(Foreign(def_id))
    }

    pub fn mk_box(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(lang_items::OwnedBoxLangItem);
        let adt_def = self.adt_def(def_id);
        let substs = Substs::for_item(self, def_id, |param, substs| {
            match param.kind {
                GenericParamDefKind::Lifetime => bug!(),
                GenericParamDefKind::Type { has_default, .. } => {
                    if param.index == 0 {
                        ty.into()
                    } else {
                        assert!(has_default);
                        self.type_of(param.def_id).subst(self, substs).into()
                    }
                }
            }
        });
        self.mk_ty(Adt(adt_def, substs))
    }

    pub fn mk_ptr(self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(RawPtr(tm))
    }

    pub fn mk_ref(self, r: Region<'tcx>, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Ref(r, tm.ty, tm.mutbl))
    }

    pub fn mk_mut_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_nil_ptr(self) -> Ty<'tcx> {
        self.mk_imm_ptr(self.mk_unit())
    }

    pub fn mk_array(self, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        self.mk_ty(Array(ty, ty::Const::from_usize(self, n)))
    }

    pub fn mk_slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Slice(ty))
    }

    pub fn intern_tup(self, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        self.mk_ty(Tuple(self.intern_type_list(ts)))
    }

    pub fn mk_tup<I: InternAs<[Ty<'tcx>], Ty<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|ts| self.mk_ty(Tuple(self.intern_type_list(ts))))
    }

    pub fn mk_unit(self) -> Ty<'tcx> {
        self.intern_tup(&[])
    }

    pub fn mk_diverging_default(self) -> Ty<'tcx> {
        if self.features().never_type {
            self.types.never
        } else {
            self.intern_tup(&[])
        }
    }

    pub fn mk_bool(self) -> Ty<'tcx> {
        self.mk_ty(Bool)
    }

    pub fn mk_fn_def(self, def_id: DefId,
                     substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnDef(def_id, substs))
    }

    pub fn mk_fn_ptr(self, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnPtr(fty))
    }

    pub fn mk_dynamic(
        self,
        obj: ty::Binder<&'tcx List<ExistentialPredicate<'tcx>>>,
        reg: ty::Region<'tcx>
    ) -> Ty<'tcx> {
        self.mk_ty(Dynamic(obj, reg))
    }

    pub fn mk_projection(self,
                         item_def_id: DefId,
                         substs: &'tcx Substs<'tcx>)
        -> Ty<'tcx> {
            self.mk_ty(Projection(ProjectionTy {
                item_def_id,
                substs,
            }))
        }

    pub fn mk_closure(self, closure_id: DefId, closure_substs: ClosureSubsts<'tcx>)
                      -> Ty<'tcx> {
        self.mk_ty(Closure(closure_id, closure_substs))
    }

    pub fn mk_generator(self,
                        id: DefId,
                        generator_substs: GeneratorSubsts<'tcx>,
                        movability: hir::GeneratorMovability)
                        -> Ty<'tcx> {
        self.mk_ty(Generator(id, generator_substs, movability))
    }

    pub fn mk_generator_witness(self, types: ty::Binder<&'tcx List<Ty<'tcx>>>) -> Ty<'tcx> {
        self.mk_ty(GeneratorWitness(types))
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
        self.mk_ty(Infer(it))
    }

    pub fn mk_ty_param(self,
                       index: u32,
                       name: InternedString) -> Ty<'tcx> {
        self.mk_ty(Param(ParamTy { idx: index, name: name }))
    }

    pub fn mk_self_type(self) -> Ty<'tcx> {
        self.mk_ty_param(0, keywords::SelfType.name().as_interned_str())
    }

    pub fn mk_param_from_def(self, param: &ty::GenericParamDef) -> Kind<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                self.mk_region(ty::ReEarlyBound(param.to_early_bound_region_data())).into()
            }
            GenericParamDefKind::Type {..} => self.mk_ty_param(param.index, param.name).into(),
        }
    }

    pub fn mk_opaque(self, def_id: DefId, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Opaque(def_id, substs))
    }

    pub fn intern_existential_predicates(self, eps: &[ExistentialPredicate<'tcx>])
        -> &'tcx List<ExistentialPredicate<'tcx>> {
        assert!(!eps.is_empty());
        assert!(eps.windows(2).all(|w| w[0].stable_cmp(self, &w[1]) != Ordering::Greater));
        self._intern_existential_predicates(eps)
    }

    pub fn intern_predicates(self, preds: &[Predicate<'tcx>])
        -> &'tcx List<Predicate<'tcx>> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        if preds.len() == 0 {
            // The macro-generated method below asserts we don't intern an empty slice.
            List::empty()
        } else {
            self._intern_predicates(preds)
        }
    }

    pub fn intern_type_list(self, ts: &[Ty<'tcx>]) -> &'tcx List<Ty<'tcx>> {
        if ts.len() == 0 {
            List::empty()
        } else {
            self._intern_type_list(ts)
        }
    }

    pub fn intern_substs(self, ts: &[Kind<'tcx>]) -> &'tcx List<Kind<'tcx>> {
        if ts.len() == 0 {
            List::empty()
        } else {
            self._intern_substs(ts)
        }
    }

    pub fn intern_projs(self, ps: &[ProjectionKind<'tcx>]) -> &'tcx List<ProjectionKind<'tcx>> {
        if ps.len() == 0 {
            List::empty()
        } else {
            self._intern_projs(ps)
        }
    }

    pub fn intern_canonical_var_infos(self, ts: &[CanonicalVarInfo]) -> CanonicalVarInfos<'gcx> {
        if ts.len() == 0 {
            List::empty()
        } else {
            self.global_tcx()._intern_canonical_var_infos(ts)
        }
    }

    pub fn intern_clauses(self, ts: &[Clause<'tcx>]) -> Clauses<'tcx> {
        if ts.len() == 0 {
            List::empty()
        } else {
            self._intern_clauses(ts)
        }
    }

    pub fn intern_goals(self, ts: &[Goal<'tcx>]) -> Goals<'tcx> {
        if ts.len() == 0 {
            List::empty()
        } else {
            self._intern_goals(ts)
        }
    }

    pub fn mk_fn_sig<I>(self,
                        inputs: I,
                        output: I::Item,
                        variadic: bool,
                        unsafety: hir::Unsafety,
                        abi: abi::Abi)
        -> <I::Item as InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>::Output
        where I: Iterator,
              I::Item: InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>
    {
        inputs.chain(iter::once(output)).intern_with(|xs| ty::FnSig {
            inputs_and_output: self.intern_type_list(xs),
            variadic, unsafety, abi
        })
    }

    pub fn mk_existential_predicates<I: InternAs<[ExistentialPredicate<'tcx>],
                                     &'tcx List<ExistentialPredicate<'tcx>>>>(self, iter: I)
                                     -> I::Output {
        iter.intern_with(|xs| self.intern_existential_predicates(xs))
    }

    pub fn mk_predicates<I: InternAs<[Predicate<'tcx>],
                                     &'tcx List<Predicate<'tcx>>>>(self, iter: I)
                                     -> I::Output {
        iter.intern_with(|xs| self.intern_predicates(xs))
    }

    pub fn mk_type_list<I: InternAs<[Ty<'tcx>],
                        &'tcx List<Ty<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_type_list(xs))
    }

    pub fn mk_substs<I: InternAs<[Kind<'tcx>],
                     &'tcx List<Kind<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_substs(xs))
    }

    pub fn mk_substs_trait(self,
                     self_ty: Ty<'tcx>,
                     rest: &[Kind<'tcx>])
                    -> &'tcx Substs<'tcx>
    {
        self.mk_substs(iter::once(self_ty.into()).chain(rest.iter().cloned()))
    }

    pub fn mk_clauses<I: InternAs<[Clause<'tcx>], Clauses<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_clauses(xs))
    }

    pub fn mk_goals<I: InternAs<[Goal<'tcx>], Goals<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_goals(xs))
    }

    pub fn lint_hir<S: Into<MultiSpan>>(self,
                                        lint: &'static Lint,
                                        hir_id: HirId,
                                        span: S,
                                        msg: &str) {
        self.struct_span_lint_hir(lint, hir_id, span.into(), msg).emit()
    }

    pub fn lint_node<S: Into<MultiSpan>>(self,
                                         lint: &'static Lint,
                                         id: NodeId,
                                         span: S,
                                         msg: &str) {
        self.struct_span_lint_node(lint, id, span.into(), msg).emit()
    }

    pub fn lint_hir_note<S: Into<MultiSpan>>(self,
                                              lint: &'static Lint,
                                              hir_id: HirId,
                                              span: S,
                                              msg: &str,
                                              note: &str) {
        let mut err = self.struct_span_lint_hir(lint, hir_id, span.into(), msg);
        err.note(note);
        err.emit()
    }

    pub fn lint_node_note<S: Into<MultiSpan>>(self,
                                              lint: &'static Lint,
                                              id: NodeId,
                                              span: S,
                                              msg: &str,
                                              note: &str) {
        let mut err = self.struct_span_lint_node(lint, id, span.into(), msg);
        err.note(note);
        err.emit()
    }

    pub fn lint_level_at_node(self, lint: &'static Lint, mut id: NodeId)
        -> (lint::Level, lint::LintSource)
    {
        // Right now we insert a `with_ignore` node in the dep graph here to
        // ignore the fact that `lint_levels` below depends on the entire crate.
        // For now this'll prevent false positives of recompiling too much when
        // anything changes.
        //
        // Once red/green incremental compilation lands we should be able to
        // remove this because while the crate changes often the lint level map
        // will change rarely.
        self.dep_graph.with_ignore(|| {
            let sets = self.lint_levels(LOCAL_CRATE);
            loop {
                let hir_id = self.hir.definitions().node_to_hir_id(id);
                if let Some(pair) = sets.level_and_source(lint, hir_id, self.sess) {
                    return pair
                }
                let next = self.hir.get_parent_node(id);
                if next == id {
                    bug!("lint traversal reached the root of the crate");
                }
                id = next;
            }
        })
    }

    pub fn struct_span_lint_hir<S: Into<MultiSpan>>(self,
                                                    lint: &'static Lint,
                                                    hir_id: HirId,
                                                    span: S,
                                                    msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let node_id = self.hir.hir_to_node_id(hir_id);
        let (level, src) = self.lint_level_at_node(lint, node_id);
        lint::struct_lint_level(self.sess, lint, level, src, Some(span.into()), msg)
    }

    pub fn struct_span_lint_node<S: Into<MultiSpan>>(self,
                                                     lint: &'static Lint,
                                                     id: NodeId,
                                                     span: S,
                                                     msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let (level, src) = self.lint_level_at_node(lint, id);
        lint::struct_lint_level(self.sess, lint, level, src, Some(span.into()), msg)
    }

    pub fn struct_lint_node(self, lint: &'static Lint, id: NodeId, msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let (level, src) = self.lint_level_at_node(lint, id);
        lint::struct_lint_level(self.sess, lint, level, src, None, msg)
    }

    pub fn in_scope_traits(self, id: HirId) -> Option<Lrc<StableVec<TraitCandidate>>> {
        self.in_scope_traits_map(id.owner)
            .and_then(|map| map.get(&id.local_id).cloned())
    }

    pub fn named_region(self, id: HirId) -> Option<resolve_lifetime::Region> {
        self.named_region_map(id.owner)
            .and_then(|map| map.get(&id.local_id).cloned())
    }

    pub fn is_late_bound(self, id: HirId) -> bool {
        self.is_late_bound_map(id.owner)
            .map(|set| set.contains(&id.local_id))
            .unwrap_or(false)
    }

    pub fn object_lifetime_defaults(self, id: HirId)
        -> Option<Lrc<Vec<ObjectLifetimeDefault>>>
    {
        self.object_lifetime_defaults_map(id.owner)
            .and_then(|map| map.get(&id.local_id).cloned())
    }
}

pub trait InternAs<T: ?Sized, R> {
    type Output;
    fn intern_with<F>(self, f: F) -> Self::Output
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
        f(&iter.collect::<SmallVec<[_; 8]>>())
    }
}

impl<'a, T, R> InternIteratorElement<T, R> for &'a T
    where T: Clone + 'a
{
    type Output = R;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        f(&iter.cloned().collect::<SmallVec<[_; 8]>>())
    }
}

impl<T, R, E> InternIteratorElement<T, R> for Result<T, E> {
    type Output = Result<R, E>;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        Ok(f(&iter.collect::<Result<SmallVec<[_; 8]>, _>>()?))
    }
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    // FIXME(#44234) - almost all of these queries have no sub-queries and
    // therefore no actual inputs, they're just reading tables calculated in
    // resolve! Does this work? Unsure! That's what the issue is about
    providers.in_scope_traits_map = |tcx, id| tcx.gcx.trait_map.get(&id).cloned();
    providers.module_exports = |tcx, id| tcx.gcx.export_map.get(&id).cloned();
    providers.crate_name = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.crate_name
    };
    providers.get_lib_features = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        Lrc::new(middle::lib_features::collect(tcx))
    };
    providers.get_lang_items = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        Lrc::new(middle::lang_items::collect(tcx))
    };
    providers.freevars = |tcx, id| tcx.gcx.freevars.get(&id).cloned();
    providers.maybe_unused_trait_import = |tcx, id| {
        tcx.maybe_unused_trait_imports.contains(&id)
    };
    providers.maybe_unused_extern_crates = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.maybe_unused_extern_crates.clone())
    };

    providers.stability_index = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(stability::Index::new(tcx))
    };
    providers.lookup_stability = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let id = tcx.hir.definitions().def_index_to_hir_id(id.index);
        tcx.stability().local_stability(id)
    };
    providers.lookup_deprecation_entry = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let id = tcx.hir.definitions().def_index_to_hir_id(id.index);
        tcx.stability().local_deprecation_entry(id)
    };
    providers.extern_mod_stmt_cnum = |tcx, id| {
        let id = tcx.hir.as_local_node_id(id).unwrap();
        tcx.cstore.extern_mod_stmt_cnum_untracked(id)
    };
    providers.all_crate_nums = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.cstore.crates_untracked())
    };
    providers.postorder_cnums = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.cstore.postorder_cnums_untracked())
    };
    providers.output_filenames = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.output_filenames.clone()
    };
    providers.features_query = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.sess.features_untracked().clone())
    };
    providers.is_panic_runtime = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        attr::contains_name(tcx.hir.krate_attrs(), "panic_runtime")
    };
    providers.is_compiler_builtins = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        attr::contains_name(tcx.hir.krate_attrs(), "compiler_builtins")
    };
}
