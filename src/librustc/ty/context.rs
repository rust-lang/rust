//! Type context book-keeping.

use crate::arena::Arena;
use crate::dep_graph::DepGraph;
use crate::dep_graph::{self, DepNode, DepConstructor};
use crate::session::Session;
use crate::session::config::{BorrowckMode, OutputFilenames};
use crate::session::config::CrateType;
use crate::middle;
use crate::hir::{TraitCandidate, HirId, ItemKind, ItemLocalId, Node};
use crate::hir::def::{Res, DefKind, Export};
use crate::hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use crate::hir::map as hir_map;
use crate::hir::map::DefPathHash;
use crate::lint::{self, Lint};
use crate::ich::{StableHashingContext, NodeIdHashingMode};
use crate::infer::canonical::{Canonical, CanonicalVarInfo, CanonicalVarInfos};
use crate::infer::outlives::free_region_map::FreeRegionMap;
use crate::middle::cstore::CrateStoreDyn;
use crate::middle::cstore::EncodedMetadata;
use crate::middle::lang_items;
use crate::middle::resolve_lifetime::{self, ObjectLifetimeDefault};
use crate::middle::stability;
use crate::mir::{self, Body, interpret, ProjectionKind};
use crate::mir::interpret::{ConstValue, Allocation, Scalar};
use crate::ty::subst::{Kind, InternalSubsts, SubstsRef, Subst};
use crate::ty::ReprOptions;
use crate::traits;
use crate::traits::{Clause, Clauses, GoalKind, Goal, Goals};
use crate::ty::{self, DefIdTree, Ty, TypeAndMut};
use crate::ty::{TyS, TyKind, List};
use crate::ty::{AdtKind, AdtDef, ClosureSubsts, GeneratorSubsts, Region, Const};
use crate::ty::{PolyFnSig, InferTy, ParamTy, ProjectionTy, ExistentialPredicate, Predicate};
use crate::ty::RegionKind;
use crate::ty::{TyVar, TyVid, IntVar, IntVid, FloatVar, FloatVid, ConstVid};
use crate::ty::TyKind::*;
use crate::ty::{InferConst, ParamConst};
use crate::ty::GenericParamDefKind;
use crate::ty::layout::{LayoutDetails, TargetDataLayout, VariantIdx};
use crate::ty::query;
use crate::ty::steal::Steal;
use crate::ty::subst::{UserSubsts, UnpackedKind};
use crate::ty::{BoundVar, BindingMode};
use crate::ty::CanonicalPolyFnSig;
use crate::util::common::ErrorReported;
use crate::util::nodemap::{DefIdMap, DefIdSet, ItemLocalMap, ItemLocalSet};
use crate::util::nodemap::{FxHashMap, FxHashSet};
use errors::DiagnosticBuilder;
use rustc_data_structures::interner::HashInterner;
use smallvec::SmallVec;
use rustc_data_structures::stable_hasher::{HashStable, hash_stable_hashmap,
                                           StableHasher, StableHasherResult,
                                           StableVec};
use arena::SyncDroplessArena;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::sync::{Lrc, Lock, WorkerLocal};
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
use rustc_macros::HashStable;
use syntax::ast;
use syntax::attr;
use syntax::source_map::MultiSpan;
use syntax::feature_gate;
use syntax::symbol::{Symbol, InternedString, kw, sym};
use syntax_pos::Span;

use crate::hir;

pub struct AllArenas {
    pub interner: SyncDroplessArena,
}

impl AllArenas {
    pub fn new() -> Self {
        AllArenas {
            interner: SyncDroplessArena::default(),
        }
    }
}

type InternedSet<'tcx, T> = Lock<FxHashMap<Interned<'tcx, T>, ()>>;

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc are allocated from
    arena: &'tcx SyncDroplessArena,

    /// Specifically use a speedy hash algorithm for these hash sets,
    /// they're accessed quite often.
    type_: InternedSet<'tcx, TyS<'tcx>>,
    type_list: InternedSet<'tcx, List<Ty<'tcx>>>,
    substs: InternedSet<'tcx, InternalSubsts<'tcx>>,
    canonical_var_infos: InternedSet<'tcx, List<CanonicalVarInfo>>,
    region: InternedSet<'tcx, RegionKind>,
    existential_predicates: InternedSet<'tcx, List<ExistentialPredicate<'tcx>>>,
    predicates: InternedSet<'tcx, List<Predicate<'tcx>>>,
    clauses: InternedSet<'tcx, List<Clause<'tcx>>>,
    goal: InternedSet<'tcx, GoalKind<'tcx>>,
    goal_list: InternedSet<'tcx, List<Goal<'tcx>>>,
    projs: InternedSet<'tcx, List<ProjectionKind>>,
    const_: InternedSet<'tcx, Const<'tcx>>,
}

impl<'tcx> CtxtInterners<'tcx> {
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
            clauses: Default::default(),
            goal: Default::default(),
            goal_list: Default::default(),
            projs: Default::default(),
            const_: Default::default(),
        }
    }

    /// Intern a type
    #[inline(never)]
    fn intern_ty(&self,
        st: TyKind<'tcx>
    ) -> Ty<'tcx> {
        self.type_.borrow_mut().intern(st, |st| {
            let flags = super::flags::FlagComputation::for_sty(&st);

            let ty_struct = TyS {
                sty: st,
                flags: flags.flags,
                outer_exclusive_binder: flags.outer_exclusive_binder,
            };


            Interned(self.arena.alloc(ty_struct))
        }).0
    }
}

pub struct Common<'tcx> {
    pub empty_predicates: ty::GenericPredicates<'tcx>,
}

pub struct CommonTypes<'tcx> {
    pub unit: Ty<'tcx>,
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

    /// Dummy type used for the `Self` of a `TraitRef` created for converting
    /// a trait object, and which gets removed in `ExistentialTraitRef`.
    /// This type must not appear anywhere in other converted types.
    pub trait_object_dummy_self: Ty<'tcx>,
}

pub struct CommonLifetimes<'tcx> {
    pub re_empty: Region<'tcx>,
    pub re_static: Region<'tcx>,
    pub re_erased: Region<'tcx>,
}

pub struct CommonConsts<'tcx> {
    pub err: &'tcx Const<'tcx>,
}

pub struct LocalTableInContext<'a, V> {
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
                    bug!("node {} with HirId::owner {:?} cannot be placed in \
                          TypeckTables with local_id_root {:?}",
                         tcx.hir().node_to_string(hir_id),
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

pub struct LocalTableInContextMut<'a, V> {
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

/// All information necessary to validate and reveal an `impl Trait` or `existential Type`
#[derive(RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ResolvedOpaqueTy<'tcx> {
    /// The revealed type as seen by this function.
    pub concrete_type: Ty<'tcx>,
    /// Generic parameters on the opaque type as passed by this function.
    /// For `existential type Foo<A, B>; fn foo<T, U>() -> Foo<T, U> { .. }` this is `[T, U]`, not
    /// `[A, B]`
    pub substs: SubstsRef<'tcx>,
}

#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct TypeckTables<'tcx> {
    /// The HirId::owner all ItemLocalIds in this table are relative to.
    pub local_id_root: Option<DefId>,

    /// Resolved definitions for `<T>::X` associated paths and
    /// method calls, including those of overloaded operators.
    type_dependent_defs: ItemLocalMap<Result<(DefKind, DefId), ErrorReported>>,

    /// Resolved field indices for field accesses in expressions (`S { field }`, `obj.field`)
    /// or patterns (`S { field }`). The index is often useful by itself, but to learn more
    /// about the field you also need definition of the variant to which the field
    /// belongs, but it may not exist if it's a tuple field (`tuple.0`).
    field_indices: ItemLocalMap<usize>,

    /// Stores the types for various nodes in the AST. Note that this table
    /// is not guaranteed to be populated until after typeck. See
    /// typeck::check::fn_ctxt for details.
    node_types: ItemLocalMap<Ty<'tcx>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node. This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    node_substs: ItemLocalMap<SubstsRef<'tcx>>,

    /// This will either store the canonicalized types provided by the user
    /// or the substitutions that the user explicitly gave (if any) attached
    /// to `id`. These will not include any inferred values. The canonical form
    /// is used to capture things like `_` or other unspecified values.
    ///
    /// For example, if the user wrote `foo.collect::<Vec<_>>()`, then the
    /// canonical substitutions would include only `for<X> { Vec<X> }`.
    ///
    /// See also `AscribeUserType` statement in MIR.
    user_provided_types: ItemLocalMap<CanonicalUserType<'tcx>>,

    /// Stores the canonicalized types provided by the user. See also
    /// `AscribeUserType` statement in MIR.
    pub user_provided_sigs: DefIdMap<CanonicalPolyFnSig<'tcx>>,

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

    /// For every coercion cast we add the HIR node ID of the cast
    /// expression to this set.
    coercion_casts: ItemLocalSet,

    /// Set of trait imports actually used in the method resolution.
    /// This is used for warning unused imports. During type
    /// checking, this `Lrc` should not be cloned: it must have a ref-count
    /// of 1 so that we can insert things into the set mutably.
    pub used_trait_imports: Lrc<DefIdSet>,

    /// If any errors occurred while type-checking this body,
    /// this field will be set to `true`.
    pub tainted_by_errors: bool,

    /// Stores the free-region relationships that were deduced from
    /// its where-clauses and parameter types. These are then
    /// read-again by borrowck.
    pub free_region_map: FreeRegionMap<'tcx>,

    /// All the existential types that are restricted to concrete types
    /// by this function
    pub concrete_existential_types: FxHashMap<DefId, ResolvedOpaqueTy<'tcx>>,

    /// Given the closure ID this map provides the list of UpvarIDs used by it.
    /// The upvarID contains the HIR node ID and it also contains the full path
    /// leading to the member of the struct or tuple that is used instead of the
    /// entire variable.
    pub upvar_list: ty::UpvarListMap,
}

impl<'tcx> TypeckTables<'tcx> {
    pub fn empty(local_id_root: Option<DefId>) -> TypeckTables<'tcx> {
        TypeckTables {
            local_id_root,
            type_dependent_defs: Default::default(),
            field_indices: Default::default(),
            user_provided_types: Default::default(),
            user_provided_sigs: Default::default(),
            node_types: Default::default(),
            node_substs: Default::default(),
            adjustments: Default::default(),
            pat_binding_modes: Default::default(),
            pat_adjustments: Default::default(),
            upvar_capture_map: Default::default(),
            closure_kind_origins: Default::default(),
            liberated_fn_sigs: Default::default(),
            fru_field_types: Default::default(),
            coercion_casts: Default::default(),
            used_trait_imports: Lrc::new(Default::default()),
            tainted_by_errors: false,
            free_region_map: Default::default(),
            concrete_existential_types: Default::default(),
            upvar_list: Default::default(),
        }
    }

    /// Returns the final resolution of a `QPath` in an `Expr` or `Pat` node.
    pub fn qpath_res(&self, qpath: &hir::QPath, id: hir::HirId) -> Res {
        match *qpath {
            hir::QPath::Resolved(_, ref path) => path.res,
            hir::QPath::TypeRelative(..) => self.type_dependent_def(id)
                .map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
        }
    }

    pub fn type_dependent_defs(
        &self,
    ) -> LocalTableInContext<'_, Result<(DefKind, DefId), ErrorReported>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.type_dependent_defs
        }
    }

    pub fn type_dependent_def(&self, id: HirId) -> Option<(DefKind, DefId)> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.type_dependent_defs.get(&id.local_id).cloned().and_then(|r| r.ok())
    }

    pub fn type_dependent_def_id(&self, id: HirId) -> Option<DefId> {
        self.type_dependent_def(id).map(|(_, def_id)| def_id)
    }

    pub fn type_dependent_defs_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, Result<(DefKind, DefId), ErrorReported>> {
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

    pub fn user_provided_types(
        &self
    ) -> LocalTableInContext<'_, CanonicalUserType<'tcx>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.user_provided_types
        }
    }

    pub fn user_provided_types_mut(
        &mut self
    ) -> LocalTableInContextMut<'_, CanonicalUserType<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.user_provided_types
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

    pub fn node_type(&self, id: hir::HirId) -> Ty<'tcx> {
        self.node_type_opt(id).unwrap_or_else(||
            bug!("node_type: no type for node `{}`",
                 tls::with(|tcx| tcx.hir().node_to_string(id)))
        )
    }

    pub fn node_type_opt(&self, id: hir::HirId) -> Option<Ty<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_types.get(&id.local_id).cloned()
    }

    pub fn node_substs_mut(&mut self) -> LocalTableInContextMut<'_, SubstsRef<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.node_substs
        }
    }

    pub fn node_substs(&self, id: hir::HirId) -> SubstsRef<'tcx> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_substs.get(&id.local_id).cloned().unwrap_or_else(|| InternalSubsts::empty())
    }

    pub fn node_substs_opt(&self, id: hir::HirId) -> Option<SubstsRef<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_substs.get(&id.local_id).cloned()
    }

    // Returns the type of a pattern as a monotype. Like @expr_ty, this function
    // doesn't provide type parameter substitutions.
    pub fn pat_ty(&self, pat: &hir::Pat) -> Ty<'tcx> {
        self.node_type(pat.hir_id)
    }

    pub fn pat_ty_opt(&self, pat: &hir::Pat) -> Option<Ty<'tcx>> {
        self.node_type_opt(pat.hir_id)
    }

    // Returns the type of an expression as a monotype.
    //
    // NB (1): This is the PRE-ADJUSTMENT TYPE for the expression.  That is, in
    // some cases, we insert `Adjustment` annotations such as auto-deref or
    // auto-ref.  The type returned by this function does not consider such
    // adjustments.  See `expr_ty_adjusted()` instead.
    //
    // NB (2): This type doesn't provide type parameter substitutions; e.g., if you
    // ask for the type of "id" in "id(3)", it will return "fn(&isize) -> isize"
    // instead of "fn(ty) -> T with T = isize".
    pub fn expr_ty(&self, expr: &hir::Expr) -> Ty<'tcx> {
        self.node_type(expr.hir_id)
    }

    pub fn expr_ty_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.node_type_opt(expr.hir_id)
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
            Some(Ok((DefKind::Method, _))) => true,
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

    pub fn is_coercion_cast(&self, hir_id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_tables(self.local_id_root, hir_id, true);
        self.coercion_casts.contains(&hir_id.local_id)
    }

    pub fn set_coercion_cast(&mut self, id: ItemLocalId) {
        self.coercion_casts.insert(id);
    }

    pub fn coercion_casts(&self) -> &ItemLocalSet {
        &self.coercion_casts
    }

}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for TypeckTables<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::TypeckTables {
            local_id_root,
            ref type_dependent_defs,
            ref field_indices,
            ref user_provided_types,
            ref user_provided_sigs,
            ref node_types,
            ref node_substs,
            ref adjustments,
            ref pat_binding_modes,
            ref pat_adjustments,
            ref upvar_capture_map,
            ref closure_kind_origins,
            ref liberated_fn_sigs,
            ref fru_field_types,

            ref coercion_casts,

            ref used_trait_imports,
            tainted_by_errors,
            ref free_region_map,
            ref concrete_existential_types,
            ref upvar_list,

        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            type_dependent_defs.hash_stable(hcx, hasher);
            field_indices.hash_stable(hcx, hasher);
            user_provided_types.hash_stable(hcx, hasher);
            user_provided_sigs.hash_stable(hcx, hasher);
            node_types.hash_stable(hcx, hasher);
            node_substs.hash_stable(hcx, hasher);
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
            coercion_casts.hash_stable(hcx, hasher);
            used_trait_imports.hash_stable(hcx, hasher);
            tainted_by_errors.hash_stable(hcx, hasher);
            free_region_map.hash_stable(hcx, hasher);
            concrete_existential_types.hash_stable(hcx, hasher);
            upvar_list.hash_stable(hcx, hasher);
        })
    }
}

newtype_index! {
    pub struct UserTypeAnnotationIndex {
        derive [HashStable]
        DEBUG_FORMAT = "UserType({})",
        const START_INDEX = 0,
    }
}

/// Mapping of type annotation indices to canonical user type annotations.
pub type CanonicalUserTypeAnnotations<'tcx> =
    IndexVec<UserTypeAnnotationIndex, CanonicalUserTypeAnnotation<'tcx>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, HashStable)]
pub struct CanonicalUserTypeAnnotation<'tcx> {
    pub user_ty: CanonicalUserType<'tcx>,
    pub span: Span,
    pub inferred_ty: Ty<'tcx>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for CanonicalUserTypeAnnotation<'tcx> {
        user_ty, span, inferred_ty
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for CanonicalUserTypeAnnotation<'a> {
        type Lifted = CanonicalUserTypeAnnotation<'tcx>;
        user_ty, span, inferred_ty
    }
}

/// Canonicalized user type annotation.
pub type CanonicalUserType<'tcx> = Canonical<'tcx, UserType<'tcx>>;

impl CanonicalUserType<'tcx> {
    /// Returns `true` if this represents a substitution of the form `[?0, ?1, ?2]`,
    /// i.e., each thing is mapped to a canonical variable with the same index.
    pub fn is_identity(&self) -> bool {
        match self.value {
            UserType::Ty(_) => false,
            UserType::TypeOf(_, user_substs) => {
                if user_substs.user_self_ty.is_some() {
                    return false;
                }

                user_substs.substs.iter().zip(BoundVar::new(0)..).all(|(kind, cvar)| {
                    match kind.unpack() {
                        UnpackedKind::Type(ty) => match ty.sty {
                            ty::Bound(debruijn, b) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(debruijn, ty::INNERMOST);
                                cvar == b.var
                            }
                            _ => false,
                        },

                        UnpackedKind::Lifetime(r) => match r {
                            ty::ReLateBound(debruijn, br) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(*debruijn, ty::INNERMOST);
                                cvar == br.assert_bound_var()
                            }
                            _ => false,
                        },

                        UnpackedKind::Const(ct) => match ct.val {
                            ConstValue::Infer(InferConst::Canonical(debruijn, b)) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(debruijn, ty::INNERMOST);
                                cvar == b
                            }
                            _ => false,
                        },
                    }
                })
            },
        }
    }
}

/// A user-given type annotation attached to a constant. These arise
/// from constants that are named via paths, like `Foo::<A>::new` and
/// so forth.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, HashStable)]
pub enum UserType<'tcx> {
    Ty(Ty<'tcx>),

    /// The canonical type is the result of `type_of(def_id)` with the
    /// given substitutions applied.
    TypeOf(DefId, UserSubsts<'tcx>),
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for UserType<'tcx> {
        (UserType::Ty)(ty),
        (UserType::TypeOf)(def, substs),
    }
}

EnumLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for UserType<'a> {
        type Lifted = UserType<'tcx>;
        (UserType::Ty)(ty),
        (UserType::TypeOf)(def, substs),
    }
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonTypes<'tcx> {
        let mk = |sty| interners.intern_ty(sty);

        CommonTypes {
            unit: mk(Tuple(List::empty())),
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

            trait_object_dummy_self: mk(Infer(ty::FreshTy(0))),
        }
    }
}

impl<'tcx> CommonLifetimes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonLifetimes<'tcx> {
        let mk = |r| {
            interners.region.borrow_mut().intern(r, |r| {
                Interned(interners.arena.alloc(r))
            }).0
        };

        CommonLifetimes {
            re_empty: mk(RegionKind::ReEmpty),
            re_static: mk(RegionKind::ReStatic),
            re_erased: mk(RegionKind::ReErased),
        }
    }
}

impl<'tcx> CommonConsts<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>, types: &CommonTypes<'tcx>) -> CommonConsts<'tcx> {
        let mk_const = |c| {
            interners.const_.borrow_mut().intern(c, |c| {
                Interned(interners.arena.alloc(c))
            }).0
        };

        CommonConsts {
            err: mk_const(ty::Const {
                val: ConstValue::Scalar(Scalar::zst()),
                ty: types.err,
            }),
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
/// [rustc guide]: https://rust-lang.github.io/rustc-guide/ty.html
#[derive(Copy, Clone)]
pub struct TyCtxt<'tcx> {
    gcx: &'tcx GlobalCtxt<'tcx>,
}

impl<'tcx> Deref for TyCtxt<'tcx> {
    type Target = &'tcx GlobalCtxt<'tcx>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.gcx
    }
}

pub struct GlobalCtxt<'tcx> {
    pub arena: WorkerLocal<Arena<'tcx>>,

    interners: CtxtInterners<'tcx>,

    cstore: &'tcx CrateStoreDyn,

    pub sess: &'tcx Session,

    pub dep_graph: DepGraph,

    /// Common objects.
    pub common: Common<'tcx>,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    /// Common lifetimes, pre-interned for your convenience.
    pub lifetimes: CommonLifetimes<'tcx>,

    /// Common consts, pre-interned for your convenience.
    pub consts: CommonConsts<'tcx>,

    /// Map indicating what traits are in scope for places where this
    /// is relevant; generated by resolve.
    trait_map: FxHashMap<DefIndex,
                         FxHashMap<ItemLocalId,
                                   StableVec<TraitCandidate>>>,

    /// Export map produced by name resolution.
    export_map: FxHashMap<DefId, Vec<Export<hir::HirId>>>,

    hir_map: hir_map::Map<'tcx>,

    /// A map from DefPathHash -> DefId. Includes DefIds from the local crate
    /// as well as all upstream crates. Only populated in incremental mode.
    pub def_path_hash_to_def_id: Option<FxHashMap<DefPathHash, DefId>>,

    pub queries: query::Queries<'tcx>,

    maybe_unused_trait_imports: FxHashSet<DefId>,
    maybe_unused_extern_crates: Vec<(DefId, Span)>,
    /// A map of glob use to a set of names it actually imports. Currently only
    /// used in save-analysis.
    glob_map: FxHashMap<DefId, FxHashSet<ast::Name>>,
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

    stability_interner: Lock<FxHashMap<&'tcx attr::Stability, ()>>,

    /// Stores the value of constants (and deduplicates the actual memory)
    allocation_interner: Lock<FxHashMap<&'tcx Allocation, ()>>,

    pub alloc_map: Lock<interpret::AllocMap<'tcx>>,

    layout_interner: Lock<FxHashMap<&'tcx LayoutDetails, ()>>,

    /// A general purpose channel to throw data out the back towards LLVM worker
    /// threads.
    ///
    /// This is intended to only get used during the codegen phase of the compiler
    /// when satisfying the query for a particular codegen unit. Internally in
    /// the query it'll send data along this channel to get processed later.
    pub tx_to_llvm_workers: Lock<mpsc::Sender<Box<dyn Any + Send>>>,

    output_filenames: Arc<OutputFilenames>,
}

impl<'tcx> TyCtxt<'tcx> {
    /// Gets the global `TyCtxt`.
    #[inline]
    pub fn global_tcx(self) -> TyCtxt<'tcx> {
        TyCtxt {
            gcx: self.gcx,
        }
    }

    #[inline(always)]
    pub fn hir(self) -> &'tcx hir_map::Map<'tcx> {
        &self.hir_map
    }

    pub fn alloc_steal_mir(self, mir: Body<'tcx>) -> &'tcx Steal<Body<'tcx>> {
        self.arena.alloc(Steal::new(mir))
    }

    pub fn alloc_adt_def(
        self,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, ty::VariantDef>,
        repr: ReprOptions,
    ) -> &'tcx ty::AdtDef {
        let def = ty::AdtDef::new(self, did, kind, variants, repr);
        self.arena.alloc(def)
    }

    pub fn intern_const_alloc(self, alloc: Allocation) -> &'tcx Allocation {
        self.allocation_interner.borrow_mut().intern(alloc, |alloc| {
            self.arena.alloc(alloc)
        })
    }

    /// Allocates a byte or string literal for `mir::interpret`, read-only
    pub fn allocate_bytes(self, bytes: &[u8]) -> interpret::AllocId {
        // create an allocation that just contains these bytes
        let alloc = interpret::Allocation::from_byte_aligned_bytes(bytes);
        let alloc = self.intern_const_alloc(alloc);
        self.alloc_map.lock().create_memory_alloc(alloc)
    }

    pub fn intern_stability(self, stab: attr::Stability) -> &'tcx attr::Stability {
        self.stability_interner.borrow_mut().intern(stab, |stab| {
            self.arena.alloc(stab)
        })
    }

    pub fn intern_layout(self, layout: LayoutDetails) -> &'tcx LayoutDetails {
        self.layout_interner.borrow_mut().intern(layout, |layout| {
            self.arena.alloc(layout)
        })
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
        (get(sym::rustc_layout_scalar_valid_range_start),
         get(sym::rustc_layout_scalar_valid_range_end))
    }

    pub fn lift<T: ?Sized + Lift<'tcx>>(self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }

    /// Like lift, but only tries in the global tcx.
    pub fn lift_to_global<T: ?Sized + Lift<'tcx>>(self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self.global_tcx())
    }

    /// Creates a type context and call the closure with a `TyCtxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_global_ctxt(
        s: &'tcx Session,
        cstore: &'tcx CrateStoreDyn,
        local_providers: ty::query::Providers<'tcx>,
        extern_providers: ty::query::Providers<'tcx>,
        arenas: &'tcx AllArenas,
        resolutions: ty::Resolutions,
        hir: hir_map::Map<'tcx>,
        on_disk_query_result_cache: query::OnDiskCache<'tcx>,
        crate_name: &str,
        tx: mpsc::Sender<Box<dyn Any + Send>>,
        output_filenames: &OutputFilenames,
    ) -> GlobalCtxt<'tcx> {
        let data_layout = TargetDataLayout::parse(&s.target.target).unwrap_or_else(|err| {
            s.fatal(&err);
        });
        let interners = CtxtInterners::new(&arenas.interner);
        let common = Common {
            empty_predicates: ty::GenericPredicates {
                parent: None,
                predicates: vec![],
            },
        };
        let common_types = CommonTypes::new(&interners);
        let common_lifetimes = CommonLifetimes::new(&interners);
        let common_consts = CommonConsts::new(&interners, &common_types);
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

        let mut trait_map: FxHashMap<_, FxHashMap<_, _>> = FxHashMap::default();
        for (k, v) in resolutions.trait_map {
            let hir_id = hir.node_to_hir_id(k);
            let map = trait_map.entry(hir_id.owner).or_default();
            map.insert(hir_id.local_id, StableVec::new(v));
        }

        GlobalCtxt {
            sess: s,
            cstore,
            arena: WorkerLocal::new(|_| Arena::default()),
            interners,
            dep_graph,
            common,
            types: common_types,
            lifetimes: common_lifetimes,
            consts: common_consts,
            trait_map,
            export_map: resolutions.export_map.into_iter().map(|(k, v)| {
                let exports: Vec<_> = v.into_iter().map(|e| {
                    e.map_id(|id| hir.node_to_hir_id(id))
                }).collect();
                (k, exports)
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
            glob_map: resolutions.glob_map.into_iter().map(|(id, names)| {
                (hir.local_def_id(id), names)
            }).collect(),
            extern_prelude: resolutions.extern_prelude,
            hir_map: hir,
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
        }
    }

    pub fn consider_optimizing<T: Fn() -> String>(&self, msg: T) -> bool {
        let cname = self.crate_name(LOCAL_CRATE).as_str();
        self.sess.consider_optimizing(&cname, msg)
    }

    pub fn lib_features(self) -> &'tcx middle::lib_features::LibFeatures {
        self.get_lib_features(LOCAL_CRATE)
    }

    pub fn lang_items(self) -> &'tcx middle::lang_items::LanguageItems {
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

    pub fn stability(self) -> &'tcx stability::Index<'tcx> {
        self.stability_index(LOCAL_CRATE)
    }

    pub fn crates(self) -> &'tcx [CrateNum] {
        self.all_crate_nums(LOCAL_CRATE)
    }

    pub fn features(self) -> &'tcx feature_gate::Features {
        self.features_query(LOCAL_CRATE)
    }

    pub fn def_key(self, id: DefId) -> hir_map::DefKey {
        if id.is_local() {
            self.hir().def_key(id)
        } else {
            self.cstore.def_key(id)
        }
    }

    /// Converts a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned def-path).
    ///
    /// Note that if `id` is not local to this crate, the result will
    ///  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> hir_map::DefPath {
        if id.is_local() {
            self.hir().def_path(id)
        } else {
            self.cstore.def_path(id)
        }
    }

    /// Returns whether or not the crate with CrateNum 'cnum'
    /// is marked as a private dependency
    pub fn is_private_dep(self, cnum: CrateNum) -> bool {
        if cnum == LOCAL_CRATE {
            false
        } else {
            self.cstore.crate_is_private_dep_untracked(cnum)
        }
    }

    #[inline]
    pub fn def_path_hash(self, def_id: DefId) -> hir_map::DefPathHash {
        if def_id.is_local() {
            self.hir().definitions().def_path_hash(def_id.index)
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

    #[inline(always)]
    pub fn create_stable_hashing_context(self) -> StableHashingContext<'tcx> {
        let krate = self.gcx.hir_map.forest.untracked_krate();

        StableHashingContext::new(self.sess,
                                  krate,
                                  self.hir().definitions(),
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
                                     |_, x| x, // No transformation needed
                                     dep_graph::hash_result,
            );
        }
    }

    pub fn serialize_query_result_cache<E>(self,
                                           encoder: &mut E)
                                           -> Result<(), E::Error>
        where E: ty::codec::TyEncoder
    {
        self.queries.on_disk_cache.serialize(self.global_tcx(), encoder)
    }

    /// If true, we should use the AST-based borrowck (we may *also* use
    /// the MIR-based borrowck).
    pub fn use_ast_borrowck(self) -> bool {
        self.borrowck_mode().use_ast()
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
        !self.sess.opts.debugging_opts.nll_dont_emit_read_for_match
    }

    /// What mode(s) of borrowck should we run? AST? MIR? both?
    /// (Also considers the `#![feature(nll)]` setting.)
    pub fn borrowck_mode(&self) -> BorrowckMode {
        // Here are the main constraints we need to deal with:
        //
        // 1. An opts.borrowck_mode of `BorrowckMode::Migrate` is
        //    synonymous with no `-Z borrowck=...` flag at all.
        //
        // 2. We want to allow developers on the Nightly channel
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
        // * Otherwise, if no `-Z borrowck=...` then use migrate mode
        //
        // * Otherwise, use the behavior requested via `-Z borrowck=...`

        if self.features().nll { return BorrowckMode::Mir; }

        self.sess.opts.borrowck_mode
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
                self.parent(ebr.def_id).unwrap(),
                ty::BoundRegion::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let hir_id = self.hir()
            .as_local_hir_id(suitable_region_binding_scope)
            .unwrap();
        let is_impl_item = match self.hir().find(hir_id) {
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
        let hir_id = self.hir().as_local_hir_id(scope_def_id).unwrap();
        match self.hir().get(hir_id) {
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

    /// Determine whether identifiers in the assembly have strict naming rules.
    /// Currently, only NVPTX* targets need it.
    pub fn has_strict_asm_symbol_naming(&self) -> bool {
        self.gcx.sess.target.target.arch.contains("nvptx")
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn encode_metadata(self)
        -> EncodedMetadata
    {
        self.cstore.encode_metadata(self)
    }
}

impl<'tcx> GlobalCtxt<'tcx> {
    /// Call the closure with a local `TyCtxt` using the given arena.
    /// `interners` is a slot passed so we can create a CtxtInterners
    /// with the same lifetime as `arena`.
    pub fn enter_local<F, R>(&'tcx self, f: F) -> R
    where
        F: FnOnce(TyCtxt<'tcx>) -> R,
    {
        let tcx = TyCtxt {
            gcx: self,
        };
        ty::tls::with_related_context(tcx.global_tcx(), |icx| {
            let new_icx = ty::tls::ImplicitCtxt {
                tcx,
                query: icx.query.clone(),
                diagnostics: icx.diagnostics,
                layout_depth: icx.layout_depth,
                task_deps: icx.task_deps,
            };
            ty::tls::enter_context(&new_icx, |_| {
                f(tcx)
            })
        })
    }
}

/// A trait implemented for all `X<'a>` types that can be safely and
/// efficiently converted to `X<'tcx>` as long as they are part of the
/// provided `TyCtxt<'tcx>`.
/// This can be done, for example, for `Ty<'tcx>` or `SubstsRef<'tcx>`
/// by looking them up in their respective interners.
///
/// However, this is still not the best implementation as it does
/// need to compare the components, even for interned values.
/// It would be more efficient if `TypedArena` provided a way to
/// determine whether the address is in the allocated range.
///
/// None is returned if the value or one of the components is not part
/// of the provided context.
/// For `Ty`, `None` can be returned if either the type interner doesn't
/// contain the `TyKind` key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g., `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx>: fmt::Debug {
    type Lifted: fmt::Debug + 'tcx;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted>;
}


macro_rules! nop_lift {
    ($ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<'tcx> for $ty {
                    type Lifted = $lifted;
                    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                        if tcx.interners.arena.in_arena(*self as *const _) {
                            Some(unsafe { mem::transmute(*self) })
                        } else {
                            None
                        }
                    }
                }
    };
}

macro_rules! nop_list_lift {
    ($ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<'tcx> for &'a List<$ty> {
                    type Lifted = &'tcx List<$lifted>;
                    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                        if self.is_empty() {
                            return Some(List::empty());
                        }
                        if tcx.interners.arena.in_arena(*self as *const _) {
                            Some(unsafe { mem::transmute(*self) })
                        } else {
                            None
                        }
                    }
                }
    };
}

nop_lift!{Ty<'a> => Ty<'tcx>}
nop_lift!{Region<'a> => Region<'tcx>}
nop_lift!{Goal<'a> => Goal<'tcx>}
nop_lift!{&'a Const<'a> => &'tcx Const<'tcx>}

nop_list_lift!{Goal<'a> => Goal<'tcx>}
nop_list_lift!{Clause<'a> => Clause<'tcx>}
nop_list_lift!{Ty<'a> => Ty<'tcx>}
nop_list_lift!{ExistentialPredicate<'a> => ExistentialPredicate<'tcx>}
nop_list_lift!{Predicate<'a> => Predicate<'tcx>}
nop_list_lift!{CanonicalVarInfo => CanonicalVarInfo}
nop_list_lift!{ProjectionKind => ProjectionKind}

// this is the impl for `&'a InternalSubsts<'a>`
nop_list_lift!{Kind<'a> => Kind<'tcx>}

pub mod tls {
    use super::{GlobalCtxt, TyCtxt, ptr_eq};

    use std::fmt;
    use std::mem;
    use syntax_pos;
    use crate::ty::query;
    use errors::{Diagnostic, TRACK_DIAGNOSTICS};
    use rustc_data_structures::OnDrop;
    use rustc_data_structures::sync::{self, Lrc, Lock};
    use rustc_data_structures::thin_vec::ThinVec;
    use crate::dep_graph::TaskDeps;

    #[cfg(not(parallel_compiler))]
    use std::cell::Cell;

    #[cfg(parallel_compiler)]
    use rustc_rayon_core as rayon_core;

    /// This is the implicit state of rustc. It contains the current
    /// TyCtxt and query. It is updated when creating a local interner or
    /// executing a new query. Whenever there's a TyCtxt value available
    /// you should also have access to an ImplicitCtxt through the functions
    /// in this module.
    #[derive(Clone)]
    pub struct ImplicitCtxt<'a, 'tcx> {
        /// The current TyCtxt. Initially created by `enter_global` and updated
        /// by `enter_local` with a new local interner
        pub tcx: TyCtxt<'tcx>,

        /// The current query job, if any. This is updated by JobOwner::start in
        /// ty::query::plumbing when executing a query
        pub query: Option<Lrc<query::QueryJob<'tcx>>>,

        /// Where to store diagnostics for the current query job, if any.
        /// This is updated by JobOwner::start in ty::query::plumbing when executing a query
        pub diagnostics: Option<&'a Lock<ThinVec<Diagnostic>>>,

        /// Used to prevent layout from recursing too deeply.
        pub layout_depth: usize,

        /// The current dep graph task. This is used to add dependencies to queries
        /// when executing them
        pub task_deps: Option<&'a Lock<TaskDeps>>,
    }

    /// Sets Rayon's thread local variable which is preserved for Rayon jobs
    /// to `value` during the call to `f`. It is restored to its previous value after.
    /// This is used to set the pointer to the new ImplicitCtxt.
    #[cfg(parallel_compiler)]
    #[inline]
    fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
        rayon_core::tlv::with(value, f)
    }

    /// Gets Rayon's thread local variable which is preserved for Rayon jobs.
    /// This is used to get the pointer to the current ImplicitCtxt.
    #[cfg(parallel_compiler)]
    #[inline]
    fn get_tlv() -> usize {
        rayon_core::tlv::get()
    }

    #[cfg(not(parallel_compiler))]
    thread_local! {
        /// A thread local variable which stores a pointer to the current ImplicitCtxt.
        static TLV: Cell<usize> = Cell::new(0);
    }

    /// Sets TLV to `value` during the call to `f`.
    /// It is restored to its previous value after.
    /// This is used to set the pointer to the new ImplicitCtxt.
    #[cfg(not(parallel_compiler))]
    #[inline]
    fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
        let old = get_tlv();
        let _reset = OnDrop(move || TLV.with(|tlv| tlv.set(old)));
        TLV.with(|tlv| tlv.set(value));
        f()
    }

    /// This is used to get the pointer to the current ImplicitCtxt.
    #[cfg(not(parallel_compiler))]
    fn get_tlv() -> usize {
        TLV.with(|tlv| tlv.get())
    }

    /// This is a callback from libsyntax as it cannot access the implicit state
    /// in librustc otherwise
    fn span_debug(span: syntax_pos::Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_opt(|tcx| {
            if let Some(tcx) = tcx {
                write!(f, "{}", tcx.sess.source_map().span_to_string(span))
            } else {
                syntax_pos::default_span_debug(span, f)
            }
        })
    }

    /// This is a callback from libsyntax as it cannot access the implicit state
    /// in librustc otherwise. It is used to when diagnostic messages are
    /// emitted and stores them in the current query, if there is one.
    fn track_diagnostic(diagnostic: &Diagnostic) {
        with_context_opt(|icx| {
            if let Some(icx) = icx {
                if let Some(ref diagnostics) = icx.diagnostics {
                    let mut diagnostics = diagnostics.lock();
                    diagnostics.extend(Some(diagnostic.clone()));
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
    #[inline]
    pub fn enter_context<'a, 'tcx, F, R>(context: &ImplicitCtxt<'a, 'tcx>, f: F) -> R
    where
        F: FnOnce(&ImplicitCtxt<'a, 'tcx>) -> R,
    {
        set_tlv(context as *const _ as usize, || {
            f(&context)
        })
    }

    /// Enters GlobalCtxt by setting up libsyntax callbacks and
    /// creating a initial TyCtxt and ImplicitCtxt.
    /// This happens once per rustc session and TyCtxts only exists
    /// inside the `f` function.
    pub fn enter_global<'tcx, F, R>(gcx: &'tcx GlobalCtxt<'tcx>, f: F) -> R
    where
        F: FnOnce(TyCtxt<'tcx>) -> R,
    {
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
        };
        let icx = ImplicitCtxt {
            tcx,
            query: None,
            diagnostics: None,
            layout_depth: 0,
            task_deps: None,
        };
        enter_context(&icx, |_| {
            f(tcx)
        })
    }

    scoped_thread_local! {
        /// Stores a pointer to the GlobalCtxt if one is available.
        /// This is used to access the GlobalCtxt in the deadlock handler given to Rayon.
        pub static GCX_PTR: Lock<usize>
    }

    /// Creates a TyCtxt and ImplicitCtxt based on the GCX_PTR thread local.
    /// This is used in the deadlock handler.
    pub unsafe fn with_global<F, R>(f: F) -> R
    where
        F: for<'tcx> FnOnce(TyCtxt<'tcx>) -> R,
    {
        let gcx = GCX_PTR.with(|lock| *lock.lock());
        assert!(gcx != 0);
        let gcx = &*(gcx as *const GlobalCtxt<'_>);
        let tcx = TyCtxt {
            gcx,
        };
        let icx = ImplicitCtxt {
            query: None,
            diagnostics: None,
            tcx,
            layout_depth: 0,
            task_deps: None,
        };
        enter_context(&icx, |_| f(tcx))
    }

    /// Allows access to the current ImplicitCtxt in a closure if one is available
    #[inline]
    pub fn with_context_opt<F, R>(f: F) -> R
    where
        F: for<'a, 'tcx> FnOnce(Option<&ImplicitCtxt<'a, 'tcx>>) -> R,
    {
        let context = get_tlv();
        if context == 0 {
            f(None)
        } else {
            // We could get a ImplicitCtxt pointer from another thread.
            // Ensure that ImplicitCtxt is Sync
            sync::assert_sync::<ImplicitCtxt<'_, '_>>();

            unsafe { f(Some(&*(context as *const ImplicitCtxt<'_, '_>))) }
        }
    }

    /// Allows access to the current ImplicitCtxt.
    /// Panics if there is no ImplicitCtxt available
    #[inline]
    pub fn with_context<F, R>(f: F) -> R
    where
        F: for<'a, 'tcx> FnOnce(&ImplicitCtxt<'a, 'tcx>) -> R,
    {
        with_context_opt(|opt_context| f(opt_context.expect("no ImplicitCtxt stored in tls")))
    }

    /// Allows access to the current ImplicitCtxt whose tcx field has the same global
    /// interner as the tcx argument passed in. This means the closure is given an ImplicitCtxt
    /// with the same 'tcx lifetime as the TyCtxt passed in.
    /// This will panic if you pass it a TyCtxt which has a different global interner from
    /// the current ImplicitCtxt's tcx field.
    #[inline]
    pub fn with_related_context<'tcx, F, R>(tcx: TyCtxt<'tcx>, f: F) -> R
    where
        F: FnOnce(&ImplicitCtxt<'_, 'tcx>) -> R,
    {
        with_context(|context| {
            unsafe {
                assert!(ptr_eq(context.tcx.gcx, tcx.gcx));
                let context: &ImplicitCtxt<'_, '_> = mem::transmute(context);
                f(context)
            }
        })
    }

    /// Allows access to the TyCtxt in the current ImplicitCtxt.
    /// Panics if there is no ImplicitCtxt available
    #[inline]
    pub fn with<F, R>(f: F) -> R
    where
        F: for<'tcx> FnOnce(TyCtxt<'tcx>) -> R,
    {
        with_context(|context| f(context.tcx))
    }

    /// Allows access to the TyCtxt in the current ImplicitCtxt.
    /// The closure is passed None if there is no ImplicitCtxt available
    #[inline]
    pub fn with_opt<F, R>(f: F) -> R
    where
        F: for<'tcx> FnOnce(Option<TyCtxt<'tcx>>) -> R,
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
            use crate::ty::{self, TyCtxt};
            use crate::ty::context::Interned;

            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                lt_infer: usize,
                ty_infer: usize,
                ct_infer: usize,
                all_infer: usize,
            }

            pub fn go(tcx: TyCtxt<'_>) {
                let mut total = DebugStat {
                    total: 0,
                    lt_infer: 0,
                    ty_infer: 0,
                    ct_infer: 0,
                    all_infer: 0,
                };
                $(let mut $variant = total;)*

                for &Interned(t) in tcx.interners.type_.borrow().keys() {
                    let variant = match t.sty {
                        ty::Bool | ty::Char | ty::Int(..) | ty::Uint(..) |
                            ty::Float(..) | ty::Str | ty::Never => continue,
                        ty::Error => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let lt = t.flags.intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags.intersects(ty::TypeFlags::HAS_TY_INFER);
                    let ct = t.flags.intersects(ty::TypeFlags::HAS_CT_INFER);

                    variant.total += 1;
                    total.total += 1;
                    if lt { total.lt_infer += 1; variant.lt_infer += 1 }
                    if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                    if ct { total.ct_infer += 1; variant.ct_infer += 1 }
                    if lt && ty && ct { total.all_infer += 1; variant.all_infer += 1 }
                }
                println!("Ty interner             total           ty lt ct all");
                $(println!("    {:18}: {uses:6} {usespc:4.1}%, \
                            {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    stringify!($variant),
                    uses = $variant.total,
                    usespc = $variant.total as f64 * 100.0 / total.total as f64,
                    ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = $variant.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = $variant.ct_infer as f64 * 100.0  / total.total as f64,
                    all = $variant.all_infer as f64 * 100.0  / total.total as f64);
                )*
                println!("                  total {uses:6}        \
                          {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    uses = total.total,
                    ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = total.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = total.ct_infer as f64 * 100.0  / total.total as f64,
                    all = total.all_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($ctxt)
    }}
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn print_debug_stats(self) {
        sty_debug_print!(
            self,
            Adt, Array, Slice, RawPtr, Ref, FnDef, FnPtr, Placeholder,
            Generator, GeneratorWitness, Dynamic, Closure, Tuple, Bound,
            Param, Infer, UnnormalizedProjection, Projection, Opaque, Foreign);

        println!("InternalSubsts interner: #{}", self.interners.substs.borrow().len());
        println!("Region interner: #{}", self.interners.region.borrow().len());
        println!("Stability interner: #{}", self.stability_interner.borrow().len());
        println!("Allocation interner: #{}", self.allocation_interner.borrow().len());
        println!("Layout interner: #{}", self.layout_interner.borrow().len());
    }
}


/// An entry in an interner.
struct Interned<'tcx, T: ?Sized>(&'tcx T);

impl<'tcx, T: 'tcx+?Sized> Clone for Interned<'tcx, T> {
    fn clone(&self) -> Self {
        Interned(self.0)
    }
}
impl<'tcx, T: 'tcx+?Sized> Copy for Interned<'tcx, T> {}

// N.B., an `Interned<Ty>` compares and hashes as a sty.
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

impl<'tcx> Borrow<TyKind<'tcx>> for Interned<'tcx, TyS<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a TyKind<'tcx> {
        &self.0.sty
    }
}

// N.B., an `Interned<List<T>>` compares and hashes as its elements.
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

impl<'tcx> Borrow<[Ty<'tcx>]> for Interned<'tcx, List<Ty<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Ty<'tcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<[CanonicalVarInfo]> for Interned<'tcx, List<CanonicalVarInfo>> {
    fn borrow(&self) -> &[CanonicalVarInfo] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<[Kind<'tcx>]> for Interned<'tcx, InternalSubsts<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a [Kind<'tcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<[ProjectionKind]>
    for Interned<'tcx, List<ProjectionKind>> {
    fn borrow(&self) -> &[ProjectionKind] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<RegionKind> for Interned<'tcx, RegionKind> {
    fn borrow(&self) -> &RegionKind {
        &self.0
    }
}

impl<'tcx> Borrow<GoalKind<'tcx>> for Interned<'tcx, GoalKind<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a GoalKind<'tcx> {
        &self.0
    }
}

impl<'tcx> Borrow<[ExistentialPredicate<'tcx>]>
    for Interned<'tcx, List<ExistentialPredicate<'tcx>>>
{
    fn borrow<'a>(&'a self) -> &'a [ExistentialPredicate<'tcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<[Predicate<'tcx>]> for Interned<'tcx, List<Predicate<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Predicate<'tcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<Const<'tcx>> for Interned<'tcx, Const<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a Const<'tcx> {
        &self.0
    }
}

impl<'tcx> Borrow<[Clause<'tcx>]> for Interned<'tcx, List<Clause<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Clause<'tcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<[Goal<'tcx>]> for Interned<'tcx, List<Goal<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Goal<'tcx>] {
        &self.0[..]
    }
}

macro_rules! intern_method {
    ($lt_tcx:tt, $name:ident: $method:ident($alloc:ty,
                                            $alloc_method:expr,
                                            $alloc_to_key:expr) -> $ty:ty) => {
        impl<$lt_tcx> TyCtxt<$lt_tcx> {
            pub fn $method(self, v: $alloc) -> &$lt_tcx $ty {
                let key = ($alloc_to_key)(&v);

                self.interners.$name.borrow_mut().intern_ref(key, || {
                    Interned($alloc_method(&self.interners.arena, v))

                }).0
            }
        }
    }
}

macro_rules! direct_interners {
    ($lt_tcx:tt, $($name:ident: $method:ident($ty:ty)),+) => {
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
                           |x| x) -> $ty);)+
    }
}

pub fn keep_local<'tcx, T: ty::TypeFoldable<'tcx>>(x: &T) -> bool {
    x.has_type_flags(ty::TypeFlags::KEEP_IN_LOCAL_TCX)
}

direct_interners!('tcx,
    region: mk_region(RegionKind),
    goal: mk_goal(GoalKind<'tcx>),
    const_: mk_const(Const<'tcx>)
);

macro_rules! slice_interners {
    ($($field:ident: $method:ident($ty:ty)),+) => (
        $(intern_method!( 'tcx, $field: $method(
            &[$ty],
            |a, v| List::from_arena(a, v),
            Deref::deref) -> List<$ty>);)+
    );
}

slice_interners!(
    existential_predicates: _intern_existential_predicates(ExistentialPredicate<'tcx>),
    predicates: _intern_predicates(Predicate<'tcx>),
    type_list: _intern_type_list(Ty<'tcx>),
    substs: _intern_substs(Kind<'tcx>),
    clauses: _intern_clauses(Clause<'tcx>),
    goal_list: _intern_goals(Goal<'tcx>),
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
        Deref::deref
    ) -> List<CanonicalVarInfo>
}

impl<'tcx> TyCtxt<'tcx> {
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
    /// e.g., if we have a sig with `Fn<(u32, i32)>` then you would get
    /// a `fn(u32, i32)`.
    /// `unsafety` determines the unsafety of the `fn` type. If you pass
    /// `hir::Unsafety::Unsafe` in the previous example, then you would get
    /// an `unsafe fn (u32, i32)`.
    /// It cannot convert a closure that requires unsafe.
    pub fn coerce_closure_fn_ty(self, sig: PolyFnSig<'tcx>, unsafety: hir::Unsafety) -> Ty<'tcx> {
        let converted_sig = sig.map_bound(|s| {
            let params_iter = match s.inputs()[0].sty {
                ty::Tuple(params) => {
                    params.into_iter().map(|k| k.expect_ty())
                }
                _ => bug!(),
            };
            self.mk_fn_sig(
                params_iter,
                s.output(),
                s.c_variadic,
                unsafety,
                abi::Abi::Rust,
            )
        });

        self.mk_fn_ptr(converted_sig)
    }

    #[inline]
    pub fn mk_ty(&self, st: TyKind<'tcx>) -> Ty<'tcx> {
        self.interners.intern_ty(st)
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

    #[inline]
    pub fn mk_str(self) -> Ty<'tcx> {
        self.mk_ty(Str)
    }

    #[inline]
    pub fn mk_static_str(self) -> Ty<'tcx> {
        self.mk_imm_ref(self.lifetimes.re_static, self.mk_str())
    }

    #[inline]
    pub fn mk_adt(self, def: &'tcx AdtDef, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(Adt(def, substs))
    }

    #[inline]
    pub fn mk_foreign(self, def_id: DefId) -> Ty<'tcx> {
        self.mk_ty(Foreign(def_id))
    }

    pub fn mk_box(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(lang_items::OwnedBoxLangItem);
        let adt_def = self.adt_def(def_id);
        let substs = InternalSubsts::for_item(self, def_id, |param, substs| {
            match param.kind {
                GenericParamDefKind::Lifetime |
                GenericParamDefKind::Const => {
                    bug!()
                }
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

    #[inline]
    pub fn mk_ptr(self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(RawPtr(tm))
    }

    #[inline]
    pub fn mk_ref(self, r: Region<'tcx>, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Ref(r, tm.ty, tm.mutbl))
    }

    #[inline]
    pub fn mk_mut_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    #[inline]
    pub fn mk_imm_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    #[inline]
    pub fn mk_mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    #[inline]
    pub fn mk_imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    #[inline]
    pub fn mk_nil_ptr(self) -> Ty<'tcx> {
        self.mk_imm_ptr(self.mk_unit())
    }

    #[inline]
    pub fn mk_array(self, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        self.mk_ty(Array(ty, ty::Const::from_usize(self.global_tcx(), n)))
    }

    #[inline]
    pub fn mk_slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Slice(ty))
    }

    #[inline]
    pub fn intern_tup(self, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        let kinds: Vec<_> = ts.into_iter().map(|&t| Kind::from(t)).collect();
        self.mk_ty(Tuple(self.intern_substs(&kinds)))
    }

    pub fn mk_tup<I: InternAs<[Ty<'tcx>], Ty<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|ts| {
            let kinds: Vec<_> = ts.into_iter().map(|&t| Kind::from(t)).collect();
            self.mk_ty(Tuple(self.intern_substs(&kinds)))
        })
    }

    #[inline]
    pub fn mk_unit(self) -> Ty<'tcx> {
        self.types.unit
    }

    #[inline]
    pub fn mk_diverging_default(self) -> Ty<'tcx> {
        if self.features().never_type {
            self.types.never
        } else {
            self.intern_tup(&[])
        }
    }

    #[inline]
    pub fn mk_bool(self) -> Ty<'tcx> {
        self.mk_ty(Bool)
    }

    #[inline]
    pub fn mk_fn_def(self, def_id: DefId,
                     substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnDef(def_id, substs))
    }

    #[inline]
    pub fn mk_fn_ptr(self, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnPtr(fty))
    }

    #[inline]
    pub fn mk_dynamic(
        self,
        obj: ty::Binder<&'tcx List<ExistentialPredicate<'tcx>>>,
        reg: ty::Region<'tcx>
    ) -> Ty<'tcx> {
        self.mk_ty(Dynamic(obj, reg))
    }

    #[inline]
    pub fn mk_projection(self,
                         item_def_id: DefId,
                         substs: SubstsRef<'tcx>)
        -> Ty<'tcx> {
            self.mk_ty(Projection(ProjectionTy {
                item_def_id,
                substs,
            }))
        }

    #[inline]
    pub fn mk_closure(self, closure_id: DefId, closure_substs: ClosureSubsts<'tcx>)
                      -> Ty<'tcx> {
        self.mk_ty(Closure(closure_id, closure_substs))
    }

    #[inline]
    pub fn mk_generator(self,
                        id: DefId,
                        generator_substs: GeneratorSubsts<'tcx>,
                        movability: hir::GeneratorMovability)
                        -> Ty<'tcx> {
        self.mk_ty(Generator(id, generator_substs, movability))
    }

    #[inline]
    pub fn mk_generator_witness(self, types: ty::Binder<&'tcx List<Ty<'tcx>>>) -> Ty<'tcx> {
        self.mk_ty(GeneratorWitness(types))
    }

    #[inline]
    pub fn mk_ty_var(self, v: TyVid) -> Ty<'tcx> {
        self.mk_ty_infer(TyVar(v))
    }

    #[inline]
    pub fn mk_const_var(self, v: ConstVid<'tcx>, ty: Ty<'tcx>) -> &'tcx Const<'tcx> {
        self.mk_const(ty::Const {
            val: ConstValue::Infer(InferConst::Var(v)),
            ty,
        })
    }

    #[inline]
    pub fn mk_int_var(self, v: IntVid) -> Ty<'tcx> {
        self.mk_ty_infer(IntVar(v))
    }

    #[inline]
    pub fn mk_float_var(self, v: FloatVid) -> Ty<'tcx> {
        self.mk_ty_infer(FloatVar(v))
    }

    #[inline]
    pub fn mk_ty_infer(self, it: InferTy) -> Ty<'tcx> {
        self.mk_ty(Infer(it))
    }

    #[inline]
    pub fn mk_const_infer(
        self,
        ic: InferConst<'tcx>,
        ty: Ty<'tcx>,
    ) -> &'tcx ty::Const<'tcx> {
        self.mk_const(ty::Const {
            val: ConstValue::Infer(ic),
            ty,
        })
    }

    #[inline]
    pub fn mk_ty_param(self, index: u32, name: InternedString) -> Ty<'tcx> {
        self.mk_ty(Param(ParamTy { index, name: name }))
    }

    #[inline]
    pub fn mk_const_param(
        self,
        index: u32,
        name: InternedString,
        ty: Ty<'tcx>
    ) -> &'tcx Const<'tcx> {
        self.mk_const(ty::Const {
            val: ConstValue::Param(ParamConst { index, name }),
            ty,
        })
    }

    #[inline]
    pub fn mk_self_type(self) -> Ty<'tcx> {
        self.mk_ty_param(0, kw::SelfUpper.as_interned_str())
    }

    pub fn mk_param_from_def(self, param: &ty::GenericParamDef) -> Kind<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                self.mk_region(ty::ReEarlyBound(param.to_early_bound_region_data())).into()
            }
            GenericParamDefKind::Type { .. } => self.mk_ty_param(param.index, param.name).into(),
            GenericParamDefKind::Const => {
                self.mk_const_param(param.index, param.name, self.type_of(param.def_id)).into()
            }
        }
    }

    #[inline]
    pub fn mk_opaque(self, def_id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
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

    pub fn intern_projs(self, ps: &[ProjectionKind]) -> &'tcx List<ProjectionKind> {
        if ps.len() == 0 {
            List::empty()
        } else {
            self._intern_projs(ps)
        }
    }

    pub fn intern_canonical_var_infos(self, ts: &[CanonicalVarInfo]) -> CanonicalVarInfos<'tcx> {
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
                        c_variadic: bool,
                        unsafety: hir::Unsafety,
                        abi: abi::Abi)
        -> <I::Item as InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>::Output
        where I: Iterator,
              I::Item: InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>
    {
        inputs.chain(iter::once(output)).intern_with(|xs| ty::FnSig {
            inputs_and_output: self.intern_type_list(xs),
            c_variadic, unsafety, abi
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
                    -> SubstsRef<'tcx>
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
                                              id: hir::HirId,
                                              span: S,
                                              msg: &str,
                                              note: &str) {
        let mut err = self.struct_span_lint_hir(lint, id, span.into(), msg);
        err.note(note);
        err.emit()
    }

    /// Walks upwards from `id` to find a node which might change lint levels with attributes.
    /// It stops at `bound` and just returns it if reached.
    pub fn maybe_lint_level_root_bounded(
        self,
        mut id: hir::HirId,
        bound: hir::HirId,
    ) -> hir::HirId {
        loop {
            if id == bound {
                return bound;
            }
            if lint::maybe_lint_level_root(self, id) {
                return id;
            }
            let next = self.hir().get_parent_node(id);
            if next == id {
                bug!("lint traversal reached the root of the crate");
            }
            id = next;
        }
    }

    pub fn lint_level_at_node(
        self,
        lint: &'static Lint,
        mut id: hir::HirId
    ) -> (lint::Level, lint::LintSource) {
        let sets = self.lint_levels(LOCAL_CRATE);
        loop {
            if let Some(pair) = sets.level_and_source(lint, id, self.sess) {
                return pair
            }
            let next = self.hir().get_parent_node(id);
            if next == id {
                bug!("lint traversal reached the root of the crate");
            }
            id = next;
        }
    }

    pub fn struct_span_lint_hir<S: Into<MultiSpan>>(self,
                                                    lint: &'static Lint,
                                                    hir_id: HirId,
                                                    span: S,
                                                    msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let (level, src) = self.lint_level_at_node(lint, hir_id);
        lint::struct_lint_level(self.sess, lint, level, src, Some(span.into()), msg)
    }

    pub fn struct_lint_node(self, lint: &'static Lint, id: HirId, msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let (level, src) = self.lint_level_at_node(lint, id);
        lint::struct_lint_level(self.sess, lint, level, src, None, msg)
    }

    pub fn in_scope_traits(self, id: HirId) -> Option<&'tcx StableVec<TraitCandidate>> {
        self.in_scope_traits_map(id.owner)
            .and_then(|map| map.get(&id.local_id))
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

    pub fn object_lifetime_defaults(self, id: HirId) -> Option<&'tcx [ObjectLifetimeDefault]> {
        self.object_lifetime_defaults_map(id.owner)
            .and_then(|map| map.get(&id.local_id).map(|v| &**v))
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

// We are comparing types with different invariant lifetimes, so `ptr::eq`
// won't work for us.
fn ptr_eq<T, U>(t: *const T, u: *const U) -> bool {
    t as *const () == u as *const ()
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    providers.in_scope_traits_map = |tcx, id| tcx.gcx.trait_map.get(&id);
    providers.module_exports = |tcx, id| tcx.gcx.export_map.get(&id).map(|v| &v[..]);
    providers.crate_name = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.crate_name
    };
    providers.get_lib_features = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.arena.alloc(middle::lib_features::collect(tcx))
    };
    providers.get_lang_items = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.arena.alloc(middle::lang_items::collect(tcx))
    };
    providers.maybe_unused_trait_import = |tcx, id| {
        tcx.maybe_unused_trait_imports.contains(&id)
    };
    providers.maybe_unused_extern_crates = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        &tcx.maybe_unused_extern_crates[..]
    };
    providers.names_imported_by_glob_use = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        Lrc::new(tcx.glob_map.get(&id).cloned().unwrap_or_default())
    };

    providers.stability_index = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.arena.alloc(stability::Index::new(tcx))
    };
    providers.lookup_stability = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let id = tcx.hir().definitions().def_index_to_hir_id(id.index);
        tcx.stability().local_stability(id)
    };
    providers.lookup_deprecation_entry = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let id = tcx.hir().definitions().def_index_to_hir_id(id.index);
        tcx.stability().local_deprecation_entry(id)
    };
    providers.extern_mod_stmt_cnum = |tcx, id| {
        let id = tcx.hir().as_local_node_id(id).unwrap();
        tcx.cstore.extern_mod_stmt_cnum_untracked(id)
    };
    providers.all_crate_nums = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.arena.alloc_slice(&tcx.cstore.crates_untracked())
    };
    providers.postorder_cnums = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.arena.alloc_slice(&tcx.cstore.postorder_cnums_untracked())
    };
    providers.output_filenames = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.output_filenames.clone()
    };
    providers.features_query = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.arena.alloc(tcx.sess.features_untracked().clone())
    };
    providers.is_panic_runtime = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        attr::contains_name(tcx.hir().krate_attrs(), sym::panic_runtime)
    };
    providers.is_compiler_builtins = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        attr::contains_name(tcx.hir().krate_attrs(), sym::compiler_builtins)
    };
}
