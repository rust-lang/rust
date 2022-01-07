//! Type context book-keeping.

use crate::arena::Arena;
use crate::dep_graph::{DepGraph, DepKind, DepKindStruct};
use crate::hir::place::Place as HirPlace;
use crate::infer::canonical::{Canonical, CanonicalVarInfo, CanonicalVarInfos};
use crate::lint::{struct_lint_level, LintDiagnosticBuilder, LintLevelSource};
use crate::middle::resolve_lifetime::{self, LifetimeScopeForPath, ObjectLifetimeDefault};
use crate::middle::stability;
use crate::mir::interpret::{self, Allocation, ConstValue, Scalar};
use crate::mir::{Body, Field, Local, Place, PlaceElem, ProjectionKind, Promoted};
use crate::thir::Thir;
use crate::traits;
use crate::ty::query::{self, TyCtxtAt};
use crate::ty::subst::{GenericArg, GenericArgKind, InternalSubsts, Subst, SubstsRef, UserSubsts};
use crate::ty::TyKind::*;
use crate::ty::{
    self, AdtDef, AdtKind, Binder, BindingMode, BoundVar, CanonicalPolyFnSig,
    ClosureSizeProfileData, Const, ConstVid, DefIdTree, ExistentialPredicate, FloatTy, FloatVar,
    FloatVid, GenericParamDefKind, InferConst, InferTy, IntTy, IntVar, IntVid, List, ParamConst,
    ParamTy, PolyFnSig, Predicate, PredicateInner, PredicateKind, ProjectionTy, Region, RegionKind,
    ReprOptions, TraitObjectVisitor, Ty, TyKind, TyS, TyVar, TyVid, TypeAndMut, UintTy,
};
use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sharded::{IntoPointer, ShardedHashMap};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::{self, Lock, Lrc, WorkerLocal};
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, LOCAL_CRATE};
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{
    Constness, ExprKind, HirId, ImplItemKind, ItemKind, ItemLocalId, ItemLocalMap, ItemLocalSet,
    Node, TraitCandidate, TraitItemKind,
};
use rustc_index::vec::{Idx, IndexVec};
use rustc_macros::HashStable;
use rustc_middle::mir::FakeReadCause;
use rustc_query_system::ich::{NodeIdHashingMode, StableHashingContext};
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use rustc_session::config::{BorrowckMode, CrateType, OutputFilenames};
use rustc_session::lint::{Level, Lint};
use rustc_session::Limit;
use rustc_session::Session;
use rustc_span::def_id::{DefPathHash, StableCrateId};
use rustc_span::source_map::{MultiSpan, SourceMap};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{Layout, TargetDataLayout, VariantIdx};
use rustc_target::spec::abi;

use smallvec::SmallVec;
use std::any::Any;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::{self, Entry};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::mem;
use std::ops::{Bound, Deref};
use std::sync::Arc;

pub trait OnDiskCache<'tcx>: rustc_data_structures::sync::Sync {
    /// Creates a new `OnDiskCache` instance from the serialized data in `data`.
    fn new(sess: &'tcx Session, data: Mmap, start_pos: usize) -> Self
    where
        Self: Sized;

    fn new_empty(source_map: &'tcx SourceMap) -> Self
    where
        Self: Sized;

    fn drop_serialized_data(&self, tcx: TyCtxt<'tcx>);

    fn serialize(&self, tcx: TyCtxt<'tcx>, encoder: &mut FileEncoder) -> FileEncodeResult;
}

/// A type that is not publicly constructable. This prevents people from making [`TyKind::Error`]s
/// except through the error-reporting functions on a [`tcx`][TyCtxt].
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[derive(TyEncodable, TyDecodable, HashStable)]
pub struct DelaySpanBugEmitted(());

type InternedSet<'tcx, T> = ShardedHashMap<Interned<'tcx, T>, ()>;

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc. are allocated from.
    arena: &'tcx WorkerLocal<Arena<'tcx>>,

    // Specifically use a speedy hash algorithm for these hash sets, since
    // they're accessed quite often.
    type_: InternedSet<'tcx, TyS<'tcx>>,
    type_list: InternedSet<'tcx, List<Ty<'tcx>>>,
    substs: InternedSet<'tcx, InternalSubsts<'tcx>>,
    canonical_var_infos: InternedSet<'tcx, List<CanonicalVarInfo<'tcx>>>,
    region: InternedSet<'tcx, RegionKind>,
    poly_existential_predicates:
        InternedSet<'tcx, List<ty::Binder<'tcx, ExistentialPredicate<'tcx>>>>,
    predicate: InternedSet<'tcx, PredicateInner<'tcx>>,
    predicates: InternedSet<'tcx, List<Predicate<'tcx>>>,
    projs: InternedSet<'tcx, List<ProjectionKind>>,
    place_elems: InternedSet<'tcx, List<PlaceElem<'tcx>>>,
    const_: InternedSet<'tcx, Const<'tcx>>,
    const_allocation: InternedSet<'tcx, Allocation>,
    bound_variable_kinds: InternedSet<'tcx, List<ty::BoundVariableKind>>,
    layout: InternedSet<'tcx, Layout>,
    adt_def: InternedSet<'tcx, AdtDef>,
}

impl<'tcx> CtxtInterners<'tcx> {
    fn new(arena: &'tcx WorkerLocal<Arena<'tcx>>) -> CtxtInterners<'tcx> {
        CtxtInterners {
            arena,
            type_: Default::default(),
            type_list: Default::default(),
            substs: Default::default(),
            region: Default::default(),
            poly_existential_predicates: Default::default(),
            canonical_var_infos: Default::default(),
            predicate: Default::default(),
            predicates: Default::default(),
            projs: Default::default(),
            place_elems: Default::default(),
            const_: Default::default(),
            const_allocation: Default::default(),
            bound_variable_kinds: Default::default(),
            layout: Default::default(),
            adt_def: Default::default(),
        }
    }

    /// Interns a type.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline(never)]
    fn intern_ty(&self, kind: TyKind<'tcx>) -> Ty<'tcx> {
        self.type_
            .intern(kind, |kind| {
                let flags = super::flags::FlagComputation::for_kind(&kind);

                let ty_struct = TyS {
                    kind,
                    flags: flags.flags,
                    outer_exclusive_binder: flags.outer_exclusive_binder,
                };

                Interned(self.arena.alloc(ty_struct))
            })
            .0
    }

    #[inline(never)]
    fn intern_predicate(
        &self,
        kind: Binder<'tcx, PredicateKind<'tcx>>,
    ) -> &'tcx PredicateInner<'tcx> {
        self.predicate
            .intern(kind, |kind| {
                let flags = super::flags::FlagComputation::for_predicate(kind);

                let predicate_struct = PredicateInner {
                    kind,
                    flags: flags.flags,
                    outer_exclusive_binder: flags.outer_exclusive_binder,
                };

                Interned(self.arena.alloc(predicate_struct))
            })
            .0
    }
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
    pub str_: Ty<'tcx>,
    pub never: Ty<'tcx>,
    pub self_param: Ty<'tcx>,

    /// Dummy type used for the `Self` of a `TraitRef` created for converting
    /// a trait object, and which gets removed in `ExistentialTraitRef`.
    /// This type must not appear anywhere in other converted types.
    pub trait_object_dummy_self: Ty<'tcx>,
}

pub struct CommonLifetimes<'tcx> {
    /// `ReEmpty` in the root universe.
    pub re_root_empty: Region<'tcx>,

    /// `ReStatic`
    pub re_static: Region<'tcx>,

    /// Erased region, used after type-checking
    pub re_erased: Region<'tcx>,
}

pub struct CommonConsts<'tcx> {
    pub unit: &'tcx Const<'tcx>,
}

pub struct LocalTableInContext<'a, V> {
    hir_owner: LocalDefId,
    data: &'a ItemLocalMap<V>,
}

/// Validate that the given HirId (respectively its `local_id` part) can be
/// safely used as a key in the maps of a TypeckResults. For that to be
/// the case, the HirId must have the same `owner` as all the other IDs in
/// this table (signified by `hir_owner`). Otherwise the HirId
/// would be in a different frame of reference and using its `local_id`
/// would result in lookup errors, or worse, in silently wrong data being
/// stored/returned.
#[inline]
fn validate_hir_id_for_typeck_results(hir_owner: LocalDefId, hir_id: hir::HirId) {
    if hir_id.owner != hir_owner {
        invalid_hir_id_for_typeck_results(hir_owner, hir_id);
    }
}

#[cold]
#[inline(never)]
fn invalid_hir_id_for_typeck_results(hir_owner: LocalDefId, hir_id: hir::HirId) {
    ty::tls::with(|tcx| {
        bug!(
            "node {} with HirId::owner {:?} cannot be placed in TypeckResults with hir_owner {:?}",
            tcx.hir().node_to_string(hir_id),
            hir_id.owner,
            hir_owner
        )
    });
}

impl<'a, V> LocalTableInContext<'a, V> {
    pub fn contains_key(&self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.contains_key(&id.local_id)
    }

    pub fn get(&self, id: hir::HirId) -> Option<&V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
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
    hir_owner: LocalDefId,
    data: &'a mut ItemLocalMap<V>,
}

impl<'a, V> LocalTableInContextMut<'a, V> {
    pub fn get_mut(&mut self, id: hir::HirId) -> Option<&mut V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.get_mut(&id.local_id)
    }

    pub fn entry(&mut self, id: hir::HirId) -> Entry<'_, hir::ItemLocalId, V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.entry(id.local_id)
    }

    pub fn insert(&mut self, id: hir::HirId, val: V) -> Option<V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.insert(id.local_id, val)
    }

    pub fn remove(&mut self, id: hir::HirId) -> Option<V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.remove(&id.local_id)
    }
}

/// Whenever a value may be live across a generator yield, the type of that value winds up in the
/// `GeneratorInteriorTypeCause` struct. This struct adds additional information about such
/// captured types that can be useful for diagnostics. In particular, it stores the span that
/// caused a given type to be recorded, along with the scope that enclosed the value (which can
/// be used to find the await that the value is live across).
///
/// For example:
///
/// ```ignore (pseudo-Rust)
/// async move {
///     let x: T = expr;
///     foo.await
///     ...
/// }
/// ```
///
/// Here, we would store the type `T`, the span of the value `x`, the "scope-span" for
/// the scope that contains `x`, the expr `T` evaluated from, and the span of `foo.await`.
#[derive(TyEncodable, TyDecodable, Clone, Debug, Eq, Hash, PartialEq, HashStable)]
#[derive(TypeFoldable)]
pub struct GeneratorInteriorTypeCause<'tcx> {
    /// Type of the captured binding.
    pub ty: Ty<'tcx>,
    /// Span of the binding that was captured.
    pub span: Span,
    /// Span of the scope of the captured binding.
    pub scope_span: Option<Span>,
    /// Span of `.await` or `yield` expression.
    pub yield_span: Span,
    /// Expr which the type evaluated from.
    pub expr: Option<hir::HirId>,
}

#[derive(TyEncodable, TyDecodable, Debug)]
pub struct TypeckResults<'tcx> {
    /// The `HirId::owner` all `ItemLocalId`s in this table are relative to.
    pub hir_owner: LocalDefId,

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
    /// for later usage in THIR lowering. For example,
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
    /// <https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md#definitions>
    pat_adjustments: ItemLocalMap<Vec<Ty<'tcx>>>,

    /// Records the reasons that we picked the kind of each closure;
    /// not all closures are present in the map.
    closure_kind_origins: ItemLocalMap<(Span, HirPlace<'tcx>)>,

    /// For each fn, records the "liberated" types of its arguments
    /// and return type. Liberated means that all bound regions
    /// (including late-bound regions) are replaced with free
    /// equivalents. This table is not used in codegen (since regions
    /// are erased there) and hence is not serialized to metadata.
    ///
    /// This table also contains the "revealed" values for any `impl Trait`
    /// that appear in the signature and whose values are being inferred
    /// by this function.
    ///
    /// # Example
    ///
    /// ```rust
    /// fn foo(x: &u32) -> impl Debug { *x }
    /// ```
    ///
    /// The function signature here would be:
    ///
    /// ```
    /// for<'a> fn(&'a u32) -> Foo
    /// ```
    ///
    /// where `Foo` is an opaque type created for this function.
    ///
    ///
    /// The *liberated* form of this would be
    ///
    /// ```
    /// fn(&'a u32) -> u32
    /// ```
    ///
    /// Note that `'a` is not bound (it would be an `ReFree`) and
    /// that the `Foo` opaque type is replaced by its hidden type.
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
    pub used_trait_imports: Lrc<FxHashSet<LocalDefId>>,

    /// If any errors occurred while type-checking this body,
    /// this field will be set to `Some(ErrorReported)`.
    pub tainted_by_errors: Option<ErrorReported>,

    /// All the opaque types that are restricted to concrete types
    /// by this function.
    pub concrete_opaque_types: FxHashSet<DefId>,

    /// Tracks the minimum captures required for a closure;
    /// see `MinCaptureInformationMap` for more details.
    pub closure_min_captures: ty::MinCaptureInformationMap<'tcx>,

    /// Tracks the fake reads required for a closure and the reason for the fake read.
    /// When performing pattern matching for closures, there are times we don't end up
    /// reading places that are mentioned in a closure (because of _ patterns). However,
    /// to ensure the places are initialized, we introduce fake reads.
    /// Consider these two examples:
    /// ``` (discriminant matching with only wildcard arm)
    /// let x: u8;
    /// let c = || match x { _ => () };
    /// ```
    /// In this example, we don't need to actually read/borrow `x` in `c`, and so we don't
    /// want to capture it. However, we do still want an error here, because `x` should have
    /// to be initialized at the point where c is created. Therefore, we add a "fake read"
    /// instead.
    /// ``` (destructured assignments)
    /// let c = || {
    ///     let (t1, t2) = t;
    /// }
    /// ```
    /// In the second example, we capture the disjoint fields of `t` (`t.0` & `t.1`), but
    /// we never capture `t`. This becomes an issue when we build MIR as we require
    /// information on `t` in order to create place `t.0` and `t.1`. We can solve this
    /// issue by fake reading `t`.
    pub closure_fake_reads: FxHashMap<DefId, Vec<(HirPlace<'tcx>, FakeReadCause, hir::HirId)>>,

    /// Stores the type, expression, span and optional scope span of all types
    /// that are live across the yield of this generator (if a generator).
    pub generator_interior_types: ty::Binder<'tcx, Vec<GeneratorInteriorTypeCause<'tcx>>>,

    /// We sometimes treat byte string literals (which are of type `&[u8; N]`)
    /// as `&[u8]`, depending on the pattern  in which they are used.
    /// This hashset records all instances where we behave
    /// like this to allow `const_to_pat` to reliably handle this situation.
    pub treat_byte_string_as_slice: ItemLocalSet,

    /// Contains the data for evaluating the effect of feature `capture_disjoint_fields`
    /// on closure size.
    pub closure_size_eval: FxHashMap<DefId, ClosureSizeProfileData<'tcx>>,
}

impl<'tcx> TypeckResults<'tcx> {
    pub fn new(hir_owner: LocalDefId) -> TypeckResults<'tcx> {
        TypeckResults {
            hir_owner,
            type_dependent_defs: Default::default(),
            field_indices: Default::default(),
            user_provided_types: Default::default(),
            user_provided_sigs: Default::default(),
            node_types: Default::default(),
            node_substs: Default::default(),
            adjustments: Default::default(),
            pat_binding_modes: Default::default(),
            pat_adjustments: Default::default(),
            closure_kind_origins: Default::default(),
            liberated_fn_sigs: Default::default(),
            fru_field_types: Default::default(),
            coercion_casts: Default::default(),
            used_trait_imports: Lrc::new(Default::default()),
            tainted_by_errors: None,
            concrete_opaque_types: Default::default(),
            closure_min_captures: Default::default(),
            closure_fake_reads: Default::default(),
            generator_interior_types: ty::Binder::dummy(Default::default()),
            treat_byte_string_as_slice: Default::default(),
            closure_size_eval: Default::default(),
        }
    }

    /// Returns the final resolution of a `QPath` in an `Expr` or `Pat` node.
    pub fn qpath_res(&self, qpath: &hir::QPath<'_>, id: hir::HirId) -> Res {
        match *qpath {
            hir::QPath::Resolved(_, ref path) => path.res,
            hir::QPath::TypeRelative(..) | hir::QPath::LangItem(..) => self
                .type_dependent_def(id)
                .map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)),
        }
    }

    pub fn type_dependent_defs(
        &self,
    ) -> LocalTableInContext<'_, Result<(DefKind, DefId), ErrorReported>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.type_dependent_defs }
    }

    pub fn type_dependent_def(&self, id: HirId) -> Option<(DefKind, DefId)> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.type_dependent_defs.get(&id.local_id).cloned().and_then(|r| r.ok())
    }

    pub fn type_dependent_def_id(&self, id: HirId) -> Option<DefId> {
        self.type_dependent_def(id).map(|(_, def_id)| def_id)
    }

    pub fn type_dependent_defs_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, Result<(DefKind, DefId), ErrorReported>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.type_dependent_defs }
    }

    pub fn field_indices(&self) -> LocalTableInContext<'_, usize> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.field_indices }
    }

    pub fn field_indices_mut(&mut self) -> LocalTableInContextMut<'_, usize> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.field_indices }
    }

    pub fn user_provided_types(&self) -> LocalTableInContext<'_, CanonicalUserType<'tcx>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.user_provided_types }
    }

    pub fn user_provided_types_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, CanonicalUserType<'tcx>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.user_provided_types }
    }

    pub fn node_types(&self) -> LocalTableInContext<'_, Ty<'tcx>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.node_types }
    }

    pub fn node_types_mut(&mut self) -> LocalTableInContextMut<'_, Ty<'tcx>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.node_types }
    }

    pub fn node_type(&self, id: hir::HirId) -> Ty<'tcx> {
        self.node_type_opt(id).unwrap_or_else(|| {
            bug!("node_type: no type for node `{}`", tls::with(|tcx| tcx.hir().node_to_string(id)))
        })
    }

    pub fn node_type_opt(&self, id: hir::HirId) -> Option<Ty<'tcx>> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.node_types.get(&id.local_id).cloned()
    }

    pub fn node_substs_mut(&mut self) -> LocalTableInContextMut<'_, SubstsRef<'tcx>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.node_substs }
    }

    pub fn node_substs(&self, id: hir::HirId) -> SubstsRef<'tcx> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.node_substs.get(&id.local_id).cloned().unwrap_or_else(|| InternalSubsts::empty())
    }

    pub fn node_substs_opt(&self, id: hir::HirId) -> Option<SubstsRef<'tcx>> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.node_substs.get(&id.local_id).cloned()
    }

    // Returns the type of a pattern as a monotype. Like @expr_ty, this function
    // doesn't provide type parameter substitutions.
    pub fn pat_ty(&self, pat: &hir::Pat<'_>) -> Ty<'tcx> {
        self.node_type(pat.hir_id)
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
    pub fn expr_ty(&self, expr: &hir::Expr<'_>) -> Ty<'tcx> {
        self.node_type(expr.hir_id)
    }

    pub fn expr_ty_opt(&self, expr: &hir::Expr<'_>) -> Option<Ty<'tcx>> {
        self.node_type_opt(expr.hir_id)
    }

    pub fn adjustments(&self) -> LocalTableInContext<'_, Vec<ty::adjustment::Adjustment<'tcx>>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.adjustments }
    }

    pub fn adjustments_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, Vec<ty::adjustment::Adjustment<'tcx>>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.adjustments }
    }

    pub fn expr_adjustments(&self, expr: &hir::Expr<'_>) -> &[ty::adjustment::Adjustment<'tcx>] {
        validate_hir_id_for_typeck_results(self.hir_owner, expr.hir_id);
        self.adjustments.get(&expr.hir_id.local_id).map_or(&[], |a| &a[..])
    }

    /// Returns the type of `expr`, considering any `Adjustment`
    /// entry recorded for that expression.
    pub fn expr_ty_adjusted(&self, expr: &hir::Expr<'_>) -> Ty<'tcx> {
        self.expr_adjustments(expr).last().map_or_else(|| self.expr_ty(expr), |adj| adj.target)
    }

    pub fn expr_ty_adjusted_opt(&self, expr: &hir::Expr<'_>) -> Option<Ty<'tcx>> {
        self.expr_adjustments(expr).last().map(|adj| adj.target).or_else(|| self.expr_ty_opt(expr))
    }

    pub fn is_method_call(&self, expr: &hir::Expr<'_>) -> bool {
        // Only paths and method calls/overloaded operators have
        // entries in type_dependent_defs, ignore the former here.
        if let hir::ExprKind::Path(_) = expr.kind {
            return false;
        }

        matches!(self.type_dependent_defs().get(expr.hir_id), Some(Ok((DefKind::AssocFn, _))))
    }

    pub fn extract_binding_mode(&self, s: &Session, id: HirId, sp: Span) -> Option<BindingMode> {
        self.pat_binding_modes().get(id).copied().or_else(|| {
            s.delay_span_bug(sp, "missing binding mode");
            None
        })
    }

    pub fn pat_binding_modes(&self) -> LocalTableInContext<'_, BindingMode> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.pat_binding_modes }
    }

    pub fn pat_binding_modes_mut(&mut self) -> LocalTableInContextMut<'_, BindingMode> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.pat_binding_modes }
    }

    pub fn pat_adjustments(&self) -> LocalTableInContext<'_, Vec<Ty<'tcx>>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.pat_adjustments }
    }

    pub fn pat_adjustments_mut(&mut self) -> LocalTableInContextMut<'_, Vec<Ty<'tcx>>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.pat_adjustments }
    }

    /// For a given closure, returns the iterator of `ty::CapturedPlace`s that are captured
    /// by the closure.
    pub fn closure_min_captures_flattened(
        &self,
        closure_def_id: DefId,
    ) -> impl Iterator<Item = &ty::CapturedPlace<'tcx>> {
        self.closure_min_captures
            .get(&closure_def_id)
            .map(|closure_min_captures| closure_min_captures.values().flat_map(|v| v.iter()))
            .into_iter()
            .flatten()
    }

    pub fn closure_kind_origins(&self) -> LocalTableInContext<'_, (Span, HirPlace<'tcx>)> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.closure_kind_origins }
    }

    pub fn closure_kind_origins_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, (Span, HirPlace<'tcx>)> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.closure_kind_origins }
    }

    pub fn liberated_fn_sigs(&self) -> LocalTableInContext<'_, ty::FnSig<'tcx>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.liberated_fn_sigs }
    }

    pub fn liberated_fn_sigs_mut(&mut self) -> LocalTableInContextMut<'_, ty::FnSig<'tcx>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.liberated_fn_sigs }
    }

    pub fn fru_field_types(&self) -> LocalTableInContext<'_, Vec<Ty<'tcx>>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.fru_field_types }
    }

    pub fn fru_field_types_mut(&mut self) -> LocalTableInContextMut<'_, Vec<Ty<'tcx>>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.fru_field_types }
    }

    pub fn is_coercion_cast(&self, hir_id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, hir_id);
        self.coercion_casts.contains(&hir_id.local_id)
    }

    pub fn set_coercion_cast(&mut self, id: ItemLocalId) {
        self.coercion_casts.insert(id);
    }

    pub fn coercion_casts(&self) -> &ItemLocalSet {
        &self.coercion_casts
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for TypeckResults<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let ty::TypeckResults {
            hir_owner,
            ref type_dependent_defs,
            ref field_indices,
            ref user_provided_types,
            ref user_provided_sigs,
            ref node_types,
            ref node_substs,
            ref adjustments,
            ref pat_binding_modes,
            ref pat_adjustments,
            ref closure_kind_origins,
            ref liberated_fn_sigs,
            ref fru_field_types,
            ref coercion_casts,
            ref used_trait_imports,
            tainted_by_errors,
            ref concrete_opaque_types,
            ref closure_min_captures,
            ref closure_fake_reads,
            ref generator_interior_types,
            ref treat_byte_string_as_slice,
            ref closure_size_eval,
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            hcx.local_def_path_hash(hir_owner);

            type_dependent_defs.hash_stable(hcx, hasher);
            field_indices.hash_stable(hcx, hasher);
            user_provided_types.hash_stable(hcx, hasher);
            user_provided_sigs.hash_stable(hcx, hasher);
            node_types.hash_stable(hcx, hasher);
            node_substs.hash_stable(hcx, hasher);
            adjustments.hash_stable(hcx, hasher);
            pat_binding_modes.hash_stable(hcx, hasher);
            pat_adjustments.hash_stable(hcx, hasher);

            closure_kind_origins.hash_stable(hcx, hasher);
            liberated_fn_sigs.hash_stable(hcx, hasher);
            fru_field_types.hash_stable(hcx, hasher);
            coercion_casts.hash_stable(hcx, hasher);
            used_trait_imports.hash_stable(hcx, hasher);
            tainted_by_errors.hash_stable(hcx, hasher);
            concrete_opaque_types.hash_stable(hcx, hasher);
            closure_min_captures.hash_stable(hcx, hasher);
            closure_fake_reads.hash_stable(hcx, hasher);
            generator_interior_types.hash_stable(hcx, hasher);
            treat_byte_string_as_slice.hash_stable(hcx, hasher);
            closure_size_eval.hash_stable(hcx, hasher);
        })
    }
}

rustc_index::newtype_index! {
    pub struct UserTypeAnnotationIndex {
        derive [HashStable]
        DEBUG_FORMAT = "UserType({})",
        const START_INDEX = 0,
    }
}

/// Mapping of type annotation indices to canonical user type annotations.
pub type CanonicalUserTypeAnnotations<'tcx> =
    IndexVec<UserTypeAnnotationIndex, CanonicalUserTypeAnnotation<'tcx>>;

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, Lift)]
pub struct CanonicalUserTypeAnnotation<'tcx> {
    pub user_ty: CanonicalUserType<'tcx>,
    pub span: Span,
    pub inferred_ty: Ty<'tcx>,
}

/// Canonicalized user type annotation.
pub type CanonicalUserType<'tcx> = Canonical<'tcx, UserType<'tcx>>;

impl<'tcx> CanonicalUserType<'tcx> {
    /// Returns `true` if this represents a substitution of the form `[?0, ?1, ?2]`,
    /// i.e., each thing is mapped to a canonical variable with the same index.
    pub fn is_identity(&self) -> bool {
        match self.value {
            UserType::Ty(_) => false,
            UserType::TypeOf(_, user_substs) => {
                if user_substs.user_self_ty.is_some() {
                    return false;
                }

                iter::zip(user_substs.substs, BoundVar::new(0)..).all(|(kind, cvar)| {
                    match kind.unpack() {
                        GenericArgKind::Type(ty) => match ty.kind() {
                            ty::Bound(debruijn, b) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(*debruijn, ty::INNERMOST);
                                cvar == b.var
                            }
                            _ => false,
                        },

                        GenericArgKind::Lifetime(r) => match r {
                            ty::ReLateBound(debruijn, br) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(*debruijn, ty::INNERMOST);
                                cvar == br.var
                            }
                            _ => false,
                        },

                        GenericArgKind::Const(ct) => match ct.val {
                            ty::ConstKind::Bound(debruijn, b) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(debruijn, ty::INNERMOST);
                                cvar == b
                            }
                            _ => false,
                        },
                    }
                })
            }
        }
    }
}

/// A user-given type annotation attached to a constant. These arise
/// from constants that are named via paths, like `Foo::<A>::new` and
/// so forth.
#[derive(Copy, Clone, Debug, PartialEq, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, Lift)]
pub enum UserType<'tcx> {
    Ty(Ty<'tcx>),

    /// The canonical type is the result of `type_of(def_id)` with the
    /// given substitutions applied.
    TypeOf(DefId, UserSubsts<'tcx>),
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonTypes<'tcx> {
        let mk = |ty| interners.intern_ty(ty);

        CommonTypes {
            unit: mk(Tuple(List::empty())),
            bool: mk(Bool),
            char: mk(Char),
            never: mk(Never),
            isize: mk(Int(ty::IntTy::Isize)),
            i8: mk(Int(ty::IntTy::I8)),
            i16: mk(Int(ty::IntTy::I16)),
            i32: mk(Int(ty::IntTy::I32)),
            i64: mk(Int(ty::IntTy::I64)),
            i128: mk(Int(ty::IntTy::I128)),
            usize: mk(Uint(ty::UintTy::Usize)),
            u8: mk(Uint(ty::UintTy::U8)),
            u16: mk(Uint(ty::UintTy::U16)),
            u32: mk(Uint(ty::UintTy::U32)),
            u64: mk(Uint(ty::UintTy::U64)),
            u128: mk(Uint(ty::UintTy::U128)),
            f32: mk(Float(ty::FloatTy::F32)),
            f64: mk(Float(ty::FloatTy::F64)),
            str_: mk(Str),
            self_param: mk(ty::Param(ty::ParamTy { index: 0, name: kw::SelfUpper })),

            trait_object_dummy_self: mk(Infer(ty::FreshTy(0))),
        }
    }
}

impl<'tcx> CommonLifetimes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonLifetimes<'tcx> {
        let mk = |r| interners.region.intern(r, |r| Interned(interners.arena.alloc(r))).0;

        CommonLifetimes {
            re_root_empty: mk(RegionKind::ReEmpty(ty::UniverseIndex::ROOT)),
            re_static: mk(RegionKind::ReStatic),
            re_erased: mk(RegionKind::ReErased),
        }
    }
}

impl<'tcx> CommonConsts<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>, types: &CommonTypes<'tcx>) -> CommonConsts<'tcx> {
        let mk_const = |c| interners.const_.intern(c, |c| Interned(interners.arena.alloc(c))).0;

        CommonConsts {
            unit: mk_const(ty::Const {
                val: ty::ConstKind::Value(ConstValue::Scalar(Scalar::ZST)),
                ty: types.unit,
            }),
        }
    }
}

// This struct contains information regarding the `ReFree(FreeRegion)` corresponding to a lifetime
// conflict.
#[derive(Debug)]
pub struct FreeRegionInfo {
    // `LocalDefId` corresponding to FreeRegion
    pub def_id: LocalDefId,
    // the bound region corresponding to FreeRegion
    pub boundregion: ty::BoundRegionKind,
    // checks if bound region is in Impl Item
    pub is_impl_item: bool,
}

/// The central data structure of the compiler. It stores references
/// to the various **arenas** and also houses the results of the
/// various **compiler queries** that have been performed. See the
/// [rustc dev guide] for more details.
///
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/ty.html
#[derive(Copy, Clone)]
#[rustc_diagnostic_item = "TyCtxt"]
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
    pub arena: &'tcx WorkerLocal<Arena<'tcx>>,

    interners: CtxtInterners<'tcx>,

    pub sess: &'tcx Session,

    /// This only ever stores a `LintStore` but we don't want a dependency on that type here.
    ///
    /// FIXME(Centril): consider `dyn LintStoreMarker` once
    /// we can upcast to `Any` for some additional type safety.
    pub lint_store: Lrc<dyn Any + sync::Sync + sync::Send>,

    pub dep_graph: DepGraph,

    pub prof: SelfProfilerRef,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    /// Common lifetimes, pre-interned for your convenience.
    pub lifetimes: CommonLifetimes<'tcx>,

    /// Common consts, pre-interned for your convenience.
    pub consts: CommonConsts<'tcx>,

    /// Output of the resolver.
    pub(crate) untracked_resolutions: ty::ResolverOutputs,

    pub(crate) untracked_crate: &'tcx hir::Crate<'tcx>,

    /// This provides access to the incremental compilation on-disk cache for query results.
    /// Do not access this directly. It is only meant to be used by
    /// `DepGraph::try_mark_green()` and the query infrastructure.
    /// This is `None` if we are not incremental compilation mode
    pub on_disk_cache: Option<&'tcx dyn OnDiskCache<'tcx>>,

    pub queries: &'tcx dyn query::QueryEngine<'tcx>,
    pub query_caches: query::QueryCaches<'tcx>,
    query_kinds: &'tcx [DepKindStruct],

    // Internal caches for metadata decoding. No need to track deps on this.
    pub ty_rcache: Lock<FxHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,
    pub pred_rcache: Lock<FxHashMap<ty::CReaderCacheKey, Predicate<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that do not have to do with the parameters in scope.
    /// Merge this with `selection_cache`?
    pub evaluation_cache: traits::EvaluationCache<'tcx>,

    /// The definite name of the current crate after taking into account
    /// attributes, commandline parameters, etc.
    crate_name: Symbol,

    /// Data layout specification for the current target.
    pub data_layout: TargetDataLayout,

    /// `#[stable]` and `#[unstable]` attributes
    stability_interner: ShardedHashMap<&'tcx attr::Stability, ()>,

    /// `#[rustc_const_stable]` and `#[rustc_const_unstable]` attributes
    const_stability_interner: ShardedHashMap<&'tcx attr::ConstStability, ()>,

    /// Stores memory for globals (statics/consts).
    pub(crate) alloc_map: Lock<interpret::AllocMap<'tcx>>,

    output_filenames: Arc<OutputFilenames>,
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn typeck_opt_const_arg(
        self,
        def: ty::WithOptConstParam<LocalDefId>,
    ) -> &'tcx TypeckResults<'tcx> {
        if let Some(param_did) = def.const_param_did {
            self.typeck_const_arg((def.did, param_did))
        } else {
            self.typeck(def.did)
        }
    }

    pub fn alloc_steal_thir(self, thir: Thir<'tcx>) -> &'tcx Steal<Thir<'tcx>> {
        self.arena.alloc(Steal::new(thir))
    }

    pub fn alloc_steal_mir(self, mir: Body<'tcx>) -> &'tcx Steal<Body<'tcx>> {
        self.arena.alloc(Steal::new(mir))
    }

    pub fn alloc_steal_promoted(
        self,
        promoted: IndexVec<Promoted, Body<'tcx>>,
    ) -> &'tcx Steal<IndexVec<Promoted, Body<'tcx>>> {
        self.arena.alloc(Steal::new(promoted))
    }

    pub fn alloc_adt_def(
        self,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, ty::VariantDef>,
        repr: ReprOptions,
    ) -> &'tcx ty::AdtDef {
        self.intern_adt_def(ty::AdtDef::new(self, did, kind, variants, repr))
    }

    /// Allocates a read-only byte or string literal for `mir::interpret`.
    pub fn allocate_bytes(self, bytes: &[u8]) -> interpret::AllocId {
        // Create an allocation that just contains these bytes.
        let alloc = interpret::Allocation::from_bytes_byte_aligned_immutable(bytes);
        let alloc = self.intern_const_alloc(alloc);
        self.create_memory_alloc(alloc)
    }

    // FIXME(eddyb) move to `direct_interners!`.
    pub fn intern_stability(self, stab: attr::Stability) -> &'tcx attr::Stability {
        self.stability_interner.intern(stab, |stab| self.arena.alloc(stab))
    }

    // FIXME(eddyb) move to `direct_interners!`.
    pub fn intern_const_stability(self, stab: attr::ConstStability) -> &'tcx attr::ConstStability {
        self.const_stability_interner.intern(stab, |stab| self.arena.alloc(stab))
    }

    /// Returns a range of the start/end indices specified with the
    /// `rustc_layout_scalar_valid_range` attribute.
    // FIXME(eddyb) this is an awkward spot for this method, maybe move it?
    pub fn layout_scalar_valid_range(self, def_id: DefId) -> (Bound<u128>, Bound<u128>) {
        let attrs = self.get_attrs(def_id);
        let get = |name| {
            let attr = match attrs.iter().find(|a| a.has_name(name)) {
                Some(attr) => attr,
                None => return Bound::Unbounded,
            };
            debug!("layout_scalar_valid_range: attr={:?}", attr);
            if let Some(
                &[
                    ast::NestedMetaItem::Literal(ast::Lit {
                        kind: ast::LitKind::Int(a, _), ..
                    }),
                ],
            ) = attr.meta_item_list().as_deref()
            {
                Bound::Included(a)
            } else {
                self.sess
                    .delay_span_bug(attr.span, "invalid rustc_layout_scalar_valid_range attribute");
                Bound::Unbounded
            }
        };
        (
            get(sym::rustc_layout_scalar_valid_range_start),
            get(sym::rustc_layout_scalar_valid_range_end),
        )
    }

    pub fn lift<T: Lift<'tcx>>(self, value: T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }

    /// Creates a type context and call the closure with a `TyCtxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_global_ctxt(
        s: &'tcx Session,
        lint_store: Lrc<dyn Any + sync::Send + sync::Sync>,
        arena: &'tcx WorkerLocal<Arena<'tcx>>,
        resolutions: ty::ResolverOutputs,
        krate: &'tcx hir::Crate<'tcx>,
        dep_graph: DepGraph,
        on_disk_cache: Option<&'tcx dyn OnDiskCache<'tcx>>,
        queries: &'tcx dyn query::QueryEngine<'tcx>,
        query_kinds: &'tcx [DepKindStruct],
        crate_name: &str,
        output_filenames: OutputFilenames,
    ) -> GlobalCtxt<'tcx> {
        let data_layout = TargetDataLayout::parse(&s.target).unwrap_or_else(|err| {
            s.fatal(&err);
        });
        let interners = CtxtInterners::new(arena);
        let common_types = CommonTypes::new(&interners);
        let common_lifetimes = CommonLifetimes::new(&interners);
        let common_consts = CommonConsts::new(&interners, &common_types);

        GlobalCtxt {
            sess: s,
            lint_store,
            arena,
            interners,
            dep_graph,
            untracked_resolutions: resolutions,
            prof: s.prof.clone(),
            types: common_types,
            lifetimes: common_lifetimes,
            consts: common_consts,
            untracked_crate: krate,
            on_disk_cache,
            queries,
            query_caches: query::QueryCaches::default(),
            query_kinds,
            ty_rcache: Default::default(),
            pred_rcache: Default::default(),
            selection_cache: Default::default(),
            evaluation_cache: Default::default(),
            crate_name: Symbol::intern(crate_name),
            data_layout,
            stability_interner: Default::default(),
            const_stability_interner: Default::default(),
            alloc_map: Lock::new(interpret::AllocMap::new()),
            output_filenames: Arc::new(output_filenames),
        }
    }

    crate fn query_kind(self, k: DepKind) -> &'tcx DepKindStruct {
        &self.query_kinds[k as usize]
    }

    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` to ensure it gets used.
    #[track_caller]
    pub fn ty_error(self) -> Ty<'tcx> {
        self.ty_error_with_message(DUMMY_SP, "TyKind::Error constructed but no error reported")
    }

    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` with the given `msg` to
    /// ensure it gets used.
    #[track_caller]
    pub fn ty_error_with_message<S: Into<MultiSpan>>(self, span: S, msg: &str) -> Ty<'tcx> {
        self.sess.delay_span_bug(span, msg);
        self.mk_ty(Error(DelaySpanBugEmitted(())))
    }

    /// Like `err` but for constants.
    #[track_caller]
    pub fn const_error(self, ty: Ty<'tcx>) -> &'tcx Const<'tcx> {
        self.sess
            .delay_span_bug(DUMMY_SP, "ty::ConstKind::Error constructed but no error reported.");
        self.mk_const(ty::Const { val: ty::ConstKind::Error(DelaySpanBugEmitted(())), ty })
    }

    pub fn consider_optimizing<T: Fn() -> String>(self, msg: T) -> bool {
        let cname = self.crate_name(LOCAL_CRATE);
        self.sess.consider_optimizing(cname.as_str(), msg)
    }

    /// Obtain all lang items of this crate and all dependencies (recursively)
    pub fn lang_items(self) -> &'tcx rustc_hir::lang_items::LanguageItems {
        self.get_lang_items(())
    }

    /// Obtain the given diagnostic item's `DefId`. Use `is_diagnostic_item` if you just want to
    /// compare against another `DefId`, since `is_diagnostic_item` is cheaper.
    pub fn get_diagnostic_item(self, name: Symbol) -> Option<DefId> {
        self.all_diagnostic_items(()).name_to_id.get(&name).copied()
    }

    /// Obtain the diagnostic item's name
    pub fn get_diagnostic_name(self, id: DefId) -> Option<Symbol> {
        self.diagnostic_items(id.krate).id_to_name.get(&id).copied()
    }

    /// Check whether the diagnostic item with the given `name` has the given `DefId`.
    pub fn is_diagnostic_item(self, name: Symbol, did: DefId) -> bool {
        self.diagnostic_items(did.krate).name_to_id.get(&name) == Some(&did)
    }

    pub fn stability(self) -> &'tcx stability::Index<'tcx> {
        self.stability_index(())
    }

    pub fn features(self) -> &'tcx rustc_feature::Features {
        self.features_query(())
    }

    pub fn def_key(self, id: DefId) -> rustc_hir::definitions::DefKey {
        // Accessing the DefKey is ok, since it is part of DefPathHash.
        if let Some(id) = id.as_local() {
            self.untracked_resolutions.definitions.def_key(id)
        } else {
            self.untracked_resolutions.cstore.def_key(id)
        }
    }

    /// Converts a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned `DefPath`).
    ///
    /// Note that if `id` is not local to this crate, the result will
    ///  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> rustc_hir::definitions::DefPath {
        // Accessing the DefPath is ok, since it is part of DefPathHash.
        if let Some(id) = id.as_local() {
            self.untracked_resolutions.definitions.def_path(id)
        } else {
            self.untracked_resolutions.cstore.def_path(id)
        }
    }

    #[inline]
    pub fn def_path_hash(self, def_id: DefId) -> rustc_hir::definitions::DefPathHash {
        // Accessing the DefPathHash is ok, it is incr. comp. stable.
        if let Some(def_id) = def_id.as_local() {
            self.untracked_resolutions.definitions.def_path_hash(def_id)
        } else {
            self.untracked_resolutions.cstore.def_path_hash(def_id)
        }
    }

    #[inline]
    pub fn stable_crate_id(self, crate_num: CrateNum) -> StableCrateId {
        if crate_num == LOCAL_CRATE {
            self.sess.local_stable_crate_id()
        } else {
            self.untracked_resolutions.cstore.stable_crate_id(crate_num)
        }
    }

    /// Maps a StableCrateId to the corresponding CrateNum. This method assumes
    /// that the crate in question has already been loaded by the CrateStore.
    #[inline]
    pub fn stable_crate_id_to_crate_num(self, stable_crate_id: StableCrateId) -> CrateNum {
        if stable_crate_id == self.sess.local_stable_crate_id() {
            LOCAL_CRATE
        } else {
            self.untracked_resolutions.cstore.stable_crate_id_to_crate_num(stable_crate_id)
        }
    }

    /// Converts a `DefPathHash` to its corresponding `DefId` in the current compilation
    /// session, if it still exists. This is used during incremental compilation to
    /// turn a deserialized `DefPathHash` into its current `DefId`.
    pub fn def_path_hash_to_def_id(self, hash: DefPathHash) -> DefId {
        debug!("def_path_hash_to_def_id({:?})", hash);

        let stable_crate_id = hash.stable_crate_id();

        // If this is a DefPathHash from the local crate, we can look up the
        // DefId in the tcx's `Definitions`.
        if stable_crate_id == self.sess.local_stable_crate_id() {
            self.untracked_resolutions.definitions.local_def_path_hash_to_def_id(hash).to_def_id()
        } else {
            // If this is a DefPathHash from an upstream crate, let the CrateStore map
            // it to a DefId.
            let cstore = &self.untracked_resolutions.cstore;
            let cnum = cstore.stable_crate_id_to_crate_num(stable_crate_id);
            cstore.def_path_hash_to_def_id(cnum, hash)
        }
    }

    pub fn def_path_debug_str(self, def_id: DefId) -> String {
        // We are explicitly not going through queries here in order to get
        // crate name and stable crate id since this code is called from debug!()
        // statements within the query system and we'd run into endless
        // recursion otherwise.
        let (crate_name, stable_crate_id) = if def_id.is_local() {
            (self.crate_name, self.sess.local_stable_crate_id())
        } else {
            let cstore = &self.untracked_resolutions.cstore;
            (cstore.crate_name(def_id.krate), cstore.stable_crate_id(def_id.krate))
        };

        format!(
            "{}[{}]{}",
            crate_name,
            // Don't print the whole stable crate id. That's just
            // annoying in debug output.
            &(format!("{:08x}", stable_crate_id.to_u64()))[..4],
            self.def_path(def_id).to_string_no_crate_verbose()
        )
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    pub fn cstore_untracked(self) -> &'tcx ty::CrateStoreDyn {
        &*self.untracked_resolutions.cstore
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    pub fn definitions_untracked(self) -> &'tcx hir::definitions::Definitions {
        &self.untracked_resolutions.definitions
    }

    #[inline(always)]
    pub fn create_stable_hashing_context(self) -> StableHashingContext<'tcx> {
        let resolutions = &self.gcx.untracked_resolutions;
        StableHashingContext::new(self.sess, &resolutions.definitions, &*resolutions.cstore)
    }

    #[inline(always)]
    pub fn create_no_span_stable_hashing_context(self) -> StableHashingContext<'tcx> {
        let resolutions = &self.gcx.untracked_resolutions;
        StableHashingContext::ignore_spans(
            self.sess,
            &resolutions.definitions,
            &*resolutions.cstore,
        )
    }

    pub fn serialize_query_result_cache(self, encoder: &mut FileEncoder) -> FileEncodeResult {
        self.on_disk_cache.as_ref().map_or(Ok(()), |c| c.serialize(self, encoder))
    }

    /// If `true`, we should use the MIR-based borrowck, but also
    /// fall back on the AST borrowck if the MIR-based one errors.
    pub fn migrate_borrowck(self) -> bool {
        self.borrowck_mode().migrate()
    }

    /// What mode(s) of borrowck should we run? AST? MIR? both?
    /// (Also considers the `#![feature(nll)]` setting.)
    pub fn borrowck_mode(self) -> BorrowckMode {
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

        if self.features().nll {
            return BorrowckMode::Mir;
        }

        self.sess.opts.borrowck_mode
    }

    /// If `true`, we should use lazy normalization for constants, otherwise
    /// we still evaluate them eagerly.
    #[inline]
    pub fn lazy_normalization(self) -> bool {
        let features = self.features();
        // Note: We only use lazy normalization for generic const expressions.
        features.generic_const_exprs
    }

    #[inline]
    pub fn local_crate_exports_generics(self) -> bool {
        debug_assert!(self.sess.opts.share_generics());

        self.sess.crate_types().iter().any(|crate_type| {
            match crate_type {
                CrateType::Executable
                | CrateType::Staticlib
                | CrateType::ProcMacro
                | CrateType::Cdylib => false,

                // FIXME rust-lang/rust#64319, rust-lang/rust#64872:
                // We want to block export of generics from dylibs,
                // but we must fix rust-lang/rust#65890 before we can
                // do that robustly.
                CrateType::Dylib => true,

                CrateType::Rlib => true,
            }
        })
    }

    // Returns the `DefId` and the `BoundRegionKind` corresponding to the given region.
    pub fn is_suitable_region(self, region: Region<'tcx>) -> Option<FreeRegionInfo> {
        let (suitable_region_binding_scope, bound_region) = match *region {
            ty::ReFree(ref free_region) => {
                (free_region.scope.expect_local(), free_region.bound_region)
            }
            ty::ReEarlyBound(ref ebr) => (
                self.parent(ebr.def_id).unwrap().expect_local(),
                ty::BoundRegionKind::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let hir_id = self.hir().local_def_id_to_hir_id(suitable_region_binding_scope);
        let is_impl_item = match self.hir().find(hir_id) {
            Some(Node::Item(..) | Node::TraitItem(..)) => false,
            Some(Node::ImplItem(..)) => {
                self.is_bound_region_in_impl_item(suitable_region_binding_scope)
            }
            _ => return None,
        };

        Some(FreeRegionInfo {
            def_id: suitable_region_binding_scope,
            boundregion: bound_region,
            is_impl_item,
        })
    }

    /// Given a `DefId` for an `fn`, return all the `dyn` and `impl` traits in its return type.
    pub fn return_type_impl_or_dyn_traits(
        self,
        scope_def_id: LocalDefId,
    ) -> Vec<&'tcx hir::Ty<'tcx>> {
        let hir_id = self.hir().local_def_id_to_hir_id(scope_def_id);
        let hir_output = match self.hir().fn_decl_by_hir_id(hir_id) {
            Some(hir::FnDecl { output: hir::FnRetTy::Return(ty), .. }) => ty,
            _ => return vec![],
        };

        let mut v = TraitObjectVisitor(vec![], self.hir());
        v.visit_ty(hir_output);
        v.0
    }

    pub fn return_type_impl_trait(self, scope_def_id: LocalDefId) -> Option<(Ty<'tcx>, Span)> {
        // `type_of()` will fail on these (#55796, #86483), so only allow `fn`s or closures.
        let hir_id = self.hir().local_def_id_to_hir_id(scope_def_id);
        match self.hir().get(hir_id) {
            Node::Item(&hir::Item { kind: ItemKind::Fn(..), .. }) => {}
            Node::TraitItem(&hir::TraitItem { kind: TraitItemKind::Fn(..), .. }) => {}
            Node::ImplItem(&hir::ImplItem { kind: ImplItemKind::Fn(..), .. }) => {}
            Node::Expr(&hir::Expr { kind: ExprKind::Closure(..), .. }) => {}
            _ => return None,
        }

        let ret_ty = self.type_of(scope_def_id);
        match ret_ty.kind() {
            ty::FnDef(_, _) => {
                let sig = ret_ty.fn_sig(self);
                let output = self.erase_late_bound_regions(sig.output());
                if output.is_impl_trait() {
                    let fn_decl = self.hir().fn_decl_by_hir_id(hir_id).unwrap();
                    Some((output, fn_decl.output.span()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // Checks if the bound region is in Impl Item.
    pub fn is_bound_region_in_impl_item(self, suitable_region_binding_scope: LocalDefId) -> bool {
        let container_id =
            self.associated_item(suitable_region_binding_scope.to_def_id()).container.id();
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

    /// Determines whether identifiers in the assembly have strict naming rules.
    /// Currently, only NVPTX* targets need it.
    pub fn has_strict_asm_symbol_naming(self) -> bool {
        self.sess.target.arch.contains("nvptx")
    }

    /// Returns `&'static core::panic::Location<'static>`.
    pub fn caller_location_ty(self) -> Ty<'tcx> {
        self.mk_imm_ref(
            self.lifetimes.re_static,
            self.type_of(self.require_lang_item(LangItem::PanicLocation, None))
                .subst(self, self.mk_substs([self.lifetimes.re_static.into()].iter())),
        )
    }

    /// Returns a displayable description and article for the given `def_id` (e.g. `("a", "struct")`).
    pub fn article_and_description(self, def_id: DefId) -> (&'static str, &'static str) {
        match self.def_kind(def_id) {
            DefKind::Generator => match self.generator_kind(def_id).unwrap() {
                rustc_hir::GeneratorKind::Async(..) => ("an", "async closure"),
                rustc_hir::GeneratorKind::Gen => ("a", "generator"),
            },
            def_kind => (def_kind.article(), def_kind.descr(def_id)),
        }
    }

    pub fn type_length_limit(self) -> Limit {
        self.limits(()).type_length_limit
    }

    pub fn recursion_limit(self) -> Limit {
        self.limits(()).recursion_limit
    }

    pub fn move_size_limit(self) -> Limit {
        self.limits(()).move_size_limit
    }

    pub fn const_eval_limit(self) -> Limit {
        self.limits(()).const_eval_limit
    }

    pub fn all_traits(self) -> impl Iterator<Item = DefId> + 'tcx {
        iter::once(LOCAL_CRATE)
            .chain(self.crates(()).iter().copied())
            .flat_map(move |cnum| self.traits_in_crate(cnum).iter().copied())
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
/// `None` is returned if the value or one of the components is not part
/// of the provided context.
/// For `Ty`, `None` can be returned if either the type interner doesn't
/// contain the `TyKind` key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g., `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx>: fmt::Debug {
    type Lifted: fmt::Debug + 'tcx;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted>;
}

macro_rules! nop_lift {
    ($set:ident; $ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<'tcx> for $ty {
            type Lifted = $lifted;
            fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                if tcx.interners.$set.contains_pointer_to(&Interned(self)) {
                    Some(unsafe { mem::transmute(self) })
                } else {
                    None
                }
            }
        }
    };
}

macro_rules! nop_list_lift {
    ($set:ident; $ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<'tcx> for &'a List<$ty> {
            type Lifted = &'tcx List<$lifted>;
            fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                if self.is_empty() {
                    return Some(List::empty());
                }
                if tcx.interners.$set.contains_pointer_to(&Interned(self)) {
                    Some(unsafe { mem::transmute(self) })
                } else {
                    None
                }
            }
        }
    };
}

nop_lift! {type_; Ty<'a> => Ty<'tcx>}
nop_lift! {region; Region<'a> => Region<'tcx>}
nop_lift! {const_; &'a Const<'a> => &'tcx Const<'tcx>}
nop_lift! {const_allocation; &'a Allocation => &'tcx Allocation}
nop_lift! {predicate; &'a PredicateInner<'a> => &'tcx PredicateInner<'tcx>}

nop_list_lift! {type_list; Ty<'a> => Ty<'tcx>}
nop_list_lift! {poly_existential_predicates; ty::Binder<'a, ExistentialPredicate<'a>> => ty::Binder<'tcx, ExistentialPredicate<'tcx>>}
nop_list_lift! {predicates; Predicate<'a> => Predicate<'tcx>}
nop_list_lift! {canonical_var_infos; CanonicalVarInfo<'a> => CanonicalVarInfo<'tcx>}
nop_list_lift! {projs; ProjectionKind => ProjectionKind}
nop_list_lift! {bound_variable_kinds; ty::BoundVariableKind => ty::BoundVariableKind}

// This is the impl for `&'a InternalSubsts<'a>`.
nop_list_lift! {substs; GenericArg<'a> => GenericArg<'tcx>}

CloneLiftImpls! { for<'tcx> { Constness, traits::WellFormedLoc, } }

pub mod tls {
    use super::{ptr_eq, GlobalCtxt, TyCtxt};

    use crate::dep_graph::{DepKind, TaskDeps};
    use crate::ty::query;
    use rustc_data_structures::sync::{self, Lock};
    use rustc_data_structures::thin_vec::ThinVec;
    use rustc_errors::Diagnostic;
    use std::mem;

    #[cfg(not(parallel_compiler))]
    use std::cell::Cell;

    #[cfg(parallel_compiler)]
    use rustc_rayon_core as rayon_core;

    /// This is the implicit state of rustc. It contains the current
    /// `TyCtxt` and query. It is updated when creating a local interner or
    /// executing a new query. Whenever there's a `TyCtxt` value available
    /// you should also have access to an `ImplicitCtxt` through the functions
    /// in this module.
    #[derive(Clone)]
    pub struct ImplicitCtxt<'a, 'tcx> {
        /// The current `TyCtxt`.
        pub tcx: TyCtxt<'tcx>,

        /// The current query job, if any. This is updated by `JobOwner::start` in
        /// `ty::query::plumbing` when executing a query.
        pub query: Option<query::QueryJobId<DepKind>>,

        /// Where to store diagnostics for the current query job, if any.
        /// This is updated by `JobOwner::start` in `ty::query::plumbing` when executing a query.
        pub diagnostics: Option<&'a Lock<ThinVec<Diagnostic>>>,

        /// Used to prevent layout from recursing too deeply.
        pub layout_depth: usize,

        /// The current dep graph task. This is used to add dependencies to queries
        /// when executing them.
        pub task_deps: Option<&'a Lock<TaskDeps>>,
    }

    impl<'a, 'tcx> ImplicitCtxt<'a, 'tcx> {
        pub fn new(gcx: &'tcx GlobalCtxt<'tcx>) -> Self {
            let tcx = TyCtxt { gcx };
            ImplicitCtxt { tcx, query: None, diagnostics: None, layout_depth: 0, task_deps: None }
        }
    }

    /// Sets Rayon's thread-local variable, which is preserved for Rayon jobs
    /// to `value` during the call to `f`. It is restored to its previous value after.
    /// This is used to set the pointer to the new `ImplicitCtxt`.
    #[cfg(parallel_compiler)]
    #[inline]
    fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
        rayon_core::tlv::with(value, f)
    }

    /// Gets Rayon's thread-local variable, which is preserved for Rayon jobs.
    /// This is used to get the pointer to the current `ImplicitCtxt`.
    #[cfg(parallel_compiler)]
    #[inline]
    pub fn get_tlv() -> usize {
        rayon_core::tlv::get()
    }

    #[cfg(not(parallel_compiler))]
    thread_local! {
        /// A thread local variable that stores a pointer to the current `ImplicitCtxt`.
        static TLV: Cell<usize> = const { Cell::new(0) };
    }

    /// Sets TLV to `value` during the call to `f`.
    /// It is restored to its previous value after.
    /// This is used to set the pointer to the new `ImplicitCtxt`.
    #[cfg(not(parallel_compiler))]
    #[inline]
    fn set_tlv<F: FnOnce() -> R, R>(value: usize, f: F) -> R {
        let old = get_tlv();
        let _reset = rustc_data_structures::OnDrop(move || TLV.with(|tlv| tlv.set(old)));
        TLV.with(|tlv| tlv.set(value));
        f()
    }

    /// Gets the pointer to the current `ImplicitCtxt`.
    #[cfg(not(parallel_compiler))]
    #[inline]
    fn get_tlv() -> usize {
        TLV.with(|tlv| tlv.get())
    }

    /// Sets `context` as the new current `ImplicitCtxt` for the duration of the function `f`.
    #[inline]
    pub fn enter_context<'a, 'tcx, F, R>(context: &ImplicitCtxt<'a, 'tcx>, f: F) -> R
    where
        F: FnOnce(&ImplicitCtxt<'a, 'tcx>) -> R,
    {
        set_tlv(context as *const _ as usize, || f(&context))
    }

    /// Allows access to the current `ImplicitCtxt` in a closure if one is available.
    #[inline]
    pub fn with_context_opt<F, R>(f: F) -> R
    where
        F: for<'a, 'tcx> FnOnce(Option<&ImplicitCtxt<'a, 'tcx>>) -> R,
    {
        let context = get_tlv();
        if context == 0 {
            f(None)
        } else {
            // We could get an `ImplicitCtxt` pointer from another thread.
            // Ensure that `ImplicitCtxt` is `Sync`.
            sync::assert_sync::<ImplicitCtxt<'_, '_>>();

            unsafe { f(Some(&*(context as *const ImplicitCtxt<'_, '_>))) }
        }
    }

    /// Allows access to the current `ImplicitCtxt`.
    /// Panics if there is no `ImplicitCtxt` available.
    #[inline]
    pub fn with_context<F, R>(f: F) -> R
    where
        F: for<'a, 'tcx> FnOnce(&ImplicitCtxt<'a, 'tcx>) -> R,
    {
        with_context_opt(|opt_context| f(opt_context.expect("no ImplicitCtxt stored in tls")))
    }

    /// Allows access to the current `ImplicitCtxt` whose tcx field is the same as the tcx argument
    /// passed in. This means the closure is given an `ImplicitCtxt` with the same `'tcx` lifetime
    /// as the `TyCtxt` passed in.
    /// This will panic if you pass it a `TyCtxt` which is different from the current
    /// `ImplicitCtxt`'s `tcx` field.
    #[inline]
    pub fn with_related_context<'tcx, F, R>(tcx: TyCtxt<'tcx>, f: F) -> R
    where
        F: FnOnce(&ImplicitCtxt<'_, 'tcx>) -> R,
    {
        with_context(|context| unsafe {
            assert!(ptr_eq(context.tcx.gcx, tcx.gcx));
            let context: &ImplicitCtxt<'_, '_> = mem::transmute(context);
            f(context)
        })
    }

    /// Allows access to the `TyCtxt` in the current `ImplicitCtxt`.
    /// Panics if there is no `ImplicitCtxt` available.
    #[inline]
    pub fn with<F, R>(f: F) -> R
    where
        F: for<'tcx> FnOnce(TyCtxt<'tcx>) -> R,
    {
        with_context(|context| f(context.tcx))
    }

    /// Allows access to the `TyCtxt` in the current `ImplicitCtxt`.
    /// The closure is passed None if there is no `ImplicitCtxt` available.
    #[inline]
    pub fn with_opt<F, R>(f: F) -> R
    where
        F: for<'tcx> FnOnce(Option<TyCtxt<'tcx>>) -> R,
    {
        with_context_opt(|opt_context| f(opt_context.map(|context| context.tcx)))
    }
}

macro_rules! sty_debug_print {
    ($fmt: expr, $ctxt: expr, $($variant: ident),*) => {{
        // Curious inner module to allow variant names to be used as
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

            pub fn go(fmt: &mut std::fmt::Formatter<'_>, tcx: TyCtxt<'_>) -> std::fmt::Result {
                let mut total = DebugStat {
                    total: 0,
                    lt_infer: 0,
                    ty_infer: 0,
                    ct_infer: 0,
                    all_infer: 0,
                };
                $(let mut $variant = total;)*

                let shards = tcx.interners.type_.lock_shards();
                let types = shards.iter().flat_map(|shard| shard.keys());
                for &Interned(t) in types {
                    let variant = match t.kind() {
                        ty::Bool | ty::Char | ty::Int(..) | ty::Uint(..) |
                            ty::Float(..) | ty::Str | ty::Never => continue,
                        ty::Error(_) => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let lt = t.flags().intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags().intersects(ty::TypeFlags::HAS_TY_INFER);
                    let ct = t.flags().intersects(ty::TypeFlags::HAS_CT_INFER);

                    variant.total += 1;
                    total.total += 1;
                    if lt { total.lt_infer += 1; variant.lt_infer += 1 }
                    if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                    if ct { total.ct_infer += 1; variant.ct_infer += 1 }
                    if lt && ty && ct { total.all_infer += 1; variant.all_infer += 1 }
                }
                writeln!(fmt, "Ty interner             total           ty lt ct all")?;
                $(writeln!(fmt, "    {:18}: {uses:6} {usespc:4.1}%, \
                            {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    stringify!($variant),
                    uses = $variant.total,
                    usespc = $variant.total as f64 * 100.0 / total.total as f64,
                    ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = $variant.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = $variant.ct_infer as f64 * 100.0  / total.total as f64,
                    all = $variant.all_infer as f64 * 100.0  / total.total as f64)?;
                )*
                writeln!(fmt, "                  total {uses:6}        \
                          {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    uses = total.total,
                    ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = total.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = total.ct_infer as f64 * 100.0  / total.total as f64,
                    all = total.all_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($fmt, $ctxt)
    }}
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn debug_stats(self) -> impl std::fmt::Debug + 'tcx {
        struct DebugStats<'tcx>(TyCtxt<'tcx>);

        impl<'tcx> std::fmt::Debug for DebugStats<'tcx> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                sty_debug_print!(
                    fmt,
                    self.0,
                    Adt,
                    Array,
                    Slice,
                    RawPtr,
                    Ref,
                    FnDef,
                    FnPtr,
                    Placeholder,
                    Generator,
                    GeneratorWitness,
                    Dynamic,
                    Closure,
                    Tuple,
                    Bound,
                    Param,
                    Infer,
                    Projection,
                    Opaque,
                    Foreign
                )?;

                writeln!(fmt, "InternalSubsts interner: #{}", self.0.interners.substs.len())?;
                writeln!(fmt, "Region interner: #{}", self.0.interners.region.len())?;
                writeln!(fmt, "Stability interner: #{}", self.0.stability_interner.len())?;
                writeln!(
                    fmt,
                    "Const Stability interner: #{}",
                    self.0.const_stability_interner.len()
                )?;
                writeln!(
                    fmt,
                    "Const Allocation interner: #{}",
                    self.0.interners.const_allocation.len()
                )?;
                writeln!(fmt, "Layout interner: #{}", self.0.interners.layout.len())?;

                Ok(())
            }
        }

        DebugStats(self)
    }
}

/// An entry in an interner.
struct Interned<'tcx, T: ?Sized>(&'tcx T);

impl<'tcx, T: 'tcx + ?Sized> Clone for Interned<'tcx, T> {
    fn clone(&self) -> Self {
        Interned(self.0)
    }
}
impl<'tcx, T: 'tcx + ?Sized> Copy for Interned<'tcx, T> {}

impl<'tcx, T: 'tcx + ?Sized> IntoPointer for Interned<'tcx, T> {
    fn into_pointer(&self) -> *const () {
        self.0 as *const _ as *const ()
    }
}
// N.B., an `Interned<Ty>` compares and hashes as a `TyKind`.
impl<'tcx> PartialEq for Interned<'tcx, TyS<'tcx>> {
    fn eq(&self, other: &Interned<'tcx, TyS<'tcx>>) -> bool {
        self.0.kind() == other.0.kind()
    }
}

impl<'tcx> Eq for Interned<'tcx, TyS<'tcx>> {}

impl<'tcx> Hash for Interned<'tcx, TyS<'tcx>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0.kind().hash(s)
    }
}

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx> Borrow<TyKind<'tcx>> for Interned<'tcx, TyS<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a TyKind<'tcx> {
        &self.0.kind()
    }
}
// N.B., an `Interned<PredicateInner>` compares and hashes as a `PredicateKind`.
impl<'tcx> PartialEq for Interned<'tcx, PredicateInner<'tcx>> {
    fn eq(&self, other: &Interned<'tcx, PredicateInner<'tcx>>) -> bool {
        self.0.kind == other.0.kind
    }
}

impl<'tcx> Eq for Interned<'tcx, PredicateInner<'tcx>> {}

impl<'tcx> Hash for Interned<'tcx, PredicateInner<'tcx>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0.kind.hash(s)
    }
}

impl<'tcx> Borrow<Binder<'tcx, PredicateKind<'tcx>>> for Interned<'tcx, PredicateInner<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a Binder<'tcx, PredicateKind<'tcx>> {
        &self.0.kind
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

impl<'tcx, T> Borrow<[T]> for Interned<'tcx, List<T>> {
    fn borrow<'a>(&'a self) -> &'a [T] {
        &self.0[..]
    }
}

macro_rules! direct_interners {
    ($($name:ident: $method:ident($ty:ty),)+) => {
        $(impl<'tcx> PartialEq for Interned<'tcx, $ty> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<'tcx> Eq for Interned<'tcx, $ty> {}

        impl<'tcx> Hash for Interned<'tcx, $ty> {
            fn hash<H: Hasher>(&self, s: &mut H) {
                self.0.hash(s)
            }
        }

        impl<'tcx> Borrow<$ty> for Interned<'tcx, $ty> {
            fn borrow<'a>(&'a self) -> &'a $ty {
                &self.0
            }
        }

        impl<'tcx> TyCtxt<'tcx> {
            pub fn $method(self, v: $ty) -> &'tcx $ty {
                self.interners.$name.intern(v, |v| {
                    Interned(self.interners.arena.alloc(v))
                }).0
            }
        })+
    }
}

direct_interners! {
    region: mk_region(RegionKind),
    const_: mk_const(Const<'tcx>),
    const_allocation: intern_const_alloc(Allocation),
    layout: intern_layout(Layout),
    adt_def: intern_adt_def(AdtDef),
}

macro_rules! slice_interners {
    ($($field:ident: $method:ident($ty:ty)),+ $(,)?) => (
        impl<'tcx> TyCtxt<'tcx> {
            $(pub fn $method(self, v: &[$ty]) -> &'tcx List<$ty> {
                self.interners.$field.intern_ref(v, || {
                    Interned(List::from_arena(&*self.arena, v))
                }).0
            })+
        }
    );
}

slice_interners!(
    type_list: _intern_type_list(Ty<'tcx>),
    substs: _intern_substs(GenericArg<'tcx>),
    canonical_var_infos: _intern_canonical_var_infos(CanonicalVarInfo<'tcx>),
    poly_existential_predicates:
        _intern_poly_existential_predicates(ty::Binder<'tcx, ExistentialPredicate<'tcx>>),
    predicates: _intern_predicates(Predicate<'tcx>),
    projs: _intern_projs(ProjectionKind),
    place_elems: _intern_place_elems(PlaceElem<'tcx>),
    bound_variable_kinds: _intern_bound_variable_kinds(ty::BoundVariableKind),
);

impl<'tcx> TyCtxt<'tcx> {
    /// Given a `fn` type, returns an equivalent `unsafe fn` type;
    /// that is, a `fn` type that is equivalent in every way for being
    /// unsafe.
    pub fn safe_to_unsafe_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        assert_eq!(sig.unsafety(), hir::Unsafety::Normal);
        self.mk_fn_ptr(sig.map_bound(|sig| ty::FnSig { unsafety: hir::Unsafety::Unsafe, ..sig }))
    }

    /// Given the def_id of a Trait `trait_def_id` and the name of an associated item `assoc_name`
    /// returns true if the `trait_def_id` defines an associated item of name `assoc_name`.
    pub fn trait_may_define_assoc_type(self, trait_def_id: DefId, assoc_name: Ident) -> bool {
        self.super_traits_of(trait_def_id).any(|trait_did| {
            self.associated_items(trait_did)
                .find_by_name_and_kind(self, assoc_name, ty::AssocKind::Type, trait_did)
                .is_some()
        })
    }

    /// Computes the def-ids of the transitive supertraits of `trait_def_id`. This (intentionally)
    /// does not compute the full elaborated super-predicates but just the set of def-ids. It is used
    /// to identify which traits may define a given associated type to help avoid cycle errors.
    /// Returns a `DefId` iterator.
    fn super_traits_of(self, trait_def_id: DefId) -> impl Iterator<Item = DefId> + 'tcx {
        let mut set = FxHashSet::default();
        let mut stack = vec![trait_def_id];

        set.insert(trait_def_id);

        iter::from_fn(move || -> Option<DefId> {
            let trait_did = stack.pop()?;
            let generic_predicates = self.super_predicates_of(trait_did);

            for (predicate, _) in generic_predicates.predicates {
                if let ty::PredicateKind::Trait(data) = predicate.kind().skip_binder() {
                    if set.insert(data.def_id()) {
                        stack.push(data.def_id());
                    }
                }
            }

            Some(trait_did)
        })
    }

    /// Given a closure signature, returns an equivalent fn signature. Detuples
    /// and so forth -- so e.g., if we have a sig with `Fn<(u32, i32)>` then
    /// you would get a `fn(u32, i32)`.
    /// `unsafety` determines the unsafety of the fn signature. If you pass
    /// `hir::Unsafety::Unsafe` in the previous example, then you would get
    /// an `unsafe fn (u32, i32)`.
    /// It cannot convert a closure that requires unsafe.
    pub fn signature_unclosure(
        self,
        sig: PolyFnSig<'tcx>,
        unsafety: hir::Unsafety,
    ) -> PolyFnSig<'tcx> {
        sig.map_bound(|s| {
            let params_iter = match s.inputs()[0].kind() {
                ty::Tuple(params) => params.into_iter().map(|k| k.expect_ty()),
                _ => bug!(),
            };
            self.mk_fn_sig(params_iter, s.output(), s.c_variadic, unsafety, abi::Abi::Rust)
        })
    }

    /// Same a `self.mk_region(kind)`, but avoids accessing the interners if
    /// `*r == kind`.
    #[inline]
    pub fn reuse_or_mk_region(self, r: Region<'tcx>, kind: RegionKind) -> Region<'tcx> {
        if *r == kind { r } else { self.mk_region(kind) }
    }

    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    pub fn mk_ty(self, st: TyKind<'tcx>) -> Ty<'tcx> {
        self.interners.intern_ty(st)
    }

    #[inline]
    pub fn mk_predicate(self, binder: Binder<'tcx, PredicateKind<'tcx>>) -> Predicate<'tcx> {
        let inner = self.interners.intern_predicate(binder);
        Predicate { inner }
    }

    #[inline]
    pub fn reuse_or_mk_predicate(
        self,
        pred: Predicate<'tcx>,
        binder: Binder<'tcx, PredicateKind<'tcx>>,
    ) -> Predicate<'tcx> {
        if pred.kind() != binder { self.mk_predicate(binder) } else { pred }
    }

    pub fn mk_mach_int(self, tm: IntTy) -> Ty<'tcx> {
        match tm {
            IntTy::Isize => self.types.isize,
            IntTy::I8 => self.types.i8,
            IntTy::I16 => self.types.i16,
            IntTy::I32 => self.types.i32,
            IntTy::I64 => self.types.i64,
            IntTy::I128 => self.types.i128,
        }
    }

    pub fn mk_mach_uint(self, tm: UintTy) -> Ty<'tcx> {
        match tm {
            UintTy::Usize => self.types.usize,
            UintTy::U8 => self.types.u8,
            UintTy::U16 => self.types.u16,
            UintTy::U32 => self.types.u32,
            UintTy::U64 => self.types.u64,
            UintTy::U128 => self.types.u128,
        }
    }

    pub fn mk_mach_float(self, tm: FloatTy) -> Ty<'tcx> {
        match tm {
            FloatTy::F32 => self.types.f32,
            FloatTy::F64 => self.types.f64,
        }
    }

    #[inline]
    pub fn mk_static_str(self) -> Ty<'tcx> {
        self.mk_imm_ref(self.lifetimes.re_static, self.types.str_)
    }

    #[inline]
    pub fn mk_adt(self, def: &'tcx AdtDef, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        // Take a copy of substs so that we own the vectors inside.
        self.mk_ty(Adt(def, substs))
    }

    #[inline]
    pub fn mk_foreign(self, def_id: DefId) -> Ty<'tcx> {
        self.mk_ty(Foreign(def_id))
    }

    fn mk_generic_adt(self, wrapper_def_id: DefId, ty_param: Ty<'tcx>) -> Ty<'tcx> {
        let adt_def = self.adt_def(wrapper_def_id);
        let substs =
            InternalSubsts::for_item(self, wrapper_def_id, |param, substs| match param.kind {
                GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => bug!(),
                GenericParamDefKind::Type { has_default, .. } => {
                    if param.index == 0 {
                        ty_param.into()
                    } else {
                        assert!(has_default);
                        self.type_of(param.def_id).subst(self, substs).into()
                    }
                }
            });
        self.mk_ty(Adt(adt_def, substs))
    }

    #[inline]
    pub fn mk_box(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(LangItem::OwnedBox, None);
        self.mk_generic_adt(def_id, ty)
    }

    #[inline]
    pub fn mk_lang_item(self, ty: Ty<'tcx>, item: LangItem) -> Option<Ty<'tcx>> {
        let def_id = self.lang_items().require(item).ok()?;
        Some(self.mk_generic_adt(def_id, ty))
    }

    #[inline]
    pub fn mk_diagnostic_item(self, ty: Ty<'tcx>, name: Symbol) -> Option<Ty<'tcx>> {
        let def_id = self.get_diagnostic_item(name)?;
        Some(self.mk_generic_adt(def_id, ty))
    }

    #[inline]
    pub fn mk_maybe_uninit(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(LangItem::MaybeUninit, None);
        self.mk_generic_adt(def_id, ty)
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
        self.mk_ref(r, TypeAndMut { ty, mutbl: hir::Mutability::Mut })
    }

    #[inline]
    pub fn mk_imm_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut { ty, mutbl: hir::Mutability::Not })
    }

    #[inline]
    pub fn mk_mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut { ty, mutbl: hir::Mutability::Mut })
    }

    #[inline]
    pub fn mk_imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut { ty, mutbl: hir::Mutability::Not })
    }

    #[inline]
    pub fn mk_array(self, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        self.mk_ty(Array(ty, ty::Const::from_usize(self, n)))
    }

    #[inline]
    pub fn mk_slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Slice(ty))
    }

    #[inline]
    pub fn intern_tup(self, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        let kinds: Vec<_> = ts.iter().map(|&t| GenericArg::from(t)).collect();
        self.mk_ty(Tuple(self.intern_substs(&kinds)))
    }

    pub fn mk_tup<I: InternAs<[Ty<'tcx>], Ty<'tcx>>>(self, iter: I) -> I::Output {
        iter.intern_with(|ts| {
            let kinds: Vec<_> = ts.iter().map(|&t| GenericArg::from(t)).collect();
            self.mk_ty(Tuple(self.intern_substs(&kinds)))
        })
    }

    #[inline]
    pub fn mk_unit(self) -> Ty<'tcx> {
        self.types.unit
    }

    #[inline]
    pub fn mk_diverging_default(self) -> Ty<'tcx> {
        if self.features().never_type_fallback { self.types.never } else { self.types.unit }
    }

    #[inline]
    pub fn mk_fn_def(self, def_id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnDef(def_id, substs))
    }

    #[inline]
    pub fn mk_fn_ptr(self, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        self.mk_ty(FnPtr(fty))
    }

    #[inline]
    pub fn mk_dynamic(
        self,
        obj: &'tcx List<ty::Binder<'tcx, ExistentialPredicate<'tcx>>>,
        reg: ty::Region<'tcx>,
    ) -> Ty<'tcx> {
        self.mk_ty(Dynamic(obj, reg))
    }

    #[inline]
    pub fn mk_projection(self, item_def_id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Projection(ProjectionTy { item_def_id, substs }))
    }

    #[inline]
    pub fn mk_closure(self, closure_id: DefId, closure_substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Closure(closure_id, closure_substs))
    }

    #[inline]
    pub fn mk_generator(
        self,
        id: DefId,
        generator_substs: SubstsRef<'tcx>,
        movability: hir::Movability,
    ) -> Ty<'tcx> {
        self.mk_ty(Generator(id, generator_substs, movability))
    }

    #[inline]
    pub fn mk_generator_witness(self, types: ty::Binder<'tcx, &'tcx List<Ty<'tcx>>>) -> Ty<'tcx> {
        self.mk_ty(GeneratorWitness(types))
    }

    #[inline]
    pub fn mk_ty_var(self, v: TyVid) -> Ty<'tcx> {
        self.mk_ty_infer(TyVar(v))
    }

    #[inline]
    pub fn mk_const_var(self, v: ConstVid<'tcx>, ty: Ty<'tcx>) -> &'tcx Const<'tcx> {
        self.mk_const(ty::Const { val: ty::ConstKind::Infer(InferConst::Var(v)), ty })
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
    pub fn mk_const_infer(self, ic: InferConst<'tcx>, ty: Ty<'tcx>) -> &'tcx ty::Const<'tcx> {
        self.mk_const(ty::Const { val: ty::ConstKind::Infer(ic), ty })
    }

    #[inline]
    pub fn mk_ty_param(self, index: u32, name: Symbol) -> Ty<'tcx> {
        self.mk_ty(Param(ParamTy { index, name }))
    }

    #[inline]
    pub fn mk_const_param(self, index: u32, name: Symbol, ty: Ty<'tcx>) -> &'tcx Const<'tcx> {
        self.mk_const(ty::Const { val: ty::ConstKind::Param(ParamConst { index, name }), ty })
    }

    pub fn mk_param_from_def(self, param: &ty::GenericParamDef) -> GenericArg<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                self.mk_region(ty::ReEarlyBound(param.to_early_bound_region_data())).into()
            }
            GenericParamDefKind::Type { .. } => self.mk_ty_param(param.index, param.name).into(),
            GenericParamDefKind::Const { .. } => {
                self.mk_const_param(param.index, param.name, self.type_of(param.def_id)).into()
            }
        }
    }

    #[inline]
    pub fn mk_opaque(self, def_id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.mk_ty(Opaque(def_id, substs))
    }

    pub fn mk_place_field(self, place: Place<'tcx>, f: Field, ty: Ty<'tcx>) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Field(f, ty))
    }

    pub fn mk_place_deref(self, place: Place<'tcx>) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Deref)
    }

    pub fn mk_place_downcast(
        self,
        place: Place<'tcx>,
        adt_def: &'tcx AdtDef,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.mk_place_elem(
            place,
            PlaceElem::Downcast(Some(adt_def.variants[variant_index].ident.name), variant_index),
        )
    }

    pub fn mk_place_downcast_unnamed(
        self,
        place: Place<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Downcast(None, variant_index))
    }

    pub fn mk_place_index(self, place: Place<'tcx>, index: Local) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Index(index))
    }

    /// This method copies `Place`'s projection, add an element and reintern it. Should not be used
    /// to build a full `Place` it's just a convenient way to grab a projection and modify it in
    /// flight.
    pub fn mk_place_elem(self, place: Place<'tcx>, elem: PlaceElem<'tcx>) -> Place<'tcx> {
        let mut projection = place.projection.to_vec();
        projection.push(elem);

        Place { local: place.local, projection: self.intern_place_elems(&projection) }
    }

    pub fn intern_poly_existential_predicates(
        self,
        eps: &[ty::Binder<'tcx, ExistentialPredicate<'tcx>>],
    ) -> &'tcx List<ty::Binder<'tcx, ExistentialPredicate<'tcx>>> {
        assert!(!eps.is_empty());
        assert!(
            eps.array_windows()
                .all(|[a, b]| a.skip_binder().stable_cmp(self, &b.skip_binder())
                    != Ordering::Greater)
        );
        self._intern_poly_existential_predicates(eps)
    }

    pub fn intern_predicates(self, preds: &[Predicate<'tcx>]) -> &'tcx List<Predicate<'tcx>> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        if preds.is_empty() {
            // The macro-generated method below asserts we don't intern an empty slice.
            List::empty()
        } else {
            self._intern_predicates(preds)
        }
    }

    pub fn intern_type_list(self, ts: &[Ty<'tcx>]) -> &'tcx List<Ty<'tcx>> {
        if ts.is_empty() { List::empty() } else { self._intern_type_list(ts) }
    }

    pub fn intern_substs(self, ts: &[GenericArg<'tcx>]) -> &'tcx List<GenericArg<'tcx>> {
        if ts.is_empty() { List::empty() } else { self._intern_substs(ts) }
    }

    pub fn intern_projs(self, ps: &[ProjectionKind]) -> &'tcx List<ProjectionKind> {
        if ps.is_empty() { List::empty() } else { self._intern_projs(ps) }
    }

    pub fn intern_place_elems(self, ts: &[PlaceElem<'tcx>]) -> &'tcx List<PlaceElem<'tcx>> {
        if ts.is_empty() { List::empty() } else { self._intern_place_elems(ts) }
    }

    pub fn intern_canonical_var_infos(
        self,
        ts: &[CanonicalVarInfo<'tcx>],
    ) -> CanonicalVarInfos<'tcx> {
        if ts.is_empty() { List::empty() } else { self._intern_canonical_var_infos(ts) }
    }

    pub fn intern_bound_variable_kinds(
        self,
        ts: &[ty::BoundVariableKind],
    ) -> &'tcx List<ty::BoundVariableKind> {
        if ts.is_empty() { List::empty() } else { self._intern_bound_variable_kinds(ts) }
    }

    pub fn mk_fn_sig<I>(
        self,
        inputs: I,
        output: I::Item,
        c_variadic: bool,
        unsafety: hir::Unsafety,
        abi: abi::Abi,
    ) -> <I::Item as InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>::Output
    where
        I: Iterator<Item: InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>,
    {
        inputs.chain(iter::once(output)).intern_with(|xs| ty::FnSig {
            inputs_and_output: self.intern_type_list(xs),
            c_variadic,
            unsafety,
            abi,
        })
    }

    pub fn mk_poly_existential_predicates<
        I: InternAs<
            [ty::Binder<'tcx, ExistentialPredicate<'tcx>>],
            &'tcx List<ty::Binder<'tcx, ExistentialPredicate<'tcx>>>,
        >,
    >(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_poly_existential_predicates(xs))
    }

    pub fn mk_predicates<I: InternAs<[Predicate<'tcx>], &'tcx List<Predicate<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_predicates(xs))
    }

    pub fn mk_type_list<I: InternAs<[Ty<'tcx>], &'tcx List<Ty<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_type_list(xs))
    }

    pub fn mk_substs<I: InternAs<[GenericArg<'tcx>], &'tcx List<GenericArg<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_substs(xs))
    }

    pub fn mk_place_elems<I: InternAs<[PlaceElem<'tcx>], &'tcx List<PlaceElem<'tcx>>>>(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_place_elems(xs))
    }

    pub fn mk_substs_trait(self, self_ty: Ty<'tcx>, rest: &[GenericArg<'tcx>]) -> SubstsRef<'tcx> {
        self.mk_substs(iter::once(self_ty.into()).chain(rest.iter().cloned()))
    }

    pub fn mk_bound_variable_kinds<
        I: InternAs<[ty::BoundVariableKind], &'tcx List<ty::BoundVariableKind>>,
    >(
        self,
        iter: I,
    ) -> I::Output {
        iter.intern_with(|xs| self.intern_bound_variable_kinds(xs))
    }

    /// Walks upwards from `id` to find a node which might change lint levels with attributes.
    /// It stops at `bound` and just returns it if reached.
    pub fn maybe_lint_level_root_bounded(self, mut id: HirId, bound: HirId) -> HirId {
        let hir = self.hir();
        loop {
            if id == bound {
                return bound;
            }

            if hir.attrs(id).iter().any(|attr| Level::from_symbol(attr.name_or_empty()).is_some()) {
                return id;
            }
            let next = hir.get_parent_node(id);
            if next == id {
                bug!("lint traversal reached the root of the crate");
            }
            id = next;
        }
    }

    pub fn lint_level_at_node(
        self,
        lint: &'static Lint,
        mut id: hir::HirId,
    ) -> (Level, LintLevelSource) {
        let sets = self.lint_levels(());
        loop {
            if let Some(pair) = sets.level_and_source(lint, id, self.sess) {
                return pair;
            }
            let next = self.hir().get_parent_node(id);
            if next == id {
                bug!("lint traversal reached the root of the crate");
            }
            id = next;
        }
    }

    pub fn struct_span_lint_hir(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a>),
    ) {
        let (level, src) = self.lint_level_at_node(lint, hir_id);
        struct_lint_level(self.sess, lint, level, src, Some(span.into()), decorate);
    }

    pub fn struct_lint_node(
        self,
        lint: &'static Lint,
        id: HirId,
        decorate: impl for<'a> FnOnce(LintDiagnosticBuilder<'a>),
    ) {
        let (level, src) = self.lint_level_at_node(lint, id);
        struct_lint_level(self.sess, lint, level, src, None, decorate);
    }

    pub fn in_scope_traits(self, id: HirId) -> Option<&'tcx [TraitCandidate]> {
        let map = self.in_scope_traits_map(id.owner)?;
        let candidates = map.get(&id.local_id)?;
        Some(&*candidates)
    }

    pub fn named_region(self, id: HirId) -> Option<resolve_lifetime::Region> {
        debug!(?id, "named_region");
        self.named_region_map(id.owner).and_then(|map| map.get(&id.local_id).cloned())
    }

    pub fn is_late_bound(self, id: HirId) -> bool {
        self.is_late_bound_map(id.owner)
            .map_or(false, |(owner, set)| owner == id.owner && set.contains(&id.local_id))
    }

    pub fn object_lifetime_defaults(self, id: HirId) -> Option<Vec<ObjectLifetimeDefault>> {
        self.object_lifetime_defaults_map(id.owner)
    }

    pub fn late_bound_vars(self, id: HirId) -> &'tcx List<ty::BoundVariableKind> {
        self.mk_bound_variable_kinds(
            self.late_bound_vars_map(id.owner)
                .and_then(|map| map.get(&id.local_id).cloned())
                .unwrap_or_else(|| {
                    bug!("No bound vars found for {:?} ({:?})", self.hir().node_to_string(id), id)
                })
                .iter(),
        )
    }

    pub fn lifetime_scope(self, id: HirId) -> Option<LifetimeScopeForPath> {
        self.lifetime_scope_map(id.owner).and_then(|mut map| map.remove(&id.local_id))
    }

    /// Whether the `def_id` counts as const fn in the current crate, considering all active
    /// feature gates
    pub fn is_const_fn(self, def_id: DefId) -> bool {
        if self.is_const_fn_raw(def_id) {
            match self.lookup_const_stability(def_id) {
                Some(stability) if stability.level.is_unstable() => {
                    // has a `rustc_const_unstable` attribute, check whether the user enabled the
                    // corresponding feature gate.
                    self.features()
                        .declared_lib_features
                        .iter()
                        .any(|&(sym, _)| sym == stability.feature)
                }
                // functions without const stability are either stable user written
                // const fn or the user is using feature gates and we thus don't
                // care what they do
                _ => true,
            }
        } else {
            false
        }
    }
}

impl<'tcx> TyCtxtAt<'tcx> {
    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` to ensure it gets used.
    #[track_caller]
    pub fn ty_error(self) -> Ty<'tcx> {
        self.tcx.ty_error_with_message(self.span, "TyKind::Error constructed but no error reported")
    }

    /// Constructs a `TyKind::Error` type and registers a `delay_span_bug` with the given `msg to
    /// ensure it gets used.
    #[track_caller]
    pub fn ty_error_with_message(self, msg: &str) -> Ty<'tcx> {
        self.tcx.ty_error_with_message(self.span, msg)
    }
}

pub trait InternAs<T: ?Sized, R> {
    type Output;
    fn intern_with<F>(self, f: F) -> Self::Output
    where
        F: FnOnce(&T) -> R;
}

impl<I, T, R, E> InternAs<[T], R> for I
where
    E: InternIteratorElement<T, R>,
    I: Iterator<Item = E>,
{
    type Output = E::Output;
    fn intern_with<F>(self, f: F) -> Self::Output
    where
        F: FnOnce(&[T]) -> R,
    {
        E::intern_with(self, f)
    }
}

pub trait InternIteratorElement<T, R>: Sized {
    type Output;
    fn intern_with<I: Iterator<Item = Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output;
}

impl<T, R> InternIteratorElement<T, R> for T {
    type Output = R;
    fn intern_with<I: Iterator<Item = Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        f(&iter.collect::<SmallVec<[_; 8]>>())
    }
}

impl<'a, T, R> InternIteratorElement<T, R> for &'a T
where
    T: Clone + 'a,
{
    type Output = R;
    fn intern_with<I: Iterator<Item = Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        f(&iter.cloned().collect::<SmallVec<[_; 8]>>())
    }
}

impl<T, R, E> InternIteratorElement<T, R> for Result<T, E> {
    type Output = Result<R, E>;
    fn intern_with<I: Iterator<Item = Self>, F: FnOnce(&[T]) -> R>(
        mut iter: I,
        f: F,
    ) -> Self::Output {
        // This code is hot enough that it's worth specializing for the most
        // common length lists, to avoid the overhead of `SmallVec` creation.
        // The match arms are in order of frequency. The 1, 2, and 0 cases are
        // typically hit in ~95% of cases. We assume that if the upper and
        // lower bounds from `size_hint` agree they are correct.
        Ok(match iter.size_hint() {
            (1, Some(1)) => {
                let t0 = iter.next().unwrap()?;
                assert!(iter.next().is_none());
                f(&[t0])
            }
            (2, Some(2)) => {
                let t0 = iter.next().unwrap()?;
                let t1 = iter.next().unwrap()?;
                assert!(iter.next().is_none());
                f(&[t0, t1])
            }
            (0, Some(0)) => {
                assert!(iter.next().is_none());
                f(&[])
            }
            _ => f(&iter.collect::<Result<SmallVec<[_; 8]>, _>>()?),
        })
    }
}

// We are comparing types with different invariant lifetimes, so `ptr::eq`
// won't work for us.
fn ptr_eq<T, U>(t: *const T, u: *const U) -> bool {
    t as *const () == u as *const ()
}

pub fn provide(providers: &mut ty::query::Providers) {
    providers.in_scope_traits_map =
        |tcx, id| tcx.hir_crate(()).owners[id].as_ref().map(|owner_info| &owner_info.trait_map);
    providers.resolutions = |tcx, ()| &tcx.untracked_resolutions;
    providers.module_exports = |tcx, id| tcx.resolutions(()).export_map.get(&id).map(|v| &v[..]);
    providers.crate_name = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.crate_name
    };
    providers.maybe_unused_trait_import =
        |tcx, id| tcx.resolutions(()).maybe_unused_trait_imports.contains(&id);
    providers.maybe_unused_extern_crates =
        |tcx, ()| &tcx.resolutions(()).maybe_unused_extern_crates[..];
    providers.names_imported_by_glob_use = |tcx, id| {
        tcx.arena.alloc(tcx.resolutions(()).glob_map.get(&id).cloned().unwrap_or_default())
    };

    providers.lookup_stability = |tcx, id| tcx.stability().local_stability(id.expect_local());
    providers.lookup_const_stability =
        |tcx, id| tcx.stability().local_const_stability(id.expect_local());
    providers.lookup_deprecation_entry =
        |tcx, id| tcx.stability().local_deprecation_entry(id.expect_local());
    providers.extern_mod_stmt_cnum =
        |tcx, id| tcx.resolutions(()).extern_crate_map.get(&id).cloned();
    providers.output_filenames = |tcx, ()| tcx.output_filenames.clone();
    providers.features_query = |tcx, ()| tcx.sess.features_untracked();
    providers.is_panic_runtime = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.sess.contains_name(tcx.hir().krate_attrs(), sym::panic_runtime)
    };
    providers.is_compiler_builtins = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.sess.contains_name(tcx.hir().krate_attrs(), sym::compiler_builtins)
    };
    providers.has_panic_handler = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        // We want to check if the panic handler was defined in this crate
        tcx.lang_items().panic_impl().map_or(false, |did| did.is_local())
    };
}
