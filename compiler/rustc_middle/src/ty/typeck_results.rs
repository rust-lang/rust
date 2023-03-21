use crate::{
    hir::place::Place as HirPlace,
    infer::canonical::Canonical,
    traits::ObligationCause,
    ty::{
        self, tls, BindingMode, BoundVar, CanonicalPolyFnSig, ClosureSizeProfileData,
        GenericArgKind, InternalSubsts, SubstsRef, Ty, UserSubsts,
    },
};
use rustc_data_structures::{
    fx::{FxHashMap, FxIndexMap},
    sync::Lrc,
    unord::{UnordItems, UnordSet},
};
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::{
    def::{DefKind, Res},
    def_id::{DefId, LocalDefId, LocalDefIdMap},
    hir_id::OwnerId,
    HirId, ItemLocalId, ItemLocalMap, ItemLocalSet,
};
use rustc_index::vec::{Idx, IndexVec};
use rustc_macros::HashStable;
use rustc_middle::mir::FakeReadCause;
use rustc_session::Session;
use rustc_span::Span;
use std::{collections::hash_map::Entry, hash::Hash, iter};

use super::RvalueScopes;

#[derive(TyEncodable, TyDecodable, Debug, HashStable)]
pub struct TypeckResults<'tcx> {
    /// The `HirId::owner` all `ItemLocalId`s in this table are relative to.
    pub hir_owner: OwnerId,

    /// Resolved definitions for `<T>::X` associated paths and
    /// method calls, including those of overloaded operators.
    type_dependent_defs: ItemLocalMap<Result<(DefKind, DefId), ErrorGuaranteed>>,

    /// Resolved field indices for field accesses in expressions (`S { field }`, `obj.field`)
    /// or patterns (`S { field }`). The index is often useful by itself, but to learn more
    /// about the field you also need definition of the variant to which the field
    /// belongs, but it may not exist if it's a tuple field (`tuple.0`).
    field_indices: ItemLocalMap<usize>,

    /// Stores the types for various nodes in the AST. Note that this table
    /// is not guaranteed to be populated outside inference. See
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
    pub user_provided_sigs: LocalDefIdMap<CanonicalPolyFnSig<'tcx>>,

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
    /// # use std::fmt::Debug;
    /// fn foo(x: &u32) -> impl Debug { *x }
    /// ```
    ///
    /// The function signature here would be:
    ///
    /// ```ignore (illustrative)
    /// for<'a> fn(&'a u32) -> Foo
    /// ```
    ///
    /// where `Foo` is an opaque type created for this function.
    ///
    ///
    /// The *liberated* form of this would be
    ///
    /// ```ignore (illustrative)
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
    pub used_trait_imports: Lrc<UnordSet<LocalDefId>>,

    /// If any errors occurred while type-checking this body,
    /// this field will be set to `Some(ErrorGuaranteed)`.
    pub tainted_by_errors: Option<ErrorGuaranteed>,

    /// All the opaque types that have hidden types set
    /// by this function. We also store the
    /// type here, so that mir-borrowck can use it as a hint for figuring out hidden types,
    /// even if they are only set in dead code (which doesn't show up in MIR).
    pub concrete_opaque_types: FxIndexMap<LocalDefId, ty::OpaqueHiddenType<'tcx>>,

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
    pub closure_fake_reads: FxHashMap<LocalDefId, Vec<(HirPlace<'tcx>, FakeReadCause, hir::HirId)>>,

    /// Tracks the rvalue scoping rules which defines finer scoping for rvalue expressions
    /// by applying extended parameter rules.
    /// Details may be find in `rustc_hir_analysis::check::rvalue_scopes`.
    pub rvalue_scopes: RvalueScopes,

    /// Stores the type, expression, span and optional scope span of all types
    /// that are live across the yield of this generator (if a generator).
    pub generator_interior_types: ty::Binder<'tcx, Vec<GeneratorInteriorTypeCause<'tcx>>>,

    /// Stores the predicates that apply on generator witness types.
    /// formatting modified file tests/ui/generator/retain-resume-ref.rs
    pub generator_interior_predicates:
        FxHashMap<LocalDefId, Vec<(ty::Predicate<'tcx>, ObligationCause<'tcx>)>>,

    /// We sometimes treat byte string literals (which are of type `&[u8; N]`)
    /// as `&[u8]`, depending on the pattern in which they are used.
    /// This hashset records all instances where we behave
    /// like this to allow `const_to_pat` to reliably handle this situation.
    pub treat_byte_string_as_slice: ItemLocalSet,

    /// Contains the data for evaluating the effect of feature `capture_disjoint_fields`
    /// on closure size.
    pub closure_size_eval: FxHashMap<LocalDefId, ClosureSizeProfileData<'tcx>>,
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
#[derive(TypeFoldable, TypeVisitable)]
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

// This type holds diagnostic information on generators and async functions across crate boundaries
// and is used to provide better error messages
#[derive(TyEncodable, TyDecodable, Clone, Debug, HashStable)]
pub struct GeneratorDiagnosticData<'tcx> {
    pub generator_interior_types: ty::Binder<'tcx, Vec<GeneratorInteriorTypeCause<'tcx>>>,
    pub hir_owner: DefId,
    pub nodes_types: ItemLocalMap<Ty<'tcx>>,
    pub adjustments: ItemLocalMap<Vec<ty::adjustment::Adjustment<'tcx>>>,
}

impl<'tcx> TypeckResults<'tcx> {
    pub fn new(hir_owner: OwnerId) -> TypeckResults<'tcx> {
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
            rvalue_scopes: Default::default(),
            generator_interior_types: ty::Binder::dummy(Default::default()),
            generator_interior_predicates: Default::default(),
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
    ) -> LocalTableInContext<'_, Result<(DefKind, DefId), ErrorGuaranteed>> {
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
    ) -> LocalTableInContextMut<'_, Result<(DefKind, DefId), ErrorGuaranteed>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.type_dependent_defs }
    }

    pub fn field_indices(&self) -> LocalTableInContext<'_, usize> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.field_indices }
    }

    pub fn field_indices_mut(&mut self) -> LocalTableInContextMut<'_, usize> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.field_indices }
    }

    pub fn field_index(&self, id: hir::HirId) -> usize {
        self.field_indices().get(id).cloned().expect("no index for a field")
    }

    pub fn opt_field_index(&self, id: hir::HirId) -> Option<usize> {
        self.field_indices().get(id).cloned()
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

    pub fn get_generator_diagnostic_data(&self) -> GeneratorDiagnosticData<'tcx> {
        let generator_interior_type = self.generator_interior_types.map_bound_ref(|vec| {
            vec.iter()
                .map(|item| {
                    GeneratorInteriorTypeCause {
                        ty: item.ty,
                        span: item.span,
                        scope_span: item.scope_span,
                        yield_span: item.yield_span,
                        expr: None, //FIXME: Passing expression over crate boundaries is impossible at the moment
                    }
                })
                .collect::<Vec<_>>()
        });
        GeneratorDiagnosticData {
            generator_interior_types: generator_interior_type,
            hir_owner: self.hir_owner.to_def_id(),
            nodes_types: self.node_types.clone(),
            adjustments: self.adjustments.clone(),
        }
    }

    pub fn node_type(&self, id: hir::HirId) -> Ty<'tcx> {
        self.node_type_opt(id).unwrap_or_else(|| {
            bug!("node_type: no type for node {}", tls::with(|tcx| tcx.hir().node_to_string(id)))
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

    /// Returns the type of a pattern as a monotype. Like [`expr_ty`], this function
    /// doesn't provide type parameter substitutions.
    ///
    /// [`expr_ty`]: TypeckResults::expr_ty
    pub fn pat_ty(&self, pat: &hir::Pat<'_>) -> Ty<'tcx> {
        self.node_type(pat.hir_id)
    }

    /// Returns the type of an expression as a monotype.
    ///
    /// NB (1): This is the PRE-ADJUSTMENT TYPE for the expression. That is, in
    /// some cases, we insert `Adjustment` annotations such as auto-deref or
    /// auto-ref. The type returned by this function does not consider such
    /// adjustments. See `expr_ty_adjusted()` instead.
    ///
    /// NB (2): This type doesn't provide type parameter substitutions; e.g., if you
    /// ask for the type of `id` in `id(3)`, it will return `fn(&isize) -> isize`
    /// instead of `fn(ty) -> T with T = isize`.
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
        closure_def_id: LocalDefId,
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

/// Validate that the given HirId (respectively its `local_id` part) can be
/// safely used as a key in the maps of a TypeckResults. For that to be
/// the case, the HirId must have the same `owner` as all the other IDs in
/// this table (signified by `hir_owner`). Otherwise the HirId
/// would be in a different frame of reference and using its `local_id`
/// would result in lookup errors, or worse, in silently wrong data being
/// stored/returned.
#[inline]
fn validate_hir_id_for_typeck_results(hir_owner: OwnerId, hir_id: hir::HirId) {
    if hir_id.owner != hir_owner {
        invalid_hir_id_for_typeck_results(hir_owner, hir_id);
    }
}

#[cold]
#[inline(never)]
fn invalid_hir_id_for_typeck_results(hir_owner: OwnerId, hir_id: hir::HirId) {
    ty::tls::with(|tcx| {
        bug!(
            "node {} cannot be placed in TypeckResults with hir_owner {:?}",
            tcx.hir().node_to_string(hir_id),
            hir_owner
        )
    });
}

pub struct LocalTableInContext<'a, V> {
    hir_owner: OwnerId,
    data: &'a ItemLocalMap<V>,
}

impl<'a, V> LocalTableInContext<'a, V> {
    pub fn contains_key(&self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.contains_key(&id.local_id)
    }

    pub fn get(&self, id: hir::HirId) -> Option<&'a V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.get(&id.local_id)
    }

    pub fn items(
        &'a self,
    ) -> UnordItems<(hir::ItemLocalId, &'a V), impl Iterator<Item = (hir::ItemLocalId, &'a V)>>
    {
        self.data.items().map(|(id, value)| (*id, value))
    }

    pub fn items_in_stable_order(&self) -> Vec<(ItemLocalId, &'a V)> {
        self.data.to_sorted_stable_ord()
    }
}

impl<'a, V> ::std::ops::Index<hir::HirId> for LocalTableInContext<'a, V> {
    type Output = V;

    fn index(&self, key: hir::HirId) -> &V {
        self.get(key).expect("LocalTableInContext: key not found")
    }
}

pub struct LocalTableInContextMut<'a, V> {
    hir_owner: OwnerId,
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

    pub fn extend(
        &mut self,
        items: UnordItems<(hir::HirId, V), impl Iterator<Item = (hir::HirId, V)>>,
    ) {
        self.data.extend(items.map(|(id, value)| {
            validate_hir_id_for_typeck_results(self.hir_owner, id);
            (id.local_id, value)
        }))
    }
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[debug_format = "UserType({})"]
    pub struct UserTypeAnnotationIndex {
        const START_INDEX = 0;
    }
}

/// Mapping of type annotation indices to canonical user type annotations.
pub type CanonicalUserTypeAnnotations<'tcx> =
    IndexVec<UserTypeAnnotationIndex, CanonicalUserTypeAnnotation<'tcx>>;

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct CanonicalUserTypeAnnotation<'tcx> {
    pub user_ty: Box<CanonicalUserType<'tcx>>,
    pub span: Span,
    pub inferred_ty: Ty<'tcx>,
}

/// Canonical user type annotation.
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

                        GenericArgKind::Lifetime(r) => match *r {
                            ty::ReLateBound(debruijn, br) => {
                                // We only allow a `ty::INNERMOST` index in substitutions.
                                assert_eq!(debruijn, ty::INNERMOST);
                                cvar == br.var
                            }
                            _ => false,
                        },

                        GenericArgKind::Const(ct) => match ct.kind() {
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
#[derive(Eq, Hash, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum UserType<'tcx> {
    Ty(Ty<'tcx>),

    /// The canonical type is the result of `type_of(def_id)` with the
    /// given substitutions applied.
    TypeOf(DefId, UserSubsts<'tcx>),
}
