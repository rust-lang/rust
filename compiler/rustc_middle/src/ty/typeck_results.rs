use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::iter;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::unord::{ExtendUnord, UnordItems, UnordSet};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, LocalDefIdMap};
use rustc_hir::hir_id::OwnerId;
use rustc_hir::{
    self as hir, BindingMode, ByRef, HirId, ItemLocalId, ItemLocalMap, ItemLocalSet, Mutability,
};
use rustc_index::IndexVec;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_session::Session;
use rustc_span::Span;

use super::RvalueScopes;
use crate::hir::place::Place as HirPlace;
use crate::infer::canonical::Canonical;
use crate::mir::FakeReadCause;
use crate::traits::ObligationCause;
use crate::ty::{
    self, BoundVar, CanonicalPolyFnSig, ClosureSizeProfileData, GenericArgKind, GenericArgs,
    GenericArgsRef, Ty, UserArgs, tls,
};

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
    field_indices: ItemLocalMap<FieldIdx>,

    /// Stores the types for various nodes in the AST. Note that this table
    /// is not guaranteed to be populated outside inference. See
    /// typeck::check::fn_ctxt for details.
    node_types: ItemLocalMap<Ty<'tcx>>,

    /// Stores the type parameters which were instantiated to obtain the type
    /// of this node. This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    node_args: ItemLocalMap<GenericArgsRef<'tcx>>,

    /// This will either store the canonicalized types provided by the user
    /// or the generic parameters that the user explicitly gave (if any) attached
    /// to `id`. These will not include any inferred values. The canonical form
    /// is used to capture things like `_` or other unspecified values.
    ///
    /// For example, if the user wrote `foo.collect::<Vec<_>>()`, then the
    /// canonical generic parameters would include only `for<X> { Vec<X> }`.
    ///
    /// See also `AscribeUserType` statement in MIR.
    user_provided_types: ItemLocalMap<CanonicalUserType<'tcx>>,

    /// Stores the canonicalized types provided by the user. See also
    /// `AscribeUserType` statement in MIR.
    pub user_provided_sigs: LocalDefIdMap<CanonicalPolyFnSig<'tcx>>,

    adjustments: ItemLocalMap<Vec<ty::adjustment::Adjustment<'tcx>>>,

    /// Stores the actual binding mode for all instances of [`BindingMode`].
    pat_binding_modes: ItemLocalMap<BindingMode>,

    /// Top-level patterns incompatible with Rust 2024's match ergonomics. These will be translated
    /// to a form valid in all Editions, either as a lint diagnostic or hard error.
    rust_2024_migration_desugared_pats: ItemLocalMap<Rust2024IncompatiblePatInfo>,

    /// Stores the types which were implicitly dereferenced in pattern binding modes or deref
    /// patterns for later usage in THIR lowering. For example,
    ///
    /// ```
    /// match &&Some(5i32) {
    ///     Some(n) => {},
    ///     _ => {},
    /// }
    /// ```
    /// leads to a `vec![&&Option<i32>, &Option<i32>]` and
    ///
    /// ```
    /// #![feature(deref_patterns)]
    /// match &Box::new(Some(5i32)) {
    ///     Some(n) => {},
    ///     _ => {},
    /// }
    /// ```
    /// leads to a `vec![&Box<Option<i32>>, Box<Option<i32>>]`. Empty vectors are not stored.
    ///
    /// See:
    /// <https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md#definitions>
    pat_adjustments: ItemLocalMap<Vec<ty::adjustment::PatAdjustment<'tcx>>>,

    /// Set of reference patterns that match against a match-ergonomics inserted reference
    /// (as opposed to against a reference in the scrutinee type).
    skipped_ref_pats: ItemLocalSet,

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
    /// Note that `'a` is not bound (it would be an `ReLateParam`) and
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
    /// This is used for warning unused imports.
    pub used_trait_imports: UnordSet<LocalDefId>,

    /// If any errors occurred while type-checking this body,
    /// this field will be set to `Some(ErrorGuaranteed)`.
    pub tainted_by_errors: Option<ErrorGuaranteed>,

    /// All the opaque types that have hidden types set by this function.
    /// We also store the type here, so that the compiler can use it as a hint
    /// for figuring out hidden types, even if they are only set in dead code
    /// (which doesn't show up in MIR).
    pub concrete_opaque_types: FxIndexMap<LocalDefId, ty::OpaqueHiddenType<'tcx>>,

    /// Tracks the minimum captures required for a closure;
    /// see `MinCaptureInformationMap` for more details.
    pub closure_min_captures: ty::MinCaptureInformationMap<'tcx>,

    /// Tracks the fake reads required for a closure and the reason for the fake read.
    /// When performing pattern matching for closures, there are times we don't end up
    /// reading places that are mentioned in a closure (because of _ patterns). However,
    /// to ensure the places are initialized, we introduce fake reads.
    /// Consider these two examples:
    /// ```ignore (discriminant matching with only wildcard arm)
    /// let x: u8;
    /// let c = || match x { _ => () };
    /// ```
    /// In this example, we don't need to actually read/borrow `x` in `c`, and so we don't
    /// want to capture it. However, we do still want an error here, because `x` should have
    /// to be initialized at the point where c is created. Therefore, we add a "fake read"
    /// instead.
    /// ```ignore (destructured assignments)
    /// let c = || {
    ///     let (t1, t2) = t;
    /// }
    /// ```
    /// In the second example, we capture the disjoint fields of `t` (`t.0` & `t.1`), but
    /// we never capture `t`. This becomes an issue when we build MIR as we require
    /// information on `t` in order to create place `t.0` and `t.1`. We can solve this
    /// issue by fake reading `t`.
    pub closure_fake_reads: LocalDefIdMap<Vec<(HirPlace<'tcx>, FakeReadCause, HirId)>>,

    /// Tracks the rvalue scoping rules which defines finer scoping for rvalue expressions
    /// by applying extended parameter rules.
    /// Details may be find in `rustc_hir_analysis::check::rvalue_scopes`.
    pub rvalue_scopes: RvalueScopes,

    /// Stores the predicates that apply on coroutine witness types.
    /// formatting modified file tests/ui/coroutine/retain-resume-ref.rs
    pub coroutine_stalled_predicates: FxIndexSet<(ty::Predicate<'tcx>, ObligationCause<'tcx>)>,

    /// Contains the data for evaluating the effect of feature `capture_disjoint_fields`
    /// on closure size.
    pub closure_size_eval: LocalDefIdMap<ClosureSizeProfileData<'tcx>>,

    /// Container types and field indices of `offset_of!` expressions
    offset_of_data: ItemLocalMap<(Ty<'tcx>, Vec<(VariantIdx, FieldIdx)>)>,
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
            node_args: Default::default(),
            adjustments: Default::default(),
            pat_binding_modes: Default::default(),
            pat_adjustments: Default::default(),
            rust_2024_migration_desugared_pats: Default::default(),
            skipped_ref_pats: Default::default(),
            closure_kind_origins: Default::default(),
            liberated_fn_sigs: Default::default(),
            fru_field_types: Default::default(),
            coercion_casts: Default::default(),
            used_trait_imports: Default::default(),
            tainted_by_errors: None,
            concrete_opaque_types: Default::default(),
            closure_min_captures: Default::default(),
            closure_fake_reads: Default::default(),
            rvalue_scopes: Default::default(),
            coroutine_stalled_predicates: Default::default(),
            closure_size_eval: Default::default(),
            offset_of_data: Default::default(),
        }
    }

    /// Returns the final resolution of a `QPath` in an `Expr` or `Pat` node.
    pub fn qpath_res(&self, qpath: &hir::QPath<'_>, id: HirId) -> Res {
        match *qpath {
            hir::QPath::Resolved(_, path) => path.res,
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

    pub fn field_indices(&self) -> LocalTableInContext<'_, FieldIdx> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.field_indices }
    }

    pub fn field_indices_mut(&mut self) -> LocalTableInContextMut<'_, FieldIdx> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.field_indices }
    }

    pub fn field_index(&self, id: HirId) -> FieldIdx {
        self.field_indices().get(id).cloned().expect("no index for a field")
    }

    pub fn opt_field_index(&self, id: HirId) -> Option<FieldIdx> {
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

    pub fn node_type(&self, id: HirId) -> Ty<'tcx> {
        self.node_type_opt(id).unwrap_or_else(|| {
            bug!("node_type: no type for node {}", tls::with(|tcx| tcx.hir_id_to_string(id)))
        })
    }

    pub fn node_type_opt(&self, id: HirId) -> Option<Ty<'tcx>> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.node_types.get(&id.local_id).cloned()
    }

    pub fn node_args_mut(&mut self) -> LocalTableInContextMut<'_, GenericArgsRef<'tcx>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.node_args }
    }

    pub fn node_args(&self, id: HirId) -> GenericArgsRef<'tcx> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.node_args.get(&id.local_id).cloned().unwrap_or_else(|| GenericArgs::empty())
    }

    pub fn node_args_opt(&self, id: HirId) -> Option<GenericArgsRef<'tcx>> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.node_args.get(&id.local_id).cloned()
    }

    /// Returns the type of a pattern as a monotype. Like [`expr_ty`], this function
    /// doesn't provide type parameter args.
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
    /// adjustments. See [`Self::expr_ty_adjusted`] instead.
    ///
    /// NB (2): This type doesn't provide type parameter args; e.g., if you
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

    /// Returns the computed binding mode for a `PatKind::Binding` pattern
    /// (after match ergonomics adjustments).
    pub fn extract_binding_mode(&self, s: &Session, id: HirId, sp: Span) -> BindingMode {
        self.pat_binding_modes().get(id).copied().unwrap_or_else(|| {
            s.dcx().span_bug(sp, "missing binding mode");
        })
    }

    pub fn pat_binding_modes(&self) -> LocalTableInContext<'_, BindingMode> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.pat_binding_modes }
    }

    pub fn pat_binding_modes_mut(&mut self) -> LocalTableInContextMut<'_, BindingMode> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.pat_binding_modes }
    }

    pub fn pat_adjustments(
        &self,
    ) -> LocalTableInContext<'_, Vec<ty::adjustment::PatAdjustment<'tcx>>> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.pat_adjustments }
    }

    pub fn pat_adjustments_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, Vec<ty::adjustment::PatAdjustment<'tcx>>> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.pat_adjustments }
    }

    pub fn rust_2024_migration_desugared_pats(
        &self,
    ) -> LocalTableInContext<'_, Rust2024IncompatiblePatInfo> {
        LocalTableInContext {
            hir_owner: self.hir_owner,
            data: &self.rust_2024_migration_desugared_pats,
        }
    }

    pub fn rust_2024_migration_desugared_pats_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, Rust2024IncompatiblePatInfo> {
        LocalTableInContextMut {
            hir_owner: self.hir_owner,
            data: &mut self.rust_2024_migration_desugared_pats,
        }
    }

    pub fn skipped_ref_pats(&self) -> LocalSetInContext<'_> {
        LocalSetInContext { hir_owner: self.hir_owner, data: &self.skipped_ref_pats }
    }

    pub fn skipped_ref_pats_mut(&mut self) -> LocalSetInContextMut<'_> {
        LocalSetInContextMut { hir_owner: self.hir_owner, data: &mut self.skipped_ref_pats }
    }

    /// Does the pattern recursively contain a `ref mut` binding in it?
    ///
    /// This is used to determined whether a `deref` pattern should emit a `Deref`
    /// or `DerefMut` call for its pattern scrutinee.
    ///
    /// This is computed from the typeck results since we want to make
    /// sure to apply any match-ergonomics adjustments, which we cannot
    /// determine from the HIR alone.
    pub fn pat_has_ref_mut_binding(&self, pat: &hir::Pat<'_>) -> bool {
        let mut has_ref_mut = false;
        pat.walk(|pat| {
            if let hir::PatKind::Binding(_, id, _, _) = pat.kind
                && let Some(BindingMode(ByRef::Yes(Mutability::Mut), _)) =
                    self.pat_binding_modes().get(id)
            {
                has_ref_mut = true;
                // No need to continue recursing
                false
            } else {
                true
            }
        });
        has_ref_mut
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

    pub fn is_coercion_cast(&self, hir_id: HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, hir_id);
        self.coercion_casts.contains(&hir_id.local_id)
    }

    pub fn set_coercion_cast(&mut self, id: ItemLocalId) {
        self.coercion_casts.insert(id);
    }

    pub fn coercion_casts(&self) -> &ItemLocalSet {
        &self.coercion_casts
    }

    pub fn offset_of_data(
        &self,
    ) -> LocalTableInContext<'_, (Ty<'tcx>, Vec<(VariantIdx, FieldIdx)>)> {
        LocalTableInContext { hir_owner: self.hir_owner, data: &self.offset_of_data }
    }

    pub fn offset_of_data_mut(
        &mut self,
    ) -> LocalTableInContextMut<'_, (Ty<'tcx>, Vec<(VariantIdx, FieldIdx)>)> {
        LocalTableInContextMut { hir_owner: self.hir_owner, data: &mut self.offset_of_data }
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
fn validate_hir_id_for_typeck_results(hir_owner: OwnerId, hir_id: HirId) {
    if hir_id.owner != hir_owner {
        invalid_hir_id_for_typeck_results(hir_owner, hir_id);
    }
}

#[cold]
#[inline(never)]
fn invalid_hir_id_for_typeck_results(hir_owner: OwnerId, hir_id: HirId) {
    ty::tls::with(|tcx| {
        bug!(
            "node {} cannot be placed in TypeckResults with hir_owner {:?}",
            tcx.hir_id_to_string(hir_id),
            hir_owner
        )
    });
}

pub struct LocalTableInContext<'a, V> {
    hir_owner: OwnerId,
    data: &'a ItemLocalMap<V>,
}

impl<'a, V> LocalTableInContext<'a, V> {
    pub fn contains_key(&self, id: HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.contains_key(&id.local_id)
    }

    pub fn get(&self, id: HirId) -> Option<&'a V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.get(&id.local_id)
    }

    pub fn items(
        &self,
    ) -> UnordItems<(hir::ItemLocalId, &'a V), impl Iterator<Item = (hir::ItemLocalId, &'a V)>>
    {
        self.data.items().map(|(id, value)| (*id, value))
    }

    pub fn items_in_stable_order(&self) -> Vec<(ItemLocalId, &'a V)> {
        self.data.items().map(|(&k, v)| (k, v)).into_sorted_stable_ord_by_key(|(k, _)| k)
    }
}

impl<'a, V> ::std::ops::Index<HirId> for LocalTableInContext<'a, V> {
    type Output = V;

    fn index(&self, key: HirId) -> &V {
        self.get(key).unwrap_or_else(|| {
            bug!("LocalTableInContext({:?}): key {:?} not found", self.hir_owner, key)
        })
    }
}

pub struct LocalTableInContextMut<'a, V> {
    hir_owner: OwnerId,
    data: &'a mut ItemLocalMap<V>,
}

impl<'a, V> LocalTableInContextMut<'a, V> {
    pub fn get_mut(&mut self, id: HirId) -> Option<&mut V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.get_mut(&id.local_id)
    }

    pub fn get(&mut self, id: HirId) -> Option<&V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.get(&id.local_id)
    }

    pub fn entry(&mut self, id: HirId) -> Entry<'_, hir::ItemLocalId, V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.entry(id.local_id)
    }

    pub fn insert(&mut self, id: HirId, val: V) -> Option<V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.insert(id.local_id, val)
    }

    pub fn remove(&mut self, id: HirId) -> Option<V> {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.remove(&id.local_id)
    }

    pub fn extend(&mut self, items: UnordItems<(HirId, V), impl Iterator<Item = (HirId, V)>>) {
        self.data.extend_unord(items.map(|(id, value)| {
            validate_hir_id_for_typeck_results(self.hir_owner, id);
            (id.local_id, value)
        }))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalSetInContext<'a> {
    hir_owner: OwnerId,
    data: &'a ItemLocalSet,
}

impl<'a> LocalSetInContext<'a> {
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn contains(&self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.contains(&id.local_id)
    }
}

#[derive(Debug)]
pub struct LocalSetInContextMut<'a> {
    hir_owner: OwnerId,
    data: &'a mut ItemLocalSet,
}

impl<'a> LocalSetInContextMut<'a> {
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn contains(&self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.contains(&id.local_id)
    }
    pub fn insert(&mut self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.insert(id.local_id)
    }

    pub fn remove(&mut self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_results(self.hir_owner, id);
        self.data.remove(&id.local_id)
    }
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "UserType({})"]
    pub struct UserTypeAnnotationIndex {
        const START_INDEX = 0;
    }
}

/// Mapping of type annotation indices to canonical user type annotations.
pub type CanonicalUserTypeAnnotations<'tcx> =
    IndexVec<UserTypeAnnotationIndex, CanonicalUserTypeAnnotation<'tcx>>;

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct CanonicalUserTypeAnnotation<'tcx> {
    pub user_ty: Box<CanonicalUserType<'tcx>>,
    pub span: Span,
    pub inferred_ty: Ty<'tcx>,
}

/// Canonical user type annotation.
pub type CanonicalUserType<'tcx> = Canonical<'tcx, UserType<'tcx>>;

#[derive(Copy, Clone, Debug, PartialEq, TyEncodable, TyDecodable)]
#[derive(Eq, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct UserType<'tcx> {
    pub kind: UserTypeKind<'tcx>,
    pub bounds: ty::Clauses<'tcx>,
}

impl<'tcx> UserType<'tcx> {
    pub fn new(kind: UserTypeKind<'tcx>) -> UserType<'tcx> {
        UserType { kind, bounds: ty::ListWithCachedTypeInfo::empty() }
    }

    /// A user type annotation with additional bounds that need to be enforced.
    /// These bounds are lowered from `impl Trait` in bindings.
    pub fn new_with_bounds(kind: UserTypeKind<'tcx>, bounds: ty::Clauses<'tcx>) -> UserType<'tcx> {
        UserType { kind, bounds }
    }
}

/// A user-given type annotation attached to a constant. These arise
/// from constants that are named via paths, like `Foo::<A>::new` and
/// so forth.
#[derive(Copy, Clone, Debug, PartialEq, TyEncodable, TyDecodable)]
#[derive(Eq, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum UserTypeKind<'tcx> {
    Ty(Ty<'tcx>),

    /// The canonical type is the result of `type_of(def_id)` with the
    /// given generic parameters applied.
    TypeOf(DefId, UserArgs<'tcx>),
}

pub trait IsIdentity {
    fn is_identity(&self) -> bool;
}

impl<'tcx> IsIdentity for CanonicalUserType<'tcx> {
    /// Returns `true` if this represents the generic parameters of the form `[?0, ?1, ?2]`,
    /// i.e., each thing is mapped to a canonical variable with the same index.
    fn is_identity(&self) -> bool {
        if !self.value.bounds.is_empty() {
            return false;
        }

        match self.value.kind {
            UserTypeKind::Ty(_) => false,
            UserTypeKind::TypeOf(_, user_args) => {
                if user_args.user_self_ty.is_some() {
                    return false;
                }

                iter::zip(user_args.args, BoundVar::ZERO..).all(|(kind, cvar)| {
                    match kind.unpack() {
                        GenericArgKind::Type(ty) => match ty.kind() {
                            ty::Bound(debruijn, b) => {
                                // We only allow a `ty::INNERMOST` index in generic parameters.
                                assert_eq!(*debruijn, ty::INNERMOST);
                                cvar == b.var
                            }
                            _ => false,
                        },

                        GenericArgKind::Lifetime(r) => match r.kind() {
                            ty::ReBound(debruijn, br) => {
                                // We only allow a `ty::INNERMOST` index in generic parameters.
                                assert_eq!(debruijn, ty::INNERMOST);
                                cvar == br.var
                            }
                            _ => false,
                        },

                        GenericArgKind::Const(ct) => match ct.kind() {
                            ty::ConstKind::Bound(debruijn, b) => {
                                // We only allow a `ty::INNERMOST` index in generic parameters.
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

impl<'tcx> std::fmt::Display for UserType<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.bounds.is_empty() {
            self.kind.fmt(f)
        } else {
            self.kind.fmt(f)?;
            write!(f, " + ")?;
            std::fmt::Debug::fmt(&self.bounds, f)
        }
    }
}

impl<'tcx> std::fmt::Display for UserTypeKind<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ty(arg0) => {
                ty::print::with_no_trimmed_paths!(write!(f, "Ty({})", arg0))
            }
            Self::TypeOf(arg0, arg1) => write!(f, "TypeOf({:?}, {:?})", arg0, arg1),
        }
    }
}

/// Information on a pattern incompatible with Rust 2024, for use by the error/migration diagnostic
/// emitted during THIR construction.
#[derive(TyEncodable, TyDecodable, Debug, HashStable)]
pub struct Rust2024IncompatiblePatInfo {
    /// Labeled spans for `&`s, `&mut`s, and binding modifiers incompatible with Rust 2024.
    pub primary_labels: Vec<(Span, String)>,
    /// Whether any binding modifiers occur under a non-`move` default binding mode.
    pub bad_modifiers: bool,
    /// Whether any `&` or `&mut` patterns occur under a non-`move` default binding mode.
    pub bad_ref_pats: bool,
    /// If `true`, we can give a simpler suggestion solely by eliding explicit binding modifiers.
    pub suggest_eliding_modes: bool,
}
