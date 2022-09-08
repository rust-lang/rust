//! A visiting traversal mechanism for complex data structures that contain type
//! information.
//!
//! This is a read-only traversal of the data structure.
//!
//! This traversal has limited flexibility. Only a small number of "types of
//! interest" within the complex data structures can receive custom
//! visitation. These are the ones containing the most important type-related
//! information, such as `Ty`, `Predicate`, `Region`, and `Const`.
//!
//! There are three groups of traits involved in each traversal.
//! - `TypeVisitable`. This is implemented once for many types, including:
//!   - Types of interest, for which the the methods delegate to the
//!     visitor.
//!   - All other types, including generic containers like `Vec` and `Option`.
//!     It defines a "skeleton" of how they should be visited.
//! - `TypeSuperVisitable`. This is implemented only for each type of interest,
//!   and defines the visiting "skeleton" for these types.
//! - `TypeVisitor`. This is implemented for each visitor. This defines how
//!   types of interest are visited.
//!
//! This means each visit is a mixture of (a) generic visiting operations, and (b)
//! custom visit operations that are specific to the visitor.
//! - The `TypeVisitable` impls handle most of the traversal, and call into
//!   `TypeVisitor` when they encounter a type of interest.
//! - A `TypeVisitor` may call into another `TypeVisitable` impl, because some of
//!   the types of interest are recursive and can contain other types of interest.
//! - A `TypeVisitor` may also call into a `TypeSuperVisitable` impl, because each
//!   visitor might provide custom handling only for some types of interest, or
//!   only for some variants of each type of interest, and then use default
//!   traversal for the remaining cases.
//!
//! For example, if you have `struct S(Ty, U)` where `S: TypeVisitable` and `U:
//! TypeVisitable`, and an instance `s = S(ty, u)`, it would be visited like so:
//! ```text
//! s.visit_with(visitor) calls
//! - ty.visit_with(visitor) calls
//!   - visitor.visit_ty(ty) may call
//!     - ty.super_visit_with(visitor)
//! - u.visit_with(visitor)
//! ```
use crate::mir;
use crate::ty::{self, flags::FlagComputation, Binder, Ty, TyCtxt, TypeFlags};
use rustc_errors::ErrorGuaranteed;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sso::SsoHashSet;
use std::fmt;
use std::ops::ControlFlow;

/// This trait is implemented for every type that can be visited,
/// providing the skeleton of the traversal.
///
/// To implement this conveniently, use the derive macro located in
/// `rustc_macros`.
pub trait TypeVisitable<'tcx>: fmt::Debug + Clone {
    /// The entry point for visiting. To visit a value `t` with a visitor `v`
    /// call: `t.visit_with(v)`.
    ///
    /// For most types, this just traverses the value, calling `visit_with` on
    /// each field/element.
    ///
    /// For types of interest (such as `Ty`), the implementation of this method
    /// that calls a visitor method specifically for that type (such as
    /// `V::visit_ty`). This is where control transfers from `TypeFoldable` to
    /// `TypeVisitor`.
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy>;

    /// Returns `true` if `self` has any late-bound regions that are either
    /// bound by `binder` or bound by some binder outside of `binder`.
    /// If `binder` is `ty::INNERMOST`, this indicates whether
    /// there are any late-bound regions that appear free.
    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.visit_with(&mut HasEscapingVarsVisitor { outer_index: binder }).is_break()
    }

    /// Returns `true` if this `self` has any regions that escape `binder` (and
    /// hence are not bound by it).
    fn has_vars_bound_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.has_vars_bound_at_or_above(binder.shifted_in(1))
    }

    fn has_escaping_bound_vars(&self) -> bool {
        self.has_vars_bound_at_or_above(ty::INNERMOST)
    }

    #[instrument(level = "trace", ret)]
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.visit_with(&mut HasTypeFlagsVisitor { flags }).break_value() == Some(FoundFlags)
    }
    fn has_projections(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PROJECTION)
    }
    fn has_opaque_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_OPAQUE)
    }
    fn references_error(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_ERROR)
    }
    fn error_reported(&self) -> Option<ErrorGuaranteed> {
        if self.references_error() {
            Some(ErrorGuaranteed::unchecked_claim_error_was_emitted())
        } else {
            None
        }
    }
    fn has_param_types_or_consts(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_PARAM | TypeFlags::HAS_CT_PARAM)
    }
    fn has_infer_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_INFER)
    }
    fn has_infer_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER)
    }
    fn has_infer_types_or_consts(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_CT_INFER)
    }
    fn needs_infer(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_INFER)
    }
    fn has_placeholders(&self) -> bool {
        self.has_type_flags(
            TypeFlags::HAS_RE_PLACEHOLDER
                | TypeFlags::HAS_TY_PLACEHOLDER
                | TypeFlags::HAS_CT_PLACEHOLDER,
        )
    }
    fn needs_subst(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_SUBST)
    }
    /// "Free" regions in this context means that it has any region
    /// that is not (a) erased or (b) late-bound.
    fn has_free_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    fn has_erased_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_ERASED)
    }

    /// True if there are any un-erased free regions.
    fn has_erasable_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    /// Indicates whether this value references only 'global'
    /// generic parameters that are the same regardless of what fn we are
    /// in. This is used for caching.
    fn is_global(&self) -> bool {
        !self.has_type_flags(TypeFlags::HAS_FREE_LOCAL_NAMES)
    }

    /// True if there are any late-bound regions
    fn has_late_bound_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_LATE_BOUND)
    }

    /// Indicates whether this value still has parameters/placeholders/inference variables
    /// which could be replaced later, in a way that would change the results of `impl`
    /// specialization.
    fn still_further_specializable(&self) -> bool {
        self.has_type_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE)
    }
}

pub trait TypeSuperVisitable<'tcx>: TypeVisitable<'tcx> {
    /// Provides a default visit for a type of interest. This should only be
    /// called within `TypeVisitor` methods, when a non-custom traversal is
    /// desired for the value of the type of interest passed to that method.
    /// For example, in `MyVisitor::visit_ty(ty)`, it is valid to call
    /// `ty.super_visit_with(self)`, but any other visiting should be done
    /// with `xyz.visit_with(self)`.
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy>;
}

/// This trait is implemented for every visiting traversal. There is a visit
/// method defined for every type of interest. Each such method has a default
/// that recurses into the type's fields in a non-custom fashion.
pub trait TypeVisitor<'tcx>: Sized {
    type BreakTy = !;

    fn visit_binder<T: TypeVisitable<'tcx>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        t.super_visit_with(self)
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        r.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        c.super_visit_with(self)
    }

    fn visit_unevaluated(&mut self, uv: ty::Unevaluated<'tcx>) -> ControlFlow<Self::BreakTy> {
        uv.super_visit_with(self)
    }

    fn visit_predicate(&mut self, p: ty::Predicate<'tcx>) -> ControlFlow<Self::BreakTy> {
        p.super_visit_with(self)
    }

    fn visit_mir_const(&mut self, c: mir::ConstantKind<'tcx>) -> ControlFlow<Self::BreakTy> {
        c.super_visit_with(self)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

impl<'tcx> TyCtxt<'tcx> {
    /// Invoke `callback` on every region appearing free in `value`.
    pub fn for_each_free_region(
        self,
        value: &impl TypeVisitable<'tcx>,
        mut callback: impl FnMut(ty::Region<'tcx>),
    ) {
        self.any_free_region_meets(value, |r| {
            callback(r);
            false
        });
    }

    /// Returns `true` if `callback` returns true for every region appearing free in `value`.
    pub fn all_free_regions_meet(
        self,
        value: &impl TypeVisitable<'tcx>,
        mut callback: impl FnMut(ty::Region<'tcx>) -> bool,
    ) -> bool {
        !self.any_free_region_meets(value, |r| !callback(r))
    }

    /// Returns `true` if `callback` returns true for some region appearing free in `value`.
    pub fn any_free_region_meets(
        self,
        value: &impl TypeVisitable<'tcx>,
        callback: impl FnMut(ty::Region<'tcx>) -> bool,
    ) -> bool {
        struct RegionVisitor<F> {
            /// The index of a binder *just outside* the things we have
            /// traversed. If we encounter a bound region bound by this
            /// binder or one outer to it, it appears free. Example:
            ///
            /// ```ignore (illustrative)
            ///       for<'a> fn(for<'b> fn(), T)
            /// // ^          ^          ^     ^
            /// // |          |          |     | here, would be shifted in 1
            /// // |          |          | here, would be shifted in 2
            /// // |          | here, would be `INNERMOST` shifted in by 1
            /// // | here, initially, binder would be `INNERMOST`
            /// ```
            ///
            /// You see that, initially, *any* bound value is free,
            /// because we've not traversed any binders. As we pass
            /// through a binder, we shift the `outer_index` by 1 to
            /// account for the new binder that encloses us.
            outer_index: ty::DebruijnIndex,
            callback: F,
        }

        impl<'tcx, F> TypeVisitor<'tcx> for RegionVisitor<F>
        where
            F: FnMut(ty::Region<'tcx>) -> bool,
        {
            type BreakTy = ();

            fn visit_binder<T: TypeVisitable<'tcx>>(
                &mut self,
                t: &Binder<'tcx, T>,
            ) -> ControlFlow<Self::BreakTy> {
                self.outer_index.shift_in(1);
                let result = t.super_visit_with(self);
                self.outer_index.shift_out(1);
                result
            }

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
                match *r {
                    ty::ReLateBound(debruijn, _) if debruijn < self.outer_index => {
                        ControlFlow::CONTINUE
                    }
                    _ => {
                        if (self.callback)(r) {
                            ControlFlow::BREAK
                        } else {
                            ControlFlow::CONTINUE
                        }
                    }
                }
            }

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                // We're only interested in types involving regions
                if ty.flags().intersects(TypeFlags::HAS_FREE_REGIONS) {
                    ty.super_visit_with(self)
                } else {
                    ControlFlow::CONTINUE
                }
            }
        }

        value.visit_with(&mut RegionVisitor { outer_index: ty::INNERMOST, callback }).is_break()
    }

    /// Returns a set of all late-bound regions that are constrained
    /// by `value`, meaning that if we instantiate those LBR with
    /// variables and equate `value` with something else, those
    /// variables will also be equated.
    pub fn collect_constrained_late_bound_regions<T>(
        self,
        value: &Binder<'tcx, T>,
    ) -> FxHashSet<ty::BoundRegionKind>
    where
        T: TypeVisitable<'tcx>,
    {
        self.collect_late_bound_regions(value, true)
    }

    /// Returns a set of all late-bound regions that appear in `value` anywhere.
    pub fn collect_referenced_late_bound_regions<T>(
        self,
        value: &Binder<'tcx, T>,
    ) -> FxHashSet<ty::BoundRegionKind>
    where
        T: TypeVisitable<'tcx>,
    {
        self.collect_late_bound_regions(value, false)
    }

    fn collect_late_bound_regions<T>(
        self,
        value: &Binder<'tcx, T>,
        just_constraint: bool,
    ) -> FxHashSet<ty::BoundRegionKind>
    where
        T: TypeVisitable<'tcx>,
    {
        let mut collector = LateBoundRegionsCollector::new(just_constraint);
        let result = value.as_ref().skip_binder().visit_with(&mut collector);
        assert!(result.is_continue()); // should never have stopped early
        collector.regions
    }
}

pub struct ValidateBoundVars<'tcx> {
    bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
    binder_index: ty::DebruijnIndex,
    // We may encounter the same variable at different levels of binding, so
    // this can't just be `Ty`
    visited: SsoHashSet<(ty::DebruijnIndex, Ty<'tcx>)>,
}

impl<'tcx> ValidateBoundVars<'tcx> {
    pub fn new(bound_vars: &'tcx ty::List<ty::BoundVariableKind>) -> Self {
        ValidateBoundVars {
            bound_vars,
            binder_index: ty::INNERMOST,
            visited: SsoHashSet::default(),
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for ValidateBoundVars<'tcx> {
    type BreakTy = ();

    fn visit_binder<T: TypeVisitable<'tcx>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if t.outer_exclusive_binder() < self.binder_index
            || !self.visited.insert((self.binder_index, t))
        {
            return ControlFlow::BREAK;
        }
        match *t.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn == self.binder_index => {
                if self.bound_vars.len() <= bound_ty.var.as_usize() {
                    bug!("Not enough bound vars: {:?} not found in {:?}", t, self.bound_vars);
                }
                let list_var = self.bound_vars[bound_ty.var.as_usize()];
                match list_var {
                    ty::BoundVariableKind::Ty(kind) => {
                        if kind != bound_ty.kind {
                            bug!(
                                "Mismatched type kinds: {:?} doesn't var in list {:?}",
                                bound_ty.kind,
                                list_var
                            );
                        }
                    }
                    _ => {
                        bug!("Mismatched bound variable kinds! Expected type, found {:?}", list_var)
                    }
                }
            }

            _ => (),
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            ty::ReLateBound(index, br) if index == self.binder_index => {
                if self.bound_vars.len() <= br.var.as_usize() {
                    bug!("Not enough bound vars: {:?} not found in {:?}", br, self.bound_vars);
                }
                let list_var = self.bound_vars[br.var.as_usize()];
                match list_var {
                    ty::BoundVariableKind::Region(kind) => {
                        if kind != br.kind {
                            bug!(
                                "Mismatched region kinds: {:?} doesn't match var ({:?}) in list ({:?})",
                                br.kind,
                                list_var,
                                self.bound_vars
                            );
                        }
                    }
                    _ => bug!(
                        "Mismatched bound variable kinds! Expected region, found {:?}",
                        list_var
                    ),
                }
            }

            _ => (),
        };

        r.super_visit_with(self)
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct FoundEscapingVars;

/// An "escaping var" is a bound var whose binder is not part of `t`. A bound var can be a
/// bound region or a bound type.
///
/// So, for example, consider a type like the following, which has two binders:
///
///    for<'a> fn(x: for<'b> fn(&'a isize, &'b isize))
///    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ outer scope
///                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~  inner scope
///
/// This type has *bound regions* (`'a`, `'b`), but it does not have escaping regions, because the
/// binders of both `'a` and `'b` are part of the type itself. However, if we consider the *inner
/// fn type*, that type has an escaping region: `'a`.
///
/// Note that what I'm calling an "escaping var" is often just called a "free var". However,
/// we already use the term "free var". It refers to the regions or types that we use to represent
/// bound regions or type params on a fn definition while we are type checking its body.
///
/// To clarify, conceptually there is no particular difference between
/// an "escaping" var and a "free" var. However, there is a big
/// difference in practice. Basically, when "entering" a binding
/// level, one is generally required to do some sort of processing to
/// a bound var, such as replacing it with a fresh/placeholder
/// var, or making an entry in the environment to represent the
/// scope to which it is attached, etc. An escaping var represents
/// a bound var for which this processing has not yet been done.
struct HasEscapingVarsVisitor {
    /// Anything bound by `outer_index` or "above" is escaping.
    outer_index: ty::DebruijnIndex,
}

impl<'tcx> TypeVisitor<'tcx> for HasEscapingVarsVisitor {
    type BreakTy = FoundEscapingVars;

    fn visit_binder<T: TypeVisitable<'tcx>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.outer_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.outer_index.shift_out(1);
        result
    }

    #[inline]
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // If the outer-exclusive-binder is *strictly greater* than
        // `outer_index`, that means that `t` contains some content
        // bound at `outer_index` or above (because
        // `outer_exclusive_binder` is always 1 higher than the
        // content in `t`). Therefore, `t` has some escaping vars.
        if t.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::CONTINUE
        }
    }

    #[inline]
    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        // If the region is bound by `outer_index` or anything outside
        // of outer index, then it escapes the binders we have
        // visited.
        if r.bound_at_or_above_binder(self.outer_index) {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::CONTINUE
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        // we don't have a `visit_infer_const` callback, so we have to
        // hook in here to catch this case (annoying...), but
        // otherwise we do want to remember to visit the rest of the
        // const, as it has types/regions embedded in a lot of other
        // places.
        match ct.kind() {
            ty::ConstKind::Bound(debruijn, _) if debruijn >= self.outer_index => {
                ControlFlow::Break(FoundEscapingVars)
            }
            _ => ct.super_visit_with(self),
        }
    }

    #[inline]
    fn visit_predicate(&mut self, predicate: ty::Predicate<'tcx>) -> ControlFlow<Self::BreakTy> {
        if predicate.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::CONTINUE
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct FoundFlags;

// FIXME: Optimize for checking for infer flags
struct HasTypeFlagsVisitor {
    flags: ty::TypeFlags,
}

impl std::fmt::Debug for HasTypeFlagsVisitor {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.flags.fmt(fmt)
    }
}

impl<'tcx> TypeVisitor<'tcx> for HasTypeFlagsVisitor {
    type BreakTy = FoundFlags;

    #[inline]
    #[instrument(skip(self), level = "trace", ret)]
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        let flags = t.flags();
        trace!(t.flags=?t.flags());
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    #[inline]
    #[instrument(skip(self), level = "trace", ret)]
    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        let flags = r.type_flags();
        trace!(r.flags=?flags);
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    #[inline]
    #[instrument(level = "trace", ret)]
    fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        let flags = FlagComputation::for_const(c);
        trace!(r.flags=?flags);
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    #[inline]
    #[instrument(level = "trace", ret)]
    fn visit_unevaluated(&mut self, uv: ty::Unevaluated<'tcx>) -> ControlFlow<Self::BreakTy> {
        let flags = FlagComputation::for_unevaluated_const(uv);
        trace!(r.flags=?flags);
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    #[inline]
    #[instrument(level = "trace", ret)]
    fn visit_predicate(&mut self, predicate: ty::Predicate<'tcx>) -> ControlFlow<Self::BreakTy> {
        debug!(
            "HasTypeFlagsVisitor: predicate={:?} predicate.flags={:?} self.flags={:?}",
            predicate,
            predicate.flags(),
            self.flags
        );
        if predicate.flags().intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }
}

/// Collects all the late-bound regions at the innermost binding level
/// into a hash set.
struct LateBoundRegionsCollector {
    current_index: ty::DebruijnIndex,
    regions: FxHashSet<ty::BoundRegionKind>,

    /// `true` if we only want regions that are known to be
    /// "constrained" when you equate this type with another type. In
    /// particular, if you have e.g., `&'a u32` and `&'b u32`, equating
    /// them constraints `'a == 'b`. But if you have `<&'a u32 as
    /// Trait>::Foo` and `<&'b u32 as Trait>::Foo`, normalizing those
    /// types may mean that `'a` and `'b` don't appear in the results,
    /// so they are not considered *constrained*.
    just_constrained: bool,
}

impl LateBoundRegionsCollector {
    fn new(just_constrained: bool) -> Self {
        LateBoundRegionsCollector {
            current_index: ty::INNERMOST,
            regions: Default::default(),
            just_constrained,
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for LateBoundRegionsCollector {
    fn visit_binder<T: TypeVisitable<'tcx>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.current_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.current_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs to a projection, as they may not appear
        // in the normalized form
        if self.just_constrained {
            if let ty::Projection(..) | ty::Opaque(..) = t.kind() {
                return ControlFlow::CONTINUE;
            }
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs of an unevaluated const, as they may not appear
        // in the normalized form
        if self.just_constrained {
            if let ty::ConstKind::Unevaluated(..) = c.kind() {
                return ControlFlow::CONTINUE;
            }
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::ReLateBound(debruijn, br) = *r {
            if debruijn == self.current_index {
                self.regions.insert(br.kind);
            }
        }
        ControlFlow::CONTINUE
    }
}

/// Finds the max universe present
pub struct MaxUniverse {
    max_universe: ty::UniverseIndex,
}

impl MaxUniverse {
    pub fn new() -> Self {
        MaxUniverse { max_universe: ty::UniverseIndex::ROOT }
    }

    pub fn max_universe(self) -> ty::UniverseIndex {
        self.max_universe
    }
}

impl<'tcx> TypeVisitor<'tcx> for MaxUniverse {
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::Placeholder(placeholder) = t.kind() {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::consts::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::ConstKind::Placeholder(placeholder) = c.kind() {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::RePlaceholder(placeholder) = *r {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        ControlFlow::CONTINUE
    }
}
