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
//! There are three traits involved in each traversal.
//! - `TypeVisitable`. This is implemented once for many types, including:
//!   - Types of interest, for which the methods delegate to the visitor.
//!   - All other types, including generic containers like `Vec` and `Option`.
//!     It defines a "skeleton" of how they should be visited.
//! - `TypeSuperVisitable`. This is implemented only for recursive types of
//!   interest, and defines the visiting "skeleton" for these types. (This
//!   excludes `Region` because it is non-recursive, i.e. it never contains
//!   other types of interest.)
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

use std::fmt;
use std::ops::ControlFlow;
use std::sync::Arc;

pub use rustc_ast_ir::visit::VisitorResult;
pub use rustc_ast_ir::{try_visit, walk_visitable_list};
use rustc_index::{Idx, IndexVec};
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::inherent::*;
use crate::{self as ty, Interner, TypeFlags};

/// This trait is implemented for every type that can be visited,
/// providing the skeleton of the traversal.
///
/// To implement this conveniently, use the derive macro located in
/// `rustc_macros`.
pub trait TypeVisitable<I: Interner>: fmt::Debug {
    /// The entry point for visiting. To visit a value `t` with a visitor `v`
    /// call: `t.visit_with(v)`.
    ///
    /// For most types, this just traverses the value, calling `visit_with` on
    /// each field/element.
    ///
    /// For types of interest (such as `Ty`), the implementation of this method
    /// that calls a visitor method specifically for that type (such as
    /// `V::visit_ty`). This is where control transfers from `TypeVisitable` to
    /// `TypeVisitor`.
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result;
}

// This trait is implemented for types of interest.
pub trait TypeSuperVisitable<I: Interner>: TypeVisitable<I> {
    /// Provides a default visit for a recursive type of interest. This should
    /// only be called within `TypeVisitor` methods, when a non-custom
    /// traversal is desired for the value of the type of interest passed to
    /// that method. For example, in `MyVisitor::visit_ty(ty)`, it is valid to
    /// call `ty.super_visit_with(self)`, but any other visiting should be done
    /// with `xyz.visit_with(self)`.
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result;
}

/// This trait is implemented for every visiting traversal. There is a visit
/// method defined for every type of interest. Each such method has a default
/// that recurses into the type's fields in a non-custom fashion.
pub trait TypeVisitor<I: Interner>: Sized {
    #[cfg(feature = "nightly")]
    type Result: VisitorResult = ();

    #[cfg(not(feature = "nightly"))]
    type Result: VisitorResult;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &ty::Binder<I, T>) -> Self::Result {
        t.super_visit_with(self)
    }

    fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
        t.super_visit_with(self)
    }

    // The default region visitor is a no-op because `Region` is non-recursive
    // and has no `super_visit_with` method to call.
    fn visit_region(&mut self, r: I::Region) -> Self::Result {
        if let ty::ReError(guar) = r.kind() {
            self.visit_error(guar)
        } else {
            Self::Result::output()
        }
    }

    fn visit_const(&mut self, c: I::Const) -> Self::Result {
        c.super_visit_with(self)
    }

    fn visit_predicate(&mut self, p: I::Predicate) -> Self::Result {
        p.super_visit_with(self)
    }

    fn visit_clauses(&mut self, c: I::Clauses) -> Self::Result {
        c.super_visit_with(self)
    }

    fn visit_error(&mut self, _guar: I::ErrorGuaranteed) -> Self::Result {
        Self::Result::output()
    }
}

///////////////////////////////////////////////////////////////////////////
// Traversal implementations.

impl<I: Interner, T: TypeVisitable<I>, U: TypeVisitable<I>> TypeVisitable<I> for (T, U) {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        try_visit!(self.0.visit_with(visitor));
        self.1.visit_with(visitor)
    }
}

impl<I: Interner, A: TypeVisitable<I>, B: TypeVisitable<I>, C: TypeVisitable<I>> TypeVisitable<I>
    for (A, B, C)
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        try_visit!(self.0.visit_with(visitor));
        try_visit!(self.1.visit_with(visitor));
        self.2.visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Option<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match self {
            Some(v) => v.visit_with(visitor),
            None => V::Result::output(),
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>, E: TypeVisitable<I>> TypeVisitable<I> for Result<T, E> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match self {
            Ok(v) => v.visit_with(visitor),
            Err(e) => e.visit_with(visitor),
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Arc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Vec<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for ThinVec<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

impl<I: Interner, T: TypeVisitable<I>, const N: usize> TypeVisitable<I> for SmallVec<[T; N]> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

// `TypeFoldable` isn't impl'd for `&[T]`. It doesn't make sense in the general
// case, because we can't return a new slice. But note that there are a couple
// of trivial impls of `TypeFoldable` for specific slice types elsewhere.
impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for &[T] {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<[T]> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

impl<I: Interner, T: TypeVisitable<I>, Ix: Idx> TypeVisitable<I> for IndexVec<Ix, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

impl<I: Interner, T: TypeVisitable<I>, S> TypeVisitable<I> for indexmap::IndexSet<T, S> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

pub trait Flags {
    fn flags(&self) -> TypeFlags;
    fn outer_exclusive_binder(&self) -> ty::DebruijnIndex;
}

pub trait TypeVisitableExt<I: Interner>: TypeVisitable<I> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool;

    /// Returns `true` if `self` has any late-bound regions that are either
    /// bound by `binder` or bound by some binder outside of `binder`.
    /// If `binder` is `ty::INNERMOST`, this indicates whether
    /// there are any late-bound regions that appear free.
    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool;

    /// Returns `true` if this type has any regions that escape `binder` (and
    /// hence are not bound by it).
    fn has_vars_bound_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.has_vars_bound_at_or_above(binder.shifted_in(1))
    }

    /// Return `true` if this type has regions that are not a part of the type.
    /// For example, `for<'a> fn(&'a i32)` return `false`, while `fn(&'a i32)`
    /// would return `true`. The latter can occur when traversing through the
    /// former.
    ///
    /// See [`HasEscapingVarsVisitor`] for more information.
    fn has_escaping_bound_vars(&self) -> bool {
        self.has_vars_bound_at_or_above(ty::INNERMOST)
    }

    fn has_aliases(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_ALIAS)
    }

    fn has_opaque_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_OPAQUE)
    }

    fn references_error(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_ERROR)
    }

    fn error_reported(&self) -> Result<(), I::ErrorGuaranteed>;

    fn has_non_region_param(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PARAM - TypeFlags::HAS_RE_PARAM)
    }

    fn has_infer_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_INFER)
    }

    fn has_infer_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER)
    }

    fn has_non_region_infer(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_INFER - TypeFlags::HAS_RE_INFER)
    }

    fn has_infer(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_INFER)
    }

    fn has_placeholders(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PLACEHOLDER)
    }

    fn has_non_region_placeholders(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PLACEHOLDER - TypeFlags::HAS_RE_PLACEHOLDER)
    }

    fn has_param(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PARAM)
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
    fn has_bound_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_BOUND)
    }
    /// True if there are any late-bound non-region variables
    fn has_non_region_bound_vars(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_BOUND_VARS - TypeFlags::HAS_RE_BOUND)
    }
    /// True if there are any bound variables
    fn has_bound_vars(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_BOUND_VARS)
    }

    /// Indicates whether this value still has parameters/placeholders/inference variables
    /// which could be replaced later, in a way that would change the results of `impl`
    /// specialization.
    fn still_further_specializable(&self) -> bool {
        self.has_type_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitableExt<I> for T {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        let res =
            self.visit_with(&mut HasTypeFlagsVisitor { flags }) == ControlFlow::Break(FoundFlags);
        res
    }

    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.visit_with(&mut HasEscapingVarsVisitor { outer_index: binder }).is_break()
    }

    fn error_reported(&self) -> Result<(), I::ErrorGuaranteed> {
        if self.references_error() {
            if let ControlFlow::Break(guar) = self.visit_with(&mut HasErrorVisitor) {
                Err(guar)
            } else {
                panic!("type flags said there was an error, but now there is not")
            }
        } else {
            Ok(())
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

// Note: this visitor traverses values down to the level of
// `Ty`/`Const`/`Predicate`, but not within those types. This is because the
// type flags at the outer layer are enough. So it's faster than it first
// looks, particular for `Ty`/`Predicate` where it's just a field access.
//
// N.B. The only case where this isn't totally true is binders, which also
// add `HAS_{RE,TY,CT}_LATE_BOUND` flag depending on the *bound variables* that
// are present, regardless of whether those bound variables are used. This
// is important for anonymization of binders in `TyCtxt::erase_regions`. We
// specifically detect this case in `visit_binder`.
impl<I: Interner> TypeVisitor<I> for HasTypeFlagsVisitor {
    type Result = ControlFlow<FoundFlags>;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &ty::Binder<I, T>) -> Self::Result {
        // If we're looking for the HAS_BINDER_VARS flag, check if the
        // binder has vars. This won't be present in the binder's bound
        // value, so we need to check here too.
        if self.flags.intersects(TypeFlags::HAS_BINDER_VARS) && !t.bound_vars().is_empty() {
            return ControlFlow::Break(FoundFlags);
        }

        t.super_visit_with(self)
    }

    #[inline]
    fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
        // Note: no `super_visit_with` call.
        let flags = t.flags();
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_region(&mut self, r: I::Region) -> Self::Result {
        // Note: no `super_visit_with` call, as usual for `Region`.
        let flags = r.flags();
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_const(&mut self, c: I::Const) -> Self::Result {
        // Note: no `super_visit_with` call.
        if c.flags().intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_predicate(&mut self, predicate: I::Predicate) -> Self::Result {
        // Note: no `super_visit_with` call.
        if predicate.flags().intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_clauses(&mut self, clauses: I::Clauses) -> Self::Result {
        // Note: no `super_visit_with` call.
        if clauses.flags().intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_error(&mut self, _guar: <I as Interner>::ErrorGuaranteed) -> Self::Result {
        if self.flags.intersects(TypeFlags::HAS_ERROR) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::Continue(())
        }
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

impl<I: Interner> TypeVisitor<I> for HasEscapingVarsVisitor {
    type Result = ControlFlow<FoundEscapingVars>;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &ty::Binder<I, T>) -> Self::Result {
        self.outer_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.outer_index.shift_out(1);
        result
    }

    #[inline]
    fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
        // If the outer-exclusive-binder is *strictly greater* than
        // `outer_index`, that means that `t` contains some content
        // bound at `outer_index` or above (because
        // `outer_exclusive_binder` is always 1 higher than the
        // content in `t`). Therefore, `t` has some escaping vars.
        if t.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_region(&mut self, r: I::Region) -> Self::Result {
        // If the region is bound by `outer_index` or anything outside
        // of outer index, then it escapes the binders we have
        // visited.
        if r.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::Continue(())
        }
    }

    fn visit_const(&mut self, ct: I::Const) -> Self::Result {
        // If the outer-exclusive-binder is *strictly greater* than
        // `outer_index`, that means that `ct` contains some content
        // bound at `outer_index` or above (because
        // `outer_exclusive_binder` is always 1 higher than the
        // content in `t`). Therefore, `t` has some escaping vars.
        if ct.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_predicate(&mut self, predicate: I::Predicate) -> Self::Result {
        if predicate.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::Continue(())
        }
    }

    #[inline]
    fn visit_clauses(&mut self, clauses: I::Clauses) -> Self::Result {
        if clauses.outer_exclusive_binder() > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::Continue(())
        }
    }
}

struct HasErrorVisitor;

impl<I: Interner> TypeVisitor<I> for HasErrorVisitor {
    type Result = ControlFlow<I::ErrorGuaranteed>;

    fn visit_error(&mut self, guar: <I as Interner>::ErrorGuaranteed) -> Self::Result {
        ControlFlow::Break(guar)
    }
}
