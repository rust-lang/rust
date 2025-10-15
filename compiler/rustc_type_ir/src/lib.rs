#![cfg_attr(feature = "nightly", rustc_diagnostic_item = "type_ir")]
// tidy-alphabetical-start
#![allow(rustc::direct_use_of_rustc_type_ir)]
#![allow(rustc::usage_of_ty_tykind)]
#![allow(rustc::usage_of_type_ir_inherent)]
#![allow(rustc::usage_of_type_ir_traits)]
#![cfg_attr(
    feature = "nightly",
    feature(associated_type_defaults, never_type, rustc_attrs, negative_impls)
)]
#![cfg_attr(feature = "nightly", allow(internal_features))]
// tidy-alphabetical-end

extern crate self as rustc_type_ir;

use std::fmt;
use std::hash::Hash;

#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable, HashStable_NoContext};

// These modules are `pub` since they are not glob-imported.
pub mod data_structures;
pub mod elaborate;
pub mod error;
pub mod fast_reject;
#[cfg_attr(feature = "nightly", rustc_diagnostic_item = "type_ir_inherent")]
pub mod inherent;
pub mod ir_print;
pub mod lang_items;
pub mod lift;
pub mod outlives;
pub mod relate;
pub mod search_graph;
pub mod solve;
pub mod walk;

// These modules are not `pub` since they are glob-imported.
#[macro_use]
mod macros;
mod binder;
mod canonical;
mod const_kind;
mod flags;
mod fold;
mod generic_arg;
mod infer_ctxt;
mod interner;
mod opaque_ty;
mod pattern;
mod predicate;
mod predicate_kind;
mod region_kind;
mod ty_info;
mod ty_kind;
mod upcast;
mod visit;

pub use AliasTyKind::*;
pub use InferTy::*;
pub use RegionKind::*;
pub use TyKind::*;
pub use Variance::*;
pub use binder::*;
pub use canonical::*;
pub use const_kind::*;
pub use flags::*;
pub use fold::*;
pub use generic_arg::*;
pub use infer_ctxt::*;
pub use interner::*;
pub use opaque_ty::*;
pub use pattern::*;
pub use predicate::*;
pub use predicate_kind::*;
pub use region_kind::*;
pub use rustc_ast_ir::{FloatTy, IntTy, Movability, Mutability, Pinnedness, UintTy};
pub use ty_info::*;
pub use ty_kind::*;
pub use upcast::*;
pub use visit::*;

rustc_index::newtype_index! {
    /// A [De Bruijn index][dbi] is a standard means of representing
    /// regions (and perhaps later types) in a higher-ranked setting. In
    /// particular, imagine a type like this:
    /// ```ignore (illustrative)
    ///    for<'a> fn(for<'b> fn(&'b isize, &'a isize), &'a char)
    /// // ^          ^            |          |           |
    /// // |          |            |          |           |
    /// // |          +------------+ 0        |           |
    /// // |                                  |           |
    /// // +----------------------------------+ 1         |
    /// // |                                              |
    /// // +----------------------------------------------+ 0
    /// ```
    /// In this type, there are two binders (the outer fn and the inner
    /// fn). We need to be able to determine, for any given region, which
    /// fn type it is bound by, the inner or the outer one. There are
    /// various ways you can do this, but a De Bruijn index is one of the
    /// more convenient and has some nice properties. The basic idea is to
    /// count the number of binders, inside out. Some examples should help
    /// clarify what I mean.
    ///
    /// Let's start with the reference type `&'b isize` that is the first
    /// argument to the inner function. This region `'b` is assigned a De
    /// Bruijn index of 0, meaning "the innermost binder" (in this case, a
    /// fn). The region `'a` that appears in the second argument type (`&'a
    /// isize`) would then be assigned a De Bruijn index of 1, meaning "the
    /// second-innermost binder". (These indices are written on the arrows
    /// in the diagram).
    ///
    /// What is interesting is that De Bruijn index attached to a particular
    /// variable will vary depending on where it appears. For example,
    /// the final type `&'a char` also refers to the region `'a` declared on
    /// the outermost fn. But this time, this reference is not nested within
    /// any other binders (i.e., it is not an argument to the inner fn, but
    /// rather the outer one). Therefore, in this case, it is assigned a
    /// De Bruijn index of 0, because the innermost binder in that location
    /// is the outer fn.
    ///
    /// [dbi]: https://en.wikipedia.org/wiki/De_Bruijn_index
    #[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
    #[encodable]
    #[orderable]
    #[debug_format = "DebruijnIndex({})"]
    #[gate_rustc_only]
    pub struct DebruijnIndex {
        const INNERMOST = 0;
    }
}

impl DebruijnIndex {
    /// Returns the resulting index when this value is moved into
    /// `amount` number of new binders. So, e.g., if you had
    ///
    ///    for<'a> fn(&'a x)
    ///
    /// and you wanted to change it to
    ///
    ///    for<'a> fn(for<'b> fn(&'a x))
    ///
    /// you would need to shift the index for `'a` into a new binder.
    #[inline]
    #[must_use]
    pub fn shifted_in(self, amount: u32) -> DebruijnIndex {
        DebruijnIndex::from_u32(self.as_u32() + amount)
    }

    /// Update this index in place by shifting it "in" through
    /// `amount` number of binders.
    #[inline]
    pub fn shift_in(&mut self, amount: u32) {
        *self = self.shifted_in(amount);
    }

    /// Returns the resulting index when this value is moved out from
    /// `amount` number of new binders.
    #[inline]
    #[must_use]
    pub fn shifted_out(self, amount: u32) -> DebruijnIndex {
        DebruijnIndex::from_u32(self.as_u32() - amount)
    }

    /// Update in place by shifting out from `amount` binders.
    #[inline]
    pub fn shift_out(&mut self, amount: u32) {
        *self = self.shifted_out(amount);
    }

    /// Adjusts any De Bruijn indices so as to make `to_binder` the
    /// innermost binder. That is, if we have something bound at `to_binder`,
    /// it will now be bound at INNERMOST. This is an appropriate thing to do
    /// when moving a region out from inside binders:
    ///
    /// ```ignore (illustrative)
    ///             for<'a>   fn(for<'b>   for<'c>   fn(&'a u32), _)
    /// // Binder:  D3           D2        D1            ^^
    /// ```
    ///
    /// Here, the region `'a` would have the De Bruijn index D3,
    /// because it is the bound 3 binders out. However, if we wanted
    /// to refer to that region `'a` in the second argument (the `_`),
    /// those two binders would not be in scope. In that case, we
    /// might invoke `shift_out_to_binder(D3)`. This would adjust the
    /// De Bruijn index of `'a` to D1 (the innermost binder).
    ///
    /// If we invoke `shift_out_to_binder` and the region is in fact
    /// bound by one of the binders we are shifting out of, that is an
    /// error (and should fail an assertion failure).
    #[inline]
    pub fn shifted_out_to_binder(self, to_binder: DebruijnIndex) -> Self {
        self.shifted_out(to_binder.as_u32() - INNERMOST.as_u32())
    }
}

pub fn debug_bound_var<T: std::fmt::Write>(
    fmt: &mut T,
    bound_index: BoundVarIndexKind,
    var: impl std::fmt::Debug,
) -> Result<(), std::fmt::Error> {
    match bound_index {
        BoundVarIndexKind::Bound(debruijn) => {
            if debruijn == INNERMOST {
                write!(fmt, "^{var:?}")
            } else {
                write!(fmt, "^{}_{:?}", debruijn.index(), var)
            }
        }
        BoundVarIndexKind::Canonical => {
            write!(fmt, "^c_{:?}", var)
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(Decodable, Encodable, HashStable_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
pub enum Variance {
    Covariant,     // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,     // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant, // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,     // T<A> <: T<B>            -- e.g., unused type parameter
}

impl Variance {
    /// `a.xform(b)` combines the variance of a context with the
    /// variance of a type with the following meaning. If we are in a
    /// context with variance `a`, and we encounter a type argument in
    /// a position with variance `b`, then `a.xform(b)` is the new
    /// variance with which the argument appears.
    ///
    /// Example 1:
    /// ```ignore (illustrative)
    /// *mut Vec<i32>
    /// ```
    /// Here, the "ambient" variance starts as covariant. `*mut T` is
    /// invariant with respect to `T`, so the variance in which the
    /// `Vec<i32>` appears is `Covariant.xform(Invariant)`, which
    /// yields `Invariant`. Now, the type `Vec<T>` is covariant with
    /// respect to its type argument `T`, and hence the variance of
    /// the `i32` here is `Invariant.xform(Covariant)`, which results
    /// (again) in `Invariant`.
    ///
    /// Example 2:
    /// ```ignore (illustrative)
    /// fn(*const Vec<i32>, *mut Vec<i32)
    /// ```
    /// The ambient variance is covariant. A `fn` type is
    /// contravariant with respect to its parameters, so the variance
    /// within which both pointer types appear is
    /// `Covariant.xform(Contravariant)`, or `Contravariant`. `*const
    /// T` is covariant with respect to `T`, so the variance within
    /// which the first `Vec<i32>` appears is
    /// `Contravariant.xform(Covariant)` or `Contravariant`. The same
    /// is true for its `i32` argument. In the `*mut T` case, the
    /// variance of `Vec<i32>` is `Contravariant.xform(Invariant)`,
    /// and hence the outermost type is `Invariant` with respect to
    /// `Vec<i32>` (and its `i32` argument).
    ///
    /// Source: Figure 1 of "Taming the Wildcards:
    /// Combining Definition- and Use-Site Variance" published in PLDI'11.
    pub fn xform(self, v: Variance) -> Variance {
        match (self, v) {
            // Figure 1, column 1.
            (Variance::Covariant, Variance::Covariant) => Variance::Covariant,
            (Variance::Covariant, Variance::Contravariant) => Variance::Contravariant,
            (Variance::Covariant, Variance::Invariant) => Variance::Invariant,
            (Variance::Covariant, Variance::Bivariant) => Variance::Bivariant,

            // Figure 1, column 2.
            (Variance::Contravariant, Variance::Covariant) => Variance::Contravariant,
            (Variance::Contravariant, Variance::Contravariant) => Variance::Covariant,
            (Variance::Contravariant, Variance::Invariant) => Variance::Invariant,
            (Variance::Contravariant, Variance::Bivariant) => Variance::Bivariant,

            // Figure 1, column 3.
            (Variance::Invariant, _) => Variance::Invariant,

            // Figure 1, column 4.
            (Variance::Bivariant, _) => Variance::Bivariant,
        }
    }
}

impl fmt::Debug for Variance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Variance::Covariant => "+",
            Variance::Contravariant => "-",
            Variance::Invariant => "o",
            Variance::Bivariant => "*",
        })
    }
}

rustc_index::newtype_index! {
    /// "Universes" are used during type- and trait-checking in the
    /// presence of `for<..>` binders to control what sets of names are
    /// visible. Universes are arranged into a tree: the root universe
    /// contains names that are always visible. Each child then adds a new
    /// set of names that are visible, in addition to those of its parent.
    /// We say that the child universe "extends" the parent universe with
    /// new names.
    ///
    /// To make this more concrete, consider this program:
    ///
    /// ```ignore (illustrative)
    /// struct Foo { }
    /// fn bar<T>(x: T) {
    ///   let y: for<'a> fn(&'a u8, Foo) = ...;
    /// }
    /// ```
    ///
    /// The struct name `Foo` is in the root universe U0. But the type
    /// parameter `T`, introduced on `bar`, is in an extended universe U1
    /// -- i.e., within `bar`, we can name both `T` and `Foo`, but outside
    /// of `bar`, we cannot name `T`. Then, within the type of `y`, the
    /// region `'a` is in a universe U2 that extends U1, because we can
    /// name it inside the fn type but not outside.
    ///
    /// Universes are used to do type- and trait-checking around these
    /// "forall" binders (also called **universal quantification**). The
    /// idea is that when, in the body of `bar`, we refer to `T` as a
    /// type, we aren't referring to any type in particular, but rather a
    /// kind of "fresh" type that is distinct from all other types we have
    /// actually declared. This is called a **placeholder** type, and we
    /// use universes to talk about this. In other words, a type name in
    /// universe 0 always corresponds to some "ground" type that the user
    /// declared, but a type name in a non-zero universe is a placeholder
    /// type -- an idealized representative of "types in general" that we
    /// use for checking generic functions.
    #[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
    #[encodable]
    #[orderable]
    #[debug_format = "U{}"]
    #[gate_rustc_only]
    pub struct UniverseIndex {}
}

impl UniverseIndex {
    pub const ROOT: UniverseIndex = UniverseIndex::ZERO;

    /// Returns the "next" universe index in order -- this new index
    /// is considered to extend all previous universes. This
    /// corresponds to entering a `forall` quantifier. So, for
    /// example, suppose we have this type in universe `U`:
    ///
    /// ```ignore (illustrative)
    /// for<'a> fn(&'a u32)
    /// ```
    ///
    /// Once we "enter" into this `for<'a>` quantifier, we are in a
    /// new universe that extends `U` -- in this new universe, we can
    /// name the region `'a`, but that region was not nameable from
    /// `U` because it was not in scope there.
    pub fn next_universe(self) -> UniverseIndex {
        UniverseIndex::from_u32(self.as_u32().checked_add(1).unwrap())
    }

    /// Returns `true` if `self` can name a name from `other` -- in other words,
    /// if the set of names in `self` is a superset of those in
    /// `other` (`self >= other`).
    pub fn can_name(self, other: UniverseIndex) -> bool {
        self >= other
    }

    /// Returns `true` if `self` cannot name some names from `other` -- in other
    /// words, if the set of names in `self` is a strict subset of
    /// those in `other` (`self < other`).
    pub fn cannot_name(self, other: UniverseIndex) -> bool {
        self < other
    }

    /// Returns `true` if `self` is the root universe, otherwise false.
    pub fn is_root(self) -> bool {
        self == Self::ROOT
    }
}

impl Default for UniverseIndex {
    fn default() -> Self {
        Self::ROOT
    }
}

rustc_index::newtype_index! {
    #[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
    #[encodable]
    #[orderable]
    #[debug_format = "{}"]
    #[gate_rustc_only]
    pub struct BoundVar {}
}

/// Represents the various closure traits in the language. This
/// will determine the type of the environment (`self`, in the
/// desugaring) argument that the closure expects.
///
/// You can get the environment type of a closure using
/// `tcx.closure_env_ty()`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum ClosureKind {
    Fn,
    FnMut,
    FnOnce,
}

impl ClosureKind {
    /// This is the initial value used when doing upvar inference.
    pub const LATTICE_BOTTOM: ClosureKind = ClosureKind::Fn;

    pub const fn as_str(self) -> &'static str {
        match self {
            ClosureKind::Fn => "Fn",
            ClosureKind::FnMut => "FnMut",
            ClosureKind::FnOnce => "FnOnce",
        }
    }

    /// Returns `true` if a type that impls this closure kind
    /// must also implement `other`.
    #[rustfmt::skip]
    pub fn extends(self, other: ClosureKind) -> bool {
        use ClosureKind::*;
        match (self, other) {
              (Fn, Fn | FnMut | FnOnce)
            | (FnMut,   FnMut | FnOnce)
            | (FnOnce,          FnOnce) => true,
            _ => false,
        }
    }
}

impl fmt::Display for ClosureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}
