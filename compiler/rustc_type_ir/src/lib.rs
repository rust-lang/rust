#![feature(associated_type_defaults)]
#![feature(fmt_helpers_for_derive)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(unwrap_infallible)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate rustc_macros;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::unify::{EqUnifyValue, UnifyKey};
use smallvec::SmallVec;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem::discriminant;

pub mod codec;
pub mod fold;
pub mod sty;
pub mod ty_info;
pub mod visit;

#[macro_use]
mod macros;
mod structural_impls;

pub use codec::*;
pub use sty::*;
pub use ty_info::*;

/// Needed so we can use #[derive(HashStable_Generic)]
pub trait HashStableContext {}

pub trait Interner: Sized {
    type AdtDef: Clone + Debug + Hash + Ord;
    type SubstsRef: Clone + Debug + Hash + Ord;
    type DefId: Clone + Debug + Hash + Ord;
    type Binder<T>;
    type Ty: Clone + Debug + Hash + Ord;
    type Const: Clone + Debug + Hash + Ord;
    type Region: Clone + Debug + Hash + Ord;
    type Predicate;
    type TypeAndMut: Clone + Debug + Hash + Ord;
    type Mutability: Clone + Debug + Hash + Ord;
    type Movability: Clone + Debug + Hash + Ord;
    type PolyFnSig: Clone + Debug + Hash + Ord;
    type ListBinderExistentialPredicate: Clone + Debug + Hash + Ord;
    type BinderListTy: Clone + Debug + Hash + Ord;
    type ListTy: Clone + Debug + Hash + Ord + IntoIterator<Item = Self::Ty>;
    type AliasTy: Clone + Debug + Hash + Ord;
    type ParamTy: Clone + Debug + Hash + Ord;
    type BoundTy: Clone + Debug + Hash + Ord;
    type PlaceholderType: Clone + Debug + Hash + Ord;
    type ErrorGuaranteed: Clone + Debug + Hash + Ord;
    type PredicateKind: Clone + Debug + Hash + PartialEq + Eq;
    type AllocId: Clone + Debug + Hash + Ord;

    type InferConst: Clone + Debug + Hash + Ord;
    type AliasConst: Clone + Debug + Hash + Ord;
    type PlaceholderConst: Clone + Debug + Hash + Ord;
    type ParamConst: Clone + Debug + Hash + Ord;
    type BoundConst: Clone + Debug + Hash + Ord;
    type InferTy: Clone + Debug + Hash + Ord;
    type ValueConst: Clone + Debug + Hash + Ord;
    type ExprConst: Clone + Debug + Hash + Ord;

    type EarlyBoundRegion: Clone + Debug + Hash + Ord;
    type BoundRegion: Clone + Debug + Hash + Ord;
    type FreeRegion: Clone + Debug + Hash + Ord;
    type RegionVid: Clone + Debug + Hash + Ord;
    type PlaceholderRegion: Clone + Debug + Hash + Ord;

    fn ty_and_mut_to_parts(ty_and_mut: Self::TypeAndMut) -> (Self::Ty, Self::Mutability);
    fn mutability_is_mut(mutbl: Self::Mutability) -> bool;
}

/// Imagine you have a function `F: FnOnce(&[T]) -> R`, plus an iterator `iter`
/// that produces `T` items. You could combine them with
/// `f(&iter.collect::<Vec<_>>())`, but this requires allocating memory for the
/// `Vec`.
///
/// This trait allows for faster implementations, intended for cases where the
/// number of items produced by the iterator is small. There is a blanket impl
/// for `T` items, but there is also a fallible impl for `Result<T, E>` items.
pub trait CollectAndApply<T, R>: Sized {
    type Output;

    /// Produce a result of type `Self::Output` from `iter`. The result will
    /// typically be produced by applying `f` on the elements produced by
    /// `iter`, though this may not happen in some impls, e.g. if an error
    /// occurred during iteration.
    fn collect_and_apply<I, F>(iter: I, f: F) -> Self::Output
    where
        I: Iterator<Item = Self>,
        F: FnOnce(&[T]) -> R;
}

/// The blanket impl that always collects all elements and applies `f`.
impl<T, R> CollectAndApply<T, R> for T {
    type Output = R;

    /// Equivalent to `f(&iter.collect::<Vec<_>>())`.
    fn collect_and_apply<I, F>(mut iter: I, f: F) -> R
    where
        I: Iterator<Item = T>,
        F: FnOnce(&[T]) -> R,
    {
        // This code is hot enough that it's worth specializing for the most
        // common length lists, to avoid the overhead of `SmallVec` creation.
        // Lengths 0, 1, and 2 typically account for ~95% of cases. If
        // `size_hint` is incorrect a panic will occur via an `unwrap` or an
        // `assert`.
        match iter.size_hint() {
            (0, Some(0)) => {
                assert!(iter.next().is_none());
                f(&[])
            }
            (1, Some(1)) => {
                let t0 = iter.next().unwrap();
                assert!(iter.next().is_none());
                f(&[t0])
            }
            (2, Some(2)) => {
                let t0 = iter.next().unwrap();
                let t1 = iter.next().unwrap();
                assert!(iter.next().is_none());
                f(&[t0, t1])
            }
            _ => f(&iter.collect::<SmallVec<[_; 8]>>()),
        }
    }
}

/// A fallible impl that will fail, without calling `f`, if there are any
/// errors during collection.
impl<T, R, E> CollectAndApply<T, R> for Result<T, E> {
    type Output = Result<R, E>;

    /// Equivalent to `Ok(f(&iter.collect::<Result<Vec<_>>>()?))`.
    fn collect_and_apply<I, F>(mut iter: I, f: F) -> Result<R, E>
    where
        I: Iterator<Item = Result<T, E>>,
        F: FnOnce(&[T]) -> R,
    {
        // This code is hot enough that it's worth specializing for the most
        // common length lists, to avoid the overhead of `SmallVec` creation.
        // Lengths 0, 1, and 2 typically account for ~95% of cases. If
        // `size_hint` is incorrect a panic will occur via an `unwrap` or an
        // `assert`, unless a failure happens first, in which case the result
        // will be an error anyway.
        Ok(match iter.size_hint() {
            (0, Some(0)) => {
                assert!(iter.next().is_none());
                f(&[])
            }
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
            _ => f(&iter.collect::<Result<SmallVec<[_; 8]>, _>>()?),
        })
    }
}

bitflags! {
    /// Flags that we track on types. These flags are propagated upwards
    /// through the type during type construction, so that we can quickly check
    /// whether the type has various kinds of types in it without recursing
    /// over the type itself.
    pub struct TypeFlags: u32 {
        // Does this have parameters? Used to determine whether substitution is
        // required.
        /// Does this have `Param`?
        const HAS_TY_PARAM                = 1 << 0;
        /// Does this have `ReEarlyBound`?
        const HAS_RE_PARAM                = 1 << 1;
        /// Does this have `ConstKind::Param`?
        const HAS_CT_PARAM                = 1 << 2;

        const HAS_PARAM                 = TypeFlags::HAS_TY_PARAM.bits
                                          | TypeFlags::HAS_RE_PARAM.bits
                                          | TypeFlags::HAS_CT_PARAM.bits;

        /// Does this have `Infer`?
        const HAS_TY_INFER                = 1 << 3;
        /// Does this have `ReVar`?
        const HAS_RE_INFER                = 1 << 4;
        /// Does this have `ConstKind::Infer`?
        const HAS_CT_INFER                = 1 << 5;

        /// Does this have inference variables? Used to determine whether
        /// inference is required.
        const HAS_INFER                 = TypeFlags::HAS_TY_INFER.bits
                                          | TypeFlags::HAS_RE_INFER.bits
                                          | TypeFlags::HAS_CT_INFER.bits;

        /// Does this have `Placeholder`?
        const HAS_TY_PLACEHOLDER          = 1 << 6;
        /// Does this have `RePlaceholder`?
        const HAS_RE_PLACEHOLDER          = 1 << 7;
        /// Does this have `ConstKind::Placeholder`?
        const HAS_CT_PLACEHOLDER          = 1 << 8;

        /// `true` if there are "names" of regions and so forth
        /// that are local to a particular fn/inferctxt
        const HAS_FREE_LOCAL_REGIONS      = 1 << 9;

        /// `true` if there are "names" of types and regions and so forth
        /// that are local to a particular fn
        const HAS_FREE_LOCAL_NAMES        = TypeFlags::HAS_TY_PARAM.bits
                                          | TypeFlags::HAS_CT_PARAM.bits
                                          | TypeFlags::HAS_TY_INFER.bits
                                          | TypeFlags::HAS_CT_INFER.bits
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits
                                          // We consider 'freshened' types and constants
                                          // to depend on a particular fn.
                                          // The freshening process throws away information,
                                          // which can make things unsuitable for use in a global
                                          // cache. Note that there is no 'fresh lifetime' flag -
                                          // freshening replaces all lifetimes with `ReErased`,
                                          // which is different from how types/const are freshened.
                                          | TypeFlags::HAS_TY_FRESH.bits
                                          | TypeFlags::HAS_CT_FRESH.bits
                                          | TypeFlags::HAS_FREE_LOCAL_REGIONS.bits
                                          | TypeFlags::HAS_RE_ERASED.bits;

        /// Does this have `Projection`?
        const HAS_TY_PROJECTION           = 1 << 10;
        /// Does this have `Inherent`?
        const HAS_TY_INHERENT             = 1 << 11;
        /// Does this have `Opaque`?
        const HAS_TY_OPAQUE               = 1 << 12;
        /// Does this have `ConstKind::Unevaluated`?
        const HAS_CT_PROJECTION           = 1 << 13;

        /// Could this type be normalized further?
        const HAS_PROJECTION              = TypeFlags::HAS_TY_PROJECTION.bits
                                          | TypeFlags::HAS_TY_OPAQUE.bits
                                          | TypeFlags::HAS_TY_INHERENT.bits
                                          | TypeFlags::HAS_CT_PROJECTION.bits;

        /// Is an error type/const reachable?
        const HAS_ERROR                   = 1 << 14;

        /// Does this have any region that "appears free" in the type?
        /// Basically anything but `ReLateBound` and `ReErased`.
        const HAS_FREE_REGIONS            = 1 << 15;

        /// Does this have any `ReLateBound` regions?
        const HAS_RE_LATE_BOUND           = 1 << 16;
        /// Does this have any `Bound` types?
        const HAS_TY_LATE_BOUND           = 1 << 17;
        /// Does this have any `ConstKind::Bound` consts?
        const HAS_CT_LATE_BOUND           = 1 << 18;
        /// Does this have any bound variables?
        /// Used to check if a global bound is safe to evaluate.
        const HAS_LATE_BOUND              = TypeFlags::HAS_RE_LATE_BOUND.bits
                                          | TypeFlags::HAS_TY_LATE_BOUND.bits
                                          | TypeFlags::HAS_CT_LATE_BOUND.bits;

        /// Does this have any `ReErased` regions?
        const HAS_RE_ERASED               = 1 << 19;

        /// Does this value have parameters/placeholders/inference variables which could be
        /// replaced later, in a way that would change the results of `impl` specialization?
        const STILL_FURTHER_SPECIALIZABLE = 1 << 20;

        /// Does this value have `InferTy::FreshTy/FreshIntTy/FreshFloatTy`?
        const HAS_TY_FRESH                = 1 << 21;

        /// Does this value have `InferConst::Fresh`?
        const HAS_CT_FRESH                = 1 << 22;

        /// Does this have `Generator` or `GeneratorWitness`?
        const HAS_TY_GENERATOR            = 1 << 23;
    }
}

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
    #[derive(HashStable_Generic)]
    #[debug_format = "DebruijnIndex({})"]
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
    debruijn: DebruijnIndex,
    var: impl std::fmt::Debug,
) -> Result<(), std::fmt::Error> {
    if debruijn == INNERMOST {
        write!(fmt, "^{:?}", var)
    } else {
        write!(fmt, "^{}_{:?}", debruijn.index(), var)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl IntTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }

    pub fn bit_width(&self) -> Option<u64> {
        Some(match *self {
            IntTy::Isize => return None,
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        })
    }

    pub fn normalize(&self, target_width: u32) -> Self {
        match self {
            IntTy::Isize => match target_width {
                16 => IntTy::I16,
                32 => IntTy::I32,
                64 => IntTy::I64,
                _ => unreachable!(),
            },
            _ => *self,
        }
    }

    pub fn to_unsigned(self) -> UintTy {
        match self {
            IntTy::Isize => UintTy::Usize,
            IntTy::I8 => UintTy::U8,
            IntTy::I16 => UintTy::U16,
            IntTy::I32 => UintTy::U32,
            IntTy::I64 => UintTy::U64,
            IntTy::I128 => UintTy::U128,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl UintTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }

    pub fn bit_width(&self) -> Option<u64> {
        Some(match *self {
            UintTy::Usize => return None,
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        })
    }

    pub fn normalize(&self, target_width: u32) -> Self {
        match self {
            UintTy::Usize => match target_width {
                16 => UintTy::U16,
                32 => UintTy::U32,
                64 => UintTy::U64,
                _ => unreachable!(),
            },
            _ => *self,
        }
    }

    pub fn to_signed(self) -> IntTy {
        match self {
            UintTy::Usize => IntTy::Isize,
            UintTy::U8 => IntTy::I8,
            UintTy::U16 => IntTy::I16,
            UintTy::U32 => IntTy::I32,
            UintTy::U64 => IntTy::I64,
            UintTy::U128 => IntTy::I128,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub fn name_str(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }

    pub fn bit_width(self) -> u64 {
        match self {
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IntVarValue {
    IntType(IntTy),
    UintType(UintTy),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FloatVarValue(pub FloatTy);

rustc_index::newtype_index! {
    /// A **ty**pe **v**ariable **ID**.
    #[debug_format = "?{}t"]
    pub struct TyVid {}
}

/// An **int**egral (`u32`, `i32`, `usize`, etc.) type **v**ariable **ID**.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub struct IntVid {
    pub index: u32,
}

/// An **float**ing-point (`f32` or `f64`) type **v**ariable **ID**.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub struct FloatVid {
    pub index: u32,
}

/// A placeholder for a type that hasn't been inferred yet.
///
/// E.g., if we have an empty array (`[]`), then we create a fresh
/// type variable for the element type since we won't know until it's
/// used what the element type is supposed to be.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub enum InferTy {
    /// A type variable.
    TyVar(TyVid),
    /// An integral type variable (`{integer}`).
    ///
    /// These are created when the compiler sees an integer literal like
    /// `1` that could be several different types (`u8`, `i32`, `u32`, etc.).
    /// We don't know until it's used what type it's supposed to be, so
    /// we create a fresh type variable.
    IntVar(IntVid),
    /// A floating-point type variable (`{float}`).
    ///
    /// These are created when the compiler sees an float literal like
    /// `1.0` that could be either an `f32` or an `f64`.
    /// We don't know until it's used what type it's supposed to be, so
    /// we create a fresh type variable.
    FloatVar(FloatVid),

    /// A [`FreshTy`][Self::FreshTy] is one that is generated as a replacement
    /// for an unbound type variable. This is convenient for caching etc. See
    /// `rustc_infer::infer::freshen` for more details.
    ///
    /// Compare with [`TyVar`][Self::TyVar].
    FreshTy(u32),
    /// Like [`FreshTy`][Self::FreshTy], but as a replacement for [`IntVar`][Self::IntVar].
    FreshIntTy(u32),
    /// Like [`FreshTy`][Self::FreshTy], but as a replacement for [`FloatVar`][Self::FloatVar].
    FreshFloatTy(u32),
}

/// Raw `TyVid` are used as the unification key for `sub_relations`;
/// they carry no values.
impl UnifyKey for TyVid {
    type Value = ();
    #[inline]
    fn index(&self) -> u32 {
        self.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> TyVid {
        TyVid::from_u32(i)
    }
    fn tag() -> &'static str {
        "TyVid"
    }
}

impl EqUnifyValue for IntVarValue {}

impl UnifyKey for IntVid {
    type Value = Option<IntVarValue>;
    #[inline] // make this function eligible for inlining - it is quite hot.
    fn index(&self) -> u32 {
        self.index
    }
    #[inline]
    fn from_index(i: u32) -> IntVid {
        IntVid { index: i }
    }
    fn tag() -> &'static str {
        "IntVid"
    }
}

impl EqUnifyValue for FloatVarValue {}

impl UnifyKey for FloatVid {
    type Value = Option<FloatVarValue>;
    #[inline]
    fn index(&self) -> u32 {
        self.index
    }
    #[inline]
    fn from_index(i: u32) -> FloatVid {
        FloatVid { index: i }
    }
    fn tag() -> &'static str {
        "FloatVid"
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Decodable, Encodable, Hash, HashStable_Generic)]
#[rustc_pass_by_value]
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

impl<CTX> HashStable<CTX> for InferTy {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        use InferTy::*;
        discriminant(self).hash_stable(ctx, hasher);
        match self {
            TyVar(_) | IntVar(_) | FloatVar(_) => {
                panic!("type variables should not be hashed: {self:?}")
            }
            FreshTy(v) | FreshIntTy(v) | FreshFloatTy(v) => v.hash_stable(ctx, hasher),
        }
    }
}

impl fmt::Debug for IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            IntVarValue::IntType(ref v) => v.fmt(f),
            IntVarValue::UintType(ref v) => v.fmt(f),
        }
    }
}

impl fmt::Debug for FloatVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for IntVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}i", self.index)
    }
}

impl fmt::Debug for FloatVid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}f", self.index)
    }
}

impl fmt::Debug for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InferTy::*;
        match *self {
            TyVar(ref v) => v.fmt(f),
            IntVar(ref v) => v.fmt(f),
            FloatVar(ref v) => v.fmt(f),
            FreshTy(v) => write!(f, "FreshTy({v:?})"),
            FreshIntTy(v) => write!(f, "FreshIntTy({v:?})"),
            FreshFloatTy(v) => write!(f, "FreshFloatTy({v:?})"),
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

impl fmt::Display for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InferTy::*;
        match *self {
            TyVar(_) => write!(f, "_"),
            IntVar(_) => write!(f, "{}", "{integer}"),
            FloatVar(_) => write!(f, "{}", "{float}"),
            FreshTy(v) => write!(f, "FreshTy({v})"),
            FreshIntTy(v) => write!(f, "FreshIntTy({v})"),
            FreshFloatTy(v) => write!(f, "FreshFloatTy({v})"),
        }
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
    #[derive(HashStable_Generic)]
    #[debug_format = "U{}"]
    pub struct UniverseIndex {}
}

impl UniverseIndex {
    pub const ROOT: UniverseIndex = UniverseIndex::from_u32(0);

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
        UniverseIndex::from_u32(self.private.checked_add(1).unwrap())
    }

    /// Returns `true` if `self` can name a name from `other` -- in other words,
    /// if the set of names in `self` is a superset of those in
    /// `other` (`self >= other`).
    pub fn can_name(self, other: UniverseIndex) -> bool {
        self.private >= other.private
    }

    /// Returns `true` if `self` cannot name some names from `other` -- in other
    /// words, if the set of names in `self` is a strict subset of
    /// those in `other` (`self < other`).
    pub fn cannot_name(self, other: UniverseIndex) -> bool {
        self.private < other.private
    }
}
