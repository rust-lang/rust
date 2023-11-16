use smallvec::SmallVec;
use std::fmt::Debug;
use std::hash::Hash;

use crate::{DebugWithInfcx, Mutability};

pub trait Interner: Sized {
    type DefId: Clone + Debug + Hash + Ord;
    type AdtDef: Clone + Debug + Hash + Ord;

    type GenericArgs: Clone
        + DebugWithInfcx<Self>
        + Hash
        + Ord
        + IntoIterator<Item = Self::GenericArg>;
    type GenericArg: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type Term: Clone + Debug + Hash + Ord;

    type Binder<T>;
    type TypeAndMut: Clone + Debug + Hash + Ord;
    type CanonicalVars: Clone + Debug + Hash + Eq;

    // Kinds of tys
    type Ty: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type Tys: Clone + Debug + Hash + Ord + IntoIterator<Item = Self::Ty>;
    type AliasTy: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type ParamTy: Clone + Debug + Hash + Ord;
    type BoundTy: Clone + Debug + Hash + Ord;
    type PlaceholderTy: Clone + Debug + Hash + Ord;
    type InferTy: Clone + DebugWithInfcx<Self> + Hash + Ord;

    // Things stored inside of tys
    type ErrorGuaranteed: Clone + Debug + Hash + Ord;
    type BoundExistentialPredicates: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type PolyFnSig: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type AllocId: Clone + Debug + Hash + Ord;

    // Kinds of consts
    type Const: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type InferConst: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type AliasConst: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type PlaceholderConst: Clone + Debug + Hash + Ord;
    type ParamConst: Clone + Debug + Hash + Ord;
    type BoundConst: Clone + Debug + Hash + Ord;
    type ValueConst: Clone + Debug + Hash + Ord;
    type ExprConst: Clone + DebugWithInfcx<Self> + Hash + Ord;

    // Kinds of regions
    type Region: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type EarlyParamRegion: Clone + Debug + Hash + Ord;
    type BoundRegion: Clone + Debug + Hash + Ord;
    type LateParamRegion: Clone + Debug + Hash + Ord;
    type InferRegion: Clone + DebugWithInfcx<Self> + Hash + Ord;
    type PlaceholderRegion: Clone + Debug + Hash + Ord;

    // Predicates
    type Predicate: Clone + Debug + Hash + Eq;
    type TraitPredicate: Clone + Debug + Hash + Eq;
    type RegionOutlivesPredicate: Clone + Debug + Hash + Eq;
    type TypeOutlivesPredicate: Clone + Debug + Hash + Eq;
    type ProjectionPredicate: Clone + Debug + Hash + Eq;
    type SubtypePredicate: Clone + Debug + Hash + Eq;
    type CoercePredicate: Clone + Debug + Hash + Eq;
    type ClosureKind: Clone + Debug + Hash + Eq;

    fn ty_and_mut_to_parts(ty_and_mut: Self::TypeAndMut) -> (Self::Ty, Mutability);
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
