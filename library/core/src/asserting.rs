// Contains the machinery necessary to print useful `assert!` messages. Not intended for public
// usage, not even nightly use-cases.
//
// Based on https://github.com/dtolnay/case-studies/tree/master/autoref-specialization. When
// 'specialization' is robust enough (5 years? 10 years? Never?), `Capture` can be specialized
// to [Printable].

#![allow(missing_debug_implementations)]
#![doc(hidden)]
#![unstable(feature = "generic_assert_internals", issue = "44838")]

use crate::fmt::{Debug, Formatter};
use crate::marker::PhantomData;

// ***** TryCapture - Generic *****

/// Marker used by [Capture]
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub struct TryCaptureWithoutDebug;

/// Catches an arbitrary `E` and modifies `to` accordingly
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub trait TryCaptureGeneric<E, M> {
    /// Similar to [TryCapturePrintable] but generic to any `E`.
    fn try_capture(&self, to: &mut Capture<E, M>);
}

impl<E> TryCaptureGeneric<E, TryCaptureWithoutDebug> for &Wrapper<&E> {
    #[inline]
    fn try_capture(&self, _: &mut Capture<E, TryCaptureWithoutDebug>) {}
}

impl<E> Debug for Capture<E, TryCaptureWithoutDebug> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.write_str("N/A")
    }
}

// ***** TryCapture - Printable *****

/// Marker used by [Capture]
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub struct TryCaptureWithDebug;

/// Catches an arbitrary `E: Printable` and modifies `to` accordingly
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub trait TryCapturePrintable<E, M> {
    /// Similar as [TryCaptureGeneric] but specialized to any `E: Printable`.
    fn try_capture(&self, to: &mut Capture<E, M>);
}

impl<E> TryCapturePrintable<E, TryCaptureWithDebug> for Wrapper<&E>
where
    E: Printable,
{
    #[inline]
    fn try_capture(&self, to: &mut Capture<E, TryCaptureWithDebug>) {
        to.elem = Some(*self.0);
    }
}

impl<E> Debug for Capture<E, TryCaptureWithDebug>
where
    E: Printable,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        match self.elem {
            None => f.write_str("N/A"),
            Some(ref value) => Debug::fmt(value, f),
        }
    }
}

// ***** Others *****

/// All possible captured `assert!` elements
///
/// # Types
///
/// * `E`: **E**lement that is going to be displayed.
/// * `M`: **M**arker used to differentiate [Capture]s in regards to [Debug].
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub struct Capture<E, M> {
    // If None, then `E` does not implements [Printable] or `E` wasn't evaluated (`assert!( ... )`
    // short-circuited).
    //
    // If Some, then `E` implements [Printable] and was evaluated.
    pub elem: Option<E>,
    phantom: PhantomData<M>,
}

impl<M, T> Capture<M, T> {
    #[inline]
    pub const fn new() -> Self {
        Self { elem: None, phantom: PhantomData }
    }
}

/// Necessary for the implementations of `TryCapture*`
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub struct Wrapper<T>(pub T);

/// Tells which elements can be copied and displayed
#[unstable(feature = "generic_assert_internals", issue = "44838")]
pub trait Printable: Copy + Debug {}

impl<T> Printable for T where T: Copy + Debug {}
