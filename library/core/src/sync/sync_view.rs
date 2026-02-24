//! Defines [`SyncView`].

use core::clone::TrivialClone;
use core::cmp::Ordering;
use core::fmt;
use core::future::Future;
use core::hash::{Hash, Hasher};
use core::marker::{StructuralPartialEq, Tuple};
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::task::{Context, Poll};

/// `SyncView` provides _mutable_ access, also referred to as _exclusive_
/// access to the underlying value. However, it only permits _immutable_, or _shared_
/// access to the underlying value when that value is [`Sync`].
///
/// While this may seem not very useful, it allows `SyncView` to _unconditionally_
/// implement `Sync`. Indeed, the safety requirements of `Sync` state that for `SyncView`
/// to be `Sync`, it must be sound to _share_ across threads, that is, it must be sound
/// for `&SyncView` to cross thread boundaries. By design, a `&SyncView<T>` for non-`Sync` T
/// has no API whatsoever, making it useless, thus harmless, thus memory safe.
///
/// Certain constructs like [`Future`]s can only be used with _exclusive_ access,
/// and are often `Send` but not `Sync`, so `SyncView` can be used as hint to the
/// Rust compiler that something is `Sync` in practice.
///
/// ## Examples
///
/// Using a non-`Sync` future prevents the wrapping struct from being `Sync`:
///
/// ```compile_fail
/// use core::cell::Cell;
///
/// async fn other() {}
/// fn assert_sync<T: Sync>(t: T) {}
/// struct State<F> {
///     future: F
/// }
///
/// assert_sync(State {
///     future: async {
///         let cell = Cell::new(1);
///         let cell_ref = &cell;
///         other().await;
///         let value = cell_ref.get();
///     }
/// });
/// ```
///
/// `SyncView` ensures the struct is `Sync` without stripping the future of its
/// functionality:
///
/// ```
/// #![feature(exclusive_wrapper)]
/// use core::cell::Cell;
/// use core::sync::SyncView;
///
/// async fn other() {}
/// fn assert_sync<T: Sync>(t: T) {}
/// struct State<F> {
///     future: SyncView<F>
/// }
///
/// assert_sync(State {
///     future: SyncView::new(async {
///         let cell = Cell::new(1);
///         let cell_ref = &cell;
///         other().await;
///         let value = cell_ref.get();
///     })
/// });
/// ```
///
/// ## Parallels with a mutex
///
/// In some sense, `SyncView` can be thought of as a _compile-time_ version of
/// a mutex, as the borrow-checker guarantees that only one `&mut` can exist
/// for any value. This is a parallel with the fact that
/// `&` and `&mut` references together can be thought of as a _compile-time_
/// version of a read-write lock.
#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[doc(alias = "SyncWrapper")]
#[doc(alias = "SyncCell")]
#[doc(alias = "Unique")]
#[doc(alias = "Exclusive")]
// `SyncView` can't have derived `PartialOrd`, `Clone`, etc. impls as they would
// use `&` access to the inner value, violating the `Sync` impl's safety
// requirements.
#[repr(transparent)]
pub struct SyncView<T: ?Sized> {
    inner: T,
}

// See `SyncView`'s docs for justification.
#[unstable(feature = "exclusive_wrapper", issue = "98407")]
unsafe impl<T: ?Sized> Sync for SyncView<T> {}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_default", issue = "143894")]
impl<T> const Default for SyncView<T>
where
    T: [const] Default,
{
    #[inline]
    fn default() -> Self {
        Self { inner: Default::default() }
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
impl<T: ?Sized> fmt::Debug for SyncView<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("SyncView").finish_non_exhaustive()
    }
}

impl<T: Sized> SyncView<T> {
    /// Wrap a value in an `SyncView`
    #[unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[rustc_const_unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[must_use]
    #[inline]
    pub const fn new(t: T) -> Self {
        Self { inner: t }
    }

    /// Unwrap the value contained in the `SyncView`
    #[unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[rustc_const_unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[must_use]
    #[inline]
    pub const fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: ?Sized> SyncView<T> {
    /// Gets pinned exclusive access to the underlying value.
    ///
    /// `SyncView` is considered to _structurally pin_ the underlying
    /// value, which means _unpinned_ `SyncView`s can produce _unpinned_
    /// access to the underlying value, but _pinned_ `SyncView`s only
    /// produce _pinned_ access to the underlying value.
    #[unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[rustc_const_unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[must_use]
    #[inline]
    pub const fn as_pin_mut(self: Pin<&mut Self>) -> Pin<&mut T> {
        // SAFETY: `SyncView` can only produce `&mut T` if itself is unpinned
        // `Pin::map_unchecked_mut` is not const, so we do this conversion manually
        unsafe { Pin::new_unchecked(&mut self.get_unchecked_mut().inner) }
    }

    /// Build a _mutable_ reference to an `SyncView<T>` from
    /// a _mutable_ reference to a `T`. This allows you to skip
    /// building an `SyncView` with [`SyncView::new`].
    #[unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[rustc_const_unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[must_use]
    #[inline]
    pub const fn from_mut(r: &'_ mut T) -> &'_ mut SyncView<T> {
        // SAFETY: repr is ≥ C, so refs have the same layout; and `SyncView` properties are `&mut`-agnostic
        unsafe { &mut *(r as *mut T as *mut SyncView<T>) }
    }

    /// Build a _pinned mutable_ reference to an `SyncView<T>` from
    /// a _pinned mutable_ reference to a `T`. This allows you to skip
    /// building an `SyncView` with [`SyncView::new`].
    #[unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[rustc_const_unstable(feature = "exclusive_wrapper", issue = "98407")]
    #[must_use]
    #[inline]
    pub const fn from_pin_mut(r: Pin<&'_ mut T>) -> Pin<&'_ mut SyncView<T>> {
        // SAFETY: `SyncView` can only produce `&mut T` if itself is unpinned
        // `Pin::map_unchecked_mut` is not const, so we do this conversion manually
        unsafe { Pin::new_unchecked(Self::from_mut(r.get_unchecked_mut())) }
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const From<T> for SyncView<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_trait_impl", issue = "143874")]
impl<F, Args> const FnOnce<Args> for SyncView<F>
where
    F: [const] FnOnce<Args>,
    Args: Tuple,
{
    type Output = F::Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output {
        self.into_inner().call_once(args)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_trait_impl", issue = "143874")]
impl<F, Args> const FnMut<Args> for SyncView<F>
where
    F: [const] FnMut<Args>,
    Args: Tuple,
{
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output {
        self.as_mut().call_mut(args)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_trait_impl", issue = "143874")]
impl<F, Args> const Fn<Args> for SyncView<F>
where
    F: Sync + [const] Fn<Args>,
    Args: Tuple,
{
    extern "rust-call" fn call(&self, args: Args) -> Self::Output {
        self.as_ref().call(args)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_iter", issue = "92476")]
impl<T> const Iterator for SyncView<T>
where
    T: [const] Iterator + ?Sized,
{
    type Item = T::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.as_mut().next()
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
impl<T> Future for SyncView<T>
where
    T: Future + ?Sized,
{
    type Output = T::Output;

    #[inline]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.as_pin_mut().poll(cx)
    }
}

#[unstable(feature = "coroutine_trait", issue = "43122")] // also #98407
impl<R, G> Coroutine<R> for SyncView<G>
where
    G: Coroutine<R> + ?Sized,
{
    type Yield = G::Yield;
    type Return = G::Return;

    #[inline]
    fn resume(self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return> {
        G::resume(self.as_pin_mut(), arg)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const AsRef<T> for SyncView<T>
where
    T: Sync + ?Sized,
{
    /// Gets shared access to the underlying value.
    #[inline]
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T> const AsMut<T> for SyncView<T>
where
    T: ?Sized,
{
    /// Gets exclusive access to the underlying value.
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_clone", issue = "142757")]
impl<T> const Clone for SyncView<T>
where
    T: Sync + [const] Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

#[doc(hidden)]
#[unstable(feature = "trivial_clone", issue = "none")]
#[rustc_const_unstable(feature = "const_clone", issue = "142757")]
unsafe impl<T> const TrivialClone for SyncView<T> where T: Sync + [const] TrivialClone {}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
impl<T> Copy for SyncView<T> where T: Sync + Copy {}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U> const PartialEq<SyncView<U>> for SyncView<T>
where
    T: Sync + [const] PartialEq<U> + ?Sized,
    U: Sync + ?Sized,
{
    #[inline]
    fn eq(&self, other: &SyncView<U>) -> bool {
        self.inner == other.inner
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
impl<T> StructuralPartialEq for SyncView<T> where T: Sync + StructuralPartialEq + ?Sized {}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T> const Eq for SyncView<T> where T: Sync + [const] Eq + ?Sized {}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
impl<T> Hash for SyncView<T>
where
    T: Sync + Hash + ?Sized,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&self.inner, state)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T, U> const PartialOrd<SyncView<U>> for SyncView<T>
where
    T: Sync + [const] PartialOrd<U> + ?Sized,
    U: Sync + ?Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &SyncView<U>) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[unstable(feature = "exclusive_wrapper", issue = "98407")]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
impl<T> const Ord for SyncView<T>
where
    T: Sync + [const] Ord + ?Sized,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}
