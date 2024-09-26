#![unstable(feature = "async_drop", issue = "126482")]

use crate::fmt;
use crate::future::{Future, IntoFuture};
use crate::intrinsics::discriminant_value;
use crate::marker::{DiscriminantKind, PhantomPinned};
use crate::mem::MaybeUninit;
use crate::pin::Pin;
use crate::task::{Context, Poll, ready};

/// Asynchronously drops a value by running `AsyncDrop::async_drop`
/// on a value and its fields recursively.
#[unstable(feature = "async_drop", issue = "126482")]
pub fn async_drop<T>(value: T) -> AsyncDropOwning<T> {
    AsyncDropOwning { value: MaybeUninit::new(value), dtor: None, _pinned: PhantomPinned }
}

/// A future returned by the [`async_drop`].
#[unstable(feature = "async_drop", issue = "126482")]
pub struct AsyncDropOwning<T> {
    value: MaybeUninit<T>,
    dtor: Option<AsyncDropInPlace<T>>,
    _pinned: PhantomPinned,
}

#[unstable(feature = "async_drop", issue = "126482")]
impl<T> fmt::Debug for AsyncDropOwning<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncDropOwning").finish_non_exhaustive()
    }
}

#[unstable(feature = "async_drop", issue = "126482")]
impl<T> Future for AsyncDropOwning<T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: Self is pinned thus it is ok to store references to self
        unsafe {
            let this = self.get_unchecked_mut();
            let dtor = Pin::new_unchecked(
                this.dtor.get_or_insert_with(|| async_drop_in_place(this.value.as_mut_ptr())),
            );
            // AsyncDestuctors are idempotent so Self gets idempotency as well
            dtor.poll(cx)
        }
    }
}

#[lang = "async_drop_in_place"]
#[allow(unconditional_recursion)]
// FIXME: Consider if `#[rustc_diagnostic_item = "ptr_drop_in_place"]` is needed?
unsafe fn async_drop_in_place_raw<T: ?Sized>(
    to_drop: *mut T,
) -> <T as AsyncDestruct>::AsyncDestructor {
    // Code here does not matter - this is replaced by the
    // real async drop glue constructor by the compiler.

    // SAFETY: see comment above
    unsafe { async_drop_in_place_raw(to_drop) }
}

/// Creates the asynchronous destructor of the pointed-to value.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `to_drop` must be [valid](crate::ptr#safety) for both reads and writes.
///
/// * `to_drop` must be properly aligned, even if `T` has size 0.
///
/// * `to_drop` must be nonnull, even if `T` has size 0.
///
/// * The value `to_drop` points to must be valid for async dropping,
///   which may mean it must uphold additional invariants. These
///   invariants depend on the type of the value being dropped. For
///   instance, when dropping a Box, the box's pointer to the heap must
///   be valid.
///
/// * While `async_drop_in_place` is executing or the returned async
///   destructor is alive, the only way to access parts of `to_drop`
///   is through the `self: Pin<&mut Self>` references supplied to
///   the `AsyncDrop::async_drop` methods that `async_drop_in_place`
///   or `AsyncDropInPlace<T>::poll` invokes. This usually means the
///   returned future stores the `to_drop` pointer and user is required
///   to guarantee that dropped value doesn't move.
///
#[unstable(feature = "async_drop", issue = "126482")]
pub unsafe fn async_drop_in_place<T: ?Sized>(to_drop: *mut T) -> AsyncDropInPlace<T> {
    // SAFETY: `async_drop_in_place_raw` has the same safety requirements
    unsafe { AsyncDropInPlace(async_drop_in_place_raw(to_drop)) }
}

/// A future returned by the [`async_drop_in_place`].
#[unstable(feature = "async_drop", issue = "126482")]
pub struct AsyncDropInPlace<T: ?Sized>(<T as AsyncDestruct>::AsyncDestructor);

#[unstable(feature = "async_drop", issue = "126482")]
impl<T: ?Sized> fmt::Debug for AsyncDropInPlace<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncDropInPlace").finish_non_exhaustive()
    }
}

#[unstable(feature = "async_drop", issue = "126482")]
impl<T: ?Sized> Future for AsyncDropInPlace<T> {
    type Output = ();

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: This code simply forwards poll call to the inner future
        unsafe { Pin::new_unchecked(&mut self.get_unchecked_mut().0) }.poll(cx)
    }
}

// FIXME(zetanumbers): Add same restrictions on AsyncDrop impls as
//   with Drop impls
/// Custom code within the asynchronous destructor.
#[unstable(feature = "async_drop", issue = "126482")]
#[lang = "async_drop"]
pub trait AsyncDrop {
    /// A future returned by the [`AsyncDrop::async_drop`] to be part
    /// of the async destructor.
    #[unstable(feature = "async_drop", issue = "126482")]
    type Dropper<'a>: Future<Output = ()>
    where
        Self: 'a;

    /// Constructs the asynchronous destructor for this type.
    #[unstable(feature = "async_drop", issue = "126482")]
    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_>;
}

#[lang = "async_destruct"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
trait AsyncDestruct {
    type AsyncDestructor: Future<Output = ()>;
}

/// Basically calls `AsyncDrop::async_drop` with pointer. Used to simplify
/// generation of the code for `async_drop_in_place_raw`
#[lang = "surface_async_drop_in_place"]
async unsafe fn surface_async_drop_in_place<T: AsyncDrop + ?Sized>(ptr: *mut T) {
    // SAFETY: We call this from async drop `async_drop_in_place_raw`
    //   which has the same safety requirements
    unsafe { <T as AsyncDrop>::async_drop(Pin::new_unchecked(&mut *ptr)).await }
}

/// Basically calls `Drop::drop` with pointer. Used to simplify generation
/// of the code for `async_drop_in_place_raw`
#[allow(drop_bounds)]
#[lang = "async_drop_surface_drop_in_place"]
async unsafe fn surface_drop_in_place<T: Drop + ?Sized>(ptr: *mut T) {
    // SAFETY: We call this from async drop `async_drop_in_place_raw`
    //   which has the same safety requirements
    unsafe { crate::ops::fallback_surface_drop(&mut *ptr) }
}

/// Wraps a future to continue outputting `Poll::Ready(())` once after
/// wrapped future completes by returning `Poll::Ready(())` on poll. This
/// is useful for constructing async destructors to guarantee this
/// "fuse" property
//
// FIXME: Consider optimizing combinators to not have to use fuse in majority
// of cases, perhaps by adding `#[(rustc_)idempotent(_future)]` attribute for
// async functions and blocks with the unit return type. However current layout
// optimizations currently encode `None` case into the async block's discriminant.
struct Fuse<T> {
    inner: Option<T>,
}

#[lang = "async_drop_fuse"]
fn fuse<T>(inner: T) -> Fuse<T> {
    Fuse { inner: Some(inner) }
}

impl<T> Future for Fuse<T>
where
    T: Future<Output = ()>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: pin projection into `self.inner`
        unsafe {
            let this = self.get_unchecked_mut();
            if let Some(inner) = &mut this.inner {
                ready!(Pin::new_unchecked(inner).poll(cx));
                this.inner = None;
            }
        }
        Poll::Ready(())
    }
}

/// Async destructor for arrays and slices.
#[lang = "async_drop_slice"]
async unsafe fn slice<T>(s: *mut [T]) {
    let len = s.len();
    let ptr = s.as_mut_ptr();
    for i in 0..len {
        // SAFETY: we iterate over elements of `s` slice
        unsafe { async_drop_in_place_raw(ptr.add(i)).await }
    }
}

/// Constructs a chain of two futures, which awaits them sequentially as
/// a future.
#[lang = "async_drop_chain"]
async fn chain<F, G>(first: F, last: G)
where
    F: IntoFuture<Output = ()>,
    G: IntoFuture<Output = ()>,
{
    first.await;
    last.await;
}

/// Basically a lazy version of `async_drop_in_place`. Returns a future
/// that would call `AsyncDrop::async_drop` on a first poll.
///
/// # Safety
///
/// Same as `async_drop_in_place` except is lazy to avoid creating
/// multiple mutable references.
#[lang = "async_drop_defer"]
async unsafe fn defer<T: ?Sized>(to_drop: *mut T) {
    // SAFETY: same safety requirements as `async_drop_in_place`
    unsafe { async_drop_in_place(to_drop) }.await
}

/// If `T`'s discriminant is equal to the stored one then awaits `M`
/// otherwise awaits the `O`.
///
/// # Safety
///
/// Users should carefully manage the returned future, since it would
/// try creating an immutable reference from `this` and get pointee's
/// discriminant.
// FIXME(zetanumbers): Send and Sync impls
#[lang = "async_drop_either"]
async unsafe fn either<O: IntoFuture<Output = ()>, M: IntoFuture<Output = ()>, T>(
    other: O,
    matched: M,
    this: *mut T,
    discr: <T as DiscriminantKind>::Discriminant,
) {
    // SAFETY: Guaranteed by the safety section of this funtion's documentation
    if unsafe { discriminant_value(&*this) } == discr {
        drop(other);
        matched.await
    } else {
        drop(matched);
        other.await
    }
}

#[lang = "async_drop_deferred_drop_in_place"]
async unsafe fn deferred_drop_in_place<T>(to_drop: *mut T) {
    // SAFETY: same safety requirements as with drop_in_place (implied by
    // function's name)
    unsafe { crate::ptr::drop_in_place(to_drop) }
}

/// Used for noop async destructors. We don't use [`core::future::Ready`]
/// because it panics after its second poll, which could be potentially
/// bad if that would happen during the cleanup.
#[derive(Clone, Copy)]
struct Noop;

#[lang = "async_drop_noop"]
fn noop() -> Noop {
    Noop
}

impl Future for Noop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}
