#![unstable(feature = "async_drop", issue = "none")]

use crate::fmt;
use crate::future::{Future, IntoFuture};
use crate::intrinsics::discriminant_value;
use crate::marker::DiscriminantKind;
use crate::mem::MaybeUninit;
use crate::pin::Pin;
use crate::task::{ready, Context, Poll};

/// Asynchronously drops a value by running `AsyncDrop::async_drop`
/// on a value and its fields recursively.
// TODO: fuse the output future
#[unstable(feature = "async_drop", issue = "none")]
pub async fn async_drop<T>(value: T) {
    let mut value = MaybeUninit::new(value);
    // SAFETY: value pointer stays valid when needed
    unsafe { async_drop_in_place(value.as_mut_ptr()) }.await;
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
///   is through the `self: Pin<&mut Self>` references supplied to the
///   `AsyncDrop::async_drop` methods that `async_drop_in_place` or
///   `AsyncDropInPlace<T>::poll` invokes.
///
#[unstable(feature = "async_drop", issue = "none")]
pub unsafe fn async_drop_in_place<T: ?Sized>(to_drop: *mut T) -> AsyncDropInPlace<T> {
    // SAFETY: `async_drop_in_place_raw` has the same safety requirements
    unsafe { AsyncDropInPlace(async_drop_in_place_raw(to_drop)) }
}

/// A future returned by the [`async_drop_in_place`].
#[unstable(feature = "async_drop", issue = "none")]
pub struct AsyncDropInPlace<T: ?Sized>(<T as AsyncDestruct>::AsyncDestructor);

#[unstable(feature = "async_drop", issue = "none")]
impl<T: ?Sized> fmt::Debug for AsyncDropInPlace<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncDropInPlace").finish_non_exhaustive()
    }
}

#[unstable(feature = "async_drop", issue = "none")]
impl<T: ?Sized> Future for AsyncDropInPlace<T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: This code simply forwards poll call to the inner future
        unsafe { self.map_unchecked_mut(|p| &mut p.0).poll(cx) }
    }
}

/// Custom code within the asynchronous destructor.
#[unstable(feature = "async_drop", issue = "none")]
#[lang = "async_drop"]
pub trait AsyncDrop {
    /// A future returned by the [`AsyncDrop::async_drop`] to be part
    /// of the async destructor.
    // FIXME: should this future be always be `!Unpin`?
    #[unstable(feature = "async_drop", issue = "none")]
    type Dropper<'a>: Future<Output = ()>
    where
        Self: 'a;

    /// Constructs the asynchronous destructor for this type.
    #[unstable(feature = "async_drop", issue = "none")]
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
unsafe fn surface_async_drop_in_place<T: AsyncDrop + ?Sized>(
    ptr: *mut T,
) -> <T as AsyncDrop>::Dropper<'static> {
    // SAFETY: We call this from async drop `async_drop_in_place_raw`
    //   which has the same safety requirements
    unsafe { <T as AsyncDrop>::async_drop(Pin::new_unchecked(&mut *ptr)) }
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

/// Wraps a future to continue outputing `Poll::Ready(())` once after
/// wrapped future completes by returning `Poll::Ready(())` on poll. This
/// is useful for constructing async destructors to guarantee this
/// "fuse" property
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
// NOTE: Returned future should be `!Unpin`.
#[lang = "async_drop_slice"]
async unsafe fn slice<T>(s: *mut [T]) {
    let len = s.len();
    let ptr = s.as_mut_ptr();
    for i in 0..len {
        // SAFETY: we iterate over elements of `s` slice
        unsafe { async_drop_in_place_raw(ptr.add(i)).await }
    }
}

/// Awaits the `F` future and then awaits `G::IntoFuture`.
enum Chain<F, G: IntoFuture> {
    First(F, Option<G>),
    Last(G::IntoFuture),
}

/// Construct a chain of two futures, which awaits them sequentially as
/// a future.
// Cannot be `async fn` because for some reason fails MIRI tests.
#[lang = "async_drop_chain"]
fn chain<F, G: IntoFuture>(first: F, last: G) -> Chain<F, G> {
    Chain::First(first, Some(last))
}

impl<F, G> Future for Chain<F, G>
where
    F: Future<Output = ()>,
    G: IntoFuture<Output = ()>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // SAFETY: We do not move any possibly immovable object from self
        let this = unsafe { self.get_unchecked_mut() };
        loop {
            match this {
                Chain::First(first, last) => {
                    // SAFETY: simple pin projection
                    ready!(unsafe { Pin::new_unchecked(first) }.poll(cx));
                    // It might be important to destroy the first
                    // future so that it wouldn't possibly hold anything
                    // important long enough to cause problems.
                    *this = Chain::Last(last.take().unwrap().into_future());
                }
                // SAFETY: simple pin projection
                Chain::Last(last) => return unsafe { Pin::new_unchecked(last) }.poll(cx),
            }
        }
    }
}

/// Basically a lazy version of `async_drop_in_place`. Returns a future
/// that would call `AsyncDrop::async_drop` on a first poll.
///
/// # Safety
///
/// Same as `async_drop_in_place` except is lazy to avoid creating
/// multiple mutable refernces.
// NOTE: Returned future should be `!Unpin`.
#[lang = "async_drop_defer"]
async unsafe fn defer<T: ?Sized>(to_drop: *mut T) {
    // SAFETY: same safety requirements as `async_drop_in_place`
    unsafe { async_drop_in_place_raw(to_drop) }.await
}

/// If `T`'s discriminant is equal to the stored one then awaits `M`
/// otherwise awaits the `O`.
///
/// # Safety
///
/// User should carefully manage returned future, since it would
/// try creating an immutable referece from `this` and get pointee's
/// discriminant.
// TODO: Wrap this with `Fuse`
// TODO: Send and Sync
// NOTE: Returned future should be `!Unpin`.
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

/// Used for nop async destructors. We don't use [`core::future::Ready`]
/// because it panics after its second poll, which could be potentially
/// bad if that would happen during the cleanup.
#[derive(Clone, Copy)]
struct Nop;

#[lang = "async_drop_nop"]
fn nop() -> Nop {
    Nop
}

impl Future for Nop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

/// Used for uninhabitable async destructors.
#[derive(Clone, Copy)]
#[lang = "async_drop_never"]
enum Never {}

impl Future for Never {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        match *self {}
    }
}
