#![unstable(feature = "async_drop", issue = "none")]

use crate::fmt;
use crate::future::{Future, IntoFuture};
use crate::marker::PhantomPinned;
use crate::mem::MaybeUninit;
use crate::pin::Pin;
use crate::ptr;
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
unsafe fn surface_async_drop_in_place<T: AsyncDrop>(
    ptr: *mut T,
) -> <T as AsyncDrop>::Dropper<'static> {
    // SAFETY: We call this from async drop `async_drop_in_place_raw`
    //   which has the same safety requirements
    unsafe { <T as AsyncDrop>::async_drop(Pin::new_unchecked(&mut *ptr)) }
}

/// Wraps a future to continue outputing `Poll::Ready(())` once after
/// wrapped future completes by returning `Poll::Ready(())` on poll. This
/// is useful for constructing async destructors to guarantee this
/// "fuse" property
#[lang = "async_drop_fuse"]
struct Fuse<T> {
    inner: Option<T>,
}

#[lang = "async_drop_fuse_ctor"]
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

/// Async destructor for arrays and slices
#[lang = "async_drop_slice"]
struct Slice<T> {
    left_slice: ptr::NonNull<[T]>,
    elem_dtor: Option<<T as AsyncDestruct>::AsyncDestructor>,
    _pinned: PhantomPinned,
}

#[lang = "async_drop_slice_ctor"]
unsafe fn slice<T>(inner: *mut [T]) -> Slice<T> {
    Slice {
        // SAFETY: We call this funtion from async drop
        //   `async_drop_in_place_raw` which has the same safety requirements
        left_slice: unsafe { ptr::NonNull::new_unchecked(inner) },
        elem_dtor: None,
        _pinned: PhantomPinned,
    }
}

impl<T> Future for Slice<T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move any possibly immovable fields (elem_dtor)
        let this = unsafe { self.get_unchecked_mut() };
        loop {
            if let Some(elem_dtor) = &mut this.elem_dtor {
                // SAFETY: Projecting pin from `Self` down to the
                //   current element's destructor.
                ready!(unsafe { Pin::new_unchecked(elem_dtor) }.poll(cx));
                // Dropping the destructor
                this.elem_dtor = None;
            }
            // Return if slice is empty
            let Some(new_len) = this.left_slice.len().checked_sub(1) else {
                return Poll::Ready(());
            };
            let cur_ptr = this.left_slice.as_non_null_ptr();
            // SAFETY: current slice is not empty (see comment above),
            //   thus it is guaranteed for new_ptr to be in bounds of our
            //   slice or one after its end.
            let new_ptr = unsafe { cur_ptr.add(1) };
            this.left_slice = ptr::NonNull::slice_from_raw_parts(new_ptr, new_len);
            // SAFETY: cur_ptr points to some element of our slice
            //   because slice wasn't empty.
            this.elem_dtor = Some(unsafe { async_drop_in_place_raw(cur_ptr.as_ptr()) });
        }
    }
}

/// Awaits the `F` future and then awaits `G::IntoFuture`.
#[lang = "async_drop_chain"]
enum Chain<F, G: IntoFuture> {
    First(F, G),
    Last(G::IntoFuture),
}

/// Construct async drop chain
#[lang = "async_drop_chain_ctor"]
fn chain<F, G: IntoFuture>(first: F, last: G) -> Chain<F, G> {
    Chain::First(first, last)
}

impl<F, G> Future for Chain<F, G>
where
    F: Future<Output = ()>,
    // Copy there so that we don't need to wrap `G` with `Option` to
    // move from it
    G: IntoFuture<Output = ()> + Copy,
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
                    *this = Chain::Last(last.into_future());
                }
                // SAFETY: simple pin projection
                Chain::Last(last) => return unsafe { Pin::new_unchecked(last) }.poll(cx),
            }
        }
    }
}

#[lang = "async_drop_into_chain"]
struct IntoChain<F, G> {
    first: F,
    last: G,
}

#[lang = "async_drop_into_chain_ctor"]
fn into_chain<F, G>(first: F, last: G) -> IntoChain<F, G> {
    IntoChain { first, last }
}

impl<F, G> IntoFuture for IntoChain<F, G>
where
    F: IntoFuture<Output = ()>,
    G: IntoFuture<Output = ()> + Copy,
{
    type Output = ();

    type IntoFuture = Chain<F::IntoFuture, G>;

    fn into_future(self) -> Self::IntoFuture {
        chain(self.first.into_future(), self.last)
    }
}

#[lang = "into_async_destructor"]
struct IntoAsyncDestructor<T: ?Sized>(ptr::NonNull<T>, PhantomPinned);

#[lang = "into_async_destructor_ctor"]
unsafe fn into_async_destructor<T: ?Sized>(to_drop: *mut T) -> IntoAsyncDestructor<T> {
    // SAFETY: same safety requirements as `async_drop_in_place`
    IntoAsyncDestructor(unsafe { ptr::NonNull::new_unchecked(to_drop) }, PhantomPinned)
}

impl<T: ?Sized> Clone for IntoAsyncDestructor<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for IntoAsyncDestructor<T> {}

impl<T: ?Sized> IntoFuture for IntoAsyncDestructor<T> {
    type Output = ();

    type IntoFuture = <T as AsyncDestruct>::AsyncDestructor;

    fn into_future(self) -> Self::IntoFuture {
        // SAFETY: same safety requirements as `async_drop_in_place`
        unsafe { async_drop_in_place_raw(self.0.as_ptr()) }
    }
}

/// Used for nop async destructors. We don't use [`core::future::Ready`]
/// because it panics after its second poll, which could be potentially
/// bad if that would happen during the cleanup.
#[lang = "async_drop_nop"]
struct Nop;

#[lang = "async_drop_nop_ctor"]
fn nop() -> Nop {
    Nop
}

impl Future for Nop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}
