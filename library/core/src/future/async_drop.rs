#![unstable(feature = "async_drop", issue = "none")]
#![allow(missing_docs)] // TODO: remove

use crate::fmt;
use crate::future::Future;
use crate::marker::PhantomPinned;
use crate::mem::MaybeUninit;
use crate::pin::Pin;
use crate::ptr;
use crate::task::{ready, Context, Poll};

#[unstable(feature = "async_drop", issue = "none")]
pub fn async_drop<'a, T: 'a>(value: T) -> impl Future<Output = ()> + 'a {
    async move {
        let mut value = MaybeUninit::new(value);
        unsafe { async_drop_in_place(value.as_mut_ptr()) }.await;
    }
}

// TODO: Returned future should be `!Unpin`
#[lang = "async_drop_in_place"]
#[allow(unconditional_recursion)]
// TODO: #[rustc_diagnostic_item = "ptr_drop_in_place"] is needed?
unsafe fn async_drop_in_place_raw<'a, T: ?Sized + 'a>(
    to_drop: *mut T,
) -> <T as AsyncDestruct<'a>>::AsyncDestructor {
    // Code here does not matter - this is replaced by the
    // real async drop glue ctor by the compiler.

    // SAFETY: see comment above
    unsafe { async_drop_in_place_raw(to_drop) }
}

#[unstable(feature = "async_drop", issue = "none")]
pub unsafe fn async_drop_in_place<'a, T: ?Sized + 'a>(to_drop: *mut T) -> AsyncDropInPlace<'a, T> {
    unsafe { AsyncDropInPlace(async_drop_in_place_raw(to_drop)) }
}

#[unstable(feature = "async_drop", issue = "none")]
pub struct AsyncDropInPlace<'a, T: ?Sized + 'a>(<T as AsyncDestruct<'a>>::AsyncDestructor);

#[unstable(feature = "async_drop", issue = "none")]
impl<'a, T: ?Sized + 'a> fmt::Debug for AsyncDropInPlace<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncDropInPlace").finish_non_exhaustive()
    }
}

#[unstable(feature = "async_drop", issue = "none")]
impl<'a, T: ?Sized + 'a> Future for AsyncDropInPlace<'a, T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe { self.map_unchecked_mut(|p| &mut p.0).poll(cx) }
    }
}

#[unstable(feature = "async_drop", issue = "none")]
#[lang = "async_drop"]
pub trait AsyncDrop {
    #[unstable(feature = "async_drop", issue = "none")]
    type Dropper<'a>: Future<Output = ()>
    where
        Self: 'a;

    #[unstable(feature = "async_drop", issue = "none")]
    fn async_drop(self: Pin<&mut Self>) -> Self::Dropper<'_>;
}

#[unstable(feature = "async_drop", issue = "none")]
#[lang = "async_destruct"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
pub trait AsyncDestruct<'a>: 'a {
    #[unstable(feature = "async_drop", issue = "none")]
    type AsyncDestructor: Future<Output = ()>;
}

#[lang = "surface_async_drop_in_place"]
unsafe fn surface_async_drop_in_place<'a, T: AsyncDrop + 'a>(
    ptr: *mut T,
) -> <T as AsyncDrop>::Dropper<'a> {
    unsafe { <T as AsyncDrop>::async_drop(Pin::new_unchecked(&mut *ptr)) }
}

#[lang = "slice_async_destructor"]
struct SliceAsyncDestuctor<'a, T: 'a> {
    left_slice: ptr::NonNull<[T]>,
    elem_dtor: Option<AsyncDropInPlace<'a, T>>,
    _pinned: PhantomPinned,
}

#[lang = "slice_async_destructor_ctor"]
const unsafe fn slice_async_destructor<'a, T: 'a>(inner: *mut [T]) -> SliceAsyncDestuctor<'a, T> {
    SliceAsyncDestuctor {
        left_slice: unsafe { ptr::NonNull::new_unchecked(inner) },
        elem_dtor: None,
        _pinned: PhantomPinned,
    }
}

impl<'a, T: 'a> Future for SliceAsyncDestuctor<'a, T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe {
            let this = self.get_unchecked_mut();
            loop {
                if let Some(elem_dtor) = &mut this.elem_dtor {
                    ready!(Pin::new_unchecked(elem_dtor).poll(cx));
                    this.elem_dtor = None;
                }
                let Some(new_len) = this.left_slice.len().checked_sub(1) else {
                    return Poll::Ready(());
                };
                let cur_ptr = this.left_slice.as_non_null_ptr();
                let new_ptr = cur_ptr.add(1);
                this.left_slice = ptr::NonNull::slice_from_raw_parts(new_ptr, new_len);
                this.elem_dtor = Some(async_drop_in_place(cur_ptr.as_ptr()));
            }
        }
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "deferred_async_drop"]
enum DeferredAsyncDrop<'a, T: 'a> {
    Init { ptr: ptr::NonNull<T>, _pinned: PhantomPinned },
    Running { dtor: AsyncDropInPlace<'a, T> },
}

#[unstable(feature = "future_combinators", issue = "none")]
impl<'a, T: 'a> Future for DeferredAsyncDrop<'a, T> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let this = self.get_unchecked_mut();
            if let DeferredAsyncDrop::Init { ptr, _pinned: _ } = *this {
                *this = DeferredAsyncDrop::Running { dtor: async_drop_in_place(ptr.as_ptr()) };
            }

            let DeferredAsyncDrop::Running { dtor } = this else { unreachable!() };
            return Pin::new_unchecked(dtor).poll(cx);
        }
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "deferred_async_drop_ctor"]
const unsafe fn deferred_async_drop<'a, T: 'a>(item: *mut T) -> DeferredAsyncDrop<'a, T> {
    unsafe {
        DeferredAsyncDrop::Init { ptr: ptr::NonNull::new_unchecked(item), _pinned: PhantomPinned }
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_chain"]
struct Chain<F1, F2> {
    first: Option<F1>,
    second: F2,
}

#[unstable(feature = "future_combinators", issue = "none")]
impl<F1, F2> Future for Chain<F1, F2>
where
    F1: Future<Output = ()>,
    F2: Future<Output = ()>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = unsafe { self.get_unchecked_mut() };
        loop {
            match &mut this.first {
                Some(first) => {
                    ready!(unsafe { Pin::new_unchecked(first) }.poll(cx));
                    this.first = None;
                }
                None => {
                    return unsafe { Pin::new_unchecked(&mut this.second) }.poll(cx);
                }
            }
        }
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_chain_ctor"]
const fn chain<F1, F2>(first: F1, second: F2) -> Chain<F1, F2> {
    Chain { first: Some(first), second }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_ready_unit"]
struct ReadyUnit;

#[unstable(feature = "future_combinators", issue = "none")]
impl Future for ReadyUnit {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

#[unstable(feature = "future_combinators", issue = "none")]
#[lang = "future_ready_unit_ctor"]
const fn ready_unit() -> ReadyUnit {
    ReadyUnit
}
