#![unstable(feature = "async_drop", issue = "none")]

use crate::future::Future;
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
#[unstable(feature = "async_drop", issue = "none")]
#[lang = "async_drop_in_place"]
#[allow(unconditional_recursion)]
// TODO: #[rustc_diagnostic_item = "ptr_drop_in_place"] is needed?
pub unsafe fn async_drop_in_place<'a, T: ?Sized + 'a>(
    to_drop: *mut T,
) -> <T as AsyncDestruct<'a>>::AsyncDestructor {
    // Code here does not matter - this is replaced by the
    // real async drop glue ctor by the compiler.

    // SAFETY: see comment above
    unsafe { async_drop_in_place(to_drop) }
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

#[lang = "slice_async_destructor"]
struct SliceAsyncDestuctor<'a, T: 'a> {
    left_slice: ptr::NonNull<[T]>,
    elem_dtor: Option<<T as AsyncDestruct<'a>>::AsyncDestructor>,
}

#[lang = "slice_async_destructor_ctor"]
const unsafe fn slice_async_destructor<'a, T: 'a>(inner: *mut [T]) -> SliceAsyncDestuctor<'a, T> {
    SliceAsyncDestuctor {
        left_slice: unsafe { ptr::NonNull::new_unchecked(inner) },
        elem_dtor: None,
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
