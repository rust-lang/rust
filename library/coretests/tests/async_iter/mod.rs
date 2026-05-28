use core::async_iter::{self, AsyncIterator, IntoAsyncIterator};
use core::pin::pin;
use core::task::Poll;

#[test]
fn into_async_iter() {
    let async_iter = async_iter::from_iter(0..3);
    let mut async_iter = pin!(async_iter.into_async_iter());

    let mut cx = &mut core::task::Context::from_waker(core::task::Waker::noop());

    assert_eq!(async_iter.as_mut().poll_next(&mut cx), Poll::Ready(Some(0)));
    assert_eq!(async_iter.as_mut().poll_next(&mut cx), Poll::Ready(Some(1)));
    assert_eq!(async_iter.as_mut().poll_next(&mut cx), Poll::Ready(Some(2)));
    assert_eq!(async_iter.as_mut().poll_next(&mut cx), Poll::Ready(None));
}
