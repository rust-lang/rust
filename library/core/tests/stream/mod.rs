use core::pin::pin;
use core::stream::{self, IntoStream, Stream};
use core::task::Poll;

#[test]
fn into_stream() {
    let stream = stream::from_iter(0..3);
    let mut stream = pin!(stream.into_stream());

    let waker = core::task::Waker::noop();
    let mut cx = &mut core::task::Context::from_waker(&waker);

    assert_eq!(stream.as_mut().poll_next(&mut cx), Poll::Ready(Some(0)));
    assert_eq!(stream.as_mut().poll_next(&mut cx), Poll::Ready(Some(1)));
    assert_eq!(stream.as_mut().poll_next(&mut cx), Poll::Ready(Some(2)));
    assert_eq!(stream.as_mut().poll_next(&mut cx), Poll::Ready(None));
}
