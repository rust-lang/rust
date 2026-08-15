//! This is a stand-alone version of the `poll_fns` test in Tokio. It hits various
//! interesting edge cases in the epoll logic, making it a good integration test.
//! It also seems to depend on Tokio internals, so if Tokio changes we have have to update
//! or remove the test.

//@only-target: linux # We only support tokio on Linux

use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::os::fd::FromRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Waker};
use std::time::Duration;

use futures::poll;
use tokio::io::unix::AsyncFd;

macro_rules! assert_pending {
    ($e:expr) => {{
        use core::task::Poll;
        match $e {
            Poll::Pending => {}
            Poll::Ready(v) => panic!("ready; value = {:?}", v),
        }
    }};
}

struct TestWaker {
    inner: Arc<TestWakerInner>,
    waker: Waker,
}

#[derive(Default)]
struct TestWakerInner {
    awoken: AtomicBool,
}

impl futures::task::ArcWake for TestWakerInner {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.awoken.store(true, Ordering::SeqCst);
    }
}

impl TestWaker {
    fn new() -> Self {
        let inner: Arc<TestWakerInner> = Default::default();

        Self { inner: inner.clone(), waker: futures::task::waker(inner) }
    }

    fn awoken(&self) -> bool {
        self.inner.awoken.swap(false, Ordering::SeqCst)
    }

    fn context(&self) -> Context<'_> {
        Context::from_waker(&self.waker)
    }
}

fn socketpair() -> (File, File) {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    assert_eq!(unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) }, 0);
    assert_eq!(unsafe { libc::fcntl(fds[1], libc::F_SETFL, libc::O_NONBLOCK) }, 0);

    unsafe { (File::from_raw_fd(fds[0]), File::from_raw_fd(fds[1])) }
}

fn drain(mut fd: &File, mut amt: usize) {
    let mut buf = [0u8; 512];
    while amt > 0 {
        match fd.read(&mut buf[..]) {
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Ok(0) => panic!("unexpected EOF"),
            Err(e) => panic!("unexpected error: {e:?}"),
            Ok(x) => amt -= x,
        }
    }
}

fn main() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(the_test());
}

async fn the_test() {
    let (a, b) = socketpair();
    let afd_a = Arc::new(AsyncFd::new(a).unwrap());
    let afd_b = Arc::new(AsyncFd::new(b).unwrap());

    // Fill up the write side of A
    let mut bytes = 0;
    while let Ok(amt) = afd_a.get_ref().write(&[0; 512]) {
        bytes += amt;
    }

    let waker = TestWaker::new();

    assert_pending!(afd_a.as_ref().poll_read_ready(&mut waker.context()));

    let afd_a_2 = afd_a.clone();
    let r_barrier = Arc::new(tokio::sync::Barrier::new(2));
    let barrier_clone = r_barrier.clone();

    let read_fut = tokio::spawn(async move {
        // Move waker onto this task first
        assert_pending!(poll!(std::future::poll_fn(|cx| afd_a_2.as_ref().poll_read_ready(cx))));
        barrier_clone.wait().await;

        let _ = std::future::poll_fn(|cx| afd_a_2.as_ref().poll_read_ready(cx)).await;
    });

    let afd_a_2 = afd_a.clone();
    let w_barrier = Arc::new(tokio::sync::Barrier::new(2));
    let barrier_clone = w_barrier.clone();

    let mut write_fut = tokio::spawn(async move {
        // Move waker onto this task first
        assert_pending!(poll!(std::future::poll_fn(|cx| afd_a_2.as_ref().poll_write_ready(cx))));
        barrier_clone.wait().await;

        let _ = std::future::poll_fn(|cx| afd_a_2.as_ref().poll_write_ready(cx)).await;
    });

    r_barrier.wait().await;
    w_barrier.wait().await;

    let readable = afd_a.readable();
    tokio::pin!(readable);

    tokio::select! {
        _ = &mut readable => unreachable!(),
        _ = tokio::task::yield_now() => {}
    }

    // Make A readable. We expect that 'readable' and 'read_fut' will both complete quickly
    afd_b.get_ref().write_all(b"0").unwrap();

    let _ = tokio::join!(readable, read_fut);

    // Our original waker should _not_ be awoken (poll_read_ready retains only the last context)
    assert!(!waker.awoken());

    // The writable side should not be awoken
    tokio::select! {
        _ = &mut write_fut => unreachable!(),
        _ = tokio::time::sleep(Duration::from_millis(50)) => {}
    }

    // Make it writable now
    drain(afd_b.get_ref(), bytes);

    // now we should be writable (ie - the waker for poll_write should still be registered after we wake the read side)
    let _ = write_fut.await;
}
