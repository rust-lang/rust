// revisions: both_off both_on
// ignore-tidy-linelength
// run-pass
// [both_off]  compile-flags: -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition:2018

use std::future::Future;

async fn wait() {}

fn test_shrinks_1(arg: [u8; 1000]) -> impl std::future::Future<Output=()> {
    async move {
        let mut local = arg;
        local[2] = 3;
        wait().await;
        assert_eq!(local[2], 3);
    }
}

fn test_noshrinks_1(mut arg: [u8; 1000]) -> impl std::future::Future<Output=()> {
    async move {
        let mut local = arg;
        local[2] = 3;
        let l2 = &mut arg;
        l2[2] = 4;
        wait().await;
        assert_eq!(local[2], 3);
        assert_eq!(l2[2], 4);
    }
}

fn test_noshrinks_2(arg: [u8; 1000]) -> impl std::future::Future<Output=()> {
    async move {
        let mut local = arg;
        local[2] = 1;
        let l2 = arg;
        wait().await;
        assert_eq!(local[2], 1);
        assert_eq!(l2[2], 0);
    }
}

fn test_noshrinks_3(arg: [u8; 1000]) -> impl std::future::Future<Output=()> {
    async move {
        let bor = &arg[2];
        let mut local = arg;
        local[2] = 1;
        wait().await;
        assert_eq!(local[2], 1);
        assert_eq!(*bor, 0);
    }
}

#[cfg(both_on)]
fn check_shrinks(which: &str, fut: impl std::future::Future<Output=()>) {
    let sz = std::mem::size_of_val(&fut);
    println!("{which}: {sz}");
    assert!((1000..=1500).contains(&sz));
    run_fut(fut)
}

fn check_no_shrinks(which: &str, fut: impl std::future::Future<Output=()>) {
    let sz = std::mem::size_of_val(&fut);
    println!("{which}: {sz}");
    assert!((2000..).contains(&sz));
    run_fut(fut);
}

#[cfg(both_on)]
fn main() {
    check_shrinks("s1", test_shrinks_1([0; 1000]));

    check_no_shrinks("n1", test_noshrinks_1([0; 1000]));
    check_no_shrinks("n2", test_noshrinks_2([0; 1000]));
    check_no_shrinks("n3", test_noshrinks_3([0; 1000]));
}

#[cfg(both_off)]
fn main() {
    check_no_shrinks("s1", test_shrinks_1([0; 1000]));
    check_no_shrinks("n1", test_noshrinks_1([0; 1000]));
    check_no_shrinks("n2", test_noshrinks_2([0; 1000]));
    check_no_shrinks("n3", test_noshrinks_3([0; 1000]));
}

fn run_fut<T>(fut: impl Future<Output = T>) -> T {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct MyWaker;
    impl Wake for MyWaker {
        fn wake(self: Arc<Self>) {
            unimplemented!()
        }
    }

    let waker = Waker::from(Arc::new(MyWaker));
    let mut context = Context::from_waker(&waker);

    let mut pinned = Box::pin(fut);
    loop {
        match pinned.as_mut().poll(&mut context) {
            Poll::Pending => continue,
            Poll::Ready(v) => return v,
        }
    }
}
