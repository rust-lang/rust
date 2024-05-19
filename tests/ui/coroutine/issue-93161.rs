//@ edition:2021
//@ run-pass

#![feature(never_type)]

use std::future::Future;

// See if we can run a basic `async fn`
pub async fn foo(x: &u32, y: u32) -> u32 {
    let y = &y;
    let z = 9;
    let z = &z;
    let y = async { *y + *z }.await;
    let a = 10;
    let a = &a;
    *x + y + *a
}

async fn add(x: u32, y: u32) -> u32 {
    let a = async { x + y };
    a.await
}

async fn build_aggregate(a: u32, b: u32, c: u32, d: u32) -> u32 {
    let x = (add(a, b).await, add(c, d).await);
    x.0 + x.1
}

enum Never {}
fn never() -> Never {
    panic!()
}

async fn includes_never(crash: bool, x: u32) -> u32 {
    let result = async { x * x }.await;
    if !crash {
        return result;
    }
    #[allow(unused)]
    let bad = never();
    result *= async { x + x }.await;
    drop(bad);
    result
}

async fn partial_init(x: u32) -> u32 {
    #[allow(unreachable_code)]
    let _x: (String, !) = (String::new(), return async { x + x }.await);
}

async fn read_exact(_from: &mut &[u8], _to: &mut [u8]) -> Option<()> {
    Some(())
}

async fn hello_world() {
    let data = [0u8; 1];
    let mut reader = &data[..];

    let mut marker = [0u8; 1];
    read_exact(&mut reader, &mut marker).await.unwrap();
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

fn main() {
    let x = 5;
    assert_eq!(run_fut(foo(&x, 7)), 31);
    assert_eq!(run_fut(build_aggregate(1, 2, 3, 4)), 10);
    assert_eq!(run_fut(includes_never(false, 4)), 16);
    assert_eq!(run_fut(partial_init(4)), 8);
    run_fut(hello_world());
}
