use std::future::Future;
use std::ptr;

// This test:
// - Compares addresses of non-Copy data before and after moving it
// - Writes to the pointer after it has moved across the await point
//
// This is only meant to assert current behavior, not guarantee that this is
// how it should work in the future. In fact, upcoming changes to rustc
// *should* break these tests.
// See: https://github.com/rust-lang/rust/issues/62958
async fn data_moved_async() {
    async fn helper(mut data: Vec<u8>, raw_pointer: *mut Vec<u8>) {
        let raw_pointer2 = ptr::addr_of_mut!(data);
        // `raw_pointer` points to the original location where the Vec was stored in the caller.
        // `data` is where that Vec (to be precise, its ptr+capacity+len on-stack data)
        // got moved to. Those will usually not be the same since the Vec got moved twice
        // (into the function call, and then into the generator upvar).
        assert_ne!(raw_pointer, raw_pointer2);
        unsafe {
            // This writes into the `x` in `data_moved_async`, re-initializing it.
            std::ptr::write(raw_pointer, vec![3]);
        }
    }
    // Vec<T> is not Copy
    let mut x: Vec<u8> = vec![2];
    let raw_pointer = ptr::addr_of_mut!(x);
    helper(x, raw_pointer).await;
    unsafe {
        assert_eq!(*raw_pointer, vec![3]);
        // Drop to prevent leak.
        std::ptr::drop_in_place(raw_pointer);
    }
}

// Same thing as above, but non-async.
fn data_moved() {
    fn helper(mut data: Vec<u8>, raw_pointer: *mut Vec<u8>) {
        let raw_pointer2 = ptr::addr_of_mut!(data);
        assert_ne!(raw_pointer, raw_pointer2);
        unsafe {
            std::ptr::write(raw_pointer, vec![3]);
        }
    }

    let mut x: Vec<u8> = vec![2];
    let raw_pointer = ptr::addr_of_mut!(x);
    helper(x, raw_pointer);
    unsafe {
        assert_eq!(*raw_pointer, vec![3]);
        std::ptr::drop_in_place(raw_pointer);
    }
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
    run_fut(data_moved_async());
    data_moved();
}
