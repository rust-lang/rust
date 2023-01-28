// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2021
// build-pass

struct A;
impl Drop for A { fn drop(&mut self) {} }

pub async fn f() {
    let mut a = A;
    a = A;
    drop(a);
    async {}.await;
}

fn assert_send<T: Send>(_: T) {}

fn main() {
    let _ = f();
}
