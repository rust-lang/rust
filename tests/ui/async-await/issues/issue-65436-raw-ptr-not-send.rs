// edition:2018
// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// [drop_tracking] check-pass
// [drop_tracking_mir] check-pass

struct Foo(*const u8);

unsafe impl Send for Foo {}

async fn bar(_: Foo) {}

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(async {
        //[no_drop_tracking]~^ ERROR future cannot be sent between threads safely
        bar(Foo(std::ptr::null())).await;
    })
}
