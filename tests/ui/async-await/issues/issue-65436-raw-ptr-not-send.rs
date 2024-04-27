//@ edition:2018
//@ check-pass

struct Foo(*const u8);

unsafe impl Send for Foo {}

async fn bar(_: Foo) {}

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(async {
        bar(Foo(std::ptr::null())).await;
    })
}
