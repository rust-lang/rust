//@ check-pass
//@ compile-flags: --crate-type lib
//@ edition:2018

fn assert_send<F: Send>(_: F) {}

async fn __post<T>() -> T {
    if false {
        todo!()
    } else {
        async {}.await;
        todo!()
    }
}

fn foo<T>() {
    assert_send(__post::<T>());
}
