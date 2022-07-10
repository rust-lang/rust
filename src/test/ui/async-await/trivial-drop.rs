// edition:2018
// build-pass
async fn trivial_drop() {
    let x: *const usize = &0;
    async {}.await;
}

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(trivial_drop());
}
