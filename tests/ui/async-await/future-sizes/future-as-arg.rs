// compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition: 2021
// run-pass

async fn test(_arg: [u8; 16]) {}

async fn use_future(fut: impl std::future::Future<Output = ()>) {
    fut.await
}

fn main() {
    let actual = std::mem::size_of_val(
        &use_future(use_future(use_future(use_future(use_future(test([0; 16])))))));

    // Using an acceptable range rather than an exact number, in order to allow
    // the actual value to vary slightly over different commits
    let expected = 16..=32;
    assert!(expected.contains(&actual), "expected: {expected:?}, actual: {actual}");
}
