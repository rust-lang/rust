//@ edition: 2021
//@ run-pass

async fn test(_arg: [u8; 16]) {}

async fn use_future(fut: impl std::future::Future<Output = ()>) {
    fut.await
}

fn main() {
    let actual = std::mem::size_of_val(
        &use_future(use_future(use_future(use_future(use_future(test([0; 16])))))));
    // Not using an exact number in case it slightly changes over different commits
    let expected = 550;
    assert!(actual > expected, "expected: >{expected}, actual: {actual}");
}
