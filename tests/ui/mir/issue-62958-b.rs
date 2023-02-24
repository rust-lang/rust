// revisions: both_off just_prop both_on
// ignore-tidy-linelength
// run-pass
// [both_off]  compile-flags: -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [just_prop] compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition:2018

async fn test(_arg: [u8; 8192]) {}

async fn use_future(fut: impl std::future::Future<Output = ()>) {
    fut.await
}

#[cfg(both_on)]
fn main() {
    let expected = 8192..=9999;
    let actual = std::mem::size_of_val(&use_future(use_future(use_future(test([0; 8192])))));
    assert!(expected.contains(&actual), "expected: {:?} actual: {}", expected, actual);
}

#[cfg(any(both_off, just_prop))]
fn main() {
    let expected = 16000..=8192*(2*2*2*2); // O(2^n) LOL
    let actual = std::mem::size_of_val(&use_future(use_future(use_future(test([0; 8192])))));
    assert!(expected.contains(&actual), "expected: {:?} actual: {}", expected, actual);
}
