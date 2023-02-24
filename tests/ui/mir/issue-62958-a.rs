// revisions: both_off just_prop both_on
// ignore-tidy-linelength
// run-pass
// [both_off]  compile-flags: -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [just_prop] compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition:2018

async fn wait() {}
#[allow(dropping_copy_types)]
async fn test(arg: [u8; 8192]) {
    wait().await;
    drop(arg);
}

#[cfg(both_off)]
fn main() {
    let expected = 16000..=17000;
    let actual = std::mem::size_of_val(&test([0; 8192]));
    assert!(expected.contains(&actual));
}

#[cfg(any(just_prop, both_on))]
fn main() {
    let expected = 8192..=9999;
    let actual = std::mem::size_of_val(&test([0; 8192]));
    assert!(expected.contains(&actual));
}
