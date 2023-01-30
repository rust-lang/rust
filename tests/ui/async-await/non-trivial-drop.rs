// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// build-pass
// edition:2018

#![feature(generators)]

fn main() {
    let _ = foo();
}

fn foo() {
    || {
        yield drop(Config {
            nickname: NonCopy,
            b: NonCopy2,
        }.nickname);
    };
}

#[derive(Default)]
struct NonCopy;
impl Drop for NonCopy {
    fn drop(&mut self) {}
}

#[derive(Default)]
struct NonCopy2;
impl Drop for NonCopy2 {
    fn drop(&mut self) {}
}

#[derive(Default)]
struct Config {
    nickname: NonCopy,
    b: NonCopy2,
}
