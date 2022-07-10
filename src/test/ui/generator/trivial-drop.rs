// build-pass
#![feature(generators)]

fn assert_send<T: Send>(_: T) {}

fn main() {
    let g = || {
        let x: *const usize = &0;
        yield;
    };
    assert_send(g);
}
