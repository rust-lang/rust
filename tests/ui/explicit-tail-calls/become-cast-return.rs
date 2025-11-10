//@ run-pass
//@ ignore-backends: gcc
//@ ignore-wasm 'tail-call' feature not enabled in target wasm32-wasip1
//@ ignore-cross-compile
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

#[inline(never)]
fn leaf(_: &Box<u8>) -> [u8; 1] {
    [1]
}

#[inline(never)]
fn dispatch(param: &Box<u8>) -> [u8; 1] {
    become leaf(param)
}

fn main() {
    let data = Box::new(0);
    let out = dispatch(&data);
    assert_eq!(out, [1]);
}
