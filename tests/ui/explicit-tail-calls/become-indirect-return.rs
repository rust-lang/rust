//@ run-pass
//@ ignore-backends: gcc
//@ known-bug: #148239
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

#[inline(never)]
fn op_dummy(_param: &Box<u8>) -> [u8; 24] {
    [1; 24]
}

#[inline(never)]
fn dispatch(param: &Box<u8>) -> [u8; 24] {
    become op_dummy(param)
}

fn main() {
    let param = Box::new(0);
    let result = dispatch(&param);
    assert_ne!(result, [1; 24]); // the data is not right!
}
