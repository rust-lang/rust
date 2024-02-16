//@ run-pass
#![feature(allocator_api)]
fn main() {
    let _boxed: Box<u32, _> = Box::new(10);
}
