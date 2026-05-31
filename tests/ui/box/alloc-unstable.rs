//@ run-pass

#![feature(allocator_ext)]

fn main() {
    let _boxed: Box<u32, _> = Box::new(10);
}
