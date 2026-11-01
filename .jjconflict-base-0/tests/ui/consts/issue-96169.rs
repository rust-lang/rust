//@ check-pass
//@ compile-flags: -Zmir-opt-level=4 --emit=mir
#![allow(unused)]
fn a() -> usize { 0 }

fn bar(_: u32) {}

fn baz() -> *const dyn Fn(u32) { unimplemented!() }

fn foo() {
    match () {
        _ if baz() == &bar as &dyn Fn(u32) => (),
        () => (),
    }
}

fn main() {
}
