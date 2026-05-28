//@ run-rustfix
#![allow(dead_code)]

trait T {
    unsafe fn foo(a: &usize, b: &usize) -> usize;
    fn bar(&self, a: &usize, b: &usize) -> usize;
}

mod foo {
    use super::T;
    impl T for () {} //~ ERROR not all trait items

    impl T for usize { //~ ERROR not all trait items
    }
}

fn main() {}
