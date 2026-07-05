//@ check-pass

#![deny(unreachable_code)]
#![allow(dead_code, unused_variables)]

trait Foo {
    fn foo();
}

fn do_stuff<F, T>(new_foo: F)
where
    F: FnOnce(i32) -> T,
    T: Foo,
{
    todo!()
}

struct UnusedFoo;

impl Foo for UnusedFoo {
    fn foo() {
        panic!("shouldn't be called")
    }
}

fn main() {
    do_stuff(|_| {
        panic!("shouldn't be called");
        UnusedFoo
    });
}
