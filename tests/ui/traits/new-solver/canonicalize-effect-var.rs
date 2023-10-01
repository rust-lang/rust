// compile-flags: -Ztrait-solver=next
// check-pass

#![feature(effects)]
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo();
}

trait Bar {}

impl const Foo for i32 {
    fn foo() {}
}

impl<T> const Foo for T where T: Bar {
    fn foo() {}
}

fn main() {}
