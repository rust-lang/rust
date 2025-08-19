//! Regression test for https://github.com/rust-lang/rust/issues/15858

//@ run-pass
// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]
static mut DROP_RAN: bool = false;

trait Bar {
    fn do_something(&mut self); //~ WARN method `do_something` is never used
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}


struct Foo<B: Bar>(#[allow(dead_code)] B);

impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        unsafe {
            DROP_RAN = true;
        }
    }
}


fn main() {
    {
       let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    unsafe {
        assert_eq!(DROP_RAN, true);
    }
}
