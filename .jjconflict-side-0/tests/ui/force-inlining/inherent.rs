//@ check-fail
#![feature(rustc_attrs)]

struct Foo;

impl Foo {
    #[rustc_force_inline]
    //~^ ERROR: `Foo::bar` is incompatible with `#[rustc_force_inline]`
    #[rustc_no_mir_inline]
    fn bar() {}
}

fn bar_caller() {
    unsafe {
        Foo::bar();
    }
}

fn main() {
    bar_caller();
}
