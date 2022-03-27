// https://github.com/rust-lang/rust/issues/55223
#![allow(const_err)]

union Foo<'a> {
    y: &'a (),
    long_live_the_unit: &'static (),
}

const FOO: &() = {
//~^ ERROR evaluation of constant value failed
    let y = ();
    unsafe { Foo { y: &y }.long_live_the_unit }
};

fn main() {}
