// https://github.com/rust-lang/rust/issues/55223

#![feature(const_let)]

union Foo<'a> {
    y: &'a (),
    long_live_the_unit: &'static (),
}

const FOO: &() = { //~ ERROR this constant cannot be used
    let y = ();
    unsafe { Foo { y: &y }.long_live_the_unit }
};

fn main() {}
