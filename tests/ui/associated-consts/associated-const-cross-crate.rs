//@ run-pass
//@ aux-build:associated-const-cc-lib.rs


extern crate associated_const_cc_lib as foolib;

pub struct LocalFoo;

impl foolib::Foo for LocalFoo {
    const BAR: usize = 1;
}

fn main() {
    assert_eq!(0, <foolib::FooNoDefault as foolib::Foo>::BAR);
    assert_eq!(1, <LocalFoo as foolib::Foo>::BAR);
    assert_eq!(3, foolib::InherentBar::BAR);
}
