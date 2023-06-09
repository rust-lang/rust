// run-pass
// aux-build:associated-const-cc-lib.rs


extern crate associated_const_cc_lib as foolib;

pub struct LocalFoo;

impl foolib::Foo for LocalFoo {
    const BAR: usize = 1;
}

const FOO_1: usize = <foolib::FooNoDefault as foolib::Foo>::BAR;
const FOO_2: usize = <LocalFoo as foolib::Foo>::BAR;
const FOO_3: usize = foolib::InherentBar::BAR;

fn main() {
    assert_eq!(0, FOO_1);
    assert_eq!(1, FOO_2);
    assert_eq!(3, FOO_3);

    match 0 {
        <foolib::FooNoDefault as foolib::Foo>::BAR => {},
        <LocalFoo as foolib::Foo>::BAR => assert!(false),
        foolib::InherentBar::BAR => assert!(false),
        _ => assert!(false)
    }
}
