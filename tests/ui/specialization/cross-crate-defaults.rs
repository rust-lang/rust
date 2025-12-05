//@ run-pass

//@ aux-build:cross_crates_defaults.rs

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

extern crate cross_crates_defaults;

use cross_crates_defaults::*;

struct LocalDefault;
struct LocalOverride;

impl Foo for LocalDefault {}

impl Foo for LocalOverride {
    fn foo(&self) -> bool { true }
}

fn test_foo() {
    assert!(!0i8.foo());
    assert!(!0i32.foo());
    assert!(0i64.foo());

    assert!(!LocalDefault.foo());
    assert!(LocalOverride.foo());
}

fn test_bar() {
    assert!(0u8.bar() == 0);
    assert!(0i32.bar() == 1);
    assert!("hello".bar() == 0);
    assert!(vec![()].bar() == 2);
    assert!(vec![0i32].bar() == 2);
    assert!(vec![0i64].bar() == 3);
}

fn main() {
    test_foo();
    test_bar();
}
