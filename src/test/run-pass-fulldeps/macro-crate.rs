#![allow(plugin_as_library)]
#![allow(dead_code)]
// aux-build:macro_crate_test.rs
// ignore-stage1

#![feature(plugin, rustc_attrs)]
#![plugin(macro_crate_test)]

#[macro_use] #[no_link]
extern crate macro_crate_test;

#[rustc_into_multi_foo]
#[derive(PartialEq, Clone, Debug)]
fn foo() -> AnotherFakeTypeThatHadBetterGoAway {}

// Check that the `#[into_multi_foo]`-generated `foo2` is configured away
fn foo2() {}

trait Qux {
    #[rustc_into_multi_foo]
    fn bar();
}

impl Qux for i32 {
    #[rustc_into_multi_foo]
    fn bar() {}
}

impl Qux for u8 {}

pub fn main() {
    assert_eq!(1, make_a_1!());
    assert_eq!(2, exported_macro!());

    assert_eq!(Foo2::Bar2, Foo2::Bar2);
    test(None::<Foo2>);

    let _ = Foo3::Bar;

    let x = 10i32;
    assert_eq!(x.foo(), 42);
    let x = 10u8;
    assert_eq!(x.foo(), 0);
}

fn test<T: PartialEq+Clone>(_: Option<T>) {}
