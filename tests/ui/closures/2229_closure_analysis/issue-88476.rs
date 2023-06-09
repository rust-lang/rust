// edition:2021

#![feature(rustc_attrs)]

// Test that we can't move out of struct that impls `Drop`.


use std::rc::Rc;

// Test that we restrict precision when moving not-`Copy` types, if any of the parent paths
// implement `Drop`. This is to ensure that we don't move out of a type that implements Drop.
pub fn test1() {
    struct Foo(Rc<i32>);

    impl Drop for Foo {
        fn drop(self: &mut Foo) {}
    }

    let f = Foo(Rc::new(1));
    let x = #[rustc_capture_analysis] move || {
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{:?}", f.0);
        //~^ NOTE: Capturing f[(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture f[] -> ByValue
    };

    x();
}

// Test that we don't restrict precision when moving `Copy` types(i.e. when copying),
// even if any of the parent paths implement `Drop`.
fn test2() {
    struct Character {
        hp: u32,
        name: String,
    }

    impl Drop for Character {
        fn drop(&mut self) {}
    }

    let character = Character { hp: 100, name: format!("A") };

    let c = #[rustc_capture_analysis] move || {
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", character.hp)
        //~^ NOTE: Capturing character[(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture character[(0, 0)] -> ByValue
    };

    c();

    println!("{}", character.name);
}

fn main() {}
