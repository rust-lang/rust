//! Regression test for https://github.com/rust-lang/rust/issues/143455.
//! Distinguish private function items from private types in their signatures.

#![feature(decl_macro)]
#![allow(dead_code, private_interfaces)]

mod chamber {
    pub(crate) macro invoke() {
        invoke()
        //~^ ERROR type `fn() {invoke}` is private
    }
    fn invoke() {}

    pub(crate) macro generic_value() {
        generic::<i32>
        //~^ ERROR type `fn(i32) -> Option<i32> {generic::<i32>}` is private
    }
    fn generic<T>(value: T) -> Option<T> {
        Some(value)
    }

    pub struct Public;
    impl Public {
        fn hidden() {}
    }
    pub(crate) macro associated() {
        Public::hidden()
        //~^ ERROR type `fn() {Public::hidden}` is private
    }

    struct Private;
    pub fn public_function() -> Private {
        Private
    }
}

fn main() {
    chamber::invoke!();
    let _ = chamber::generic_value!();
    chamber::associated!();
    let _ = chamber::public_function();
    //~^ ERROR type `Private` is private
}
