const _: () = assert_eq!(1, 1);
//~^ ERROR `core::panicking::assert_failed` is not yet stable as a const fn
//~| HELP add `#![feature(const_assert_eq)]` to the crate attributes to enable

fn main() {}
