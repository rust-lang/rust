// This test serves as a regression test for issue #114468 and it also ensures that we consider
// type aliases from external crates that don't have `lazy_type_alias` enabled to be eager.

//@ aux-crate:eager=eager.rs
//@ edition: 2021
//@ check-pass

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

// This used to crash when we were computing the variances of `Struct` since we would convert
// `eager::Alias<T>` to a weak alias due to the presence of `#![feature(lazy_type_alias)]` in
// this (!) crate and subsequently attempt to obtain the variances of the type alias associated with
// the weak alias which would panic because we don't compute this information for eager type
// aliases at all.
struct Struct<T>(eager::Alias<T>);

fn main() {
    // We want to ignore (or rather “end up ignoring”) the bound `T: Copy` since `Alias` should be
    // treated as an eager type alias not just inside the crate it is defined in but also in
    // dependent crates (like this one).
    let _: eager::Alias<String>;
}
