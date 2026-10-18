// This test serves as a regression test for issue #114468 and it also ensures that we consider
// type aliases from external crates that don't have `checked_type_aliases` enabled to be unchecked.

//@ aux-crate:unchecked=unchecked.rs
//@ edition: 2021
//@ check-pass
#![feature(checked_type_aliases)]

// This used to crash when we were computing the variances of `Struct` since we would convert
// `unchecked::Alias<T>` to a free alias type due to the presence of
// `#![feature(checked_type_aliases)]` in this (!) crate and subsequently attempt to
// obtain the variances of the type alias associated with the free alias type which would
// panic because we don't compute this information for unchecked type aliases at all.
struct Struct<T>(unchecked::Alias<T>);

fn main() {
    // We want to ignore (or rather “end up ignoring”) the bound `T: Copy` since `Alias` should be
    // treated as an unchecked type alias not just inside the crate it is defined in but also in
    // dependent crates (like this one).
    let _: unchecked::Alias<String>;
}
