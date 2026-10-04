#![feature(negative_impls)]
//@ edition: 2021
// Test to ensure we are printing the polarity of the impl trait ref
// when printing out conflicting trait impls

struct MyType;

impl !Clone for &mut MyType {}
//~^ ERROR conflicting implementations of trait `Clone` for type `&mut MyType`
fn main() {}
