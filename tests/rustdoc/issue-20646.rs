// aux-build:issue-20646.rs
// ignore-cross-compile

#![feature(associated_types)]

extern crate issue_20646;

// @has issue_20646/trait.Trait.html \
//      '//*[@id="associatedtype.Output"]' \
//      'type Output'
pub trait Trait {
    type Output;
}

// @has issue_20646/fn.fun.html \
//      '//*[@class="rust fn"]' 'where T: Trait<Output = i32>'
pub fn fun<T>(_: T) where T: Trait<Output=i32> {}

pub mod reexport {
    // @has issue_20646/reexport/trait.Trait.html \
    //      '//*[@id="associatedtype.Output"]' \
    //      'type Output'
    // @has issue_20646/reexport/fn.fun.html \
    //      '//*[@class="rust fn"]' 'where T: Trait<Output = i32>'
    pub use issue_20646::{Trait, fun};
}
