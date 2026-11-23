// Check that we don't ICE when recovering a missing `fn`/`struct` keyword before a forced keyword.
//
//@ edition: 2021..
#![feature(forced_keywords)]

pub k#self() {}
//~^ ERROR visibility `pub` is not followed by an item
//~| ERROR expected item, found keyword `k#self`

fn main() {}
