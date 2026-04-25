#![feature(allow_internal_unstable)]
#![allow(internal_features)]

#[allow_internal_unstable(cat = "meow")] //~ ERROR `allow_internal_unstable` expects feature names
macro_rules! foo {
    () => {}
}

fn main() {}
