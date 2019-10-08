// aux-build:private-nested-unused_must_use.rs

#![deny(unused_must_use)]

extern crate private_nested_unused_must_use;

use self::private_nested_unused_must_use::{B, C};

fn main() {
    B::new(); // ok: ignores private `must_use` type
    C::new(); //~ ERROR unused `private_nested_unused_must_use::S` in field `s` that must be used
}
