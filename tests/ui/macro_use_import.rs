// compile-flags: --edition 2018
#![warn(clippy::macro_use_import)]

use std::collections::HashMap;
#[macro_use]
use std::prelude;

fn main() {
    let _ = HashMap::<u8, u8>::new();
    println!();
}
