// compile-pass

#![allow(unused)]

use std::collections::HashMap;

#[deny(default_hash_types)] //~ WARNING unknown lint: `default_hash_types`
fn main() {}
