//@ compile-flags: -Zunstable-options
//@ check-pass
#![warn(rustc::internal)]

#[allow(rustc::foo::bar::default_hash_types)]
//~^ WARN unknown lint: `rustc::foo`
#[allow(rustc::foo::default_hash_types)]
//~^ WARN unknown lint: `rustc::foo`
fn main() {
    let _ = std::collections::HashMap::<String, String>::new();
    //~^ WARN prefer `FxHashMap` over `HashMap`, it has better performance
}
