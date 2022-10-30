// compile-flags: -Wrust-2021-incompatible-closure-captures

pub struct A {}

impl A {
    async fn create(path: impl AsRef<std::path::Path>)  { //~ ERROR  `async fn` is not permitted in Rust 2015
    //~^ WARN changes to closure capture in Rust 2021 will affect drop order [rust_2021_incompatible_closure_captures]
    ;
    crate(move || {} ).await //~ ERROR expected function, found module `crate`
    }
}

trait C{async fn new(val: T) {} //~ ERROR  `async fn` is not permitted in Rust 2015
//~^ ERROR functions in traits cannot be declared `async`
//~| ERROR mismatched types
//~| ERROR cannot find type `T` in this scope
//~| WARN changes to closure capture in Rust 2021 will affect drop order [rust_2021_incompatible_closure_captures]

//~ ERROR  this file contains an unclosed delimiter
