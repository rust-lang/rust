//! Regression test for https://github.com/rust-lang/rust/issues/33571
#[derive(Clone,
         Sync, //~ ERROR cannot find derive macro `Sync` in this scope
               //~| ERROR cannot find derive macro `Sync` in this scope
         Copy)]
enum Foo {}

fn main() {}
