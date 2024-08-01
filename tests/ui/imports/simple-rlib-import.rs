// A simple test, where foo.rs has a dependency
// on the rlib (a static Rust-specific library format) bar.rs. If the test passes,
// rlibs can be built and linked into another file successfully..

//@ aux-crate:bar=simple-rlib.rs
//@ run-pass

extern crate bar;

fn main() {
    bar::bar();
}
