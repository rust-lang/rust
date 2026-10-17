// A simple test, where foo.rs has a dependency
// on the dynamic library simple-dylib.rs. If the test passes,
// dylibs can be built and linked into another file successfully..

//@ aux-crate:bar=simple-dylib.rs
//@ run-pass

extern crate bar;

fn main() {
    bar::bar();
}
