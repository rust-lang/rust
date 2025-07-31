//! Regression test for #121176
//! KnownPanicsLint used to assert ABI compatibility in the interpreter,
//! which ICEs with unsized statics.
//@ needs-rustc-debug-assertions

use std::fmt::Debug;

static STATIC_1: dyn Debug + Sync = *();
//~^ ERROR the size for values of type `(dyn Debug + Sync + 'static)` cannot be known
//~| ERROR type `()` cannot be dereferenced

fn main() {
    println!("{:?}", &STATIC_1);
}
