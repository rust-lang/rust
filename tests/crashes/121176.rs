//@ known-bug: #121176
//@ needs-rustc-debug-assertions
use std::fmt::Debug;

static STATIC_1: dyn Debug + Sync = *();

fn main() {
    println!("{:?}", &STATIC_1);
}
