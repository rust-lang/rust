// run-rustfix

#![warn(clippy::redundant_closure_call)]
#![allow(unused)]

fn main() {
    let a = (|| 42)();
}
