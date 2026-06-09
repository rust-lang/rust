// https://github.com/rust-lang/rust/issues/53333
//@ run-pass
#![allow(unused_imports)]
//@ edition:2018

fn main() {
    use std;
    let std = "std";
    println!("{}", std);
}
