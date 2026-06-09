// https://github.com/rust-lang/rust/issues/9737
//@ run-pass
#![allow(unused_variables)]
macro_rules! f {
    (v: $x:expr) => ( println!("{}", $x) )
}

fn main () {
    let v = 5;
    f!(v: 3);
}
