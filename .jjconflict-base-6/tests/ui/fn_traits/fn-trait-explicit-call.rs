//! regression test for <https://github.com/rust-lang/rust/issues/20847>
//@ run-pass
#![feature(fn_traits)]

fn say(x: u32, y: u32) {
    println!("{} {}", x, y);
}

fn main() {
    Fn::call(&say, (1, 2));
}
