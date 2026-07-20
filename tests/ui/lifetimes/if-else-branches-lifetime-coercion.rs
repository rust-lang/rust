//! Regression test for <https://github.com/rust-lang/rust/issues/4025>.
//! `if` expression requires both paths to have a common supertype.
//@ check-pass

#![allow(dead_code)]
#![allow(unused_mut)]

fn print1(b: bool, s1: &str, s2: &str) {
    println!("{}", if b { s1 } else { s2 });
}
fn print2<'a, 'b>(b: bool, s1: &'a str, s2: &'b str) {
    println!("{}", if b { s1 } else { s2 });
}
fn print3(b: bool, s1: &str, s2: &str) {
    let mut s: &str;
    if b { s = s1; } else { s = s2; }
    println!("{}", s);
}
fn print4<'a, 'b>(b: bool, s1: &'a str, s2: &'b str) {
    let mut s: &str;
    if b { s = s1; } else { s = s2; }
    println!("{}", s);
}

pub fn main() {}
