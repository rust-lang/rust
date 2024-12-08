//@ check-pass
//@ edition: 2021
// issue: rust-lang/rust#123697

#![feature(async_closure)]

struct S { t: i32 }

fn test(s: &S, t: &i32) {
    async || {
        println!("{}", s.t);
        println!("{}", t);
    };
}

fn main() {}
