//@ pretty-mode:expanded
//@ pp-exact:never-pattern.pp
//@ only-x86_64

#![allow(incomplete_features)]
#![feature(never_patterns)]

fn f(x: Result<u32, !>) {
    _ = match x {
        Ok(x) => x,
        Err(!),
    };
}

fn main() {}
