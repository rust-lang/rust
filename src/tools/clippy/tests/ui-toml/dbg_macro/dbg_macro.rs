//@compile-flags: --test
#![warn(clippy::dbg_macro)]
#![allow(clippy::unnecessary_operation, clippy::no_effect)]

fn foo(n: u32) -> u32 {
    if let Some(n) = dbg!(n.checked_sub(4)) { n } else { n }
}

fn factorial(n: u32) -> u32 {
    if dbg!(n <= 1) {
        dbg!(1)
    } else {
        dbg!(n * factorial(n - 1))
    }
}

fn main() {
    dbg!(42);
    foo(3) + dbg!(factorial(4));
    dbg!(1, 2, 3, 4, 5);
}

#[test]
pub fn issue8481() {
    dbg!(1);
}

#[cfg(test)]
fn foo2() {
    dbg!(1);
}

#[cfg(test)]
mod mod1 {
    fn func() {
        dbg!(1);
    }
}
