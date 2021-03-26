#![warn(clippy::dbg_macro)]

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
    dbg!(dbg!(dbg!(42)));
    foo(3) + dbg!(factorial(4));
}
