//@ check-pass
#![warn(unused_parens)]

fn main() {
    let a: i32 = 1;
    let b: i64 = 1;

    if b + a as (i64) < 0 {
        println!(":D");
    }
    if b + b + a as (i64) < 0 {
        println!(":D");
    }
    let c = a + b as (i32) < 0;
    let mut x = false;
    x |= false || (b as (i32) < 0);

    let d = 1 + 2 + 3 * 4 as (i32) < 10;
}
