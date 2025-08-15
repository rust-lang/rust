// Test for #120601, which causes an ice bug cause of unexpected type
//
//@ compile-flags: -Z threads=40
//@ compare-output-by-lines

struct T;
struct Tuple(i32);

async fn foo() -> Result<(), ()> {
    Unstable2(())
}
//~^^^ ERROR `async fn` is not permitted in Rust 2015
//~^^^ ERROR cannot find function, tuple struct or tuple variant `Unstable2` in this scope

async fn tuple() -> Tuple {
    Tuple(1i32)
}
//~^^^ ERROR `async fn` is not permitted in Rust 2015

async fn match_() {
    match tuple() {
        Tuple(_) => {}
    }
}
//~^^^^^ ERROR `async fn` is not permitted in Rust 2015
//~^^^^ ERROR  mismatched types

fn main() {}
