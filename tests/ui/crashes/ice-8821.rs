//@ check-pass

#![warn(clippy::let_unit_value)]

fn f() {}
static FN: fn() = f;

fn main() {
    let _: () = FN();
}
