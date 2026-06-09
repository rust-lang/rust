//@ compile-flags: -Znext-solver
//@ check-pass

fn require_fn(_: impl Fn() -> i32) {}

fn main() {
    require_fn(|| -> i32 { 1i32 });
}
