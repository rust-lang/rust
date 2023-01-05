// check-fail
// known-bug: unknown
// compile-flags: -Z trait-solver=chalk --edition=2021

fn main() -> () {}

async fn foo(x: u32) -> u32 {
    x
}
