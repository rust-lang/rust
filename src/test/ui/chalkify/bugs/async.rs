// check-fail
// known-bug
// compile-flags: -Z chalk --edition=2021

fn main() -> () {}

async fn foo(x: u32) -> u32 {
    x
}
