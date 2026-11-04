// Check that repeating a diverging constant 0 times doesn't lead to unsoundness.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>

trait T {
    const V: !;
}

struct Panic;
impl T for Panic {
    const V: ! = panic!();
}

fn ex_nihilo() -> u32 {
    //~^ error: mismatched types
    [Panic::V; 0];
}

fn main() {
    dbg!(ex_nihilo());
}
