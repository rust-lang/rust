//@ build-fail

// Check that repeating a diverging constant 0 times doesn't lead to unsoundness.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>

trait T {
    const V: !;
}

struct Panic;
impl T for Panic {
    const V: ! = panic!();
    //~^ ERROR evaluation panicked: explicit panic
}

fn ex_nihilo() -> u32 {
    [Panic::V; 0];
}

fn main() {
    dbg!(ex_nihilo());
}
