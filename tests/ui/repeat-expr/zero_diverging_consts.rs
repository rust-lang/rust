//@ build-fail

// Check that repeating a diverging constant 0 times doesn't lead to unsoundness.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>

fn ex_nihilo() -> u32 {
    [const { panic!() } as !; 0];
    //~^ ERROR evaluation panicked: explicit panic
}

fn main() {
    dbg!(ex_nihilo());
}
