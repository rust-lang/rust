//@ build-fail

// Check that repeating a diverging constant 0 times (inferred via `_`)
// doesn't lead to unsoundness.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>

fn ex_nihilo() -> u32 {
    let _: [_; 0] = [const { panic!() } as !; _];
    //~^ ERROR evaluation panicked: explicit panic
}

fn ex_nihilo2() -> u32 {
    let a = [const { panic!() } as !; _];
    //~^ ERROR evaluation panicked: explicit panic
    let _: [_; 0] = a;
}

fn main() {
    dbg!(ex_nihilo());
    dbg!(ex_nihilo2());
}
