// Check that repeating a diverging constant 0 times (inferred via `_`)
// doesn't lead to unsoundness.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>

fn ex_nihilo() -> u32 {
    //~^ error: mismatched types
    let _: [_; 0] = [const { panic!() } as !; _];
}

fn ex_nihilo2() -> u32 {
    //~^ error: mismatched types
    let a = [const { panic!() } as !; _];
    let _: [_; 0] = a;
}

fn main() {
    dbg!(ex_nihilo());
    dbg!(ex_nihilo2());
}
