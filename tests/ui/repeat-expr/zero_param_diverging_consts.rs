// Check that repeating a diverging constant 0 times doesn't lead to unsoundness.
//
// This specific test checks that using a generic const argument that is substited with 0, rather
// than just plain literal 0, doesn't allow the unsoundness either.
//
// Issue: <https://github.com/rust-lang/rust/issues/165323>
//
//@ build-fail

fn ex_nihilo<const N: usize>() -> u32 {
    [const { panic!() } as !; N];
    //~^ error: evaluation panicked: explicit panic
}

fn main() {
    dbg!(ex_nihilo::<0>());
}
