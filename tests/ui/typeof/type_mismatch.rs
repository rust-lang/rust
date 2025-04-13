// Test that using typeof results in the correct type mismatch errors instead of always assuming
// `usize`, in addition to the pre-existing "typeof is reserved and unimplemented" error

//@ dont-require-annotations: NOTE

fn main() {
    const a: u8 = 1;
    let b: typeof(a) = 1i8;
    //~^ ERROR `typeof` is a reserved keyword but unimplemented
    //~| ERROR mismatched types
    //~| NOTE expected `u8`, found `i8`
}
