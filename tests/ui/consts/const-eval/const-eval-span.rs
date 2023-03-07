// Check that error in constant evaluation of enum discriminant
// provides the context for what caused the evaluation.

struct S(i32);

const CONSTANT: S = S(0);

enum E {
    V = CONSTANT,
    //~^ ERROR mismatched types
    //~| expected `isize`, found `S`
}

fn main() {}
