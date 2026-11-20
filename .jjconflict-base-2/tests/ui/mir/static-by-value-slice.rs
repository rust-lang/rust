//! Regression test for #140332
//! KnownPanicsLint used to assert ABI compatibility in the interpreter,
//! which ICEs with unsized statics.

static mut S: [i8] = ["Some thing"; 1];
//~^ ERROR the size for values of type `[i8]` cannot be known

fn main() {
    assert_eq!(S, [0; 1]);
}
