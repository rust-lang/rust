//! Regression test for #159487: when suggesting `const` for a value used as an
//! array length or repeat count, prefer `usize` over `/* Type */`.

fn main() {
    let length = 3;
    let values: [i32; length] = [0; length];
    //~^ ERROR attempt to use a non-constant value in a constant
    //~| ERROR attempt to use a non-constant value in a constant
    println!("{}", values.len());
}
