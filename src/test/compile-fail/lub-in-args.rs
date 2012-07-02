fn two_args<T>(x: T, y: T) { }

fn main() {
    let x: ~[mut int] = ~[mut 3];
    let y: ~[int] = ~[3];
    let a: @mut int = @mut 3;
    let b: @int = @3;

    // NOTE:
    //
    // The fact that this test fails to compile reflects a known
    // shortcoming of the current inference algorithm.  These errors
    // are *not* desirable.

    two_args(x, y); //~ ERROR (values differ in mutability)
    two_args(a, b); //~ ERROR (values differ in mutability)
}