// This is a separate test from `issue-66693.rs` because array lengths are evaluated
// in a separate stage before `const`s and `statics` and so the error below is hit and
// the compiler exits before generating errors for the others.

fn main() {
    let _ = [0i32; panic!(2f32)];
    //~^ ERROR: argument to `panic!()` in a const context must have type `&str`

    // ensure that conforming panics are handled correctly
    let _ = [false; panic!()];
    //~^ ERROR: evaluation of constant value failed

    // typechecking halts before getting to this one
    let _ = ['a', panic!("panic in array len")];
}
