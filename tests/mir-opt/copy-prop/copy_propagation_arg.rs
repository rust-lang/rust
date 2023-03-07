// Check that CopyProp does not propagate an assignment to a function argument
// (doing so can break usages of the original argument value)
// unit-test: CopyProp
fn dummy(x: u8) -> u8 {
    x
}

// EMIT_MIR copy_propagation_arg.foo.CopyProp.diff
fn foo(mut x: u8) {
    // calling `dummy` to make a use of `x` that copyprop cannot eliminate
    x = dummy(x); // this will assign a local to `x`
}

// EMIT_MIR copy_propagation_arg.bar.CopyProp.diff
fn bar(mut x: u8) {
    dummy(x);
    x = 5;
}

// EMIT_MIR copy_propagation_arg.baz.CopyProp.diff
fn baz(mut x: i32) -> i32 {
    // self-assignment to a function argument should be eliminated
    x = x;
    x
}

// EMIT_MIR copy_propagation_arg.arg_src.CopyProp.diff
fn arg_src(mut x: i32) -> i32 {
    let y = x;
    x = 123; // Don't propagate this assignment to `y`
    y
}

fn main() {
    // Make sure the function actually gets instantiated.
    foo(0);
    bar(0);
    baz(0);
    arg_src(0);
}
