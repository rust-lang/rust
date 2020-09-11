// Check that CopyPropagation does not propagate an assignment to a function argument
// (doing so can break usages of the original argument value)

fn dummy(x: u8) -> u8 {
    x
}

// EMIT_MIR copy_propagation_arg.foo.CopyPropagation.diff
fn foo(mut x: u8) {
    // calling `dummy` to make an use of `x` that copyprop cannot eliminate
    x = dummy(x); // this will assign a local to `x`
}

// EMIT_MIR copy_propagation_arg.bar.CopyPropagation.diff
fn bar(mut x: u8) {
    dummy(x);
    x = 5;
}

// EMIT_MIR copy_propagation_arg.baz.CopyPropagation.diff
fn baz(mut x: i32) {
    // self-assignment to a function argument should be eliminated
    x = x;
}

// EMIT_MIR copy_propagation_arg.arg_src.CopyPropagation.diff
fn arg_src(mut x: i32) -> i32 {
    let y = x;
    x = 123; // Don't propagate this assignment to `y`
    y
}

// EMIT_MIR copy_propagation_arg.maybe.CopyPropagation.diff
fn maybe(m: &[bool], s1: String) {
    if m[0] {
        let s2 = s1;
        if m[1] {
            let s3 = s2;
            if m[2] {
                let s4 = s3;
                &s4;
            }
        }
    }
}

fn main() {
    // Make sure the function actually gets instantiated.
    foo(0);
    bar(0);
    baz(0);
    arg_src(0);
    maybe(&[true, false, true], "Hello!".to_owned());
}
