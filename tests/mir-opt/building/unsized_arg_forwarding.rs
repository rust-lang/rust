#![feature(unsized_fn_params)]

fn nop(_: [usize], _: ()) {}

// EMIT_MIR unsized_arg_forwarding.forward.built.after.mir
fn forward(mut x: [usize], i: usize) {
    // Check that we do not copy `x` into a temporary. Unfortunately this means that the write to
    // `x[0]` does not result in a borrowck error. There is currently no way to generate Mir that
    // does not exhibit this bug.
    nop(x, {
        x[i] = i;
        ()
    });
}

fn main() {
    let x: Box<[usize]> = Box::new([1]);
    forward(*x, 0);
}
