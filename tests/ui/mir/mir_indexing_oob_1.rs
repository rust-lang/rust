// run-fail
//@error-in-other-file:index out of bounds: the len is 5 but the index is 10
//@ignore-target-emscripten no processes

const C: [u32; 5] = [0; 5];

#[allow(unconditional_panic)]
fn test() -> u32 {
    C[10]
}

fn main() {
    test();
}
