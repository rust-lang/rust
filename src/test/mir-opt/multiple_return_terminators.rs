// compile-flags: -Z mir-opt-level=3
// EMIT_MIR multiple_return_terminators.test.MultipleReturnTerminators.diff

fn test(x: bool) {
    if x {
        // test
    } else {
        // test
    }
}

fn main() {
    test(true)
}
