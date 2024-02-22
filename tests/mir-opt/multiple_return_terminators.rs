// skip-filecheck
//@ compile-flags: -Z mir-opt-level=4
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
