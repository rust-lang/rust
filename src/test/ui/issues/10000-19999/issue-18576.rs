// run-fail
// error-pattern:stop
// ignore-emscripten no processes

// #18576
// Make sure that calling an extern function pointer in an unreachable
// context doesn't cause an LLVM assertion

#[allow(unreachable_code)]
fn main() {
    panic!("stop");
    let pointer = other;
    pointer();
}

extern "C" fn other() {}
