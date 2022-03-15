// EMIT_MIR const_folding_test.f.PreCodegen.after.mir

fn f() -> usize {
    1 + if true {1} else {2}
}

fn main() {
    let _a = f();
}