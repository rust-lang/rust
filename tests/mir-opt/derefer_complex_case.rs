// unit-test: Derefer
// EMIT_MIR derefer_complex_case.main.Derefer.diff
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

fn main() {
    for &foo in &[42, 43] { drop(foo) }
}
