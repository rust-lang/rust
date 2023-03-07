// unit-test: Derefer
// EMIT_MIR derefer_complex_case.main.Derefer.diff
// ignore-wasm32

fn main() {
    for &foo in &[42, 43] { drop(foo) }
}
