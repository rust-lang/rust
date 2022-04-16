// EMIT_MIR derefer_complex_case.main.Derefer.diff

fn main() {
    for &foo in &[42, 43] { drop(foo) }
}
