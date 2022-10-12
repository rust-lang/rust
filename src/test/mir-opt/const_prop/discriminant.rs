// unit-test: ConstProp
// compile-flags: -O

// FIXME(wesleywiser): Ideally, we could const-prop away all of this and just be left with
// `let x = 42` but that doesn't work because const-prop doesn't support `Operand::Indirect`
// and `InterpCx::eval_place()` always forces an allocation which creates the `Indirect`.
// Fixing either of those will allow us to const-prop this away.

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR discriminant.main.ConstProp.diff
fn main() {
    let x = (if let Some(true) = Some(true) { 42 } else { 10 }) + 0;
}
