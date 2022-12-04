// compile-flags: -Zmir-enable-passes=-SimplifyLocals-before-const-prop
// EMIT_MIR ref_deref.main.PromoteTemps.diff
// EMIT_MIR ref_deref.main.ConstProp.diff

fn main() {
    *(&4);
}
