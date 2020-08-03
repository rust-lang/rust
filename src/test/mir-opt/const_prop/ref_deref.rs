// EMIT_MIR ref_deref.main.PromoteTemps.diff
// EMIT_MIR ref_deref.main.ConstProp.diff

fn main() {
    *(&4);
}
