// EMIT_MIR ref_deref.main.PromoteTemps.diff
// EMIT_MIR ref_deref.main.DataflowConstProp.diff

fn main() {
    *(&4);
}
