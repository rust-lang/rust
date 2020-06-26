// EMIT_MIR rustc.main.PromoteTemps.diff
// EMIT_MIR rustc.main.ConstProp.diff

fn main() {
    *(&4);
}
