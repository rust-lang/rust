// EMIT_MIR rustc.main.PromoteTemps.diff
// EMIT_MIR rustc.main.ConstProp.diff

fn main() {
    *(&(4, 5).1); // This does not currently propagate (#67862)
}
