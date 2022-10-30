// unit-test
// EMIT_MIR ref_deref_project.main.PromoteTemps.diff
// EMIT_MIR ref_deref_project.main.ConstProp.diff

fn main() {
    *(&(4, 5).1); // This does not currently propagate (#67862)
}
