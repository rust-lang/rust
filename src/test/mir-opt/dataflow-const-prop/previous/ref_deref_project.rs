// unit-test: DataflowConstProp
// EMIT_MIR ref_deref_project.main.PromoteTemps.diff
// EMIT_MIR ref_deref_project.main.DataflowConstProp.diff

fn main() {
    *(&(4, 5).1); // This does not currently propagate (#67862)
}
