// skip-filecheck
// unit-test: DeadStoreElimination
// compile-flags: -Zmir-emit-retag

// Test that we don't remove pointer to int casts or retags

// EMIT_MIR provenance_soundness.pointer_to_int.DeadStoreElimination.diff
fn pointer_to_int(p: *mut i32) {
    let _x = p as usize;
    let _y = p as isize;
}

// EMIT_MIR provenance_soundness.retags.DeadStoreElimination.diff
fn retags(_r: &mut i32) {}

fn main() {
    pointer_to_int(&mut 5 as *mut _);
    retags(&mut 5);
}
