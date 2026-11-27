// Test that we don't remove pointer to int casts or retags
//@ test-mir-pass: DeadStoreElimination-initial

// EMIT_MIR provenance_soundness.pointer_to_int.DeadStoreElimination-initial.diff
fn pointer_to_int(p: *mut i32) {
    // CHECK-LABEL: fn pointer_to_int(
    // CHECK: {{_.*}} = {{.*}} as usize (PointerExposeProvenance);
    // CHECK: {{_.*}} = {{.*}} as isize (PointerExposeProvenance);
    let _x = p as usize;
    let _y = p as isize;
}

fn main() {
    pointer_to_int(&mut 5 as *mut _);
}
