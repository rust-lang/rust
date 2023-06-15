// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: RemoveStorageMarkers

// Checks that storage markers are removed at opt-level=0.
//
// compile-flags: -C opt-level=0 -Coverflow-checks=off

// EMIT_MIR remove_storage_markers.main.RemoveStorageMarkers.diff
fn main() {
    let mut sum = 0;
    for i in 0..10 {
        sum += i;
    }
}
