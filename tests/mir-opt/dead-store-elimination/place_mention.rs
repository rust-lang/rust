// skip-filecheck
// unit-test: DeadStoreElimination
// compile-flags: -Zmir-keep-place-mention

// EMIT_MIR place_mention.main.DeadStoreElimination.diff
fn main() {
    // Verify that we account for the `PlaceMention` statement as a use of the tuple,
    // and don't remove it as a dead store.
    let (_, _) = ("Hello", "World");
}
