//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks if a local access gets correctly triggered during wildcard access,
/// if we know that the only exposed reference is local to the node.
pub fn main() {
    let mut x: u32 = 0;

    let ref1 = &mut x;

    let int = ref1 as *mut u32 as usize;
    let wild = int as *mut u32;

    // Activates ref1.
    unsafe { wild.write(41) };

    // Reads from x causing a foreign read on ref1, freezing it
    // (because it was active).
    let _y = x;

    // The only exposed reference (ref1) is frozen, so wildcard writes are UB.
    unsafe { wild.write(0) }; //~ ERROR: /write access through <wildcard> at .* is forbidden/
}
