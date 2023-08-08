fn main() {
    // This pointer *could* be NULL so we cannot load from it, not even at ZST
    let ptr = (&0u8 as *const u8).wrapping_sub(0x800) as *const ();
    let _x: () = unsafe { *ptr }; //~ ERROR: out-of-bounds
}
