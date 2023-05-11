//@compile-flags: -Zmiri-permissive-provenance

fn main() {
    unsafe {
        let root = &mut 42;
        let addr = root as *mut i32 as usize;
        let exposed_ptr = addr as *mut i32;
        // From the exposed ptr, we get a new SRO ptr.
        let root2 = &*exposed_ptr;
        // Stack: Unknown(<N), SRO(N), SRO(N+1)
        // And we test that it is read-only by doing a conflicting write.
        // (The write is still fine, using the `root as *mut i32` provenance which got exposed.)
        *exposed_ptr = 0;
        // Stack: Unknown(<N)
        let _val = *root2; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}
