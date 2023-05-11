//@compile-flags: -Zmiri-permissive-provenance

fn main() {
    unsafe {
        let root = &mut 42;
        let addr = root as *mut i32 as usize;
        let exposed_ptr = addr as *mut i32;
        // From the exposed ptr, we get a new unique ptr.
        let root2 = &mut *exposed_ptr;
        let _fool = root2 as *mut _; // this would have fooled the old untagged pointer logic
        // Stack: Unknown(<N), Unique(N), SRW(N+1)
        // And we test that it has uniqueness by doing a conflicting write.
        *exposed_ptr = 0;
        // Stack: Unknown(<N)
        let _val = *root2; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}
