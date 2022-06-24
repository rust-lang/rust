// compile-flags: -Zmiri-permissive-provenance

fn main() {
    unsafe {
        let root = &mut 42;
        let addr = root as *mut i32 as usize;
        let exposed_ptr = addr as *mut i32;
        // From the exposed ptr, we get a new unique ptr.
        let root2 = &mut *exposed_ptr;
        // let _fool = root2 as *mut _; // this would [fool] us, since SRW(N+1) remains on the stack
        // Stack: Unknown(<N), Unique(N) [, SRW(N+1)]
        // And we test that it has uniqueness by doing a conflicting read.
        let _val = *exposed_ptr;
        // Stack: Unknown(<N), Disabled(N) [, SRW(N+1)]
        // collapsed to Unknown(<N) [Unknown(<N+2)]
        let _val = *root2; //~ ERROR: borrow stack
    }
}
