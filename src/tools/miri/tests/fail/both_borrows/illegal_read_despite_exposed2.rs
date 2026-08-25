//@revisions: stack tree
//@compile-flags: -Zmiri-permissive-provenance
//@[tree]compile-flags: -Zmiri-tree-borrows

fn main() {
    unsafe {
        let root = &mut 42;
        let addr = root as *mut i32 as usize;
        let exposed_ptr = addr as *mut i32;
        // From the exposed ptr, we get a new unique ptr.
        let root2 = &mut *exposed_ptr;
        // Activate the reference (unnecessary on Stacked Borrows).
        *root2 = 42;
        // let _fool = root2 as *mut _; // this would fool us, since SRW(N+1) remains on the stack
        // Stack: Unknown(<N), Unique(N)
        // Stack if _fool existed: Unknown(<N), Unique(N), SRW(N+1)
        // And we test that it has uniqueness by doing a conflicting read.
        let _val = *exposed_ptr;
        // Stack: Unknown(<N), Disabled(N)
        // collapsed to Unknown(<N)
        // Stack if _fool existed: Unknown(<N), Disabled(N), SRW(N+1); collapsed to Unknown(<N+2) which would not cause an ERROR

        // Stack borrows would also fail if we replaced this with a read, but tree borrows would let it pass.
        *root2 = 3;
        //~[stack]^ ERROR: /write access .* tag does not exist in the borrow stack/
        //~[tree]| ERROR: /write access through .* is forbidden/
    }
}
