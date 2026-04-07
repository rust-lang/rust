//@compile-flags: -Zmiri-tree-borrows

// These tests fail Stacked Borrows, but pass Tree Borrows.

// The first three have in common that in SB a mutable reborrow is enough to produce
// write access errors, but in TB an actual write is needed.
// A modified version of each is also available that fails Tree Borrows.

mod pass_invalid_mut {
    // Copied directly from fail/stacked_borrows/pass_invalid_mut.rs
    // Version that fails TB: fail/tree_borrows/pass_invalid_mut.rs
    fn foo(_: &mut i32) {}

    pub fn main() {
        let x = &mut 42;
        let xraw = x as *mut _;
        let xref = unsafe { &mut *xraw };
        let _val = unsafe { *xraw }; // In SB this invalidates xref...
        foo(xref); // ...which then cannot be reborrowed here.
        // But in TB xref is Reserved and thus still writeable.
    }
}

mod return_invalid_mut {
    // Copied directly from fail/stacked_borrows/return_invalid_mut.rs
    // Version that fails TB: fail/tree_borrows/return_invalid_mut.rs
    fn foo(x: &mut (i32, i32)) -> &mut i32 {
        let xraw = x as *mut (i32, i32);
        let ret = unsafe { &mut (*xraw).1 };
        let _val = unsafe { *xraw }; // In SB this invalidates ret...
        ret // ...which then cannot be reborrowed here.
        // But in TB ret is Reserved and thus still writeable.
    }

    pub fn main() {
        foo(&mut (1, 2));
    }
}

mod static_memory_modification {
    // Copied directly from fail/stacked_borrows/static_memory_modification.rs
    // Version that fails TB: fail/tree_borrows/static_memory_modification.rs
    static X: usize = 5;

    #[allow(mutable_transmutes)]
    pub fn main() {
        let x = unsafe {
            std::mem::transmute::<&usize, &mut usize>(&X) // In SB this mutable reborrow fails.
            // But in TB we are allowed to transmute as long as we don't write.
        };
        assert_eq!(*&*x, 5);
    }
}

// This one is about direct writes to local variables not being in conflict
// with interior mutable reborrows.
#[allow(unused_assignments)] // spurious warning
fn interior_mut_reborrow() {
    use std::cell::UnsafeCell;

    let mut c = UnsafeCell::new(42);
    let ptr = c.get(); // first create interior mutable ptr
    c = UnsafeCell::new(13); // then write to parent
    assert_eq!(unsafe { ptr.read() }, 13); // then read through previous ptr
}

fn main() {
    pass_invalid_mut::main();
    return_invalid_mut::main();
    static_memory_modification::main();
    interior_mut_reborrow();
}
