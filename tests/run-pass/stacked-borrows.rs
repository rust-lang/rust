// Test various stacked-borrows-related things.
fn main() {
    deref_partially_dangling_raw();
    read_does_not_invalidate1();
    read_does_not_invalidate2();
    ref_raw_int_raw();
    mut_shr_raw();
    mut_raw_then_mut_shr();
    mut_raw_mut();
}

// Deref a raw ptr to access a field of a large struct, where the field
// is allocated but not the entire struct is.
// For now, we want to allow this.
fn deref_partially_dangling_raw() {
    let x = (1, 13);
    let xptr = &x as *const _ as *const (i32, i32, i32);
    let val = unsafe { (*xptr).1 };
    assert_eq!(val, 13);
}

// Make sure that reading from an `&mut` does, like reborrowing to `&`,
// NOT invalidate other reborrows.
fn read_does_not_invalidate1() {
    fn foo(x: &mut (i32, i32)) -> &i32 {
        let xraw = x as *mut (i32, i32);
        let ret = unsafe { &(*xraw).1 };
        let _val = x.1; // we just read, this does NOT invalidate the reborrows.
        ret
    }
    assert_eq!(*foo(&mut (1, 2)), 2);
}
// Same as above, but this time we first create a raw, then read from `&mut`
// and then freeze from the raw.
fn read_does_not_invalidate2() {
    fn foo(x: &mut (i32, i32)) -> &i32 {
        let xraw = x as *mut (i32, i32);
        let _val = x.1; // we just read, this does NOT invalidate the raw reborrow.
        let ret = unsafe { &(*xraw).1 };
        ret
    }
    assert_eq!(*foo(&mut (1, 2)), 2);
}

// Just to make sure that casting a ref to raw, to int and back to raw
// and only then using it works.  This rules out ideas like "do escape-to-raw lazily":
// After casting to int and back, we lost the tag that could have let us do that.
fn ref_raw_int_raw() {
    let mut x = 3;
    let xref = &mut x;
    let xraw = xref as *mut i32 as usize as *mut i32;
    assert_eq!(unsafe { *xraw }, 3);
}

// Creating a raw from a `&mut` through an `&` works, even if we
// write through that raw.
fn mut_shr_raw() {
    let mut x = 2;
    {
        let xref = &mut x;
        let xraw = &*xref as *const i32 as *mut i32;
        unsafe { *xraw = 4; }
    }
    assert_eq!(x, 4);
}

// Escape a mut to raw, then share the same mut and use the share, then the raw.
// That should work.
fn mut_raw_then_mut_shr() {
    let mut x = 2;
    {
        let xref = &mut x;
        let xraw = &mut *xref as *mut _;
        let xshr = &*xref;
        assert_eq!(*xshr, 2);
        unsafe { *xraw = 4; }
    }
    assert_eq!(x, 4);
}

// Ensure that if we derive from a mut a raw, and then from that a mut,
// and then read through the original mut, that does not invalidate the raw.
// This shows that the read-exception for `&mut` applies even if the `Shr` item
// on the stack is not at the top.
fn mut_raw_mut() {
    let mut x = 2;
    {
        let xref1 = &mut x;
        let xraw = xref1 as *mut _;
        let _xref2 = unsafe { &mut *xraw };
        let _val = *xref1;
        unsafe { *xraw = 4; }
        // we can now use both xraw and xref1, for reading
        assert_eq!(*xref1, 4);
        assert_eq!(unsafe { *xraw }, 4);
        assert_eq!(*xref1, 4);
        assert_eq!(unsafe { *xraw }, 4);
        // we cannot use xref2; see `compile-fail/stacked-borows/illegal_read4.rs`
    }
    assert_eq!(x, 4);
}
