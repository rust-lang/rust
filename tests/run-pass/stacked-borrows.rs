// Test various stacked-borrows-related things.
fn main() {
    deref_partially_dangling_raw();
    read_does_not_invalidate1();
    read_does_not_invalidate2();
    ref_raw_int_raw();
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
