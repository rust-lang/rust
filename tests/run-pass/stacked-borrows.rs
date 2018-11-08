// Test various stacked-borrows-related things.
fn main() {
    deref_partially_dangling_raw();
    read_does_not_invalidate1();
    read_does_not_invalidate2();
}

// Deref a raw ptr to access a field of a large struct, where the field
// is allocated but not the entire struct is.
// For now, we want to allow this.
fn deref_partially_dangling_raw() {
    let x = (1, 1);
    let xptr = &x as *const _ as *const (i32, i32, i32);
    let _val = unsafe { (*xptr).1 };
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
    foo(&mut (1, 2));
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
    foo(&mut (1, 2));
}
