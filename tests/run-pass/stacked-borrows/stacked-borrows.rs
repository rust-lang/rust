// Test various stacked-borrows-related things.
fn main() {
    deref_partially_dangling_raw();
    read_does_not_invalidate1();
    read_does_not_invalidate2();
    ref_raw_int_raw();
    mut_raw_then_mut_shr();
    mut_shr_then_mut_raw();
    mut_raw_mut();
    partially_invalidate_mut();
    drop_after_sharing();
    direct_mut_to_const_raw();
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
// and only then using it works. This rules out ideas like "do escape-to-raw lazily";
// after casting to int and back, we lost the tag that could have let us do that.
fn ref_raw_int_raw() {
    let mut x = 3;
    let xref = &mut x;
    let xraw = xref as *mut i32 as usize as *mut i32;
    assert_eq!(unsafe { *xraw }, 3);
}

// Escape a mut to raw, then share the same mut and use the share, then the raw.
// That should work.
fn mut_raw_then_mut_shr() {
    let mut x = 2;
    let xref = &mut x;
    let xraw = &mut *xref as *mut _;
    let xshr = &*xref;
    assert_eq!(*xshr, 2);
    unsafe { *xraw = 4; }
    assert_eq!(x, 4);
}

// Create first a shared reference and then a raw pointer from a `&mut`
// should permit mutation through that raw pointer.
fn mut_shr_then_mut_raw() {
    let xref = &mut 2;
    let _xshr = &*xref;
    let xraw = xref as *mut _;
    unsafe { *xraw = 3; }
    assert_eq!(*xref, 3);
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

fn partially_invalidate_mut() {
    let data = &mut (0u8, 0u8);
    let reborrow = &mut *data as *mut (u8, u8);
    let shard = unsafe { &mut (*reborrow).0 };
    data.1 += 1; // the deref overlaps with `shard`, but that is ok; the access does not overlap.
    *shard += 1; // so we can still use `shard`.
    assert_eq!(*data, (1, 1));
}

// Make sure that we can handle the situation where a loaction is frozen when being dropped.
fn drop_after_sharing() {
    let x = String::from("hello!");
    let _len = x.len();
}

// Make sure that coercing &mut T to *const T produces a writeable pointer.
fn direct_mut_to_const_raw() {
    // FIXME: This is currently disabled, waiting on a fix for <https://github.com/rust-lang/rust/issues/56604>
    /*let x = &mut 0;
    let y: *const i32 = x;
    unsafe { *(y as *mut i32) = 1; }
    assert_eq!(*x, 1);
    */
}
