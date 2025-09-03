//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(allocator_api)]
use std::cell::Cell;
use std::ptr;

// Test various aliasing-model-related things.
fn main() {
    read_does_not_invalidate1();
    read_does_not_invalidate2();
    mut_raw_then_mut_shr();
    mut_shr_then_mut_raw();
    mut_raw_mut();
    partially_invalidate_mut();
    drop_after_sharing();
    two_raw();
    shr_and_raw();
    disjoint_mutable_subborrows();
    raw_ref_to_part();
    array_casts();
    mut_below_shr();
    wide_raw_ptr_in_tuple();
    not_unpin_not_protected();
    write_does_not_invalidate_all_aliases();
    box_into_raw_allows_interior_mutable_alias();
    cell_inside_struct();
    zst();
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

// Escape a mut to raw, then share the same mut and use the share, then the raw.
// That should work.
fn mut_raw_then_mut_shr() {
    let mut x = 2;
    let xref = &mut x;
    let xraw = &mut *xref as *mut _;
    let xshr = &*xref;
    assert_eq!(*xshr, 2);
    unsafe {
        *xraw = 4;
    }
    assert_eq!(x, 4);
}

// Create first a shared reference and then a raw pointer from a `&mut`
// should permit mutation through that raw pointer.
fn mut_shr_then_mut_raw() {
    let xref = &mut 2;
    let _xshr = &*xref;
    let xraw = xref as *mut _;
    unsafe {
        *xraw = 3;
    }
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
        unsafe {
            *xraw = 4;
        }
        // we can now use both xraw and xref1, for reading
        assert_eq!(*xref1, 4);
        assert_eq!(unsafe { *xraw }, 4);
        assert_eq!(*xref1, 4);
        assert_eq!(unsafe { *xraw }, 4);
        // we cannot use xref2; see `compile-fail/stacked-borrows/illegal_read4.rs`
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

// Make sure that we can handle the situation where a location is frozen when being dropped.
fn drop_after_sharing() {
    let x = String::from("hello!");
    let _len = x.len();
}

// Make sure that we can create two raw pointers from a mutable reference and use them both.
fn two_raw() {
    unsafe {
        let x = &mut 0;
        let y1 = x as *mut _;
        let y2 = x as *mut _;
        *y1 += 2;
        *y2 += 1;
    }
}

// Make sure that creating a *mut does not invalidate existing shared references.
fn shr_and_raw() {
    unsafe {
        use std::mem;
        let x = &mut 0;
        let y1: &i32 = mem::transmute(&*x); // launder lifetimes
        let y2 = x as *mut _;
        let _val = *y1;
        *y2 += 1;
    }
}

fn disjoint_mutable_subborrows() {
    struct Foo {
        a: String,
        b: Vec<u32>,
    }

    unsafe fn borrow_field_a<'a>(this: *mut Foo) -> &'a mut String {
        &mut (*this).a
    }

    unsafe fn borrow_field_b<'a>(this: *mut Foo) -> &'a mut Vec<u32> {
        &mut (*this).b
    }

    let mut foo = Foo { a: "hello".into(), b: vec![0, 1, 2] };

    let ptr = &mut foo as *mut Foo;

    let a = unsafe { borrow_field_a(ptr) };
    let b = unsafe { borrow_field_b(ptr) };
    b.push(4);
    a.push_str(" world");
    assert_eq!(format!("{:?} {:?}", a, b), r#""hello world" [0, 1, 2, 4]"#);
}

fn raw_ref_to_part() {
    struct Part {
        _lame: i32,
    }

    #[repr(C)]
    struct Whole {
        part: Part,
        extra: i32,
    }

    let it = Box::new(Whole { part: Part { _lame: 0 }, extra: 42 });
    let whole = ptr::addr_of_mut!(*Box::leak(it));
    let part = unsafe { ptr::addr_of_mut!((*whole).part) };
    let typed = unsafe { &mut *(part as *mut Whole) };
    assert!(typed.extra == 42);
    drop(unsafe { Box::from_raw(whole) });
}

/// When casting an array reference to a raw element ptr, that should cover the whole array.
fn array_casts() {
    let mut x: [usize; 2] = [0, 0];
    let p = &mut x as *mut usize;
    unsafe {
        *p.add(1) = 1;
    }

    let x: [usize; 2] = [0, 1];
    let p = &x as *const usize;
    assert_eq!(unsafe { *p.add(1) }, 1);
}

/// Transmuting &&i32 to &&mut i32 is fine.
fn mut_below_shr() {
    let x = 0;
    let y = &x;
    let p = unsafe { core::mem::transmute::<&&i32, &&mut i32>(&y) };
    let r = &**p;
    let _val = *r;
}

fn wide_raw_ptr_in_tuple() {
    let mut x: Box<dyn std::any::Any> = Box::new("ouch");
    let r = &mut *x as *mut dyn std::any::Any;
    // This triggers the visitor-based recursive retagging. It is *not* supposed to retag raw
    // pointers, but then the visitor might recurse into the "fields" of a wide raw pointer and
    // finds a reference (to a vtable) there that it wants to retag... and that would be Wrong.
    let pair = (r, &0);
    let r = unsafe { &mut *pair.0 };
    // Make sure the fn ptr part of the vtable is still fine.
    r.type_id();
}

fn not_unpin_not_protected() {
    // `&mut !Unpin`, at least for now, does not get `noalias` nor `dereferenceable`, so we also
    // don't add protectors. (We could, but until we have a better idea for where we want to go with
    // the self-referential-coroutine situation, it does not seem worth the potential trouble.)
    use std::marker::PhantomPinned;

    pub struct NotUnpin(#[allow(dead_code)] i32, PhantomPinned);

    fn inner(x: &mut NotUnpin, f: fn(&mut NotUnpin)) {
        // `f` is allowed to deallocate `x`.
        f(x)
    }

    inner(Box::leak(Box::new(NotUnpin(0, PhantomPinned))), |x| {
        let raw = x as *mut _;
        drop(unsafe { Box::from_raw(raw) });
    });
}

fn write_does_not_invalidate_all_aliases() {
    mod other {
        /// Some private memory to store stuff in.
        static mut S: *mut i32 = 0 as *mut i32;

        pub fn lib1(x: &&mut i32) {
            unsafe {
                S = (x as *const &mut i32).cast::<*mut i32>().read();
            }
        }

        pub fn lib2() {
            unsafe {
                *S = 1337;
            }
        }
    }

    let x = &mut 0;
    other::lib1(&x);
    *x = 42; // a write to x -- invalidates other pointers?
    other::lib2();
    assert_eq!(*x, 1337); // oops, the value changed! I guess not all pointers were invalidated
}

fn box_into_raw_allows_interior_mutable_alias() {
    unsafe {
        let b = Box::new(Cell::new(42));
        let raw = Box::into_raw(b);
        let c = &*raw;
        let d = raw.cast::<i32>(); // bypassing `Cell` -- only okay in Miri tests
        // `c` and `d` should permit arbitrary aliasing with each other now.
        *d = 1;
        c.set(2);
        drop(Box::from_raw(raw));
    }
}

fn cell_inside_struct() {
    struct Foo {
        field1: u32,
        field2: Cell<u32>,
    }

    let mut root = Foo { field1: 42, field2: Cell::new(88) };
    let a = &mut root;

    // Writing to `field2`, which is interior mutable, should be allowed.
    (*a).field2.set(10);

    // Writing to `field1`, which is reserved, should also be allowed.
    (*a).field1 = 88;
}

/// ZST reborrows on various kinds of dangling pointers are valid.
fn zst() {
    unsafe {
        // Integer pointer.
        let ptr = ptr::without_provenance_mut::<()>(15);
        let _ref = &mut *ptr;

        // Out-of-bounds pointer.
        let mut b = Box::new(0u8);
        let ptr = (&raw mut *b).wrapping_add(15) as *mut ();
        let _ref = &mut *ptr;

        // Deallocated pointer.
        let ptr = &raw mut *b as *mut ();
        drop(b);
        let _ref = &mut *ptr;
    }
}
