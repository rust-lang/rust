//@revisions: default uniq
//@compile-flags: -Zmiri-tree-borrows
//@[uniq]compile-flags: -Zmiri-unique-is-unique
#![feature(allocator_api)]

use std::{mem, ptr};

fn main() {
    aliasing_read_only_mutable_refs();
    string_as_mut_ptr();
    two_mut_protected_same_alloc();
    direct_mut_to_const_raw();
    local_addr_of_mut();
    returned_mut_is_usable();

    // Stacked Borrows tests
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
}

#[allow(unused_assignments)]
fn local_addr_of_mut() {
    let mut local = 0;
    let ptr = ptr::addr_of_mut!(local);
    // In SB, `local` and `*ptr` would have different tags, but in TB they have the same tag.
    local = 1;
    unsafe { *ptr = 2 };
    local = 3;
    unsafe { *ptr = 4 };
}

// Tree Borrows has no issue with several mutable references existing
// at the same time, as long as they are used only immutably.
// I.e. multiple Reserved can coexist.
pub fn aliasing_read_only_mutable_refs() {
    unsafe {
        let base = &mut 42u64;
        let r1 = &mut *(base as *mut u64);
        let r2 = &mut *(base as *mut u64);
        let _l = *r1;
        let _l = *r2;
    }
}

pub fn string_as_mut_ptr() {
    // This errors in Stacked Borrows since as_mut_ptr restricts the provenance,
    // but with Tree Borrows it should work.
    unsafe {
        let mut s = String::from("hello");
        s.reserve(1); // make the `str` that `s` derefs to not cover the entire `s`.

        // Prevent automatically dropping the String's data
        let mut s = mem::ManuallyDrop::new(s);

        let ptr = s.as_mut_ptr();
        let len = s.len();
        let capacity = s.capacity();

        let s = String::from_raw_parts(ptr, len, capacity);

        assert_eq!(String::from("hello"), s);
    }
}

// This function checks that there is no issue with having two mutable references
// from the same allocation both under a protector.
// This is safe code, it must absolutely not be UB.
// This test failing is a symptom of forgetting to check that only initialized
// locations can cause protector UB.
fn two_mut_protected_same_alloc() {
    fn write_second(_x: &mut u8, y: &mut u8) {
        // write through `y` will make some locations of `x` (protected)
        // become Disabled. Those locations are outside of the range on which
        // `x` is initialized, and the protector must not trigger.
        *y = 1;
    }

    let mut data = (0u8, 1u8);
    write_second(&mut data.0, &mut data.1);
}

// This checks that a reborrowed mutable reference returned from a function
// is actually writeable.
// The fact that this is not obvious is due to the addition of
// implicit reads on function exit that might freeze the return value.
fn returned_mut_is_usable() {
    fn reborrow(x: &mut u8) -> &mut u8 {
        let y = &mut *x;
        // Activate the reference so that it is vulnerable to foreign reads.
        *y = *y;
        y
        // An implicit read through `x` is inserted here.
    }
    let mut data = 0;
    let x = &mut data;
    let y = reborrow(x);
    *y = 1;
}

// ----- The tests below were taken from Stacked Borrows ----

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

// Make sure that coercing &mut T to *const T produces a writeable pointer.
fn direct_mut_to_const_raw() {
    let x = &mut 0;
    let y: *const i32 = x;
    unsafe {
        *(y as *mut i32) = 1;
    }
    assert_eq!(*x, 1);
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
    // In TB there are other ways to do that (`addr_of!(*x)` has the same tag as `x`),
    // but let's still make sure this SB test keeps working.

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
