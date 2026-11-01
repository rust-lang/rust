//@ check-pass
//@ run-rustfix
//@ rustfix-only-machine-applicable

#![allow(dead_code, unused_variables)]
#![warn(raw_borrows_via_references)]

struct A {
    a: i32,
    b: u8,
}

fn via_ref(x: *const (i32, i32)) -> *const i32 {
    unsafe { &(*x).0 as *const i32 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
}

fn via_ref_struct(x: *const A) -> *const u8 {
    unsafe { &(*x).b as *const u8 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
}

fn via_ref_mut(x: *mut (i32, i32)) -> *mut i32 {
    unsafe { &mut (*x).0 as *mut i32 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
}

fn via_ref_struct_mut(x: *mut A) -> *mut i32 {
    unsafe { &mut (*x).a as *mut i32 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
}

fn multiple_casts(x: *const (i32, i32)) -> *const u8 {
    unsafe { &(*x).0 as *const i32 as *const u8 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
}

fn ret_i32() -> i32 {
    0
}

fn temporaries() {
    let _ = &1 as *const i32;
    let _ = &(1 + 2) as *const i32;
    let _ = &ret_i32() as *const i32;
    let _ = &(1, 2) as *const (i32, i32);
    let _ = &[1, 2, 3] as *const [i32; 3];
    let _ = &A { a: 0, b: 0 } as *const A;
    let _ = &mut 4 as *mut i32;
}

fn inner_blocks() {
    let x = 0;
    let _ = { &x as *const i32 };
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]

    let _ = {
        let y = 0;
        { &y as *const i32 }
        //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
    };

    let _ = { &1 as *const i32 };
}

macro_rules! ref_cast {
    ($e:expr) => {
        &$e as *const i32
        //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]
    };
}

fn from_macro(x: *const i32) -> *const i32 {
    unsafe { ref_cast!(*x) }
}

fn main() {
    let a = 0;
    let a = &a as *const i32;
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]

    let mut b = 0;
    let b = &mut b as *mut i32;
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately cast to a raw pointers [raw_borrows_via_references]

    let i = &1 as *const i32;
}
