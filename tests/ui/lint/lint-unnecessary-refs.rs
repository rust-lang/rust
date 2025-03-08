//@ check-pass
//@ run-rustfix

#![allow(dead_code, unused_variables)]

struct A {
    a: i32,
    b: u8,
}

fn via_ref(x: *const (i32, i32)) -> *const i32 {
    unsafe { &(*x).0 as *const i32 }
    //~^ WARN creating a intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn via_ref_struct(x: *const A) -> *const u8 {
    unsafe { &(*x).b as *const u8 }
    //~^ WARN creating a intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn via_ref_mut(x: *mut (i32, i32)) -> *mut i32 {
    unsafe { &mut (*x).0 as *mut i32 }
    //~^ WARN creating a intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn via_ref_struct_mut(x: *mut A) -> *mut i32 {
    unsafe { &mut (*x).a as *mut i32 }
    //~^ WARN creating a intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn main() {
    let a = 0;
    let a = &a as *const i32;
    //~^ WARN creating a intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]

    let mut b = 0;
    let b = &mut b as *mut i32;
    //~^ WARN creating a intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}
