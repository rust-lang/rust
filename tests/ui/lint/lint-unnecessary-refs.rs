//@ check-pass
//@ run-rustfix

#![allow(dead_code, unused_variables)]
#![warn(unnecessary_refs)]

struct A {
    a: i32,
    b: u8,
}

fn via_ref(x: *const (i32, i32)) -> *const i32 {
    unsafe { &(*x).0 as *const i32 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn via_ref_struct(x: *const A) -> *const u8 {
    unsafe { &(*x).b as *const u8 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn via_ref_mut(x: *mut (i32, i32)) -> *mut i32 {
    unsafe { &mut (*x).0 as *mut i32 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn via_ref_struct_mut(x: *mut A) -> *mut i32 {
    unsafe { &mut (*x).a as *mut i32 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
}

fn multiple_casts(x: *const (i32, i32)) -> *const u8 {
    unsafe { &(*x).0 as *const i32 as *const u8 }
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
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
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]

    let _ = {
        let y = 0;
        { &y as *const i32 }
        //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]
    };

    let _ = { &1 as *const i32 };
}

fn main() {
    let a = 0;
    let a = &a as *const i32;
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]

    let mut b = 0;
    let b = &mut b as *mut i32;
    //~^ WARN creating an intermediate reference implies aliasing requirements even when immediately casting to raw pointers [unnecessary_refs]

    let i = &1 as *const i32;
}
