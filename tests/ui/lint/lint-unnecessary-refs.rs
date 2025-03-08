#![deny(unnecessary_refs)]

struct A {
    a: i32,
    b: u8,
}

fn via_ref(x: *const (i32, i32)) -> *const i32 {
    unsafe { &(*x).0 as *const i32 }
    //~^ ERROR creating unecessary reference is discouraged [unnecessary_refs]
}

fn via_ref_struct(x: *const A) -> *const u8 {
    unsafe { &(*x).b as *const u8 }
    //~^ ERROR creating unecessary reference is discouraged [unnecessary_refs]
}

fn main() {}
