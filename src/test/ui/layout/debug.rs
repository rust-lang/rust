// normalize-stderr-test "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
#![feature(never_type, rustc_attrs, type_alias_impl_trait)]
#![crate_type = "lib"]

#[rustc_layout(debug)]
enum E { Foo, Bar(!, i32, i32) } //~ ERROR: layout_of

#[rustc_layout(debug)]
struct S { f1: i32, f2: (), f3: i32 } //~ ERROR: layout_of

#[rustc_layout(debug)]
union U { f1: (i32, i32), f3: i32 } //~ ERROR: layout_of

#[rustc_layout(debug)]
type Test = Result<i32, i32>; //~ ERROR: layout_of

#[rustc_layout(debug)]
type T = impl std::fmt::Debug; //~ ERROR: layout_of

#[rustc_layout(debug)]
pub union V { //~ ERROR: layout_of
    a: [u16; 0],
    b: u8,
}

#[rustc_layout(debug)]
pub union W { //~ ERROR: layout_of
    b: u8,
    a: [u16; 0],
}

#[rustc_layout(debug)]
pub union Y { //~ ERROR: layout_of
    b: [u8; 0],
    a: [u16; 0],
}

#[rustc_layout(debug)]
type X = std::mem::MaybeUninit<u8>; //~ ERROR: layout_of

fn f() -> T {
    0i32
}
