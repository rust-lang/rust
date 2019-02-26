#![feature(rustc_attrs)]

// Show that `homogeneous_aggregate` code ignores zero-length C
// arrays.  This matches the recent C standard, though not the
// behavior of all older compilers, which somtimes consider `T[0]` to
// be a "flexible array member" (see discussion on #56877 for
// details).

#[repr(C)]
pub struct Foo {
    x: u32
}

#[repr(C)]
pub struct Middle {
    pub a: f32,
    pub foo: [Foo; 0],
    pub b: f32,
}

#[rustc_layout(homogeneous_aggregate)]
pub type TestMiddle = Middle;
//~^ ERROR homogeneous_aggregate: Homogeneous

#[repr(C)]
pub struct Final {
    pub a: f32,
    pub b: f32,
    pub foo: [Foo; 0],
}

#[rustc_layout(homogeneous_aggregate)]
pub type TestFinal = Final;
//~^ ERROR homogeneous_aggregate: Homogeneous

fn main() { }
