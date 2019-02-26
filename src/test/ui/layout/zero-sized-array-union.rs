#![feature(rustc_attrs)]

// Various tests around the behavior of zero-sized arrays and
// unions. This matches the behavior of modern C compilers, though
// older compilers (and sometimes clang) treat `T[0]` as a "flexible
// array member". See more
// details in #56877.

#[derive(Copy, Clone)]
#[repr(C)]
struct Empty { }

#[derive(Copy, Clone)]
#[repr(C)]
struct Empty2 {
    e: Empty
}

#[derive(Copy, Clone)]
#[repr(C)]
struct Empty3 {
    z: [f32; 0],
}

#[derive(Copy, Clone)]
#[repr(C)]
struct Empty4 {
    e: Empty3
}

#[repr(C)]
union U1 {
    s: Empty
}

#[repr(C)]
union U2 {
    s: Empty2
}

#[repr(C)]
union U3 {
    s: Empty3
}

#[repr(C)]
union U4 {
    s: Empty4
}

#[repr(C)]
struct Baz1 {
    x: f32,
    y: f32,
    u: U1,
}

#[rustc_layout(homogeneous_aggregate)]
type TestBaz1 = Baz1;
//~^ ERROR homogeneous_aggregate: Homogeneous

#[repr(C)]
struct Baz2 {
    x: f32,
    y: f32,
    u: U2,
}

#[rustc_layout(homogeneous_aggregate)]
type TestBaz2 = Baz2;
//~^ ERROR homogeneous_aggregate: Homogeneous

#[repr(C)]
struct Baz3 {
    x: f32,
    y: f32,
    u: U3,
}

#[rustc_layout(homogeneous_aggregate)]
type TestBaz3 = Baz3;
//~^ ERROR homogeneous_aggregate: Homogeneous

#[repr(C)]
struct Baz4 {
    x: f32,
    y: f32,
    u: U4,
}

#[rustc_layout(homogeneous_aggregate)]
type TestBaz4 = Baz4;
//~^ ERROR homogeneous_aggregate: Homogeneous

fn main() { }
