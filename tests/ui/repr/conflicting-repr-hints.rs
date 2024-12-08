#![allow(dead_code)]

#[repr(C)]
enum A {
    A,
}

#[repr(u64)]
enum B {
    B,
}

#[repr(C, u64)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
enum C {
    C,
}

#[repr(u32, u64)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
enum D {
    D,
}

#[repr(C, packed)]
struct E(i32);

#[repr(packed, align(8))]
struct F(i32); //~ ERROR type has conflicting packed and align representation hints

#[repr(packed)]
#[repr(align(8))]
struct G(i32); //~ ERROR type has conflicting packed and align representation hints

#[repr(align(8))]
#[repr(packed)]
struct H(i32); //~ ERROR type has conflicting packed and align representation hints

#[repr(packed, packed(2))]
struct I(i32); //~ ERROR type has conflicting packed representation hints

#[repr(packed(2))]
#[repr(packed)]
struct J(i32); //~ ERROR type has conflicting packed representation hints

#[repr(packed, packed(1))]
struct K(i32);

#[repr(packed, align(8))]
union X {
    //~^ ERROR type has conflicting packed and align representation hints
    i: i32,
}

#[repr(packed)]
#[repr(align(8))]
union Y {
    //~^ ERROR type has conflicting packed and align representation hints
    i: i32,
}

#[repr(align(8))]
#[repr(packed)]
union Z {
    //~^ ERROR type has conflicting packed and align representation hints
    i: i32,
}

#[repr(packed, align(0x100))]
pub struct S(u16); //~ ERROR type has conflicting packed and align representation hints

#[repr(packed, align(0x100))]
pub union U { //~ ERROR type has conflicting packed and align representation hints
    u: u16
}

static B: U = U { u: 0 };
static A: S = S(0);

fn main() {}
