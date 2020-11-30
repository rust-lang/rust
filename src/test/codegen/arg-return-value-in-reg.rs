//! Check that types of up to 128 bits are passed and returned by-value instead of via pointer.

// compile-flags: -C no-prepopulate-passes -O
// only-x86_64

#![crate_type = "lib"]

pub struct S {
    a: u64,
    b: u32,
    c: u32,
}

// CHECK: define i128 @modify(i128{{( %0)?}})
#[no_mangle]
pub fn modify(s: S) -> S {
    S { a: s.a + s.a, b: s.b + s.b, c: s.c + s.c }
}

#[repr(packed)]
pub struct TooBig {
    a: u64,
    b: u32,
    c: u32,
    d: u8,
}

// CHECK: define void @m_big(%TooBig* [[ATTRS:.*sret.*]], %TooBig* [[ATTRS2:.*]] %s)
#[no_mangle]
pub fn m_big(s: TooBig) -> TooBig {
    TooBig { a: s.a + s.a, b: s.b + s.b, c: s.c + s.c, d: s.d + s.d }
}
