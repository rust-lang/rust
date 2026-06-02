#![feature(repr_simd)]

#[repr(C)] //~ ERROR: attribute cannot be used on
fn f() {}

#[repr(C)]
struct SExtern(f64, f64);

#[repr(packed)]
struct SPacked(f64, f64);

#[repr(simd)]
struct SSimd([f64; 2]);

#[repr(i8)] //~ ERROR: `#[repr]` attribute cannot be used on
struct SInt(f64, f64);

#[repr(C)]
enum EExtern {
    A,
    B,
}

#[repr(align(8))]
enum EAlign {
    A,
    B,
}

#[repr(packed)] //~ ERROR: attribute cannot be used on
enum EPacked {
    A,
    B,
}

#[repr(simd)] //~ ERROR: attribute cannot be used on
enum ESimd {
    A,
    B,
}

#[repr(i8)]
enum EInt {
    A,
    B,
}

#[repr()] //~ ERROR attribute cannot be used on
//~^ WARN unused attribute
type SirThisIsAType = i32;

#[repr()]
//~^ WARN unused attribute
struct EmptyReprArgumentList(i32);

fn main() {}
