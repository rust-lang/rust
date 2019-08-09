#![feature(repr_simd)]

#[repr(C)] //~ ERROR: attribute should be applied to struct, enum, or union
fn f() {}

#[repr(C)]
struct SExtern(f64, f64);

#[repr(packed)]
struct SPacked(f64, f64);

#[repr(simd)]
struct SSimd(f64, f64);

#[repr(i8)] //~ ERROR: attribute should be applied to enum
struct SInt(f64, f64);

#[repr(C)]
enum EExtern { A, B }

#[repr(align(8))]
enum EAlign { A, B }

#[repr(packed)] //~ ERROR: attribute should be applied to struct
enum EPacked { A, B }

#[repr(simd)] //~ ERROR: attribute should be applied to struct
enum ESimd { A, B }

#[repr(i8)]
enum EInt { A, B }

fn main() {}
