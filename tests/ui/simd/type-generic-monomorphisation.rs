//@ build-fail

#![feature(repr_simd, intrinsics)]


//@ error-pattern:monomorphising SIMD type `Simd2<X>` with a non-primitive-scalar (integer/float/pointer) element type `X`

struct X(Vec<i32>);
#[repr(simd)]
struct Simd2<T>([T; 2]);

fn main() {
    let _ = Simd2([X(vec![]), X(vec![])]);
}
