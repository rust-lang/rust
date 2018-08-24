#![feature(repr_simd, platform_intrinsics)]

// error-pattern:monomorphising SIMD type `Simd2<X>` with a non-machine element type `X`

struct X(Vec<i32>);
#[repr(simd)]
struct Simd2<T>(T, T);

fn main() {
    let _ = Simd2(X(vec![]), X(vec![]));
}
