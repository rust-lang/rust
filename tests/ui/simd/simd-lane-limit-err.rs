#![feature(rustc_attrs, repr_simd)]

//@ build-fail

#[repr(simd, packed)]
#[rustc_simd_monomorphize_lane_limit = "4"]
struct V<T, const N: usize>([T; N]);

fn main() {
    let _a: V<i32, 8> = V([0; 8]); //~ ERROR the SIMD type `V<i32, 8>` has more elements than the limit 4
}
