#![feature(rustc_attrs, repr_simd)]

//@ build-pass

#[repr(simd, packed)]
#[rustc_simd_monomorphize_lane_limit(8)]
struct V<T, const N: usize>([T; N]);

const LANES: usize = 4;

fn main() {
    let _x: V<i32, LANES> = V([0; LANES]);
}
