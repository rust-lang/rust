//@ run-pass

use std::fmt::Debug;
use std::hint::black_box;

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug, Default)]
struct Regular(f32, f64);

#[repr(C, packed)]
#[derive(Copy, Clone, PartialEq, Debug, Default)]
struct Packed(f32, f64);

#[repr(C, align(64))]
#[derive(Copy, Clone, PartialEq, Debug, Default)]
struct AlignedF32(f32);

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug, Default)]
struct Aligned(f64, AlignedF32);

#[inline(never)]
extern "C" fn read<T: Copy>(x: &T) -> T {
    *black_box(x)
}

#[inline(never)]
extern "C" fn write<T: Copy>(x: T, dest: &mut T) {
    *dest = black_box(x)
}

#[track_caller]
fn check<T: Copy + PartialEq + Debug + Default>(x: T) {
    assert_eq!(read(&x), x);
    let mut out = T::default();
    write(x, &mut out);
    assert_eq!(out, x);
}

fn main() {
    check(Regular(1.0, 2.0));
    check(Packed(3.0, 4.0));
    check(Aligned(5.0, AlignedF32(6.0)));
}
