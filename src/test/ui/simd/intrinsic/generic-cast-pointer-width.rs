// run-pass
#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 4]);

fn main() {
    let u = V::<usize>([0, 1, 2, 3]);
    let uu32: V<u32> = unsafe { simd_cast(u) };
    let ui64: V<i64> = unsafe { simd_cast(u) };

    for (u, (uu32, ui64)) in u.0.iter().zip(uu32.0.iter().zip(ui64.0.iter())) {
        assert_eq!(*u as u32, *uu32);
        assert_eq!(*u as i64, *ui64);
    }
}
