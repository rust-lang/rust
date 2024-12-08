//@ run-pass
//@ ignore-emscripten

// Short form of the generic gather/scatter tests,
// verifying simd([*const T; N]) and simd([*mut T; N]) pass typeck and work.
#![feature(repr_simd, intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct cptrx4<T>([*const T; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct mptrx4<T>([*mut T; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct f32x4([f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct i32x4([i32; 4]);

extern "rust-intrinsic" {
    fn simd_gather<T, U, V>(x: T, y: U, z: V) -> T;
    fn simd_scatter<T, U, V>(x: T, y: U, z: V) -> ();
}

fn main() {
    let mut x = [0_f32, 1., 2., 3., 4., 5., 6., 7.];

    let default = f32x4([-3_f32, -3., -3., -3.]);
    let s_strided = f32x4([0_f32, 2., -3., 6.]);
    let mask = i32x4([-1_i32, -1, 0, -1]);

    // reading from *const
    unsafe {
        let pointer = &x as *const f32;
        let pointers =  cptrx4([
            pointer.offset(0) as *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        ]);

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = &mut x as *mut f32;
        let pointers = mptrx4([
            pointer.offset(0) as *mut f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        ]);

        let values = f32x4([42_f32, 43_f32, 44_f32, 45_f32]);
        simd_scatter(values, pointers, mask);

        assert_eq!(x, [42., 1., 43., 3., 4., 5., 45., 7.]);
    }
}
