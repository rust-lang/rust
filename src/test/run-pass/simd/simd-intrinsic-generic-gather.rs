// run-pass
// ignore-emscripten

// Test that the simd_{gather,scatter} intrinsics produce the correct results.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct x4<T>(pub T, pub T, pub T, pub T);

extern "platform-intrinsic" {
    fn simd_gather<T, U, V>(x: T, y: U, z: V) -> T;
    fn simd_scatter<T, U, V>(x: T, y: U, z: V) -> ();
}

fn main() {
    let mut x = [0_f32, 1., 2., 3., 4., 5., 6., 7.];

    let default = x4(-3_f32, -3., -3., -3.);
    let s_strided = x4(0_f32, 2., -3., 6.);
    let mask = x4(-1_i32, -1, 0, -1);

    // reading from *const
    unsafe {
        let pointer = &x[0] as *const f32;
        let pointers =  x4(
            pointer.offset(0) as *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // reading from *mut
    unsafe {
        let pointer = &mut x[0] as *mut f32;
        let pointers = x4(
            pointer.offset(0) as *mut f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = &mut x[0] as *mut f32;
        let pointers = x4(
            pointer.offset(0) as *mut f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let values = x4(42_f32, 43_f32, 44_f32, 45_f32);
        simd_scatter(values, pointers, mask);

        assert_eq!(x, [42., 1., 43., 3., 4., 5., 45., 7.]);
    }

    // test modifying array of *const f32
    let mut y = [
        &x[0] as *const f32,
        &x[1] as *const f32,
        &x[2] as *const f32,
        &x[3] as *const f32,
        &x[4] as *const f32,
        &x[5] as *const f32,
        &x[6] as *const f32,
        &x[7] as *const f32
    ];

    let default = x4(y[0], y[0], y[0], y[0]);
    let s_strided = x4(y[0], y[2], y[0], y[6]);

    // reading from *const
    unsafe {
        let pointer = &y[0] as *const *const f32;
        let pointers = x4(
            pointer.offset(0) as *const *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // reading from *mut
    unsafe {
        let pointer = &mut y[0] as *mut *const f32;
        let pointers = x4(
            pointer.offset(0) as *mut *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let r_strided = simd_gather(default, pointers, mask);

        assert_eq!(r_strided, s_strided);
    }

    // writing to *mut
    unsafe {
        let pointer = &mut y[0] as *mut *const f32;
        let pointers = x4(
            pointer.offset(0) as *mut *const f32,
            pointer.offset(2),
            pointer.offset(4),
            pointer.offset(6)
        );

        let values = x4(y[7], y[6], y[5], y[1]);
        simd_scatter(values, pointers, mask);

        let s = [
            &x[7] as *const f32,
            &x[1] as *const f32,
            &x[6] as *const f32,
            &x[3] as *const f32,
            &x[4] as *const f32,
            &x[5] as *const f32,
            &x[1] as *const f32,
            &x[7] as *const f32
        ];
        assert_eq!(y, s);
    }
}
