// run-pass

// Test that simd gather instructions on slice of usize don't cause crash
// See issue #89183 - https://github.com/rust-lang/rust/issues/89193

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct x4<T>(pub T, pub T, pub T, pub T);

extern "platform-intrinsic" {
    fn simd_gather<T, U, V>(x: T, y: U, z: V) -> T;
}

fn main() {
    let x: [usize; 4] = [10, 11, 12, 13];
    let default = x4(0_usize, 1, 2, 3);
    let all_set = u8::MAX as i8; // aka -1
    let mask = x4(all_set, all_set, all_set, all_set);
    let expected = x4(10_usize, 11, 12, 13);

    unsafe {
        let pointer = x.as_ptr();
        let pointers =  x4(
            pointer.offset(0),
            pointer.offset(1),
            pointer.offset(2),
            pointer.offset(3)
        );
        let result = simd_gather(default, pointers, mask);
        assert_eq!(result, expected);
    }

    // and again for isize
    let x: [isize; 4] = [10, 11, 12, 13];
    let default = x4(0_isize, 1, 2, 3);
    let expected = x4(10_isize, 11, 12, 13);

    unsafe {
        let pointer = x.as_ptr();
        let pointers =  x4(
            pointer.offset(0),
            pointer.offset(1),
            pointer.offset(2),
            pointer.offset(3)
        );
        let result = simd_gather(default, pointers, mask);
        assert_eq!(result, expected);
    }
}
