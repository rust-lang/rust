// run-pass
// ignore-emscripten FIXME(#45351) hits an LLVM assert

#![feature(repr_simd, platform_intrinsics)]
#![feature(inline_const)]

#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[allow(non_camel_case_types)]
struct i32x2(i32, i32);
#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[allow(non_camel_case_types)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[allow(non_camel_case_types)]
struct i32x8(i32, i32, i32, i32,
             i32, i32, i32, i32);

extern "platform-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;

    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
    fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
    fn simd_shuffle8<T, U>(x: T, y: T, idx: [u32; 8]) -> U;
}

macro_rules! all_eq {
    ($a: expr, $b: expr) => {{
        let a = $a;
        let b = $b;
        // type inference works better with the concrete type on the
        // left, but humans work better with the expected on the
        // right.
        assert!(b == a,
                "{:?} != {:?}", a, b);
    }}
}

fn main() {
    let x2 = i32x2(20, 21);
    let x4 = i32x4(40, 41, 42, 43);
    let x8 = i32x8(80, 81, 82, 83, 84, 85, 86, 87);
    unsafe {
        all_eq!(simd_insert(x2, 0, 100), i32x2(100, 21));
        all_eq!(simd_insert(x2, 1, 100), i32x2(20, 100));

        all_eq!(simd_insert(x4, 0, 100), i32x4(100, 41, 42, 43));
        all_eq!(simd_insert(x4, 1, 100), i32x4(40, 100, 42, 43));
        all_eq!(simd_insert(x4, 2, 100), i32x4(40, 41, 100, 43));
        all_eq!(simd_insert(x4, 3, 100), i32x4(40, 41, 42, 100));

        all_eq!(simd_insert(x8, 0, 100), i32x8(100, 81, 82, 83, 84, 85, 86, 87));
        all_eq!(simd_insert(x8, 1, 100), i32x8(80, 100, 82, 83, 84, 85, 86, 87));
        all_eq!(simd_insert(x8, 2, 100), i32x8(80, 81, 100, 83, 84, 85, 86, 87));
        all_eq!(simd_insert(x8, 3, 100), i32x8(80, 81, 82, 100, 84, 85, 86, 87));
        all_eq!(simd_insert(x8, 4, 100), i32x8(80, 81, 82, 83, 100, 85, 86, 87));
        all_eq!(simd_insert(x8, 5, 100), i32x8(80, 81, 82, 83, 84, 100, 86, 87));
        all_eq!(simd_insert(x8, 6, 100), i32x8(80, 81, 82, 83, 84, 85, 100, 87));
        all_eq!(simd_insert(x8, 7, 100), i32x8(80, 81, 82, 83, 84, 85, 86, 100));

        all_eq!(simd_extract(x2, 0), 20);
        all_eq!(simd_extract(x2, 1), 21);

        all_eq!(simd_extract(x4, 0), 40);
        all_eq!(simd_extract(x4, 1), 41);
        all_eq!(simd_extract(x4, 2), 42);
        all_eq!(simd_extract(x4, 3), 43);

        all_eq!(simd_extract(x8, 0), 80);
        all_eq!(simd_extract(x8, 1), 81);
        all_eq!(simd_extract(x8, 2), 82);
        all_eq!(simd_extract(x8, 3), 83);
        all_eq!(simd_extract(x8, 4), 84);
        all_eq!(simd_extract(x8, 5), 85);
        all_eq!(simd_extract(x8, 6), 86);
        all_eq!(simd_extract(x8, 7), 87);
    }

    let y2 = i32x2(120, 121);
    let y4 = i32x4(140, 141, 142, 143);
    let y8 = i32x8(180, 181, 182, 183, 184, 185, 186, 187);
    unsafe {
        all_eq!(simd_shuffle2(x2, y2, const { [3u32, 0] }), i32x2(121, 20));
        all_eq!(simd_shuffle4(x2, y2, const { [3u32, 0, 1, 2] }), i32x4(121, 20, 21, 120));
        all_eq!(simd_shuffle8(x2, y2, const { [3u32, 0, 1, 2, 1, 2, 3, 0] }),
                i32x8(121, 20, 21, 120, 21, 120, 121, 20));

        all_eq!(simd_shuffle2(x4, y4, const { [7u32, 2] }), i32x2(143, 42));
        all_eq!(simd_shuffle4(x4, y4, const { [7u32, 2, 5, 0] }), i32x4(143, 42, 141, 40));
        all_eq!(simd_shuffle8(x4, y4, const { [7u32, 2, 5, 0, 3, 6, 4, 1] }),
                i32x8(143, 42, 141, 40, 43, 142, 140, 41));

        all_eq!(simd_shuffle2(x8, y8, const { [11u32, 5] }), i32x2(183, 85));
        all_eq!(simd_shuffle4(x8, y8, const { [11u32, 5, 15, 0] }), i32x4(183, 85, 187, 80));
        all_eq!(simd_shuffle8(x8, y8, const { [11u32, 5, 15, 0, 3, 8, 12, 1] }),
                i32x8(183, 85, 187, 80, 83, 180, 184, 81));
    }

}
