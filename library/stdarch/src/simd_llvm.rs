extern "platform-intrinsic" {
    pub fn simd_eq<T, U>(x: T, y: T) -> U;
    pub fn simd_ne<T, U>(x: T, y: T) -> U;
    pub fn simd_lt<T, U>(x: T, y: T) -> U;
    pub fn simd_le<T, U>(x: T, y: T) -> U;
    pub fn simd_gt<T, U>(x: T, y: T) -> U;
    pub fn simd_ge<T, U>(x: T, y: T) -> U;

    pub fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
    pub fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
    pub fn simd_shuffle8<T, U>(x: T, y: T, idx: [u32; 8]) -> U;
    pub fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
    pub fn simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U;

    pub fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
    pub fn simd_extract<T, U>(x: T, idx: u32) -> U;

    pub fn simd_cast<T, U>(x: T) -> U;

    pub fn simd_add<T>(x: T, y: T) -> T;
    pub fn simd_sub<T>(x: T, y: T) -> T;
    pub fn simd_mul<T>(x: T, y: T) -> T;
    pub fn simd_div<T>(x: T, y: T) -> T;
    pub fn simd_shl<T>(x: T, y: T) -> T;
    pub fn simd_shr<T>(x: T, y: T) -> T;
    pub fn simd_and<T>(x: T, y: T) -> T;
    pub fn simd_or<T>(x: T, y: T) -> T;
    pub fn simd_xor<T>(x: T, y: T) -> T;
}
