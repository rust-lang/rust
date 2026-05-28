//@ build-pass

#[cfg(target_arch = "wasm32")]
fn main() {
    unsafe {
        a::api_with_simd_feature();
    }
}

#[cfg(target_arch = "wasm32")]
mod a {
    use std::arch::wasm32::*;

    #[target_feature(enable = "simd128")]
    pub unsafe fn api_with_simd_feature() {
        crate::b::api_takes_v128(u64x2(0, 1));
    }
}

#[cfg(target_arch = "wasm32")]
mod b {
    use std::arch::wasm32::*;

    #[inline(never)]
    pub fn api_takes_v128(a: v128) -> v128 {
        a
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {}
