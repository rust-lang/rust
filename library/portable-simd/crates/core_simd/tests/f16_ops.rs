#![feature(portable_simd)]
#![feature(f16)]

#[macro_use]
mod ops_macros;

// FIXME: some f16 operations cause rustc to hang on wasm simd
// https://github.com/llvm/llvm-project/issues/189251
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
impl_float_tests! { f16, i16 }
