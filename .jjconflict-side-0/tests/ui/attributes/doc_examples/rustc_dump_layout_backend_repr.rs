//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]
#![feature(portable_simd)]

#[rustc_dump_layout(backend_repr)]
type Simd = std::simd::u32x4;
