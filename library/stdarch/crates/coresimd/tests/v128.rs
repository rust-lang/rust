//! coresimd 128-bit wide vector tests

#![cfg_attr(feature = "strict", deny(warnings))]
#![feature(stdsimd, link_llvm_intrinsics, simd_ffi, core_float,
           cfg_target_feature)]
#![allow(unused_imports, dead_code)]

#[cfg(test)]
extern crate coresimd;

#[cfg(test)]
macro_rules! test_v16 {
    ($item: item) => {};
}
#[cfg(test)]
macro_rules! test_v32 {
    ($item: item) => {};
}
#[cfg(test)]
macro_rules! test_v64 {
    ($item: item) => {};
}
#[cfg(test)]
macro_rules! test_v128 {
    ($item: item) => {
        $item
    };
}
#[cfg(test)]
macro_rules! test_v256 {
    ($item: item) => {};
}
#[cfg(test)]
macro_rules! test_v512 {
    ($item: item) => {};
}

#[cfg(test)]
macro_rules! vector_impl {
    ($([$f: ident, $($args: tt)*]),*) => {};
}

#[cfg(test)]
#[path = "../../../coresimd/ppsv/mod.rs"]
mod ppsv;

#[cfg(test)]
use std::{marker, mem};

#[cfg(all(test, target_arch = "aarch64"))]
use std::cmp;

#[cfg(all(test, target_arch = "aarch64"))]
extern crate core as _core;

#[cfg(all(test, target_arch = "aarch64"))]
use _core::num;
