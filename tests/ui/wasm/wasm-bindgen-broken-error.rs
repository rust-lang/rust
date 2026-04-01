//@ only-wasm32
//@ revisions: v0_1_0 v0_2_87 v0_2_88 v0_3_0 v1_0_0
//@[v0_1_0] check-fail
//@[v0_1_0] rustc-env:CARGO_PKG_VERSION_MAJOR=0
//@[v0_1_0] rustc-env:CARGO_PKG_VERSION_MINOR=1
//@[v0_1_0] rustc-env:CARGO_PKG_VERSION_PATCH=0
//@[v0_2_87] check-fail
//@[v0_2_87] rustc-env:CARGO_PKG_VERSION_MAJOR=0
//@[v0_2_87] rustc-env:CARGO_PKG_VERSION_MINOR=2
//@[v0_2_87] rustc-env:CARGO_PKG_VERSION_PATCH=87
//@[v0_2_88] check-pass
//@[v0_2_88] rustc-env:CARGO_PKG_VERSION_MAJOR=0
//@[v0_2_88] rustc-env:CARGO_PKG_VERSION_MINOR=2
//@[v0_2_88] rustc-env:CARGO_PKG_VERSION_PATCH=88
//@[v0_3_0] check-pass
//@[v0_3_0] rustc-env:CARGO_PKG_VERSION_MAJOR=0
//@[v0_3_0] rustc-env:CARGO_PKG_VERSION_MINOR=3
//@[v0_3_0] rustc-env:CARGO_PKG_VERSION_PATCH=0
//@[v1_0_0] check-pass
//@[v1_0_0] rustc-env:CARGO_PKG_VERSION_MAJOR=1
//@[v1_0_0] rustc-env:CARGO_PKG_VERSION_MINOR=0
//@[v1_0_0] rustc-env:CARGO_PKG_VERSION_PATCH=0

#![crate_name = "wasm_bindgen"]
//[v0_1_0]~^ ERROR: older versions of the `wasm-bindgen` crate
//[v0_2_87]~^^ ERROR: older versions of the `wasm-bindgen` crate

fn main() {}
