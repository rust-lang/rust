//@ ignore-aarch64
//@ ignore-riscv32
//@ ignore-riscv64

// Confirm that non-AArch64 and non-RISC-V targets error when compiling scalable vectors
// (see #153593)

#![crate_type = "lib"]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(4)]
//~^ ERROR: scalable vectors are not supported on this architecture
struct ScalableVec(i32);
