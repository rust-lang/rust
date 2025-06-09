//! RISC-V RV32 specific intrinsics

mod zk;

#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub use zk::*;
