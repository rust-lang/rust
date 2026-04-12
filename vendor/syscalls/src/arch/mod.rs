#[macro_use]
mod macros;

#[cfg(any(target_arch = "aarch64", feature = "aarch64"))]
pub mod aarch64;
#[cfg(any(target_arch = "arm", feature = "arm"))]
pub mod arm;
#[cfg(any(target_arch = "mips", feature = "mips"))]
pub mod mips;
#[cfg(any(target_arch = "mips64", feature = "mips64"))]
pub mod mips64;
#[cfg(any(target_arch = "powerpc", feature = "powerpc"))]
pub mod powerpc;
#[cfg(any(target_arch = "powerpc64", feature = "powerpc64"))]
pub mod powerpc64;
#[cfg(any(target_arch = "riscv32", feature = "riscv32"))]
pub mod riscv32;
#[cfg(any(target_arch = "riscv64", feature = "riscv64"))]
pub mod riscv64;
#[cfg(any(target_arch = "s390x", feature = "s390x"))]
pub mod s390x;
#[cfg(any(target_arch = "sparc", feature = "sparc"))]
pub mod sparc;
#[cfg(any(target_arch = "sparc64", feature = "sparc64"))]
pub mod sparc64;
#[cfg(any(target_arch = "x86", feature = "x86"))]
pub mod x86;
#[cfg(any(target_arch = "x86_64", feature = "x86_64"))]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(target_arch = "arm")]
pub use arm::*;

#[cfg(target_arch = "mips")]
pub use mips::*;

#[cfg(target_arch = "mips64")]
pub use mips64::*;

#[cfg(target_arch = "powerpc")]
pub use powerpc::*;

#[cfg(target_arch = "powerpc64")]
pub use powerpc64::*;

#[cfg(target_arch = "riscv32")]
pub use riscv32::*;

#[cfg(target_arch = "riscv64")]
pub use riscv64::*;

#[cfg(target_arch = "s390x")]
pub use s390x::*;

#[cfg(target_arch = "sparc")]
pub use sparc::*;

#[cfg(target_arch = "sparc64")]
pub use sparc64::*;

#[cfg(target_arch = "x86")]
pub use x86::*;

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;
