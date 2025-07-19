//@ run-pass

#[cfg(target_arch = "x86")]
pub fn main() { }

#[cfg(target_arch = "x86_64")]
pub fn main() { }

#[cfg(target_arch = "arm")]
pub fn main() { }

#[cfg(target_arch = "aarch64")]
pub fn main() { }

#[cfg(target_arch = "mips")]
pub fn main() { }

#[cfg(target_arch = "mips64")]
pub fn main() { }

#[cfg(target_arch = "powerpc")]
pub fn main() { }

#[cfg(target_arch = "powerpc64")]
pub fn main() { }

#[cfg(target_arch = "s390x")]
pub fn main() { }

#[cfg(target_arch = "wasm32")]
pub fn main() { }

#[cfg(target_arch = "sparc64")]
pub fn main() { }

#[cfg(target_arch = "riscv64")]
pub fn main() { }

#[cfg(target_arch = "loongarch64")]
pub fn main() { }

#[cfg(target_arch = "arm64ec")]
pub fn main() { }
