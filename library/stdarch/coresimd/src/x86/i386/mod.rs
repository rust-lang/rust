//! `i386` intrinsics

mod eflags;
pub use self::eflags::*;

#[cfg(dont_compile_me)] // TODO: need to upstream `fxsr` target feature
mod fxsr;
#[cfg(dont_compile_me)] // TODO: need to upstream `fxsr` target feature
pub use self::fxsr::*;
