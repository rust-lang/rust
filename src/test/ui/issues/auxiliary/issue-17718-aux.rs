use std::sync::atomic;

pub const C1: usize = 1;
pub const C2: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
pub const C3: fn() = { fn foo() {} foo };
pub const C4: usize = C1 * C1 + C1 / C1;
pub const C5: &'static usize = &C4;

pub static S1: usize = 3;
pub static S2: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
