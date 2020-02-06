//! A version of `std::time::Instant` that doesn't panic in WASM.

#[cfg(not(feature = "wasm"))]
pub use std::time::Instant;

#[cfg(feature = "wasm")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(feature = "wasm")]
impl Instant {
    pub fn now() -> Self {
        Self
    }

    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::new(0, 0)
    }
}
