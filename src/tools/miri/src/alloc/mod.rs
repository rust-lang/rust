mod alloc_bytes;
#[cfg(target_os = "linux")]
pub mod isolated_alloc;

pub use self::alloc_bytes::{MiriAllocBytes, MiriAllocParams};
