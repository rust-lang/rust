//! Linux and Android-specific definitions for socket options.

#![unstable(feature = "tcp_quickack", issue = "96256")]
#![doc(cfg(any(target_os = "linux", target_os = "android",)))]
pub mod tcp;
#[cfg(test)]
mod tests;
