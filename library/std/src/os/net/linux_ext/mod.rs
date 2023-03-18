//! Linux and Android-specific networking functionality.

#![doc(cfg(any(target_os = "linux", target_os = "android")))]

#[stable(feature = "unix_socket_abstract", since = "CURRENT_RUSTC_VERSION")]
pub(crate) mod addr;

#[unstable(feature = "tcp_quickack", issue = "96256")]
pub(crate) mod tcp;

#[cfg(test)]
mod tests;
