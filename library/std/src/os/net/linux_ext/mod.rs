//! Linux and Android-specific networking functionality.

#![doc(cfg(any(target_os = "linux", target_os = "android")))]

#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
pub(crate) mod addr;

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub(crate) mod socket;

#[stable(feature = "tcp_quickack", since = "CURRENT_RUSTC_VERSION")]
pub(crate) mod tcp;

#[cfg(test)]
mod tests;
