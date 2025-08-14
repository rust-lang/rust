//! Linux and Android-specific networking functionality.

#![doc(cfg(any(target_os = "linux", target_os = "android")))]

#[stable(feature = "unix_socket_abstract", since = "1.70.0")]
pub(crate) mod addr;

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub(crate) mod socket;

#[unstable(feature = "tcp_quickack", issue = "96256")]
pub(crate) mod tcp;

#[cfg(test)]
mod tests;
