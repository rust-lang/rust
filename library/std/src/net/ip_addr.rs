// Tests for this module
#[cfg(all(test, not(any(target_os = "emscripten", all(target_os = "wasi", target_env = "p1")))))]
mod tests;

#[stable(feature = "ip_addr", since = "1.7.0")]
pub use core::net::IpAddr;
#[unstable(feature = "ip", issue = "27709")]
pub use core::net::Ipv6MulticastScope;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::net::{Ipv4Addr, Ipv6Addr};
