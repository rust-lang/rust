//! WASI-specific networking functionality

#![unstable(feature = "wasi_ext", issue = "71213")]

mod addr;
mod listener;
mod stream;

#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::addr::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::listener::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::stream::*;
