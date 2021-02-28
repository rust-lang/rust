mod inner;
#[cfg(any(doc, target_os = "android", target_os = "emscripten", target_os = "linux"))]
mod ip;
mod unix;

pub use inner::Messages;
#[cfg(any(doc, target_os = "android", target_os = "emscripten", target_os = "linux"))]
pub use ip::{IpAncillary, IpAncillaryData};
pub use unix::{SocketCred, UnixAncillary, UnixAncillaryData};
