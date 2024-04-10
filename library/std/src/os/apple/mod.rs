#![cfg_attr(target_os = "ios", doc = "iOS-specific definitions")]
#![cfg_attr(target_os = "macos", doc = "macOS-specific definitions")]
#![cfg_attr(target_os = "tvos", doc = "tvOS-specific definitions")]
#![cfg_attr(target_os = "visionos", doc = "visionOS-specific definitions")]
#![cfg_attr(target_os = "watchos", doc = "watchOS-specific definitions")]
#![stable(feature = "raw_ext", since = "1.1.0")]

pub mod fs;
pub mod raw;
