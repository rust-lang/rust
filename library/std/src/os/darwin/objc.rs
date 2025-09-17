//! Defines types and macros for Objective-C interoperability.
//!
//! This module re-exports all the items in [`core::os::darwin::objc`].
//!
//! [`core::os::darwin::objc`]: ../../../../core/os/darwin/objc/index.html "mod core::os::darwin::objc"

#![unstable(feature = "darwin_objc", issue = "145496")]

// We can't generate an intra-doc link for this automatically since `core::os::darwin` isn't
// compiled into `core` on every platform even though it's documented on every platform.
// We just link to it directly in the module documentation above instead.
#[cfg(not(doc))]
pub use core::os::darwin::objc::*;
