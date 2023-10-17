//! This module contains integral components of the build and configuration process, providing
//! support for a wide range of tasks and operations such as caching, tarballs, release
//! channels, job management, etc.

pub(crate) mod cache;
pub(crate) mod cc_detect;
pub(crate) mod channel;
pub(crate) mod dylib;
pub(crate) mod helpers;
pub(crate) mod job;
#[cfg(feature = "build-metrics")]
pub(crate) mod metrics;
pub(crate) mod render_tests;
pub(crate) mod tarball;
