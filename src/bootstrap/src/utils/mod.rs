//! This module contains integral components of the build and configuration process, providing
//! support for a wide range of tasks and operations such as caching, tarballs, release
//! channels, job management, etc.

pub(crate) mod build_stamp;
pub(crate) mod cache;
pub(crate) mod cc_detect;
pub(crate) mod change_tracker;
pub(crate) mod channel;
pub(crate) mod exec;
pub(crate) mod execution_context;
pub(crate) mod helpers;
pub(crate) mod job;
pub(crate) mod render_tests;
pub(crate) mod shared_helpers;
pub(crate) mod tarball;

pub(crate) mod tracing;

#[cfg(feature = "build-metrics")]
pub(crate) mod metrics;

#[cfg(test)]
pub(crate) mod tests;
