//! Library interface for `proc-macro-srv-cli`.
//!
//! This module exposes the server main loop and protocol format for integration testing.

#[cfg(feature = "sysroot-abi")]
pub mod main_loop;
