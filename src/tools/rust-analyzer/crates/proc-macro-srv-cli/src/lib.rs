//! Library interface for `proc-macro-srv-cli`.
//!
//! This module exposes the server main loop and protocol format for integration testing.

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

#[cfg(feature = "sysroot-abi")]
pub mod main_loop;
