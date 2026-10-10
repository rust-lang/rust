//! Library interface for `proc-macro-srv-cli`.
//!
//! This module exposes the server main loop and protocol format for integration testing.

#![cfg(feature = "in-rust-tree")]
#![feature(rustc_private)]

extern crate rustc_driver as _;

pub mod main_loop;
