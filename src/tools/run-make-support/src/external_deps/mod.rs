//! This module contains external tool dependencies that we assume are available in the environment,
//! such as `cc` or `python`.
//!
//! # Notes
//!
//! - This is not the *only* place where external dependencies are assumed or referenced. For
//!   example, see [`cygpath_windows`][crate::path_helpers::cygpath_windows].

pub mod cc;
pub mod clang;
pub mod htmldocck;
pub mod llvm;
pub mod python;
pub mod rustc;
pub mod rustdoc;
