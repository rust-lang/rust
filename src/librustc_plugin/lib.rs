// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Infrastructure for compiler plugins.
//!
//! Plugins are Rust libraries which extend the behavior of `rustc`
//! in various ways.
//!
//! Plugin authors will use the `Registry` type re-exported by
//! this module, along with its methods.  The rest of the module
//! is for use by `rustc` itself.
//!
//! To define a plugin, build a dylib crate with a
//! `#[plugin_registrar]` function:
//!
//! ```rust,ignore
//! #![crate_name = "myplugin"]
//! #![crate_type = "dylib"]
//! #![feature(plugin_registrar)]
//!
//! extern crate rustc;
//!
//! use rustc::plugin::Registry;
//!
//! #[plugin_registrar]
//! pub fn plugin_registrar(reg: &mut Registry) {
//!     reg.register_macro("mymacro", expand_mymacro);
//! }
//!
//! fn expand_mymacro(...) {  // details elided
//! ```
//!
//! WARNING: We currently don't check that the registrar function
//! has the appropriate type!
//!
//! To use a plugin while compiling another crate:
//!
//! ```rust
//! #![feature(plugin)]
//! #![plugin(myplugin)]
//! ```
//!
//! See the [Plugins Chapter](../../book/compiler-plugins.html) of the book
//! for more examples.

#![crate_name = "rustc_plugin"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(dynamic_lib)]
#![feature(staged_api)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
#[macro_use] #[no_link] extern crate rustc_bitflags;

extern crate rustc;
extern crate rustc_front;
extern crate rustc_metadata;

pub use self::registry::Registry;

pub mod diagnostics;
pub mod registry;
pub mod load;
pub mod build;
