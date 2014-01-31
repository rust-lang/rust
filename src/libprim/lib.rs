// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id = "prim#0.10-pre"];
#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];

#[no_std];
#[feature(globs)];
#[feature(phase)];

#[cfg(test)] #[phase(syntax)] extern mod std;

#[cfg(test)] extern mod realprim = "prim";
#[cfg(test)] extern mod std;
#[cfg(test)] extern mod rustuv;
#[cfg(test)] extern mod green;
#[cfg(test)] extern mod native;

#[cfg(test)] pub use kinds = realprim::kinds;

pub mod cast;
pub mod intrinsics;
#[cfg(not(test))] pub mod kinds;
pub mod mem;