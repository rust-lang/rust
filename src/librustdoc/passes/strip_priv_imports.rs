// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clean;
use core::DocContext;
use fold::DocFolder;
use passes::{ImportStripper, Pass};

pub const STRIP_PRIV_IMPORTS: Pass = Pass::early("strip-priv-imports", strip_priv_imports,
     "strips all private import statements (`use`, `extern crate`) from a crate");

pub fn strip_priv_imports(krate: clean::Crate, _: &DocContext)  -> clean::Crate {
    ImportStripper.fold_crate(krate)
}
