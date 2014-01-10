// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The diagnostic registry
//!
//! All diagnostic codes must be registered here. To add a new
//! diagnostic code just go to the end of the file and add a new
//! line with a code that is one greater than the previous.

#[cfg(not(stage0))];

reg_diag!(A0000)
