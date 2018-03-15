// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that crate-level `#![macro_escape]` is not gated beyond a
// depecation warning.  This file sits on its own, because crate-level
// `#![macro_escape]` is incompatible with crate-level `#![macro_use]`
// already present in issue-43106-gating-of-builtin-attrs.

#![macro_escape]
//~^ WARN macro_escape is a deprecated synonym for macro_use
