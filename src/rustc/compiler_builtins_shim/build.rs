// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This file is left intentionally empty (and not removed) to avoid an issue
// where this crate is always considered dirty due to compiler-builtins'
// `cargo:rerun-if-changed=build.rs` directive; since the path is relative, it
// refers to this file when this shim crate is being built, and the absence of
// this file is considered by cargo to be equivalent to it having changed.
