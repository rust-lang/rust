// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains `HashStable` implementations for various data types
//! that don't fit into any of the other impls_xxx modules.

impl_stable_hash_for!(enum ::session::search_paths::PathKind {
    Native,
    Crate,
    Dependency,
    Framework,
    ExternFlag,
    All
});

impl_stable_hash_for!(enum ::rustc_back::PanicStrategy {
    Abort,
    Unwind
});
