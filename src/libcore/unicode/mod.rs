// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "unicode_internals", issue = "0")]
#![allow(missing_docs)]

mod bool_trie;
pub(crate) mod printable;
pub(crate) mod tables;
pub(crate) mod version;

// For use in liballoc, not re-exported in libstd.
pub mod derived_property {
    pub use unicode::tables::derived_property::{Case_Ignorable, Cased};
}

// For use in libsyntax
pub mod property {
    pub use unicode::tables::property::Pattern_White_Space;
}
