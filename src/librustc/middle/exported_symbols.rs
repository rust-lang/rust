// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The SymbolExportLevel of a symbols specifies from which kinds of crates
/// the symbol will be exported. `C` symbols will be exported from any
/// kind of crate, including cdylibs which export very few things.
/// `Rust` will only be exported if the crate produced is a Rust
/// dylib.
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SymbolExportLevel {
    C,
    Rust,
}

impl_stable_hash_for!(enum self::SymbolExportLevel {
    C,
    Rust
});

impl SymbolExportLevel {
    pub fn is_below_threshold(self, threshold: SymbolExportLevel) -> bool {
        if threshold == SymbolExportLevel::Rust {
            // We export everything from Rust dylibs
            true
        } else {
            self == SymbolExportLevel::C
        }
    }
}
