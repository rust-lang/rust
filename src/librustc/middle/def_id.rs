// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use metadata::cstore::LOCAL_CRATE;
use middle::ty;
use syntax::ast::CrateNum;
use std::fmt;
use std::u32;

/// A DefIndex is an index into the hir-map for a crate, identifying a
/// particular definition. It should really be considered an interned
/// shorthand for a particular DefPath.
#[derive(Clone, Debug, Eq, Ord, PartialOrd, PartialEq, RustcEncodable,
           RustcDecodable, Hash, Copy)]
pub struct DefIndex(u32);

impl DefIndex {
    pub fn new(x: usize) -> DefIndex {
        assert!(x < (u32::MAX as usize));
        DefIndex(x as u32)
    }

    pub fn from_u32(x: u32) -> DefIndex {
        DefIndex(x)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// The crate root is always assigned index 0 by the AST Map code,
/// thanks to `NodeCollector::new`.
pub const CRATE_DEF_INDEX: DefIndex = DefIndex(0);

/// A DefId identifies a particular *definition*, by combining a crate
/// index and a def index.
#[derive(Clone, Eq, Ord, PartialOrd, PartialEq, RustcEncodable,
           RustcDecodable, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub index: DefIndex,
}

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "DefId {{ krate: {:?}, node: {:?}",
                    self.krate, self.index));

        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        if self.is_local() { // (1)
            // (1) side-step fact that not all external things have paths at
            // the moment, such as type parameters
            try!(ty::tls::with_opt(|opt_tcx| {
                if let Some(tcx) = opt_tcx {
                    try!(write!(f, " => {}", tcx.item_path_str(*self)));
                }
                Ok(())
            }));
        }

        write!(f, " }}")
    }
}


impl DefId {
    pub fn local(index: DefIndex) -> DefId {
        DefId { krate: LOCAL_CRATE, index: index }
    }

    pub fn is_local(&self) -> bool {
        self.krate == LOCAL_CRATE
    }
}
