// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty;

use rustc_data_structures::indexed_vec::Idx;
use serialize::{self, Encoder, Decoder};

use std::fmt;
use std::u32;

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq, Hash, Debug)]
pub struct CrateNum(u32);

impl Idx for CrateNum {
    fn new(value: usize) -> Self {
        assert!(value < (u32::MAX) as usize);
        CrateNum(value as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Item definitions in the currently-compiled crate would have the CrateNum
/// LOCAL_CRATE in their DefId.
pub const LOCAL_CRATE: CrateNum = CrateNum(0);

/// Virtual crate for builtin macros
// FIXME(jseyfried): this is also used for custom derives until proc-macro crates get `CrateNum`s.
pub const BUILTIN_MACROS_CRATE: CrateNum = CrateNum(!0);

impl CrateNum {
    pub fn new(x: usize) -> CrateNum {
        assert!(x < (u32::MAX as usize));
        CrateNum(x as u32)
    }

    pub fn from_u32(x: u32) -> CrateNum {
        CrateNum(x)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for CrateNum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl serialize::UseSpecializedEncodable for CrateNum {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(self.0)
    }
}

impl serialize::UseSpecializedDecodable for CrateNum {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<CrateNum, D::Error> {
        d.read_u32().map(CrateNum)
    }
}

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
#[derive(Clone, Eq, Ord, PartialOrd, PartialEq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub index: DefIndex,
}

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DefId {{ krate: {:?}, node: {:?}",
               self.krate, self.index)?;

        ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                write!(f, " => {}", tcx.def_path(*self).to_string(tcx))?;
            }
            Ok(())
        })?;

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
