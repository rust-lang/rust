// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Calculation and management of a Strict Version Hash for crates
//!
//! # Today's ABI problem
//!
//! In today's implementation of rustc, it is incredibly difficult to achieve
//! forward binary compatibility without resorting to C-like interfaces. Within
//! rust code itself, abi details such as symbol names suffer from a variety of
//! unrelated factors to code changing such as the "def id drift" problem. This
//! ends up yielding confusing error messages about metadata mismatches and
//! such.
//!
//! The core of this problem is when an upstream dependency changes and
//! downstream dependents are not recompiled. This causes compile errors because
//! the upstream crate's metadata has changed but the downstream crates are
//! still referencing the older crate's metadata.
//!
//! This problem exists for many reasons, the primary of which is that rust does
//! not currently support forwards ABI compatibility (in place upgrades of a
//! crate).
//!
//! # SVH and how it alleviates the problem
//!
//! With all of this knowledge on hand, this module contains the implementation
//! of a notion of a "Strict Version Hash" for a crate. This is essentially a
//! hash of all contents of a crate which can somehow be exposed to downstream
//! crates.
//!
//! This hash is currently calculated by just hashing the AST, but this is
//! obviously wrong (doc changes should not result in an incompatible ABI).
//! Implementation-wise, this is required at this moment in time.
//!
//! By encoding this strict version hash into all crate's metadata, stale crates
//! can be detected immediately and error'd about by rustc itself.
//!
//! # Relevant links
//!
//! Original issue: https://github.com/rust-lang/rust/issues/10207

use std::fmt;

#[derive(Clone, Eq, Hash, PartialEq, Debug)]
pub struct Svh {
    hash: String,
}

impl Svh {
    /// Create a new `Svh` given the hash. If you actually want to
    /// compute the SVH from some HIR, you want the `calculate_svh`
    /// function found in `librustc_trans`.
    pub fn new(hash: String) -> Svh {
        assert!(hash.len() == 16);
        Svh { hash: hash }
    }

    pub fn from_hash(hash: u64) -> Svh {
        return Svh::new((0..64).step_by(4).map(|i| hex(hash >> i)).collect());

        fn hex(b: u64) -> char {
            let b = (b & 0xf) as u8;
            let b = match b {
                0 ... 9 => '0' as u8 + b,
                _ => 'a' as u8 + b - 10,
            };
            b as char
        }
    }

    pub fn as_str<'a>(&'a self) -> &'a str {
        &self.hash
    }
}

impl fmt::Display for Svh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(self.as_str())
    }
}
