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
//! Original issue: https://github.com/mozilla/rust/issues/10207

use std::fmt;
use std::hash::Hash;
use std::hash::sip::SipState;
use std::iter::range_step;
use syntax::ast;

#[deriving(Clone, Eq)]
pub struct Svh {
    hash: StrBuf,
}

impl Svh {
    pub fn new(hash: &str) -> Svh {
        assert!(hash.len() == 16);
        Svh { hash: hash.to_strbuf() }
    }

    pub fn as_str<'a>(&'a self) -> &'a str {
        self.hash.as_slice()
    }

    pub fn calculate(krate: &ast::Crate) -> Svh {
        // FIXME: see above for why this is wrong, it shouldn't just hash the
        //        crate.  Fixing this would require more in-depth analysis in
        //        this function about what portions of the crate are reachable
        //        in tandem with bug fixes throughout the rest of the compiler.
        //
        //        Note that for now we actually exclude some top-level things
        //        from the crate like the CrateConfig/span. The CrateConfig
        //        contains command-line `--cfg` flags, so this means that the
        //        stage1/stage2 AST for libstd and such is different hash-wise
        //        when it's actually the exact same representation-wise.
        //
        //        As a first stab at only hashing the relevant parts of the
        //        AST, this only hashes the module/attrs, not the CrateConfig
        //        field.
        //
        // FIXME: this should use SHA1, not SipHash. SipHash is not built to
        //        avoid collisions.
        let mut state = SipState::new();
        krate.module.hash(&mut state);
        krate.attrs.hash(&mut state);

        let hash = state.result();
        return Svh {
            hash: range_step(0, 64, 4).map(|i| hex(hash >> i)).collect()
        };

        fn hex(b: u64) -> char {
            let b = (b & 0xf) as u8;
            let b = match b {
                0 .. 9 => '0' as u8 + b,
                _ => 'a' as u8 + b - 10,
            };
            b as char
        }
    }
}

impl fmt::Show for Svh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(self.as_str())
    }
}
