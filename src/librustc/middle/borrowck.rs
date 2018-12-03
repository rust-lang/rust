// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ich::StableHashingContext;
use hir::HirId;
use util::nodemap::FxHashSet;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum SignalledError { SawSomeError, NoErrorsSeen }

impl Default for SignalledError {
    fn default() -> SignalledError {
        SignalledError::NoErrorsSeen
    }
}

impl_stable_hash_for!(enum self::SignalledError { SawSomeError, NoErrorsSeen });

#[derive(Debug, Default, RustcEncodable, RustcDecodable)]
pub struct BorrowCheckResult {
    pub used_mut_nodes: FxHashSet<HirId>,
    pub signalled_any_error: SignalledError,
}

impl<'a> HashStable<StableHashingContext<'a>> for BorrowCheckResult {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let BorrowCheckResult {
            ref used_mut_nodes,
            ref signalled_any_error,
        } = *self;
        used_mut_nodes.hash_stable(hcx, hasher);
        signalled_any_error.hash_stable(hcx, hasher);
    }
}
