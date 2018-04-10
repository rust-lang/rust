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
//! from rustc::middle::cstore in no particular order.

use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};
use ich::StableHashingContext;

use middle;

impl_stable_hash_for!(enum middle::cstore::DepKind {
    UnexportedMacrosOnly,
    MacrosOnly,
    Implicit,
    Explicit
});

impl_stable_hash_for!(enum middle::cstore::NativeLibraryKind {
    NativeStatic,
    NativeStaticNobundle,
    NativeFramework,
    NativeUnknown
});

impl_stable_hash_for!(struct middle::cstore::NativeLibrary {
    kind,
    name,
    cfg,
    foreign_module
});

impl_stable_hash_for!(struct middle::cstore::ForeignModule {
    foreign_items,
    def_id
});

impl_stable_hash_for!(enum middle::cstore::LinkagePreference {
    RequireDynamic,
    RequireStatic
});

impl_stable_hash_for!(struct middle::cstore::ExternCrate {
    src,
    span,
    direct
});

impl<'a> HashStable<StableHashingContext<'a>> for middle::cstore::ExternCrateSource {
    fn hash_stable<W: StableHasherResult>(
        &self,
        hcx: &mut StableHashingContext<'a>,
        hasher: &mut StableHasher<W>,
    ) {
        use middle::cstore::ExternCrateSource::*;

        ::std::mem::discriminant(self).hash_stable(hcx, hasher);

        match *self {
            Extern { def_id, path_len } => {
                def_id.hash_stable(hcx, hasher);
                path_len.hash_stable(hcx, hasher);
            }
            Use { path_len } => path_len.hash_stable(hcx, hasher),
            Path => {}
        }
    }
}

impl_stable_hash_for!(struct middle::cstore::CrateSource {
    dylib,
    rlib,
    rmeta
});

impl<HCX> HashStable<HCX> for middle::cstore::ExternBodyNestedBodies {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        let middle::cstore::ExternBodyNestedBodies {
            nested_bodies: _,
            fingerprint,
        } = *self;

        fingerprint.hash_stable(hcx, hasher);
    }
}

impl<'a, HCX> HashStable<HCX> for middle::cstore::ExternConstBody<'a> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        let middle::cstore::ExternConstBody {
            body: _,
            fingerprint,
        } = *self;

        fingerprint.hash_stable(hcx, hasher);
    }
}
