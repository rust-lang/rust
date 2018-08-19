// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::BindingAnnotation::*;
use hir::BindingAnnotation;
use hir::Mutability;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BindingMode {
    BindByReference(Mutability),
    BindByValue(Mutability),
}

CloneTypeFoldableAndLiftImpls! { BindingMode, }

impl BindingMode {
    pub fn convert(ba: BindingAnnotation) -> BindingMode {
        match ba {
            Unannotated => BindingMode::BindByValue(Mutability::MutImmutable),
            Mutable => BindingMode::BindByValue(Mutability::MutMutable),
            Ref => BindingMode::BindByReference(Mutability::MutImmutable),
            RefMut => BindingMode::BindByReference(Mutability::MutMutable),
        }
    }
}

impl_stable_hash_for!(enum self::BindingMode {
    BindByReference(mutability),
    BindByValue(mutability)
});
