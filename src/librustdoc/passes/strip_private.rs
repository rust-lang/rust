// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::util::nodemap::DefIdSet;

use clean;
use core::DocContext;
use fold::DocFolder;
use passes::{ImplStripper, ImportStripper, Stripper, Pass};

pub const STRIP_PRIVATE: Pass =
    Pass::early("strip-private", strip_private,
        "strips all private items from a crate which cannot be seen externally, \
         implies strip-priv-imports");

/// Strip private items from the point of view of a crate or externally from a
/// crate, specified by the `xcrate` flag.
pub fn strip_private(mut krate: clean::Crate, cx: &DocContext) -> clean::Crate {
    // This stripper collects all *retained* nodes.
    let mut retained = DefIdSet::default();
    let access_levels = cx.renderinfo.borrow().access_levels.clone();

    // strip all private items
    {
        let mut stripper = Stripper {
            retained: &mut retained,
            access_levels: &access_levels,
            update_retained: true,
        };
        krate = ImportStripper.fold_crate(stripper.fold_crate(krate));
    }

    // strip all impls referencing private items
    let mut stripper = ImplStripper { retained: &retained };
    stripper.fold_crate(krate)
}
