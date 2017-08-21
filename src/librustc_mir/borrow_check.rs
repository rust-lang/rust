// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::{DefId};
use rustc::mir::transform::{MirSource};
use rustc::ty::{TyCtxt};
use rustc::ty::maps::Providers;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        mir_borrowck,
        ..*providers
    };
}

fn mir_borrowck<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    let mir = tcx.mir_validated(def_id);
    let src = MirSource::from_local_def_id(tcx, def_id);
    debug!("run query mir_borrowck: {}", tcx.node_path_str(src.item_id()));

    if tcx.has_attr(def_id, "rustc_mir_borrowck") || tcx.sess.opts.debugging_opts.borrowck_mir {
        ::transform::borrow_check::borrowck_mir(tcx, src, &mir.borrow());
    }
}
