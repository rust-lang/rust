// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty;
use std::fs;
use std::path::PathBuf;

pub fn dep_graph_path<'tcx>(tcx: &ty::TyCtxt<'tcx>) -> Option<PathBuf> {
    // For now, just save/load dep-graph from
    // directory/dep_graph.rbml
    tcx.sess.opts.incremental.as_ref().and_then(|incr_dir| {
        match fs::create_dir_all(&incr_dir){
            Ok(()) => {}
            Err(err) => {
                tcx.sess.err(
                    &format!("could not create the directory `{}`: {}",
                             incr_dir.display(), err));
                return None;
            }
        }

        Some(incr_dir.join("dep_graph.rbml"))
    })
}

