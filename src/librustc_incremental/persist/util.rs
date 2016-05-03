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
use std::io;
use std::path::{PathBuf, Path};

pub fn dep_graph_path<'tcx>(tcx: &ty::TyCtxt<'tcx>) -> Option<PathBuf> {
    // For now, just save/load dep-graph from
    // directory/dep_graph.rbml
    tcx.sess.opts.incremental.as_ref().and_then(|incr_dir| {
        match create_dir_racy(&incr_dir) {
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

// Like std::fs::create_dir_all, except handles concurrent calls among multiple
// threads or processes.
fn create_dir_racy(path: &Path) -> io::Result<()> {
    match fs::create_dir(path) {
        Ok(()) => return Ok(()),
        Err(ref e) if e.kind() == io::ErrorKind::AlreadyExists => return Ok(()),
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    match path.parent() {
        Some(p) => try!(create_dir_racy(p)),
        None => return Err(io::Error::new(io::ErrorKind::Other,
                                          "failed to create whole tree")),
    }
    match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(ref e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e),
    }
}
