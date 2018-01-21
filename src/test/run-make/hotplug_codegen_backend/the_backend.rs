// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]

extern crate syntax;
extern crate rustc;
extern crate rustc_trans_utils;

use std::any::Any;
use std::sync::mpsc;
use syntax::symbol::Symbol;
use rustc::session::{Session, CompileIncomplete};
use rustc::session::config::OutputFilenames;
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;
use rustc::middle::cstore::MetadataLoader;
use rustc::dep_graph::DepGraph;
use rustc_trans_utils::trans_crate::{TransCrate, MetadataOnlyTransCrate};

struct TheBackend(Box<TransCrate>);

impl TransCrate for TheBackend {
    fn metadata_loader(&self) -> Box<MetadataLoader> {
        self.0.metadata_loader()
    }

    fn provide(&self, providers: &mut Providers) {
        self.0.provide(providers);
    }

    fn provide_extern(&self, providers: &mut Providers) {
        self.0.provide_extern(providers);
    }

    fn trans_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Box<Any> {
        use rustc::hir::def_id::LOCAL_CRATE;

        Box::new(tcx.crate_name(LOCAL_CRATE) as Symbol)
    }

    fn join_trans_and_link(
        &self,
        trans: Box<Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        use std::io::Write;
        use rustc::session::config::CrateType;
        use rustc_trans_utils::link::out_filename;
        let crate_name = trans.downcast::<Symbol>()
            .expect("in join_trans_and_link: trans is not a Symbol");
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::CrateTypeExecutable {
                sess.fatal(&format!("Crate type is {:?}", crate_type));
            }
            let output_name =
                out_filename(sess, crate_type, &outputs, &*crate_name.as_str());
            let mut out_file = ::std::fs::File::create(output_name).unwrap();
            write!(out_file, "This has been \"compiled\" succesfully.").unwrap();
        }
        Ok(())
    }
}

/// This is the entrypoint for a hot plugged rustc_trans
#[no_mangle]
pub fn __rustc_codegen_backend(sess: &Session) -> Box<TransCrate> {
    Box::new(TheBackend(MetadataOnlyTransCrate::new(sess)))
}
