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
extern crate rustc_codegen_utils;

use std::any::Any;
use std::sync::mpsc;
use syntax::symbol::Symbol;
use rustc::session::{Session, CompileIncomplete};
use rustc::session::config::OutputFilenames;
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc::middle::cstore::MetadataLoader;
use rustc::dep_graph::DepGraph;
use rustc_codegen_utils::codegen_backend::{CodegenBackend, MetadataOnlyCodegenBackend};

struct TheBackend(Box<CodegenBackend>);

impl CodegenBackend for TheBackend {
    fn metadata_loader(&self) -> Box<MetadataLoader + Sync> {
        self.0.metadata_loader()
    }

    fn provide(&self, providers: &mut Providers) {
        self.0.provide(providers);
    }

    fn provide_extern(&self, providers: &mut Providers) {
        self.0.provide_extern(providers);
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Box<Any> {
        use rustc::hir::def_id::LOCAL_CRATE;

        Box::new(tcx.crate_name(LOCAL_CRATE) as Symbol)
    }

    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        use std::io::Write;
        use rustc::session::config::CrateType;
        use rustc_codegen_utils::link::out_filename;
        let crate_name = ongoing_codegen.downcast::<Symbol>()
            .expect("in join_codegen_and_link: ongoing_codegen is not a Symbol");
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::CrateTypeExecutable {
                sess.fatal(&format!("Crate type is {:?}", crate_type));
            }
            let output_name =
                out_filename(sess, crate_type, &outputs, &*crate_name.as_str());
            let mut out_file = ::std::fs::File::create(output_name).unwrap();
            write!(out_file, "This has been \"compiled\" successfully.").unwrap();
        }
        Ok(())
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_llvm
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<CodegenBackend> {
    Box::new(TheBackend(MetadataOnlyCodegenBackend::new()))
}
