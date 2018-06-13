// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_syntax)]

use std::any::Any;
use std::io::prelude::*;
use std::io::{self, Cursor};
use std::fs::File;
use std::path::Path;
use std::sync::mpsc;

use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::sync::Lrc;
use ar::{Archive, Builder, Header};
use flate2::Compression;
use flate2::write::DeflateEncoder;

use syntax::symbol::Symbol;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::session::{Session, CompileIncomplete};
use rustc::session::config::{CrateType, OutputFilenames, PrintRequest};
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc::middle::cstore::EncodedMetadata;
use rustc::middle::cstore::MetadataLoader;
use rustc::dep_graph::DepGraph;
use rustc_target::spec::Target;
use rustc_data_structures::fx::FxHashMap;
use rustc_mir::monomorphize::collector;
use link::{build_link_meta, out_filename};

pub use rustc_data_structures::sync::MetadataRef;

pub trait CodegenBackend {
    fn init(&self, _sess: &Session) {}
    fn print(&self, _req: PrintRequest, _sess: &Session) {}
    fn target_features(&self, _sess: &Session) -> Vec<Symbol> { vec![] }
    fn print_passes(&self) {}
    fn print_version(&self) {}
    fn diagnostics(&self) -> &[(&'static str, &'static str)] { &[] }

    fn metadata_loader(&self) -> Box<MetadataLoader + Sync>;
    fn provide(&self, _providers: &mut Providers);
    fn provide_extern(&self, _providers: &mut Providers);
    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Box<Any>;

    /// This is called on the returned `Box<Any>` from `codegen_backend`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<Any>` was not returned by `codegen_backend`.
    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<Any>,
        sess: &Session,
        dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete>;
}

pub struct DummyCodegenBackend;

impl CodegenBackend for DummyCodegenBackend {
    fn metadata_loader(&self) -> Box<MetadataLoader + Sync> {
        box DummyMetadataLoader(())
    }

    fn provide(&self, _providers: &mut Providers) {
        bug!("DummyCodegenBackend::provide");
    }

    fn provide_extern(&self, _providers: &mut Providers) {
        bug!("DummyCodegenBackend::provide_extern");
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        _tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Box<Any> {
        bug!("DummyCodegenBackend::codegen_backend");
    }

    fn join_codegen_and_link(
        &self,
        _ongoing_codegen: Box<Any>,
        _sess: &Session,
        _dep_graph: &DepGraph,
        _outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        bug!("DummyCodegenBackend::join_codegen_and_link");
    }
}

pub struct DummyMetadataLoader(());

impl MetadataLoader for DummyMetadataLoader {
    fn get_rlib_metadata(
        &self,
        _target: &Target,
        _filename: &Path
    ) -> Result<MetadataRef, String> {
        bug!("DummyMetadataLoader::get_rlib_metadata");
    }

    fn get_dylib_metadata(
        &self,
        _target: &Target,
        _filename: &Path
    ) -> Result<MetadataRef, String> {
        bug!("DummyMetadataLoader::get_dylib_metadata");
    }
}

pub struct NoLlvmMetadataLoader;

impl MetadataLoader for NoLlvmMetadataLoader {
    fn get_rlib_metadata(&self, _: &Target, filename: &Path) -> Result<MetadataRef, String> {
        let file = File::open(filename)
            .map_err(|e| format!("metadata file open err: {:?}", e))?;
        let mut archive = Archive::new(file);

        while let Some(entry_result) = archive.next_entry() {
            let mut entry = entry_result
                .map_err(|e| format!("metadata section read err: {:?}", e))?;
            if entry.header().identifier() == "rust.metadata.bin" {
                let mut buf = Vec::new();
                io::copy(&mut entry, &mut buf).unwrap();
                let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(buf).into();
                return Ok(rustc_erase_owner!(buf.map_owner_box()));
            }
        }

        Err("Couldn't find metadata section".to_string())
    }

    fn get_dylib_metadata(
        &self,
        _target: &Target,
        _filename: &Path,
    ) -> Result<MetadataRef, String> {
        // FIXME: Support reading dylibs from llvm enabled rustc
        self.get_rlib_metadata(_target, _filename)
    }
}

pub struct MetadataOnlyCodegenBackend(());
pub struct OngoingCodegen {
    metadata: EncodedMetadata,
    metadata_version: Vec<u8>,
    crate_name: Symbol,
}

impl MetadataOnlyCodegenBackend {
    pub fn new() -> Box<CodegenBackend> {
        box MetadataOnlyCodegenBackend(())
    }
}

impl CodegenBackend for MetadataOnlyCodegenBackend {
    fn init(&self, sess: &Session) {
        for cty in sess.opts.crate_types.iter() {
            match *cty {
                CrateType::CrateTypeRlib | CrateType::CrateTypeDylib |
                CrateType::CrateTypeExecutable => {},
                _ => {
                    sess.parse_sess.span_diagnostic.warn(
                        &format!("LLVM unsupported, so output type {} is not supported", cty)
                    );
                },
            }
        }
    }

    fn metadata_loader(&self) -> Box<MetadataLoader + Sync> {
        box NoLlvmMetadataLoader
    }

    fn provide(&self, providers: &mut Providers) {
        ::symbol_names::provide(providers);

        providers.target_features_whitelist = |_tcx, _cnum| {
            Lrc::new(FxHashMap()) // Just a dummy
        };
    }
    fn provide_extern(&self, _providers: &mut Providers) {}

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Box<Any> {
        use rustc_mir::monomorphize::item::MonoItem;

        ::check_for_rustc_errors_attr(tcx);
        ::symbol_names_test::report_symbol_names(tcx);
        ::rustc_incremental::assert_dep_graph(tcx);
        ::rustc_incremental::assert_module_sources::assert_module_sources(tcx);
        ::rustc_mir::monomorphize::assert_symbols_are_distinct(tcx,
            collector::collect_crate_mono_items(
                tcx,
                collector::MonoItemCollectionMode::Eager
            ).0.iter()
        );
        ::rustc::middle::dependency_format::calculate(tcx);
        let _ = tcx.link_args(LOCAL_CRATE);
        let _ = tcx.native_libraries(LOCAL_CRATE);
        for mono_item in
            collector::collect_crate_mono_items(
                tcx,
                collector::MonoItemCollectionMode::Eager
            ).0 {
            match mono_item {
                MonoItem::Fn(inst) => {
                    let def_id = inst.def_id();
                    if def_id.is_local()  {
                        let _ = inst.def.is_inline(tcx);
                        let _ = tcx.codegen_fn_attrs(def_id);
                    }
                }
                _ => {}
            }
        }
        tcx.sess.abort_if_errors();

        let link_meta = build_link_meta(tcx.crate_hash(LOCAL_CRATE));
        let metadata = tcx.encode_metadata(&link_meta);

        box OngoingCodegen {
            metadata: metadata,
            metadata_version: tcx.metadata_encoding_version().to_vec(),
            crate_name: tcx.crate_name(LOCAL_CRATE),
        }
    }

    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        let ongoing_codegen = ongoing_codegen.downcast::<OngoingCodegen>()
            .expect("Expected MetadataOnlyCodegenBackend's OngoingCodegen, found Box<Any>");
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::CrateTypeRlib && crate_type != CrateType::CrateTypeDylib {
                continue;
            }
            let output_name =
                out_filename(sess, crate_type, &outputs, &ongoing_codegen.crate_name.as_str());
            let mut compressed = ongoing_codegen.metadata_version.clone();
            let metadata = if crate_type == CrateType::CrateTypeDylib {
                DeflateEncoder::new(&mut compressed, Compression::fast())
                    .write_all(&ongoing_codegen.metadata.raw_data)
                    .unwrap();
                &compressed
            } else {
                &ongoing_codegen.metadata.raw_data
            };
            let mut builder = Builder::new(File::create(&output_name).unwrap());
            let header = Header::new("rust.metadata.bin".to_string(), metadata.len() as u64);
            builder.append(&header, Cursor::new(metadata)).unwrap();
        }

        sess.abort_if_errors();
        if !sess.opts.crate_types.contains(&CrateType::CrateTypeRlib)
            && !sess.opts.crate_types.contains(&CrateType::CrateTypeDylib)
        {
            sess.fatal("Executables are not supported by the metadata-only backend.");
        }
        Ok(())
    }
}
