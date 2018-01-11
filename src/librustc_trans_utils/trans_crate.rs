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

use rustc_data_structures::owning_ref::{ErasedBoxRef, OwningRef};
use ar::{Archive, Builder, Header};
use flate2::Compression;
use flate2::write::DeflateEncoder;

use syntax::symbol::Symbol;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::session::Session;
use rustc::session::config::{CrateType, OutputFilenames};
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;
use rustc::middle::cstore::EncodedMetadata;
use rustc::middle::cstore::MetadataLoader as MetadataLoaderTrait;
use rustc::dep_graph::DepGraph;
use rustc_back::target::Target;
use link::{build_link_meta, out_filename};

pub trait TransCrate {
    type MetadataLoader: MetadataLoaderTrait;
    type OngoingCrateTranslation;
    type TranslatedCrate;

    fn metadata_loader() -> Box<MetadataLoaderTrait>;
    fn provide(_providers: &mut Providers);
    fn provide_extern(_providers: &mut Providers);
    fn trans_crate<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Self::OngoingCrateTranslation;
    fn join_trans(
        trans: Self::OngoingCrateTranslation,
        sess: &Session,
        dep_graph: &DepGraph
    ) -> Self::TranslatedCrate;
    fn link_binary(sess: &Session, trans: &Self::TranslatedCrate, outputs: &OutputFilenames);
    fn dump_incremental_data(trans: &Self::TranslatedCrate);
}

pub struct DummyTransCrate;

impl TransCrate for DummyTransCrate {
    type MetadataLoader = DummyMetadataLoader;
    type OngoingCrateTranslation = ();
    type TranslatedCrate = ();

    fn metadata_loader() -> Box<MetadataLoaderTrait> {
        box DummyMetadataLoader(())
    }

    fn provide(_providers: &mut Providers) {
        bug!("DummyTransCrate::provide");
    }

    fn provide_extern(_providers: &mut Providers) {
        bug!("DummyTransCrate::provide_extern");
    }

    fn trans_crate<'a, 'tcx>(
        _tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Self::OngoingCrateTranslation {
        bug!("DummyTransCrate::trans_crate");
    }

    fn join_trans(
        _trans: Self::OngoingCrateTranslation,
        _sess: &Session,
        _dep_graph: &DepGraph
    ) -> Self::TranslatedCrate {
        bug!("DummyTransCrate::join_trans");
    }

    fn link_binary(_sess: &Session, _trans: &Self::TranslatedCrate, _outputs: &OutputFilenames) {
        bug!("DummyTransCrate::link_binary");
    }

    fn dump_incremental_data(_trans: &Self::TranslatedCrate) {
        bug!("DummyTransCrate::dump_incremental_data");
    }
}

pub struct DummyMetadataLoader(());

impl MetadataLoaderTrait for DummyMetadataLoader {
    fn get_rlib_metadata(
        &self,
        _target: &Target,
        _filename: &Path
    ) -> Result<ErasedBoxRef<[u8]>, String> {
        bug!("DummyMetadataLoader::get_rlib_metadata");
    }

    fn get_dylib_metadata(
        &self,
        _target: &Target,
        _filename: &Path
    ) -> Result<ErasedBoxRef<[u8]>, String> {
        bug!("DummyMetadataLoader::get_dylib_metadata");
    }
}

pub struct NoLlvmMetadataLoader;

impl MetadataLoaderTrait for NoLlvmMetadataLoader {
    fn get_rlib_metadata(&self, _: &Target, filename: &Path) -> Result<ErasedBoxRef<[u8]>, String> {
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
                return Ok(buf.map_owner_box().erase_owner());
            }
        }

        Err("Couldnt find metadata section".to_string())
    }

    fn get_dylib_metadata(
        &self,
        _target: &Target,
        _filename: &Path,
    ) -> Result<ErasedBoxRef<[u8]>, String> {
        // FIXME: Support reading dylibs from llvm enabled rustc
        self.get_rlib_metadata(_target, _filename)
    }
}

pub struct MetadataOnlyTransCrate;
pub struct OngoingCrateTranslation {
    metadata: EncodedMetadata,
    metadata_version: Vec<u8>,
    crate_name: Symbol,
}
pub struct TranslatedCrate(OngoingCrateTranslation);

impl MetadataOnlyTransCrate {
    #[allow(dead_code)]
    pub fn new() -> Self {
        MetadataOnlyTransCrate
    }
}

impl TransCrate for MetadataOnlyTransCrate {
    type MetadataLoader = NoLlvmMetadataLoader;
    type OngoingCrateTranslation = OngoingCrateTranslation;
    type TranslatedCrate = TranslatedCrate;

    fn metadata_loader() -> Box<MetadataLoaderTrait> {
        box NoLlvmMetadataLoader
    }

    fn provide(_providers: &mut Providers) {}
    fn provide_extern(_providers: &mut Providers) {}

    fn trans_crate<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Self::OngoingCrateTranslation {
        ::check_for_rustc_errors_attr(tcx);
        let _ = tcx.link_args(LOCAL_CRATE);
        let _ = tcx.native_libraries(LOCAL_CRATE);
        tcx.sess.abort_if_errors();

        let link_meta = build_link_meta(tcx.crate_hash(LOCAL_CRATE));
        let exported_symbols = ::find_exported_symbols(tcx);
        let metadata = tcx.encode_metadata(&link_meta, &exported_symbols);

        OngoingCrateTranslation {
            metadata: metadata,
            metadata_version: tcx.metadata_encoding_version().to_vec(),
            crate_name: tcx.crate_name(LOCAL_CRATE),
        }
    }

    fn join_trans(
        trans: Self::OngoingCrateTranslation,
        _sess: &Session,
        _dep_graph: &DepGraph,
    ) -> Self::TranslatedCrate {
        TranslatedCrate(trans)
    }

    fn link_binary(sess: &Session, trans: &Self::TranslatedCrate, outputs: &OutputFilenames) {
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::CrateTypeRlib && crate_type != CrateType::CrateTypeDylib {
                continue;
            }
            let output_name =
                out_filename(sess, crate_type, &outputs, &trans.0.crate_name.as_str());
            let mut compressed = trans.0.metadata_version.clone();
            let metadata = if crate_type == CrateType::CrateTypeDylib {
                DeflateEncoder::new(&mut compressed, Compression::fast())
                    .write_all(&trans.0.metadata.raw_data)
                    .unwrap();
                &compressed
            } else {
                &trans.0.metadata.raw_data
            };
            let mut builder = Builder::new(File::create(&output_name).unwrap());
            let header = Header::new("rust.metadata.bin".to_string(), metadata.len() as u64);
            builder.append(&header, Cursor::new(metadata)).unwrap();
        }

        if !sess.opts.crate_types.contains(&CrateType::CrateTypeRlib)
            && !sess.opts.crate_types.contains(&CrateType::CrateTypeDylib) {
            sess.fatal("Executables are not supported by the metadata-only backend.");
        }
    }

    fn dump_incremental_data(_trans: &Self::TranslatedCrate) {}
}
