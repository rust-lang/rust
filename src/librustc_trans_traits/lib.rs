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

extern crate owning_ref;

#[macro_use]
extern crate rustc;
extern crate rustc_back;
extern crate rustc_incremental;

use std::path::Path;
use owning_ref::ErasedBoxRef;

use rustc::session::Session;
use rustc::session::config::OutputFilenames;
use rustc::ty::{TyCtxt, CrateAnalysis};
use rustc::ty::maps::Providers;
use rustc::middle::cstore::MetadataLoader as MetadataLoaderTrait;
use rustc::dep_graph::DepGraph;
use rustc_back::target::Target;
use rustc_incremental::IncrementalHashesMap;

pub trait TransCrate {
    type MetadataLoader: MetadataLoaderTrait;
    type OngoingCrateTranslation;
    type TranslatedCrate;

    fn metadata_loader() -> Box<MetadataLoaderTrait>;
    fn provide(_providers: &mut Providers);
    fn trans_crate<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        analysis: CrateAnalysis,
        incr_hashes_map: IncrementalHashesMap,
        output_filenames: &OutputFilenames
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

    fn trans_crate<'a, 'tcx>(
        _tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _analysis: CrateAnalysis,
        _incr_hashes_map: IncrementalHashesMap,
        _output_filenames: &OutputFilenames
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
    fn get_rlib_metadata(&self, _target: &Target, _filename: &Path) -> Result<ErasedBoxRef<[u8]>, String> {
        bug!("DummyMetadataLoader::get_rlib_metadata");
    }

    fn get_dylib_metadata(&self, _target: &Target, _filename: &Path) -> Result<ErasedBoxRef<[u8]>, String> {
        bug!("DummyMetadataLoader::get_dylib_metadata");
    }
}
