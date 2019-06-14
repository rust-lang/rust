//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]
#![feature(box_syntax)]

use std::any::Any;
use std::sync::mpsc;

use syntax::symbol::Symbol;
use rustc::session::Session;
use rustc::util::common::ErrorReported;
use rustc::session::config::{OutputFilenames, PrintRequest};
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc::dep_graph::DepGraph;

pub use rustc_data_structures::sync::MetadataRef;

pub trait CodegenBackend {
    fn init(&self, _sess: &Session) {}
    fn print(&self, _req: PrintRequest, _sess: &Session) {}
    fn target_features(&self, _sess: &Session) -> Vec<Symbol> { vec![] }
    fn print_passes(&self) {}
    fn print_version(&self) {}
    fn diagnostics(&self) -> &[(&'static str, &'static str)] { &[] }

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync>;
    fn provide(&self, _providers: &mut Providers<'_>);
    fn provide_extern(&self, _providers: &mut Providers<'_>);
    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
        rx: mpsc::Receiver<Box<dyn Any + Send>>,
    ) -> Box<dyn Any>;

    /// This is called on the returned `Box<dyn Any>` from `codegen_backend`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by `codegen_backend`.
    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported>;
}
