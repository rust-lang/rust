//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

use std::any::Any;

use rustc::dep_graph::DepGraph;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoaderDyn};
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use rustc::util::common::ErrorReported;
use rustc_session::config::{OutputFilenames, PrintRequest};
use rustc_session::Session;
use rustc_span::symbol::Symbol;

pub use rustc_data_structures::sync::MetadataRef;

pub trait CodegenBackend {
    fn init(&self, _sess: &Session) {}
    fn print(&self, _req: PrintRequest, _sess: &Session) {}
    fn target_features(&self, _sess: &Session) -> Vec<Symbol> {
        vec![]
    }
    fn print_passes(&self) {}
    fn print_version(&self) {}

    fn metadata_loader(&self) -> Box<MetadataLoaderDyn>;
    fn provide(&self, _providers: &mut Providers<'_>);
    fn provide_extern(&self, _providers: &mut Providers<'_>);
    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any>;

    /// This is called on the returned `Box<dyn Any>` from `codegen_backend`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by `codegen_backend`.
    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        dep_graph: &DepGraph,
    ) -> Result<Box<dyn Any>, ErrorReported>;

    /// This is called on the returned `Box<dyn Any>` from `join_codegen`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by `join_codegen`.
    fn link(
        &self,
        sess: &Session,
        codegen_results: Box<dyn Any>,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported>;
}
