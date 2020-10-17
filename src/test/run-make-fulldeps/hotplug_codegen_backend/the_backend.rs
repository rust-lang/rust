#![feature(rustc_private)]

extern crate rustc_codegen_ssa;
extern crate rustc_errors;
extern crate rustc_middle;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;

use rustc_codegen_ssa::back::linker::LinkerInfo;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CodegenResults, CrateInfo};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::MetadataRef;
use rustc_errors::ErrorReported;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::cstore::{EncodedMetadata, MetadataLoader, MetadataLoaderDyn};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::OutputFilenames;
use rustc_session::Session;
use rustc_target::spec::Target;
use std::any::Any;
use std::path::Path;

pub struct NoLlvmMetadataLoader;

impl MetadataLoader for NoLlvmMetadataLoader {
    fn get_rlib_metadata(&self, _: &Target, filename: &Path) -> Result<MetadataRef, String> {
        unreachable!("some_crate.rs shouldn't depend on any external crates");
    }

    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String> {
        unreachable!("some_crate.rs shouldn't depend on any external crates");
    }
}

struct TheBackend;

impl CodegenBackend for TheBackend {
    fn metadata_loader(&self) -> Box<MetadataLoaderDyn> {
        Box::new(NoLlvmMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {}
    fn provide_extern(&self, providers: &mut Providers) {}

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        _need_metadata_module: bool,
    ) -> Box<dyn Any> {
        use rustc_hir::def_id::LOCAL_CRATE;

        Box::new(CodegenResults {
            crate_name: tcx.crate_name(LOCAL_CRATE),
            modules: vec![],
            allocator_module: None,
            metadata_module: None,
            metadata,
            windows_subsystem: None,
            linker_info: LinkerInfo::new(tcx),
            crate_info: CrateInfo::new(tcx),
        })
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        _sess: &Session,
    ) -> Result<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>), ErrorReported> {
        let codegen_results = ongoing_codegen
            .downcast::<CodegenResults>()
            .expect("in join_codegen: ongoing_codegen is not a CodegenResults");
        Ok((*codegen_results, FxHashMap::default()))
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported> {
        use rustc_session::{config::CrateType, output::out_filename};
        use std::io::Write;
        let crate_name = codegen_results.crate_name;
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::Rlib {
                sess.fatal(&format!("Crate type is {:?}", crate_type));
            }
            let output_name = out_filename(sess, crate_type, &outputs, &*crate_name.as_str());
            let mut out_file = ::std::fs::File::create(output_name).unwrap();
            write!(out_file, "This has been \"compiled\" successfully.").unwrap();
        }
        Ok(())
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_llvm
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(TheBackend)
}
