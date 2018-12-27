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
use std::io::Write;
use std::fs;
use std::path::Path;
use std::sync::{mpsc, Arc};

use rustc_data_structures::owning_ref::OwningRef;
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
use rustc_mir::monomorphize::collector;
use link::out_filename;

pub use rustc_data_structures::sync::MetadataRef;

pub trait CodegenBackend {
    fn init(&self, _sess: &Session) {}
    fn print(&self, _req: PrintRequest, _sess: &Session) {}
    fn target_features(&self, _sess: &Session) -> Vec<Symbol> { vec![] }
    fn print_passes(&self) {}
    fn print_version(&self) {}
    fn diagnostics(&self) -> &[(&'static str, &'static str)] { &[] }

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync>;
    fn provide(&self, _providers: &mut Providers);
    fn provide_extern(&self, _providers: &mut Providers);
    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        rx: mpsc::Receiver<Box<dyn Any + Send>>
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
    ) -> Result<(), CompileIncomplete>;
}

pub struct NoLlvmMetadataLoader;

impl MetadataLoader for NoLlvmMetadataLoader {
    fn get_rlib_metadata(&self, _: &Target, filename: &Path) -> Result<MetadataRef, String> {
        let buf = fs::read(filename).map_err(|e| format!("metadata file open err: {:?}", e))?;
        let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(buf);
        Ok(rustc_erase_owner!(buf.map_owner_box()))
    }

    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String> {
        self.get_rlib_metadata(target, filename)
    }
}

pub struct MetadataOnlyCodegenBackend(());
pub struct OngoingCodegen {
    metadata: EncodedMetadata,
    metadata_version: Vec<u8>,
    crate_name: Symbol,
}

impl MetadataOnlyCodegenBackend {
    pub fn boxed() -> Box<dyn CodegenBackend> {
        box MetadataOnlyCodegenBackend(())
    }
}

impl CodegenBackend for MetadataOnlyCodegenBackend {
    fn init(&self, sess: &Session) {
        for cty in sess.opts.crate_types.iter() {
            match *cty {
                CrateType::Rlib | CrateType::Dylib | CrateType::Executable => {},
                _ => {
                    sess.diagnostic().warn(
                        &format!("LLVM unsupported, so output type {} is not supported", cty)
                    );
                },
            }
        }
    }

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync> {
        box NoLlvmMetadataLoader
    }

    fn provide(&self, providers: &mut Providers) {
        ::symbol_names::provide(providers);

        providers.target_features_whitelist = |_tcx, _cnum| {
            Default::default() // Just a dummy
        };
        providers.is_reachable_non_generic = |_tcx, _defid| true;
        providers.exported_symbols = |_tcx, _crate| Arc::new(Vec::new());
    }
    fn provide_extern(&self, providers: &mut Providers) {
        providers.is_reachable_non_generic = |_tcx, _defid| true;
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<dyn Any + Send>>
    ) -> Box<dyn Any> {
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
        // FIXME: Fix this
        // ::rustc::middle::dependency_format::calculate(tcx);
        let _ = tcx.link_args(LOCAL_CRATE);
        let _ = tcx.native_libraries(LOCAL_CRATE);
        for mono_item in
            collector::collect_crate_mono_items(
                tcx,
                collector::MonoItemCollectionMode::Eager
            ).0 {
            if let MonoItem::Fn(inst) = mono_item {
                let def_id = inst.def_id();
                if def_id.is_local()  {
                    let _ = inst.def.is_inline(tcx);
                    let _ = tcx.codegen_fn_attrs(def_id);
                }
            }
        }
        tcx.sess.abort_if_errors();

        let metadata = tcx.encode_metadata();

        box OngoingCodegen {
            metadata,
            metadata_version: tcx.metadata_encoding_version().to_vec(),
            crate_name: tcx.crate_name(LOCAL_CRATE),
        }
    }

    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        let ongoing_codegen = ongoing_codegen.downcast::<OngoingCodegen>()
            .expect("Expected MetadataOnlyCodegenBackend's OngoingCodegen, found Box<dyn Any>");
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::Rlib &&
               crate_type != CrateType::Dylib {
                continue;
            }
            let output_name =
                out_filename(sess, crate_type, &outputs, &ongoing_codegen.crate_name.as_str());
            let mut compressed = ongoing_codegen.metadata_version.clone();
            let metadata = if crate_type == CrateType::Dylib {
                DeflateEncoder::new(&mut compressed, Compression::fast())
                    .write_all(&ongoing_codegen.metadata.raw_data)
                    .unwrap();
                &compressed
            } else {
                &ongoing_codegen.metadata.raw_data
            };
            fs::write(&output_name, metadata).unwrap();
        }

        sess.abort_if_errors();
        if !sess.opts.crate_types.contains(&CrateType::Rlib)
            && !sess.opts.crate_types.contains(&CrateType::Dylib)
        {
            sess.fatal("Executables are not supported by the metadata-only backend.");
        }
        Ok(())
    }
}
