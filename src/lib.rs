#![feature(rustc_private)]
#![allow(intra_doc_link_resolution_failure)]

extern crate syntax;
#[macro_use]
extern crate rustc;
extern crate rustc_mir;
extern crate rustc_codegen_utils;
extern crate rustc_incremental;
extern crate rustc_data_structures;

extern crate cretonne;
extern crate cretonne_module;
extern crate cretonne_simplejit;
extern crate cretonne_faerie;

use syntax::symbol::Symbol;
use rustc::session::{
    CompileIncomplete,
    config::{
        CrateType,
        OutputFilenames,
    },
};
use rustc::middle::cstore::{MetadataLoader, EncodedMetadata};
use rustc::dep_graph::DepGraph;
use rustc::ty::query::Providers;
use rustc_codegen_utils::codegen_backend::{CodegenBackend, NoLlvmMetadataLoader};
use rustc_codegen_utils::link::{out_filename, build_link_meta};

use std::any::Any;
use std::sync::{mpsc, Arc};
use std::fs::File;
use std::io::Write;

mod base;
mod common;

mod prelude {
    pub use std::collections::HashMap;

    pub use rustc::hir::def_id::{DefId, LOCAL_CRATE};
    pub use rustc::mir;
    pub use rustc::mir::*;
    pub use rustc::session::Session;
    pub use rustc::ty::layout::{self, LayoutOf, TyLayout};
    pub use rustc::ty::{
        self, subst::Substs, FnSig, Instance, InstanceDef, ParamEnv, PolyFnSig, Ty, TyCtxt,
        TypeFoldable, TypeVariants,
    };
    pub use rustc_data_structures::{indexed_vec::Idx, sync::Lrc};
    pub use rustc_mir::monomorphize::collector;

    pub use cretonne::codegen::ir::{
        condcodes::IntCC, function::Function, ExternalName, FuncRef, StackSlot,
    };
    pub use cretonne::codegen::Context;
    pub use cretonne::prelude::*;

    pub use common::Variable;
    pub use common::*;
}

use prelude::*;

struct CretonneCodegenBackend(());

struct OngoingCodegen {
    metadata: EncodedMetadata,
    //translated_module: Module<cretonne_faerie::FaerieBackend>,
    crate_name: Symbol,
}

impl CretonneCodegenBackend {
    fn new() -> Box<CodegenBackend> {
        Box::new(CretonneCodegenBackend(()))
    }
}

impl CodegenBackend for CretonneCodegenBackend {
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
        Box::new(NoLlvmMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {
        rustc_codegen_utils::symbol_names::provide(providers);

        providers.target_features_whitelist = |_tcx, _cnum| {
            /*Lrc::new(rustc_codegen_utils::llvm_target_features::all_known_features()
                .map(|(a, b)| (a.to_string(), b.map(|s| s.to_string())))
                .collect())*/
            Lrc::new(Default::default())
        };
        providers.is_reachable_non_generic = |_tcx, _defid| true;
        providers.exported_symbols = |_tcx, _crate| Arc::new(Vec::new());
        providers.wasm_custom_sections = |_tcx, _crate| Lrc::new(Vec::new());
    }
    fn provide_extern(&self, providers: &mut Providers) {
        providers.is_reachable_non_generic = |_tcx, _defid| true;
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>
    ) -> Box<Any> {
        use rustc_mir::monomorphize::item::MonoItem;

        rustc_codegen_utils::check_for_rustc_errors_attr(tcx);
        rustc_codegen_utils::symbol_names_test::report_symbol_names(tcx);
        rustc_incremental::assert_dep_graph(tcx);
        rustc_incremental::assert_module_sources::assert_module_sources(tcx);
        rustc_mir::monomorphize::assert_symbols_are_distinct(tcx,
            collector::collect_crate_mono_items(
                tcx,
                collector::MonoItemCollectionMode::Eager
            ).0.iter()
        );
        //::rustc::middle::dependency_format::calculate(tcx);
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

        base::trans_crate(tcx)
    }

    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        if true {
            unimplemented!();
        }

        let ongoing_codegen = ongoing_codegen.downcast::<OngoingCodegen>()
            .expect("Expected MetadataOnlyCodegenBackend's OngoingCodegen, found Box<Any>");
        for &crate_type in sess.opts.crate_types.iter() {
            if crate_type != CrateType::CrateTypeRlib && crate_type != CrateType::CrateTypeDylib {
                continue;
            }
            let output_name =
                out_filename(sess, crate_type, &outputs, &ongoing_codegen.crate_name.as_str());
            let metadata = &ongoing_codegen.metadata.raw_data;
            let mut file = File::create(&output_name).unwrap();
            file.write_all(metadata).unwrap();
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

/// This is the entrypoint for a hot plugged rustc_codegen_cretonne
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<CodegenBackend> {
    CretonneCodegenBackend::new()
}
