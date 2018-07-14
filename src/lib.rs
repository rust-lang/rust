#![feature(rustc_private)]
#![allow(intra_doc_link_resolution_failure)]

extern crate syntax;
#[macro_use]
extern crate rustc;
extern crate rustc_mir;
extern crate rustc_codegen_utils;
extern crate rustc_incremental;
extern crate rustc_data_structures;

extern crate cranelift;
extern crate cranelift_module;
extern crate cranelift_simplejit;
//extern crate cranelift_faerie;

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
mod constant;
mod common;
mod pretty_clif;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::HashMap;

    pub use syntax::codemap::DUMMY_SP;
    pub use rustc::hir::def_id::{DefId, LOCAL_CRATE};
    pub use rustc::mir;
    pub use rustc::mir::*;
    pub use rustc::session::Session;
    pub use rustc::ty::layout::{self, LayoutOf, TyLayout, Size};
    pub use rustc::ty::{
        self, subst::Substs, FnSig, Instance, InstanceDef, ParamEnv, PolyFnSig, Ty, TyCtxt,
        TypeFoldable, TypeVariants, TypeAndMut,
    };
    pub use rustc_data_structures::{indexed_vec::Idx, sync::Lrc};
    pub use rustc_mir::monomorphize::{MonoItem, collector};

    pub use cranelift::codegen::ir::{
        condcodes::IntCC, function::Function, ExternalName, FuncRef, StackSlot, Inst
    };
    pub use cranelift::codegen::Context;
    pub use cranelift::prelude::*;
    pub use cranelift_module::{Module, Backend, FuncId, Linkage};
    pub use cranelift_simplejit::{SimpleJITBuilder, SimpleJITBackend};

    pub use common::Variable;
    pub use common::*;

    pub use CodegenCx;
}

use prelude::*;

pub struct CodegenCx<'a, 'tcx: 'a, B: Backend + 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub module: &'a mut Module<B>,
    pub def_id_fn_id_map: &'a mut HashMap<Instance<'tcx>, FuncId>,
}

struct CraneliftCodegenBackend(());

struct OngoingCodegen {
    metadata: EncodedMetadata,
    //translated_module: Module<cranelift_faerie::FaerieBackend>,
    crate_name: Symbol,
}

impl CraneliftCodegenBackend {
    fn new() -> Box<CodegenBackend> {
        Box::new(CraneliftCodegenBackend(()))
    }
}

impl CodegenBackend for CraneliftCodegenBackend {
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

        let link_meta = ::build_link_meta(tcx.crate_hash(LOCAL_CRATE));
        let metadata = tcx.encode_metadata(&link_meta);

        let mut module: Module<SimpleJITBackend> = Module::new(SimpleJITBuilder::new());
        let mut context = Context::new();
        let mut def_id_fn_id_map = HashMap::new();

        {
            let mut cx = CodegenCx {
                tcx,
                module: &mut module,
                def_id_fn_id_map: &mut def_id_fn_id_map,
            };

            for mono_item in
                collector::collect_crate_mono_items(
                    tcx,
                    collector::MonoItemCollectionMode::Eager
                ).0 {
                base::trans_mono_item(&mut cx, &mut context, mono_item)
            }
        }

        tcx.sess.warn("Compiled everything");

        module.finalize_all();

        tcx.sess.warn("Finalized everything");

        for (inst, func_id) in def_id_fn_id_map.iter() {
            //if tcx.absolute_item_path_str(inst.def_id()) != "example::ret_42" {
            if tcx.absolute_item_path_str(inst.def_id()) != "example::option_unwrap_or" {
                continue;
            }
            let finalized_function: *const u8 = module.finalize_function(*func_id);
            /*let f: extern "C" fn(&mut u32) = unsafe { ::std::mem::transmute(finalized_function) };
            let mut res = 0u32;
            f(&mut res);
            tcx.sess.warn(&format!("ret_42 returned {}", res));*/
            let f: extern "C" fn(&mut bool, &u8, bool) = unsafe { ::std::mem::transmute(finalized_function) };
            let mut res = false;
            f(&mut res, &3, false);
            tcx.sess.warn(&format!("option_unwrap_or returned {}", res));
        }

        module.finish();

        tcx.sess.fatal("unimplemented");

        Box::new(::OngoingCodegen {
            metadata: metadata,
            //translated_module: Module::new(::cranelift_faerie::FaerieBuilder::new(,
            crate_name: tcx.crate_name(LOCAL_CRATE),
        })
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

/// This is the entrypoint for a hot plugged rustc_codegen_cranelift
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<CodegenBackend> {
    CraneliftCodegenBackend::new()
}
