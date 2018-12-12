#![feature(rustc_private, never_type, decl_macro)]
#![allow(intra_doc_link_resolution_failure)]

extern crate log;
extern crate rustc;
extern crate rustc_allocator;
extern crate rustc_codegen_ssa;
extern crate rustc_codegen_utils;
extern crate rustc_data_structures;
extern crate rustc_fs_util;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_target;
extern crate syntax;

use std::any::Any;
use std::fs::File;
use std::sync::mpsc;

use rustc::dep_graph::DepGraph;
use rustc::middle::cstore::MetadataLoader;
use rustc::session::{
    config::{OutputFilenames, OutputType},
    CompileIncomplete,
};
use rustc::ty::query::Providers;
use rustc_codegen_ssa::back::linker::LinkerInfo;
use rustc_codegen_ssa::CrateInfo;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_codegen_utils::link::out_filename;

use cranelift::codegen::settings;
use cranelift_faerie::*;

use crate::constant::ConstantCx;
use crate::prelude::*;

mod abi;
mod allocator;
mod analyze;
mod archive;
mod base;
mod common;
mod constant;
mod intrinsics;
mod link;
mod link_copied;
mod main_shim;
mod metadata;
mod pretty_clif;
mod trap;
mod unimpl;
mod vtable;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::{HashMap, HashSet};

    pub use syntax::ast::{FloatTy, IntTy, UintTy};
    pub use syntax::source_map::DUMMY_SP;

    pub use rustc::bug;
    pub use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
    pub use rustc::mir::{self, interpret::AllocId, *};
    pub use rustc::session::{
        config::{CrateType, Lto},
        Session,
    };
    pub use rustc::ty::layout::{self, Abi, LayoutOf, Scalar, Size, TyLayout, VariantIdx};
    pub use rustc::ty::{
        self, subst::Substs, FnSig, Instance, InstanceDef, ParamEnv, PolyFnSig, Ty, TyCtxt,
        TypeAndMut, TypeFoldable,
    };
    pub use rustc_data_structures::{
        fx::{FxHashMap, FxHashSet},
        indexed_vec::Idx,
        sync::Lrc,
    };
    pub use rustc_mir::monomorphize::{collector, MonoItem};

    pub use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
    pub use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleKind};

    pub use cranelift::codegen::ir::{
        condcodes::IntCC, function::Function, ExternalName, FuncRef, Inst, StackSlot,
    };
    pub use cranelift::codegen::isa::CallConv;
    pub use cranelift::codegen::Context;
    pub use cranelift::prelude::*;
    pub use cranelift_module::{Backend, DataContext, DataId, FuncId, Linkage, Module};
    pub use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

    pub use crate::abi::*;
    pub use crate::base::{trans_operand, trans_place};
    pub use crate::common::*;
    pub use crate::trap::*;
    pub use crate::unimpl::{unimpl, with_unimpl_span};
    pub use crate::Caches;
}

pub struct Caches<'tcx> {
    pub context: Context,
    pub vtables: HashMap<(Ty<'tcx>, ty::PolyExistentialTraitRef<'tcx>), DataId>,
}

impl<'tcx> Caches<'tcx> {
    fn new() -> Self {
        Caches {
            context: Context::new(),
            vtables: HashMap::new(),
        }
    }
}

struct CraneliftCodegenBackend;

impl CodegenBackend for CraneliftCodegenBackend {
    fn init(&self, sess: &Session) {
        for cty in sess.opts.crate_types.iter() {
            match *cty {
                CrateType::Rlib | CrateType::Dylib | CrateType::Executable => {}
                _ => {
                    sess.err(&format!(
                        "Rustc codegen cranelift doesn't support output type {}",
                        cty
                    ));
                }
            }
        }
        match sess.lto() {
            Lto::Fat | Lto::Thin | Lto::ThinLocal => {
                sess.warn("Rustc codegen cranelift doesn't support lto");
            }
            Lto::No => {}
        }
        if sess.opts.cg.rpath {
            sess.err("rpath is not yet supported");
        }
        if sess.opts.debugging_opts.pgo_gen.is_some() {
            sess.err("pgo is not supported");
        }
    }

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync> {
        Box::new(crate::metadata::CraneliftMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {
        rustc_codegen_utils::symbol_names::provide(providers);
        rustc_codegen_ssa::back::symbol_export::provide(providers);

        providers.target_features_whitelist = |_tcx, _cnum| Lrc::new(Default::default());
    }
    fn provide_extern(&self, providers: &mut Providers) {
        rustc_codegen_ssa::back::symbol_export::provide_extern(providers);
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<dyn Any + Send>>,
    ) -> Box<dyn Any> {
        if !tcx.sess.crate_types.get().contains(&CrateType::Executable)
            && std::env::var("SHOULD_RUN").is_ok()
        {
            tcx.sess
                .err("Can't JIT run non executable (SHOULD_RUN env var is set)");
        }

        tcx.sess.abort_if_errors();

        let metadata = tcx.encode_metadata();

        // TODO: move to the end of this function when compiling libcore doesn't have unimplemented stuff anymore
        save_incremental(tcx);
        tcx.sess.warn("Saved incremental data");

        let mut log = if cfg!(debug_assertions) {
            Some(File::create(concat!(env!("CARGO_MANIFEST_DIR"), "/target/out/log.txt")).unwrap())
        } else {
            None
        };

        if std::env::var("SHOULD_RUN").is_ok() {
            let mut jit_module: Module<SimpleJITBackend> = Module::new(SimpleJITBuilder::new());
            assert_eq!(pointer_ty(tcx), jit_module.target_config().pointer_type());

            let sig = Signature {
                params: vec![
                    AbiParam::new(jit_module.target_config().pointer_type()),
                    AbiParam::new(jit_module.target_config().pointer_type()),
                ],
                returns: vec![AbiParam::new(
                    jit_module.target_config().pointer_type(), /*isize*/
                )],
                call_conv: CallConv::SystemV,
            };
            let main_func_id = jit_module
                .declare_function("main", Linkage::Import, &sig)
                .unwrap();

            codegen_mono_items(tcx, &mut jit_module, &mut log);

            tcx.sess.abort_if_errors();
            println!("Compiled everything");
            println!("Rustc codegen cranelift will JIT run the executable, because the SHOULD_RUN env var is set");

            let finalized_main: *const u8 = jit_module.get_finalized_function(main_func_id);
            println!("🎉 Finalized everything");

            let f: extern "C" fn(isize, *const *const u8) -> isize =
                unsafe { ::std::mem::transmute(finalized_main) };
            let res = f(0, 0 as *const _);
            tcx.sess.warn(&format!("🚀 main returned {}", res));

            jit_module.finish();
            ::std::process::exit(0);
        } else {
            let isa = build_isa(tcx.sess);
            let mut faerie_module: Module<FaerieBackend> = Module::new(
                FaerieBuilder::new(
                    isa,
                    "some_file.o".to_string(),
                    FaerieTrapCollection::Disabled,
                    FaerieBuilder::default_libcall_names(),
                )
                .unwrap(),
            );
            assert_eq!(
                pointer_ty(tcx),
                faerie_module.target_config().pointer_type()
            );

            codegen_mono_items(tcx, &mut faerie_module, &mut log);

            tcx.sess.abort_if_errors();

            let artifact = faerie_module.finish().artifact;

            let tmp_file = tcx
                .output_filenames(LOCAL_CRATE)
                .temp_path(OutputType::Object, None);
            let obj = artifact.emit().unwrap();
            std::fs::write(&tmp_file, obj).unwrap();

            return Box::new(CodegenResults {
                crate_name: tcx.crate_name(LOCAL_CRATE),
                modules: vec![CompiledModule {
                    name: "dummy_name".to_string(),
                    kind: ModuleKind::Regular,
                    object: Some(tmp_file),
                    bytecode: None,
                    bytecode_compressed: None,
                }],
                allocator_module: None,
                metadata_module: CompiledModule {
                    name: "dummy_metadata".to_string(),
                    kind: ModuleKind::Metadata,
                    object: None,
                    bytecode: None,
                    bytecode_compressed: None,
                },
                crate_hash: tcx.crate_hash(LOCAL_CRATE),
                metadata,
                windows_subsystem: None, // Windows is not yet supported
                linker_info: LinkerInfo::new(tcx),
                crate_info: CrateInfo::new(tcx),
            });
        }
    }

    fn join_codegen_and_link(
        &self,
        res: Box<dyn Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        let res = *res
            .downcast::<CodegenResults>()
            .expect("Expected CraneliftCodegenBackend's CodegenResult, found Box<Any>");

        for &crate_type in sess.opts.crate_types.iter() {
            let output_name = out_filename(sess, crate_type, &outputs, &res.crate_name.as_str());
            match crate_type {
                CrateType::Rlib => link::link_rlib(sess, &res, output_name),
                CrateType::Dylib | CrateType::Executable => {
                    link::link_natively(sess, crate_type, &res, &output_name);
                }
                _ => sess.fatal(&format!("Unsupported crate type: {:?}", crate_type)),
            }
        }
        Ok(())
    }
}

fn build_isa(sess: &Session) -> Box<isa::TargetIsa + 'static> {
    use rustc::session::config::OptLevel;

    let mut flags_builder = settings::builder();
    flags_builder.enable("is_pic").unwrap();
    flags_builder.set("enable_verifier", if cfg!(debug_assertions) {
        "true"
    } else {
        "false"
    }).unwrap();

    match sess.opts.optimize {
        OptLevel::No => {
            flags_builder.set("opt_level", "fastest").unwrap();
        }
        OptLevel::Less | OptLevel::Default => {}
        OptLevel::Aggressive => {
            flags_builder.set("opt_level", "best").unwrap();
        }
        OptLevel::Size | OptLevel::SizeMin => {
            sess.warn("Optimizing for size is not supported. Just ignoring the request");
        }
    }

    let flags = settings::Flags::new(flags_builder);
    cranelift::codegen::isa::lookup(sess.target.target.llvm_target.parse().unwrap())
        .unwrap()
        .finish(flags)
}

fn codegen_mono_items<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
    log: &mut Option<File>,
) {
    let mut caches = Caches::new();
    let mut ccx = ConstantCx::default();

    let (_, cgus) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);
    let mono_items = cgus
        .iter()
        .map(|cgu| cgu.items().iter())
        .flatten()
        .collect::<FxHashSet<(_, _)>>();

    let before = ::std::time::Instant::now();
    println!("[codegen mono items] start");

    for (&mono_item, &(_linkage, _vis)) in mono_items {
        unimpl::try_unimpl(tcx, log, || {
            base::trans_mono_item(tcx, module, &mut caches, &mut ccx, mono_item);
        });
    }

    crate::main_shim::maybe_create_entry_wrapper(tcx, module);

    let any_dynamic_crate = tcx
        .sess
        .dependency_formats
        .borrow()
        .iter()
        .any(|(_, list)| {
            use rustc::middle::dependency_format::Linkage;
            list.iter().any(|&linkage| linkage == Linkage::Dynamic)
        });
    if any_dynamic_crate {
    } else if let Some(kind) = *tcx.sess.allocator_kind.get() {
        allocator::codegen(module, kind);
    }

    ccx.finalize(tcx, module);
    module.finalize_definitions();

    let after = ::std::time::Instant::now();
    println!("[codegen mono items] end time: {:?}", after - before);
}

fn save_incremental<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    rustc_incremental::assert_dep_graph(tcx);
    rustc_incremental::save_dep_graph(tcx);
    rustc_incremental::finalize_session_directory(tcx.sess, tcx.crate_hash(LOCAL_CRATE));
}

/// This is the entrypoint for a hot plugged rustc_codegen_cranelift
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(CraneliftCodegenBackend)
}
