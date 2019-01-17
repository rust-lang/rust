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
    config::{DebugInfo, OutputFilenames, OutputType},
    CompileIncomplete,
};
use rustc::ty::query::Providers;
use rustc::mir::mono::{Linkage as RLinkage, Visibility};
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
mod debuginfo;
mod intrinsics;
mod link;
mod link_copied;
mod main_shim;
mod metadata;
mod pretty_clif;
mod trap;
mod unimpl;
mod unsize;
mod vtable;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::{HashMap, HashSet};

    pub use syntax::ast::{FloatTy, IntTy, UintTy};
    pub use syntax::source_map::{DUMMY_SP, Span, Pos};

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
    pub use rustc_codegen_ssa::traits::*;

    pub use cranelift::codegen::ir::{
        condcodes::IntCC, function::Function, ExternalName, FuncRef, Inst, StackSlot, SourceLoc,
    };
    pub use cranelift::codegen::isa::CallConv;
    pub use cranelift::codegen::Context;
    pub use cranelift::prelude::*;
    pub use cranelift_module::{
        self, Backend, DataContext, DataId, FuncId, FuncOrDataId, Linkage,
        Module,
    };
    pub use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

    pub use crate::abi::*;
    pub use crate::base::{trans_operand, trans_place};
    pub use crate::common::*;
    pub use crate::debuginfo::{DebugContext, FunctionDebugContext};
    pub use crate::trap::*;
    pub use crate::unimpl::{unimpl, with_unimpl_span};
    pub use crate::{Caches, CodegenCx};
}

pub struct Caches<'tcx> {
    pub context: Context,
    pub vtables: HashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), DataId>,
}

impl<'tcx> Default for Caches<'tcx> {
    fn default() -> Self {
        Caches {
            context: Context::new(),
            vtables: HashMap::new(),
        }
    }
}

pub struct CodegenCx<'a, 'clif, 'tcx, B: Backend + 'static> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &'clif mut Module<B>,
    ccx: ConstantCx,
    caches: Caches<'tcx>,
    debug_context: Option<&'clif mut DebugContext<'tcx>>,
}

impl<'a, 'clif, 'tcx, B: Backend + 'static> CodegenCx<'a, 'clif, 'tcx, B> {
    fn new(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        module: &'clif mut Module<B>,
        debug_context: Option<&'clif mut DebugContext<'tcx>>,
    ) -> Self {
        CodegenCx {
            tcx,
            module,
            ccx: ConstantCx::default(),
            caches: Caches::default(),
            debug_context,
        }
    }

    fn finalize(self) {
        self.ccx.finalize(self.tcx, self.module);
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
        env_logger::init();
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

            codegen_cgus(tcx, &mut jit_module, &mut None, &mut log);
            crate::allocator::codegen(tcx.sess, &mut jit_module);
            jit_module.finalize_definitions();

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
            let new_module = |name: String| {
                let module: Module<FaerieBackend> = Module::new(
                    FaerieBuilder::new(
                        build_isa(tcx.sess),
                        name + ".o",
                        FaerieTrapCollection::Disabled,
                        FaerieBuilder::default_libcall_names(),
                    )
                    .unwrap(),
                );
                assert_eq!(
                    pointer_ty(tcx),
                    module.target_config().pointer_type()
                );
                module
            };

            let emit_module = |name: &str, kind: ModuleKind, mut module: Module<FaerieBackend>, debug: Option<DebugContext>| {
                module.finalize_definitions();
                let mut artifact = module.finish().artifact;

                if let Some(mut debug) = debug {
                    debug.emit(&mut artifact);
                }

                let tmp_file = tcx
                    .output_filenames(LOCAL_CRATE)
                    .temp_path(OutputType::Object, Some(name));
                let obj = artifact.emit().unwrap();
                std::fs::write(&tmp_file, obj).unwrap();
                CompiledModule {
                    name: name.to_string(),
                    kind,
                    object: Some(tmp_file),
                    bytecode: None,
                    bytecode_compressed: None,
                }
            };

            let mut faerie_module = new_module("some_file".to_string());

            let mut debug = if tcx.sess.opts.debuginfo != DebugInfo::None {
                let debug = DebugContext::new(tcx, faerie_module.target_config().pointer_type().bytes() as u8);
                Some(debug)
            } else {
                None
            };

            codegen_cgus(tcx, &mut faerie_module, &mut debug, &mut log);

            tcx.sess.abort_if_errors();

            let mut allocator_module = new_module("allocator_shim.o".to_string());
            let created_alloc_shim =
                crate::allocator::codegen(tcx.sess, &mut allocator_module);

            return Box::new(CodegenResults {
                crate_name: tcx.crate_name(LOCAL_CRATE),
                modules: vec![emit_module("dummy_name", ModuleKind::Regular, faerie_module, debug)],
                allocator_module: if created_alloc_shim {
                    Some(emit_module("allocator_shim", ModuleKind::Allocator, allocator_module, None))
                } else {
                    None
                },
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
    flags_builder.set("probestack_enabled", "false").unwrap(); // ___cranelift_probestack is not provided
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

fn codegen_cgus<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
    debug: &mut Option<DebugContext<'tcx>>,
    log: &mut Option<File>,
) {
    let (_, cgus) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);
    let mono_items = cgus
        .iter()
        .map(|cgu| cgu.items().iter())
        .flatten()
        .map(|(&mono_item, &(linkage, vis))| (mono_item, (linkage, vis)))
        .collect::<FxHashMap<_, (_, _)>>();

    codegen_mono_items(tcx, module, debug.as_mut(), log, mono_items);

    crate::main_shim::maybe_create_entry_wrapper(tcx, module);
}

fn codegen_mono_items<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
    debug_context: Option<&mut DebugContext<'tcx>>,
    log: &mut Option<File>,
    mono_items: FxHashMap<MonoItem<'tcx>, (RLinkage, Visibility)>,
) {
    let mut cx = CodegenCx::new(tcx, module, debug_context);
    time("codegen mono items", move || {
        for (mono_item, (linkage, vis)) in mono_items {
            unimpl::try_unimpl(tcx, log, || {
                let linkage = match (linkage, vis) {
                    (RLinkage::External, Visibility::Default) => Linkage::Export,
                    (RLinkage::Internal, Visibility::Default) => Linkage::Local,
                    // FIXME this should get external linkage, but hidden visibility,
                    // not internal linkage and default visibility
                    | (RLinkage::External, Visibility::Hidden) => Linkage::Local,
                    _ => panic!("{:?} = {:?} {:?}", mono_item, linkage, vis),
                };
                base::trans_mono_item(&mut cx, mono_item, linkage);
            });
        }

        cx.finalize();
    });
}

fn time<R>(name: &str, f: impl FnOnce() -> R) -> R {
    println!("[{}] start", name);
    let before = ::std::time::Instant::now();
    let res = f();
    let after = ::std::time::Instant::now();
    println!("[{}] end time: {:?}", name, after - before);
    res
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
