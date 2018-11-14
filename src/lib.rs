#![feature(
    rustc_private,
    macro_at_most_once_rep,
    never_type,
    extern_crate_item_prelude
)]
#![allow(intra_doc_link_resolution_failure)]

extern crate byteorder;
extern crate syntax;
#[macro_use]
extern crate rustc;
extern crate rustc_allocator;
extern crate rustc_codegen_utils;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_target;
#[macro_use]
extern crate rustc_data_structures;
extern crate rustc_fs_util;
#[macro_use]
extern crate log;

extern crate ar;
#[macro_use]
extern crate bitflags;
extern crate faerie;
//extern crate goblin;
extern crate cranelift;
extern crate cranelift_faerie;
extern crate cranelift_module;
extern crate cranelift_simplejit;
extern crate target_lexicon;

use std::any::Any;
use std::fs::File;
use std::io::Write;
use std::sync::mpsc;

use syntax::symbol::Symbol;

use rustc::dep_graph::DepGraph;
use rustc::middle::cstore::{
    self, CrateSource, LibSource, LinkagePreference, MetadataLoader, NativeLibrary,
};
use rustc::middle::lang_items::LangItem;
use rustc::middle::weak_lang_items;
use rustc::session::{
    config::{self, OutputFilenames, OutputType},
    CompileIncomplete,
};
use rustc::ty::query::Providers;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_codegen_utils::link::out_filename;
use rustc_codegen_utils::linker::LinkerInfo;

use cranelift::codegen::settings;
use cranelift_faerie::*;

use crate::constant::ConstantCx;
use crate::prelude::*;

struct NonFatal(pub String);

macro_rules! unimpl {
    ($($tt:tt)*) => {
        panic!(crate::NonFatal(format!($($tt)*)));
    };
}

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
mod vtable;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::{HashMap, HashSet};

    pub use syntax::ast::{FloatTy, IntTy, UintTy};
    pub use syntax::source_map::DUMMY_SP;

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
    pub use rustc_codegen_utils::{CompiledModule, ModuleKind};
    pub use rustc_data_structures::{
        fx::{FxHashMap, FxHashSet},
        indexed_vec::Idx,
        sync::Lrc,
    };
    pub use rustc_mir::monomorphize::{collector, MonoItem};

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
    pub use crate::{Caches, CodegenResults, CrateInfo};
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

pub struct CrateInfo {
    panic_runtime: Option<CrateNum>,
    compiler_builtins: Option<CrateNum>,
    profiler_runtime: Option<CrateNum>,
    sanitizer_runtime: Option<CrateNum>,
    is_no_builtins: FxHashSet<CrateNum>,
    native_libraries: FxHashMap<CrateNum, Lrc<Vec<NativeLibrary>>>,
    crate_name: FxHashMap<CrateNum, String>,
    used_libraries: Lrc<Vec<NativeLibrary>>,
    link_args: Lrc<Vec<String>>,
    used_crate_source: FxHashMap<CrateNum, Lrc<CrateSource>>,
    used_crates_static: Vec<(CrateNum, LibSource)>,
    used_crates_dynamic: Vec<(CrateNum, LibSource)>,
    wasm_imports: FxHashMap<String, String>,
    lang_item_to_crate: FxHashMap<LangItem, CrateNum>,
    missing_lang_items: FxHashMap<CrateNum, Vec<LangItem>>,
}

impl CrateInfo {
    pub fn new(tcx: TyCtxt) -> CrateInfo {
        let mut info = CrateInfo {
            panic_runtime: None,
            compiler_builtins: None,
            profiler_runtime: None,
            sanitizer_runtime: None,
            is_no_builtins: Default::default(),
            native_libraries: Default::default(),
            used_libraries: tcx.native_libraries(LOCAL_CRATE),
            link_args: tcx.link_args(LOCAL_CRATE),
            crate_name: Default::default(),
            used_crates_dynamic: cstore::used_crates(tcx, LinkagePreference::RequireDynamic),
            used_crates_static: cstore::used_crates(tcx, LinkagePreference::RequireStatic),
            used_crate_source: Default::default(),
            wasm_imports: Default::default(),
            lang_item_to_crate: Default::default(),
            missing_lang_items: Default::default(),
        };
        let lang_items = tcx.lang_items();

        let load_wasm_items = tcx
            .sess
            .crate_types
            .borrow()
            .iter()
            .any(|c| *c != config::CrateType::Rlib)
            && tcx.sess.opts.target_triple.triple() == "wasm32-unknown-unknown";

        if load_wasm_items {
            info.load_wasm_imports(tcx, LOCAL_CRATE);
        }

        let crates = tcx.crates();

        let n_crates = crates.len();
        info.native_libraries.reserve(n_crates);
        info.crate_name.reserve(n_crates);
        info.used_crate_source.reserve(n_crates);
        info.missing_lang_items.reserve(n_crates);

        for &cnum in crates.iter() {
            info.native_libraries
                .insert(cnum, tcx.native_libraries(cnum));
            info.crate_name
                .insert(cnum, tcx.crate_name(cnum).to_string());
            info.used_crate_source
                .insert(cnum, tcx.used_crate_source(cnum));
            if tcx.is_panic_runtime(cnum) {
                info.panic_runtime = Some(cnum);
            }
            if tcx.is_compiler_builtins(cnum) {
                info.compiler_builtins = Some(cnum);
            }
            if tcx.is_profiler_runtime(cnum) {
                info.profiler_runtime = Some(cnum);
            }
            if tcx.is_sanitizer_runtime(cnum) {
                info.sanitizer_runtime = Some(cnum);
            }
            if tcx.is_no_builtins(cnum) {
                info.is_no_builtins.insert(cnum);
            }
            if load_wasm_items {
                info.load_wasm_imports(tcx, cnum);
            }
            let missing = tcx.missing_lang_items(cnum);
            for &item in missing.iter() {
                if let Ok(id) = lang_items.require(item) {
                    info.lang_item_to_crate.insert(item, id.krate);
                }
            }

            // No need to look for lang items that are whitelisted and don't
            // actually need to exist.
            let missing = missing
                .iter()
                .cloned()
                .filter(|&l| !weak_lang_items::whitelisted(tcx, l))
                .collect();
            info.missing_lang_items.insert(cnum, missing);
        }

        return info;
    }

    fn load_wasm_imports(&mut self, tcx: TyCtxt, cnum: CrateNum) {
        self.wasm_imports.extend(
            tcx.wasm_import_module_map(cnum)
                .iter()
                .map(|(&id, module)| {
                    let instance = Instance::mono(tcx, id);
                    let import_name = tcx.symbol_name(instance);

                    (import_name.to_string(), module.clone())
                }),
        );
    }
}

pub struct CodegenResults {
    artifact: faerie::Artifact,
    modules: Vec<CompiledModule>,
    allocator_module: Option<CompiledModule>,
    metadata: Vec<u8>,
    crate_name: Symbol,
    crate_info: CrateInfo,
    linker_info: LinkerInfo,
}

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

    fn metadata_loader(&self) -> Box<MetadataLoader + Sync> {
        Box::new(crate::metadata::CraneliftMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {
        rustc_codegen_utils::symbol_names::provide(providers);
        rustc_codegen_utils::symbol_export::provide(providers);

        providers.target_features_whitelist = |_tcx, _cnum| Lrc::new(Default::default());
    }
    fn provide_extern(&self, providers: &mut Providers) {
        rustc_codegen_utils::symbol_export::provide_extern(providers);
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        _rx: mpsc::Receiver<Box<Any + Send>>,
    ) -> Box<Any> {
        if !tcx.sess.crate_types.get().contains(&CrateType::Executable)
            && std::env::var("SHOULD_RUN").is_ok()
        {
            tcx.sess
                .err("Can't JIT run non executable (SHOULD_RUN env var is set)");
        }

        tcx.sess.abort_if_errors();

        let metadata = tcx.encode_metadata();

        let mut flags_builder = settings::builder();
        flags_builder.enable("is_pic").unwrap();
        let flags = settings::Flags::new(flags_builder);
        let isa =
            cranelift::codegen::isa::lookup(tcx.sess.target.target.llvm_target.parse().unwrap())
                .unwrap()
                .finish(flags);

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
                artifact,
                metadata: metadata.raw_data,
                crate_name: tcx.crate_name(LOCAL_CRATE),
                crate_info: CrateInfo::new(tcx),
                linker_info: LinkerInfo::new(tcx),
                modules: vec![CompiledModule {
                    name: "dummy".to_string(),
                    kind: ModuleKind::Regular,
                    object: Some(tmp_file),
                    bytecode: None,
                    bytecode_compressed: None,
                }],
                //modules: vec![],
                allocator_module: None,
            });
        }
    }

    fn join_codegen_and_link(
        &self,
        res: Box<Any>,
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
                CrateType::Executable => link::link_bin(sess, &res, &output_name),
                _ => sess.fatal(&format!("Unsupported crate type: {:?}", crate_type)),
            }
        }
        Ok(())
    }
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
        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
            base::trans_mono_item(tcx, module, &mut caches, &mut ccx, mono_item);
        }));

        if let Err(err) = res {
            match err.downcast::<NonFatal>() {
                Ok(non_fatal) => {
                    if cfg!(debug_assertions) {
                        writeln!(log.as_mut().unwrap(), "{}", &non_fatal.0).unwrap();
                    }
                    tcx.sess.err(&non_fatal.0)
                }
                Err(err) => ::std::panic::resume_unwind(err),
            }
        }
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
pub fn __rustc_codegen_backend() -> Box<CodegenBackend> {
    Box::new(CraneliftCodegenBackend)
}
