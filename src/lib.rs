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
use std::sync::mpsc;

use rustc::dep_graph::DepGraph;
use rustc::middle::cstore::MetadataLoader;
use rustc::session::{config::OutputFilenames, CompileIncomplete};
use rustc::ty::query::Providers;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_codegen_utils::link::out_filename;
use syntax::symbol::Symbol;

use cranelift::codegen::settings;
use cranelift_faerie::*;

struct NonFatal(pub String);

macro_rules! unimpl {
    ($($tt:tt)*) => {
        panic!(crate::NonFatal(format!($($tt)*)));
    };
}

mod abi;
mod allocator;
mod analyze;
mod base;
mod common;
mod constant;
mod intrinsics;
mod main_shim;
mod metadata;
mod pretty_clif;
mod vtable;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::{HashMap, HashSet};

    pub use rustc::hir::def_id::{DefId, LOCAL_CRATE};
    pub use rustc::mir::{self, interpret::AllocId, *};
    pub use rustc::session::{config::CrateType, Session};
    pub use rustc::ty::layout::{self, Abi, LayoutOf, Scalar, Size, TyLayout};
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
    pub use syntax::ast::{FloatTy, IntTy, UintTy};
    pub use syntax::source_map::DUMMY_SP;

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
    pub use crate::Caches;
}

use crate::constant::ConstantCx;
use crate::prelude::*;

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

struct OngoingCodegen {
    product: cranelift_faerie::FaerieProduct,
    metadata: Vec<u8>,
    crate_name: Symbol,
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

        if std::env::var("SHOULD_RUN").is_ok() {
            let mut jit_module: Module<SimpleJITBackend> = Module::new(SimpleJITBuilder::new());
            assert_eq!(pointer_ty(tcx), jit_module.target_config().pointer_type());

            codegen_mono_items(tcx, &mut jit_module);

            tcx.sess.abort_if_errors();
            println!("Compiled everything");
            println!("Rustc codegen cranelift will JIT run the executable, because the SHOULD_RUN env var is set");

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

            jit_module.finalize_definitions();
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

            codegen_mono_items(tcx, &mut faerie_module);

            tcx.sess.abort_if_errors();

            faerie_module.finalize_definitions();

            return Box::new(OngoingCodegen {
                product: faerie_module.finish(),
                metadata: metadata.raw_data,
                crate_name: tcx.crate_name(LOCAL_CRATE),
            });
        }
    }

    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete> {
        let ongoing_codegen = *ongoing_codegen
            .downcast::<OngoingCodegen>()
            .expect("Expected CraneliftCodegenBackend's OngoingCodegen, found Box<Any>");

        let artifact = ongoing_codegen.product.artifact;
        let metadata = ongoing_codegen.metadata;

        /*
        artifact
            .declare_with(
                &metadata_name,
                faerie::artifact::Decl::Data {
                    global: true,
                    writable: false,
                },
                metadata.clone(),
            )
            .unwrap();
        */

        for &crate_type in sess.opts.crate_types.iter() {
            match crate_type {
                // TODO: link executable
                CrateType::Executable | CrateType::Rlib => {
                    let output_name = out_filename(
                        sess,
                        crate_type,
                        &outputs,
                        &ongoing_codegen.crate_name.as_str(),
                    );
                    let file = File::create(&output_name).unwrap();
                    let mut builder = ar::Builder::new(file);

                    // Add main object file
                    let obj = artifact.emit().unwrap();
                    builder
                        .append(
                            &ar::Header::new(b"data.o".to_vec(), obj.len() as u64),
                            ::std::io::Cursor::new(obj),
                        )
                        .unwrap();

                    // Non object files need to be added after object files, because ranlib will
                    // try to read the native architecture from the first file, even if it isn't
                    // an object file
                    if crate_type != CrateType::Executable {
                        builder
                            .append(
                                &ar::Header::new(
                                    metadata::METADATA_FILE.to_vec(),
                                    metadata.len() as u64,
                                ),
                                ::std::io::Cursor::new(metadata.clone()),
                            )
                            .unwrap();
                    }

                    // Finalize archive
                    std::mem::drop(builder);

                    // Run ranlib to be able to link the archive
                    let status = std::process::Command::new("ranlib")
                        .arg(output_name)
                        .status()
                        .expect("Couldn't run ranlib");
                    if !status.success() {
                        sess.fatal(&format!("Ranlib exited with code {:?}", status.code()));
                    }
                }
                _ => sess.fatal(&format!("Unsupported crate type: {:?}", crate_type)),
            }
        }
        Ok(())
    }
}

fn codegen_mono_items<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
) {
    use std::io::Write;

    let mut caches = Caches::new();
    let mut ccx = ConstantCx::default();

    let mut log = if cfg!(debug_assertions) {
        Some(
            ::std::fs::File::create(concat!(env!("CARGO_MANIFEST_DIR"), "/target/out/log.txt"))
                .unwrap(),
        )
    } else {
        None
    };

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
