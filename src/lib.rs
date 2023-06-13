#![feature(rustc_private)]
// Note: please avoid adding other feature gates where possible
#![warn(rust_2018_idioms)]
#![warn(unused_lifetimes)]
#![warn(unreachable_pub)]

extern crate jobserver;
#[macro_use]
extern crate rustc_middle;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_fs_util;
extern crate rustc_hir;
extern crate rustc_incremental;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

// This prevents duplicating functions and statics that are already part of the host rustc process.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::sync::Arc;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::CodegenResults;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::ErrorGuaranteed;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::config::OutputFilenames;
use rustc_session::Session;
use rustc_span::Symbol;

use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};

pub use crate::config::*;
use crate::prelude::*;

mod abi;
mod allocator;
mod analyze;
mod archive;
mod base;
mod cast;
mod codegen_i128;
mod common;
mod compiler_builtins;
mod concurrency_limiter;
mod config;
mod constant;
mod debuginfo;
mod discriminant;
mod driver;
mod global_asm;
mod inline_asm;
mod intrinsics;
mod linkage;
mod main_shim;
mod num;
mod optimize;
mod pointer;
mod pretty_clif;
mod toolchain;
mod trap;
mod unsize;
mod value_and_place;
mod vtable;

mod prelude {
    pub(crate) use rustc_span::{FileNameDisplayPreference, Span};

    pub(crate) use rustc_hir::def_id::{DefId, LOCAL_CRATE};
    pub(crate) use rustc_middle::bug;
    pub(crate) use rustc_middle::mir::{self, *};
    pub(crate) use rustc_middle::ty::layout::{self, LayoutOf, TyAndLayout};
    pub(crate) use rustc_middle::ty::{
        self, FloatTy, Instance, InstanceDef, IntTy, ParamEnv, Ty, TyCtxt, TypeAndMut,
        TypeFoldable, TypeVisitableExt, UintTy,
    };
    pub(crate) use rustc_target::abi::{Abi, FieldIdx, Scalar, Size, VariantIdx, FIRST_VARIANT};

    pub(crate) use rustc_data_structures::fx::{FxHashMap, FxIndexMap};

    pub(crate) use rustc_index::Idx;

    pub(crate) use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    pub(crate) use cranelift_codegen::ir::function::Function;
    pub(crate) use cranelift_codegen::ir::types;
    pub(crate) use cranelift_codegen::ir::{
        AbiParam, Block, FuncRef, Inst, InstBuilder, MemFlags, Signature, SourceLoc, StackSlot,
        StackSlotData, StackSlotKind, TrapCode, Type, Value,
    };
    pub(crate) use cranelift_codegen::isa::{self, CallConv};
    pub(crate) use cranelift_codegen::Context;
    pub(crate) use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    pub(crate) use cranelift_module::{self, DataDescription, FuncId, Linkage, Module};

    pub(crate) use crate::abi::*;
    pub(crate) use crate::base::{codegen_operand, codegen_place};
    pub(crate) use crate::cast::*;
    pub(crate) use crate::common::*;
    pub(crate) use crate::debuginfo::{DebugContext, UnwindContext};
    pub(crate) use crate::pointer::Pointer;
    pub(crate) use crate::value_and_place::{CPlace, CValue};
}

struct PrintOnPanic<F: Fn() -> String>(F);
impl<F: Fn() -> String> Drop for PrintOnPanic<F> {
    fn drop(&mut self) {
        if ::std::thread::panicking() {
            println!("{}", (self.0)());
        }
    }
}

/// The codegen context holds any information shared between the codegen of individual functions
/// inside a single codegen unit with the exception of the Cranelift [`Module`](cranelift_module::Module).
struct CodegenCx {
    profiler: SelfProfilerRef,
    output_filenames: Arc<OutputFilenames>,
    should_write_ir: bool,
    global_asm: String,
    inline_asm_index: Cell<usize>,
    debug_context: Option<DebugContext>,
    unwind_context: UnwindContext,
    cgu_name: Symbol,
}

impl CodegenCx {
    fn new(
        tcx: TyCtxt<'_>,
        backend_config: BackendConfig,
        isa: &dyn TargetIsa,
        debug_info: bool,
        cgu_name: Symbol,
    ) -> Self {
        assert_eq!(pointer_ty(tcx), isa.pointer_type());

        let unwind_context =
            UnwindContext::new(isa, matches!(backend_config.codegen_mode, CodegenMode::Aot));
        let debug_context = if debug_info && !tcx.sess.target.options.is_like_windows {
            Some(DebugContext::new(tcx, isa))
        } else {
            None
        };
        CodegenCx {
            profiler: tcx.prof.clone(),
            output_filenames: tcx.output_filenames(()).clone(),
            should_write_ir: crate::pretty_clif::should_write_ir(tcx),
            global_asm: String::new(),
            inline_asm_index: Cell::new(0),
            debug_context,
            unwind_context,
            cgu_name,
        }
    }
}

pub struct CraneliftCodegenBackend {
    pub config: RefCell<Option<BackendConfig>>,
}

impl CodegenBackend for CraneliftCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        // FIXME(rust-lang/rust#100717) - cranelift codegen backend is not yet translated
        ""
    }

    fn init(&self, sess: &Session) {
        use rustc_session::config::Lto;
        match sess.lto() {
            Lto::No | Lto::ThinLocal => {}
            Lto::Thin | Lto::Fat => sess.warn("LTO is not supported. You may get a linker error."),
        }

        let mut config = self.config.borrow_mut();
        if config.is_none() {
            let new_config = BackendConfig::from_opts(&sess.opts.cg.llvm_args)
                .unwrap_or_else(|err| sess.fatal(err));
            *config = Some(new_config);
        }
    }

    fn target_features(&self, _sess: &Session, _allow_unstable: bool) -> Vec<rustc_span::Symbol> {
        vec![]
    }

    fn print_version(&self) {
        println!("Cranelift version: {}", cranelift_codegen::VERSION);
    }

    fn codegen_crate(
        &self,
        tcx: TyCtxt<'_>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any> {
        tcx.sess.abort_if_errors();
        let config = self.config.borrow().clone().unwrap();
        match config.codegen_mode {
            CodegenMode::Aot => driver::aot::run_aot(tcx, config, metadata, need_metadata_module),
            CodegenMode::Jit | CodegenMode::JitLazy => {
                #[cfg(feature = "jit")]
                driver::jit::run_jit(tcx, config);

                #[cfg(not(feature = "jit"))]
                tcx.sess.fatal("jit support was disabled when compiling rustc_codegen_cranelift");
            }
        }
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> Result<(CodegenResults, FxIndexMap<WorkProductId, WorkProduct>), ErrorGuaranteed> {
        Ok(ongoing_codegen
            .downcast::<driver::aot::OngoingCodegen>()
            .unwrap()
            .join(sess, self.config.borrow().as_ref().unwrap()))
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorGuaranteed> {
        use rustc_codegen_ssa::back::link::link_binary;

        link_binary(sess, &crate::archive::ArArchiveBuilderBuilder, &codegen_results, outputs)
    }
}

fn target_triple(sess: &Session) -> target_lexicon::Triple {
    match sess.target.llvm_target.parse() {
        Ok(triple) => triple,
        Err(err) => sess.fatal(format!("target not recognized: {}", err)),
    }
}

fn build_isa(sess: &Session, backend_config: &BackendConfig) -> Arc<dyn isa::TargetIsa + 'static> {
    use target_lexicon::BinaryFormat;

    let target_triple = crate::target_triple(sess);

    let mut flags_builder = settings::builder();
    flags_builder.enable("is_pic").unwrap();
    let enable_verifier = if backend_config.enable_verifier { "true" } else { "false" };
    flags_builder.set("enable_verifier", enable_verifier).unwrap();
    flags_builder.set("regalloc_checker", enable_verifier).unwrap();

    let tls_model = match target_triple.binary_format {
        BinaryFormat::Elf => "elf_gd",
        BinaryFormat::Macho => "macho",
        BinaryFormat::Coff => "coff",
        _ => "none",
    };
    flags_builder.set("tls_model", tls_model).unwrap();

    flags_builder.set("enable_simd", "true").unwrap();

    flags_builder.set("enable_llvm_abi_extensions", "true").unwrap();

    use rustc_session::config::OptLevel;
    match sess.opts.optimize {
        OptLevel::No => {
            flags_builder.set("opt_level", "none").unwrap();
        }
        OptLevel::Less | OptLevel::Default => {}
        OptLevel::Size | OptLevel::SizeMin | OptLevel::Aggressive => {
            flags_builder.set("opt_level", "speed_and_size").unwrap();
        }
    }

    if let target_lexicon::Architecture::Aarch64(_)
    | target_lexicon::Architecture::Riscv64(_)
    | target_lexicon::Architecture::X86_64 = target_triple.architecture
    {
        // Windows depends on stack probes to grow the committed part of the stack.
        // On other platforms it helps prevents stack smashing.
        flags_builder.enable("enable_probestack").unwrap();
        flags_builder.set("probestack_strategy", "inline").unwrap();
    } else {
        // __cranelift_probestack is not provided and inline stack probes are only supported on
        // AArch64, Riscv64 and x86_64.
        flags_builder.set("enable_probestack", "false").unwrap();
    }

    let flags = settings::Flags::new(flags_builder);

    let isa_builder = match sess.opts.cg.target_cpu.as_deref() {
        Some("native") => {
            let builder = cranelift_native::builder_with_options(true).unwrap();
            builder
        }
        Some(value) => {
            let mut builder =
                cranelift_codegen::isa::lookup(target_triple.clone()).unwrap_or_else(|err| {
                    sess.fatal(format!("can't compile for {}: {}", target_triple, err));
                });
            if let Err(_) = builder.enable(value) {
                sess.fatal("the specified target cpu isn't currently supported by Cranelift.");
            }
            builder
        }
        None => {
            let mut builder =
                cranelift_codegen::isa::lookup(target_triple.clone()).unwrap_or_else(|err| {
                    sess.fatal(format!("can't compile for {}: {}", target_triple, err));
                });
            if target_triple.architecture == target_lexicon::Architecture::X86_64 {
                // Don't use "haswell" as the default, as it implies `has_lzcnt`.
                // macOS CI is still at Ivy Bridge EP, so `lzcnt` is interpreted as `bsr`.
                builder.enable("nehalem").unwrap();
            }
            builder
        }
    };

    match isa_builder.finish(flags) {
        Ok(target_isa) => target_isa,
        Err(err) => sess.fatal(format!("failed to build TargetIsa: {}", err)),
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_cranelift
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(CraneliftCodegenBackend { config: RefCell::new(None) })
}
