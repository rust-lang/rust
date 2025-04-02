// tidy-alphabetical-start
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![cfg_attr(doc, allow(internal_features))]
#![cfg_attr(doc, doc(rust_logo))]
#![cfg_attr(doc, feature(rustdoc_internals))]
// Note: please avoid adding other feature gates where possible
#![feature(rustc_private)]
// Note: please avoid adding other feature gates where possible
#![warn(rust_2018_idioms)]
#![warn(unreachable_pub)]
#![warn(unused_lifetimes)]
// tidy-alphabetical-end

#[macro_use]
extern crate rustc_middle;
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_fs_util;
extern crate rustc_hir;
extern crate rustc_incremental;
extern crate rustc_index;
extern crate rustc_metadata;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;
#[macro_use]
extern crate tracing;

// This prevents duplicating functions and statics that are already part of the host rustc process.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use std::any::Any;
use std::env;
use std::sync::Arc;

use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use rustc_codegen_ssa::CodegenResults;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::Session;
use rustc_session::config::OutputFilenames;
use rustc_span::{Symbol, sym};

pub use crate::config::*;
use crate::prelude::*;

mod abi;
mod allocator;
mod analyze;
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
mod unwind_module;
mod value_and_place;
mod vtable;

mod prelude {
    pub(crate) use cranelift_codegen::Context;
    pub(crate) use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    pub(crate) use cranelift_codegen::ir::function::Function;
    pub(crate) use cranelift_codegen::ir::{
        AbiParam, Block, FuncRef, Inst, InstBuilder, MemFlags, Signature, SourceLoc, StackSlot,
        StackSlotData, StackSlotKind, TrapCode, Type, Value, types,
    };
    pub(crate) use cranelift_module::{self, DataDescription, FuncId, Linkage, Module};
    pub(crate) use rustc_abi::{BackendRepr, FIRST_VARIANT, FieldIdx, Scalar, Size, VariantIdx};
    pub(crate) use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
    pub(crate) use rustc_hir::def_id::{DefId, LOCAL_CRATE};
    pub(crate) use rustc_index::Idx;
    pub(crate) use rustc_middle::mir::{self, *};
    pub(crate) use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
    pub(crate) use rustc_middle::ty::{
        self, FloatTy, Instance, InstanceKind, IntTy, Ty, TyCtxt, UintTy,
    };
    pub(crate) use rustc_span::Span;

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
    output_filenames: Arc<OutputFilenames>,
    should_write_ir: bool,
    global_asm: String,
    inline_asm_index: usize,
    debug_context: Option<DebugContext>,
    cgu_name: Symbol,
}

impl CodegenCx {
    fn new(tcx: TyCtxt<'_>, isa: &dyn TargetIsa, debug_info: bool, cgu_name: Symbol) -> Self {
        assert_eq!(pointer_ty(tcx), isa.pointer_type());

        let debug_context = if debug_info && !tcx.sess.target.options.is_like_windows {
            Some(DebugContext::new(tcx, isa, cgu_name.as_str()))
        } else {
            None
        };
        CodegenCx {
            output_filenames: tcx.output_filenames(()).clone(),
            should_write_ir: crate::pretty_clif::should_write_ir(tcx),
            global_asm: String::new(),
            inline_asm_index: 0,
            debug_context,
            cgu_name,
        }
    }
}

pub struct CraneliftCodegenBackend {
    pub config: Option<BackendConfig>,
}

impl CodegenBackend for CraneliftCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        // FIXME(rust-lang/rust#100717) - cranelift codegen backend is not yet translated
        ""
    }

    fn init(&self, sess: &Session) {
        use rustc_session::config::{InstrumentCoverage, Lto};
        match sess.lto() {
            Lto::No | Lto::ThinLocal => {}
            Lto::Thin | Lto::Fat => {
                sess.dcx().warn("LTO is not supported. You may get a linker error.")
            }
        }

        if sess.opts.cg.instrument_coverage() != InstrumentCoverage::No {
            sess.dcx()
                .fatal("`-Cinstrument-coverage` is LLVM specific and not supported by Cranelift");
        }
    }

    fn target_features_cfg(&self, sess: &Session) -> (Vec<Symbol>, Vec<Symbol>) {
        // FIXME return the actually used target features. this is necessary for #[cfg(target_feature)]
        let target_features = if sess.target.arch == "x86_64" && sess.target.os != "none" {
            // x86_64 mandates SSE2 support and rustc requires the x87 feature to be enabled
            vec![sym::fsxr, sym::sse, sym::sse2, Symbol::intern("x87")]
        } else if sess.target.arch == "aarch64" {
            match &*sess.target.os {
                "none" => vec![],
                // On macOS the aes, sha2 and sha3 features are enabled by default and ring
                // fails to compile on macOS when they are not present.
                "macos" => vec![sym::neon, sym::aes, sym::sha2, sym::sha3],
                // AArch64 mandates Neon support
                _ => vec![sym::neon],
            }
        } else {
            vec![]
        };
        // FIXME do `unstable_target_features` properly
        let unstable_target_features = target_features.clone();
        (target_features, unstable_target_features)
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
        info!("codegen crate {}", tcx.crate_name(LOCAL_CRATE));
        let config = self.config.clone().unwrap_or_else(|| {
            BackendConfig::from_opts(&tcx.sess.opts.cg.llvm_args)
                .unwrap_or_else(|err| tcx.sess.dcx().fatal(err))
        });
        if config.jit_mode {
            #[cfg(feature = "jit")]
            driver::jit::run_jit(tcx, config.jit_args);

            #[cfg(not(feature = "jit"))]
            tcx.dcx().fatal("jit support was disabled when compiling rustc_codegen_cranelift");
        } else {
            driver::aot::run_aot(tcx, metadata, need_metadata_module)
        }
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        ongoing_codegen.downcast::<driver::aot::OngoingCodegen>().unwrap().join(sess, outputs)
    }
}

/// Determine if the Cranelift ir verifier should run.
///
/// Returns true when `-Zverify-llvm-ir` is passed, the `CG_CLIF_ENABLE_VERIFIER` env var is set to
/// 1 or when cg_clif is compiled with debug assertions enabled or false otherwise.
fn enable_verifier(sess: &Session) -> bool {
    sess.verify_llvm_ir()
        || cfg!(debug_assertions)
        || env::var("CG_CLIF_ENABLE_VERIFIER").as_deref() == Ok("1")
}

fn target_triple(sess: &Session) -> target_lexicon::Triple {
    match sess.target.llvm_target.parse() {
        Ok(triple) => triple,
        Err(err) => sess.dcx().fatal(format!("target not recognized: {}", err)),
    }
}

fn build_isa(sess: &Session, jit: bool) -> Arc<dyn TargetIsa + 'static> {
    use target_lexicon::BinaryFormat;

    let target_triple = crate::target_triple(sess);

    let mut flags_builder = settings::builder();
    flags_builder.set("is_pic", if jit { "false" } else { "true" }).unwrap();
    let enable_verifier = if enable_verifier(sess) { "true" } else { "false" };
    flags_builder.set("enable_verifier", enable_verifier).unwrap();
    flags_builder.set("regalloc_checker", enable_verifier).unwrap();

    let mut frame_ptr = sess.target.options.frame_pointer.clone();
    frame_ptr.ratchet(sess.opts.cg.force_frame_pointers);
    let preserve_frame_pointer = frame_ptr != rustc_target::spec::FramePointer::MayOmit;
    flags_builder
        .set("preserve_frame_pointers", if preserve_frame_pointer { "true" } else { "false" })
        .unwrap();

    let tls_model = match target_triple.binary_format {
        BinaryFormat::Elf => "elf_gd",
        BinaryFormat::Macho => "macho",
        BinaryFormat::Coff => "coff",
        _ => "none",
    };
    flags_builder.set("tls_model", tls_model).unwrap();

    flags_builder.set("enable_llvm_abi_extensions", "true").unwrap();

    use rustc_session::config::OptLevel;
    match sess.opts.optimize {
        OptLevel::No => {
            flags_builder.set("opt_level", "none").unwrap();
        }
        OptLevel::Less
        | OptLevel::More
        | OptLevel::Size
        | OptLevel::SizeMin
        | OptLevel::Aggressive => {
            flags_builder.set("opt_level", "speed_and_size").unwrap();
        }
    }

    if let target_lexicon::OperatingSystem::Windows = target_triple.operating_system {
        // FIXME remove dependency on this from the Rust ABI. cc bytecodealliance/wasmtime#9510
        flags_builder.enable("enable_multi_ret_implicit_sret").unwrap();
    }

    if let target_lexicon::Architecture::S390x = target_triple.architecture {
        // FIXME remove dependency on this from the Rust ABI. cc bytecodealliance/wasmtime#9510
        flags_builder.enable("enable_multi_ret_implicit_sret").unwrap();
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
        Some("native") => cranelift_native::builder_with_options(true).unwrap(),
        Some(value) => {
            let mut builder =
                cranelift_codegen::isa::lookup(target_triple.clone()).unwrap_or_else(|err| {
                    sess.dcx().fatal(format!("can't compile for {}: {}", target_triple, err));
                });
            if builder.enable(value).is_err() {
                sess.dcx()
                    .fatal("the specified target cpu isn't currently supported by Cranelift.");
            }
            builder
        }
        None => {
            let mut builder =
                cranelift_codegen::isa::lookup(target_triple.clone()).unwrap_or_else(|err| {
                    sess.dcx().fatal(format!("can't compile for {}: {}", target_triple, err));
                });
            if target_triple.architecture == target_lexicon::Architecture::X86_64 {
                // Only set the target cpu on x86_64 as Cranelift is missing
                // the target cpu list for most other targets.
                builder.enable(sess.target.cpu.as_ref()).unwrap();
            }
            builder
        }
    };

    match isa_builder.finish(flags) {
        Ok(target_isa) => target_isa,
        Err(err) => sess.dcx().fatal(format!("failed to build TargetIsa: {}", err)),
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_cranelift
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(CraneliftCodegenBackend { config: None })
}
