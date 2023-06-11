/*
 * TODO(antoyo): implement equality in libgccjit based on https://zpz.github.io/blog/overloading-equality-operator-in-cpp-class-hierarchy/ (for type equality?)
 * TODO(antoyo): support #[inline] attributes.
 * TODO(antoyo): support LTO (gcc's equivalent to Full LTO is -flto -flto-partition=one — https://documentation.suse.com/sbp/all/html/SBP-GCC-10/index.html).
 *
 * TODO(antoyo): remove the patches.
 */

#![feature(
    rustc_private,
    decl_macro,
    associated_type_bounds,
    never_type,
    trusted_len,
    hash_raw_entry
)]
#![allow(broken_intra_doc_links)]
#![recursion_limit="256"]
#![warn(rust_2018_idioms)]
#![warn(unused_lifetimes)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

extern crate rustc_apfloat;
extern crate rustc_ast;
extern crate rustc_attr;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_fluent_macro;
extern crate rustc_hir;
extern crate rustc_macros;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate tempfile;

// This prevents duplicating functions and statics that are already part of the host rustc process.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

mod abi;
mod allocator;
mod archive;
mod asm;
mod attributes;
mod back;
mod base;
mod builder;
mod callee;
mod common;
mod consts;
mod context;
mod coverageinfo;
mod debuginfo;
mod declare;
mod errors;
mod int;
mod intrinsic;
mod mono_item;
mod type_;
mod type_of;

use std::any::Any;
use std::sync::{Arc, Mutex};

use crate::errors::LTONotSupported;
use gccjit::{Context, OptimizationLevel, CType};
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleCodegen};
use rustc_codegen_ssa::base::codegen_crate;
use rustc_codegen_ssa::back::write::{CodegenContext, FatLTOInput, ModuleConfig, TargetMachineFactoryFn};
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule};
use rustc_codegen_ssa::target_features::supported_target_features;
use rustc_codegen_ssa::traits::{CodegenBackend, ExtraBackendMethods, ModuleBufferMethods, ThinBufferMethods, WriteBackendMethods};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::{DiagnosticMessage, ErrorGuaranteed, Handler, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{Lto, OptLevel, OutputFilenames};
use rustc_session::Session;
use rustc_span::Symbol;
use rustc_span::fatal_error::FatalError;
use tempfile::TempDir;

fluent_messages! { "../messages.ftl" }

pub struct PrintOnPanic<F: Fn() -> String>(pub F);

impl<F: Fn() -> String> Drop for PrintOnPanic<F> {
    fn drop(&mut self) {
        if ::std::thread::panicking() {
            println!("{}", (self.0)());
        }
    }
}

#[derive(Clone)]
pub struct GccCodegenBackend {
    supports_128bit_integers: Arc<Mutex<bool>>,
}

impl CodegenBackend for GccCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        crate::DEFAULT_LOCALE_RESOURCE
    }

    fn init(&self, sess: &Session) {
        if sess.lto() != Lto::No {
            sess.emit_warning(LTONotSupported {});
        }

        let temp_dir = TempDir::new().expect("cannot create temporary directory");
        let temp_file = temp_dir.into_path().join("result.asm");
        let check_context = Context::default();
        check_context.set_print_errors_to_stderr(false);
        let _int128_ty = check_context.new_c_type(CType::UInt128t);
        // NOTE: we cannot just call compile() as this would require other files than libgccjit.so.
        check_context.compile_to_file(gccjit::OutputKind::Assembler, temp_file.to_str().expect("path to str"));
        *self.supports_128bit_integers.lock().expect("lock") = check_context.get_last_error() == Ok(None);
    }

    fn provide(&self, providers: &mut Providers) {
        // FIXME(antoyo) compute list of enabled features from cli flags
        providers.global_backend_features = |_tcx, ()| vec![];
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>, metadata: EncodedMetadata, need_metadata_module: bool) -> Box<dyn Any> {
        let target_cpu = target_cpu(tcx.sess);
        let res = codegen_crate(self.clone(), tcx, target_cpu.to_string(), metadata, need_metadata_module);

        Box::new(res)
    }

    fn join_codegen(&self, ongoing_codegen: Box<dyn Any>, sess: &Session, _outputs: &OutputFilenames) -> Result<(CodegenResults, FxIndexMap<WorkProductId, WorkProduct>), ErrorGuaranteed> {
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<GccCodegenBackend>>()
            .expect("Expected GccCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess);

        Ok((codegen_results, work_products))
    }

    fn link(&self, sess: &Session, codegen_results: CodegenResults, outputs: &OutputFilenames) -> Result<(), ErrorGuaranteed> {
        use rustc_codegen_ssa::back::link::link_binary;

        link_binary(
            sess,
            &crate::archive::ArArchiveBuilderBuilder,
            &codegen_results,
            outputs,
        )
    }

    fn target_features(&self, sess: &Session, allow_unstable: bool) -> Vec<Symbol> {
        target_features(sess, allow_unstable)
    }
}

impl ExtraBackendMethods for GccCodegenBackend {
    fn codegen_allocator<'tcx>(&self, tcx: TyCtxt<'tcx>, module_name: &str, kind: AllocatorKind, alloc_error_handler_kind: AllocatorKind) -> Self::Module {
        let mut mods = GccContext {
            context: Context::default(),
        };
        unsafe { allocator::codegen(tcx, &mut mods, module_name, kind, alloc_error_handler_kind); }
        mods
    }

    fn compile_codegen_unit(&self, tcx: TyCtxt<'_>, cgu_name: Symbol) -> (ModuleCodegen<Self::Module>, u64) {
        base::compile_codegen_unit(tcx, cgu_name, *self.supports_128bit_integers.lock().expect("lock"))
    }

    fn target_machine_factory(&self, _sess: &Session, _opt_level: OptLevel, _features: &[String]) -> TargetMachineFactoryFn<Self> {
        // TODO(antoyo): set opt level.
        Arc::new(|_| {
            Ok(())
        })
    }
}

pub struct ModuleBuffer;

impl ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        unimplemented!();
    }
}

pub struct ThinBuffer;

impl ThinBufferMethods for ThinBuffer {
    fn data(&self) -> &[u8] {
        unimplemented!();
    }
}

pub struct GccContext {
    context: Context<'static>,
}

unsafe impl Send for GccContext {}
// FIXME(antoyo): that shouldn't be Sync. Parallel compilation is currently disabled with "-Zno-parallel-llvm". Try to disable it here.
unsafe impl Sync for GccContext {}

impl WriteBackendMethods for GccCodegenBackend {
    type Module = GccContext;
    type TargetMachine = ();
    type TargetMachineError = ();
    type ModuleBuffer = ModuleBuffer;
    type ThinData = ();
    type ThinBuffer = ThinBuffer;

    fn run_fat_lto(_cgcx: &CodegenContext<Self>, mut modules: Vec<FatLTOInput<Self>>, _cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>) -> Result<LtoModuleCodegen<Self>, FatalError> {
        // TODO(antoyo): implement LTO by sending -flto to libgccjit and adding the appropriate gcc linker plugins.
        // NOTE: implemented elsewhere.
        // TODO(antoyo): what is implemented elsewhere ^ ?
        let module =
            match modules.remove(0) {
                FatLTOInput::InMemory(module) => module,
                FatLTOInput::Serialized { .. } => {
                    unimplemented!();
                }
            };
        Ok(LtoModuleCodegen::Fat { module, _serialized_bitcode: vec![] })
    }

    fn run_thin_lto(_cgcx: &CodegenContext<Self>, _modules: Vec<(String, Self::ThinBuffer)>, _cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError> {
        unimplemented!();
    }

    fn print_pass_timings(&self) {
        unimplemented!();
    }

    unsafe fn optimize(_cgcx: &CodegenContext<Self>, _diag_handler: &Handler, module: &ModuleCodegen<Self::Module>, config: &ModuleConfig) -> Result<(), FatalError> {
        module.module_llvm.context.set_optimization_level(to_gcc_opt_level(config.opt_level));
        Ok(())
    }

    fn optimize_fat(_cgcx: &CodegenContext<Self>, _module: &mut ModuleCodegen<Self::Module>) -> Result<(), FatalError> {
        // TODO(antoyo)
        Ok(())
    }

    unsafe fn optimize_thin(_cgcx: &CodegenContext<Self>, _thin: ThinModule<Self>) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        unimplemented!();
    }

    unsafe fn codegen(cgcx: &CodegenContext<Self>, diag_handler: &Handler, module: ModuleCodegen<Self::Module>, config: &ModuleConfig) -> Result<CompiledModule, FatalError> {
        back::write::codegen(cgcx, diag_handler, module, config)
    }

    fn prepare_thin(_module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        unimplemented!();
    }

    fn serialize_module(_module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        unimplemented!();
    }

    fn run_link(cgcx: &CodegenContext<Self>, diag_handler: &Handler, modules: Vec<ModuleCodegen<Self::Module>>) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::write::link(cgcx, diag_handler, modules)
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_gccjit
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(GccCodegenBackend {
        supports_128bit_integers: Arc::new(Mutex::new(false)),
    })
}

fn to_gcc_opt_level(optlevel: Option<OptLevel>) -> OptimizationLevel {
    match optlevel {
        None => OptimizationLevel::None,
        Some(level) => {
            match level {
                OptLevel::No => OptimizationLevel::None,
                OptLevel::Less => OptimizationLevel::Limited,
                OptLevel::Default => OptimizationLevel::Standard,
                OptLevel::Aggressive => OptimizationLevel::Aggressive,
                OptLevel::Size | OptLevel::SizeMin => OptimizationLevel::Limited,
            }
        },
    }
}

fn handle_native(name: &str) -> &str {
    if name != "native" {
        return name;
    }

    unimplemented!();
}

pub fn target_cpu(sess: &Session) -> &str {
    match sess.opts.cg.target_cpu {
        Some(ref name) => handle_native(name),
        None => handle_native(sess.target.cpu.as_ref()),
    }
}

pub fn target_features(sess: &Session, allow_unstable: bool) -> Vec<Symbol> {
    supported_target_features(sess)
        .iter()
        .filter_map(
            |&(feature, gate)| {
                if sess.is_nightly_build() || allow_unstable || gate.is_none() { Some(feature) } else { None }
            },
        )
        .filter(|_feature| {
            // TODO(antoyo): implement a way to get enabled feature in libgccjit.
            // Probably using the equivalent of __builtin_cpu_supports.
            // TODO(antoyo): maybe use whatever outputs the following command:
            // gcc -march=native -Q --help=target
            #[cfg(feature="master")]
            {
                // NOTE: the CPU in the CI doesn't support sse4a, so disable it to make the stdarch tests pass in the CI.
                (_feature.contains("sse") || _feature.contains("avx")) && !_feature.contains("avx512") && !_feature.contains("sse4a")
            }
            #[cfg(not(feature="master"))]
            {
                false
            }
            /*
               adx, aes, avx, avx2, avx512bf16, avx512bitalg, avx512bw, avx512cd, avx512dq, avx512er, avx512f, avx512ifma,
               avx512pf, avx512vbmi, avx512vbmi2, avx512vl, avx512vnni, avx512vp2intersect, avx512vpopcntdq,
               bmi1, bmi2, cmpxchg16b, ermsb, f16c, fma, fxsr, gfni, lzcnt, movbe, pclmulqdq, popcnt, rdrand, rdseed, rtm,
               sha, sse, sse2, sse3, sse4.1, sse4.2, sse4a, ssse3, tbm, vaes, vpclmulqdq, xsave, xsavec, xsaveopt, xsaves
             */
            //false
        })
        .map(|feature| Symbol::intern(feature))
        .collect()
}
