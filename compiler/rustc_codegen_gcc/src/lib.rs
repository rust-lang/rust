/*
 * TODO(antoyo): implement equality in libgccjit based on https://zpz.github.io/blog/overloading-equality-operator-in-cpp-class-hierarchy/ (for type equality?)
 * TODO(antoyo): support #[inline] attributes.
 * TODO(antoyo): support LTO (gcc's equivalent to Full LTO is -flto -flto-partition=one — https://documentation.suse.com/sbp/all/html/SBP-GCC-10/index.html).
 * For Thin LTO, this might be helpful:
 * In gcc 4.6 -fwhopr was removed and became default with -flto. The non-whopr path can still be executed via -flto-partition=none.
 * Or the new incremental LTO (https://www.phoronix.com/news/GCC-Incremental-LTO-Patches)?
 *
 * Maybe some missing optizations enabled by rustc's LTO is in there: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
 * Like -fipa-icf (should be already enabled) and maybe -fdevirtualize-at-ltrans.
 * TODO: disable debug info always being emitted. Perhaps this slows down things?
 *
 * TODO(antoyo): remove the patches.
 */

#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![feature(rustc_private, decl_macro, never_type, trusted_len)]
#![allow(broken_intra_doc_links)]
#![recursion_limit = "256"]
#![warn(rust_2018_idioms)]
#![warn(unused_lifetimes)]
#![deny(clippy::pattern_type_mismatch)]
#![allow(clippy::needless_lifetimes, clippy::uninlined_format_args)]

// Some "regular" crates we want to share with rustc
extern crate object;
extern crate smallvec;
// FIXME(antoyo): clippy bug: remove the #[allow] when it's fixed.
#[allow(unused_extern_crates)]
extern crate tempfile;
#[macro_use]
extern crate tracing;

// The rustc crates we need
extern crate rustc_abi;
extern crate rustc_apfloat;
extern crate rustc_ast;
extern crate rustc_attr_data_structures;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_fluent_macro;
extern crate rustc_fs_util;
extern crate rustc_hir;
extern crate rustc_index;
#[cfg(feature = "master")]
extern crate rustc_interface;
extern crate rustc_macros;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;

// This prevents duplicating functions and statics that are already part of the host rustc process.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

mod abi;
mod allocator;
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
mod gcc_util;
mod int;
mod intrinsic;
mod mono_item;
mod type_;
mod type_of;

use std::any::Any;
use std::fmt::Debug;
use std::ops::Deref;
#[cfg(not(feature = "master"))]
use std::sync::atomic::AtomicBool;
#[cfg(not(feature = "master"))]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use back::lto::{ThinBuffer, ThinData};
use gccjit::{CType, Context, OptimizationLevel};
#[cfg(feature = "master")]
use gccjit::{TargetInfo, Version};
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_ast::expand::autodiff_attrs::AutoDiffItem;
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::base::codegen_crate;
use rustc_codegen_ssa::traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods};
use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleCodegen, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_errors::DiagCtxtHandle;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputFilenames};
use rustc_span::Symbol;
use rustc_span::fatal_error::FatalError;
use rustc_target::spec::RelocModel;
use tempfile::TempDir;

use crate::back::lto::ModuleBuffer;
use crate::gcc_util::target_cpu;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub struct PrintOnPanic<F: Fn() -> String>(pub F);

impl<F: Fn() -> String> Drop for PrintOnPanic<F> {
    fn drop(&mut self) {
        if ::std::thread::panicking() {
            println!("{}", (self.0)());
        }
    }
}

#[cfg(not(feature = "master"))]
#[derive(Debug)]
pub struct TargetInfo {
    supports_128bit_integers: AtomicBool,
}

#[cfg(not(feature = "master"))]
impl TargetInfo {
    fn cpu_supports(&self, _feature: &str) -> bool {
        false
    }

    fn supports_target_dependent_type(&self, typ: CType) -> bool {
        match typ {
            CType::UInt128t | CType::Int128t => {
                if self.supports_128bit_integers.load(Ordering::SeqCst) {
                    return true;
                }
            }
            _ => (),
        }
        false
    }
}

#[derive(Clone)]
pub struct LockedTargetInfo {
    info: Arc<Mutex<IntoDynSyncSend<TargetInfo>>>,
}

impl Debug for LockedTargetInfo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.info.lock().expect("lock").fmt(formatter)
    }
}

impl LockedTargetInfo {
    fn cpu_supports(&self, feature: &str) -> bool {
        self.info.lock().expect("lock").cpu_supports(feature)
    }

    fn supports_target_dependent_type(&self, typ: CType) -> bool {
        self.info.lock().expect("lock").supports_target_dependent_type(typ)
    }
}

#[derive(Clone)]
pub struct GccCodegenBackend {
    target_info: LockedTargetInfo,
}

impl CodegenBackend for GccCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        crate::DEFAULT_LOCALE_RESOURCE
    }

    fn init(&self, _sess: &Session) {
        #[cfg(feature = "master")]
        {
            let target_cpu = target_cpu(_sess);

            // Get the second TargetInfo with the correct CPU features by setting the arch.
            let context = Context::default();
            if target_cpu != "generic" {
                context.add_command_line_option(format!("-march={}", target_cpu));
            }

            **self.target_info.info.lock().expect("lock") = context.get_target_info();
        }

        #[cfg(feature = "master")]
        gccjit::set_global_personality_function_name(b"rust_eh_personality\0");

        #[cfg(not(feature = "master"))]
        {
            let temp_dir = TempDir::new().expect("cannot create temporary directory");
            let temp_file = temp_dir.into_path().join("result.asm");
            let check_context = Context::default();
            check_context.set_print_errors_to_stderr(false);
            let _int128_ty = check_context.new_c_type(CType::UInt128t);
            // NOTE: we cannot just call compile() as this would require other files than libgccjit.so.
            check_context.compile_to_file(
                gccjit::OutputKind::Assembler,
                temp_file.to_str().expect("path to str"),
            );
            self.target_info
                .info
                .lock()
                .expect("lock")
                .supports_128bit_integers
                .store(check_context.get_last_error() == Ok(None), Ordering::SeqCst);
        }
    }

    fn provide(&self, providers: &mut Providers) {
        providers.global_backend_features = |tcx, ()| gcc_util::global_gcc_features(tcx.sess, true)
    }

    fn codegen_crate(
        &self,
        tcx: TyCtxt<'_>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any> {
        let target_cpu = target_cpu(tcx.sess);
        let res = codegen_crate(
            self.clone(),
            tcx,
            target_cpu.to_string(),
            metadata,
            need_metadata_module,
        );

        Box::new(res)
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<GccCodegenBackend>>()
            .expect("Expected GccCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess)
    }

    fn target_config(&self, sess: &Session) -> TargetConfig {
        target_config(sess, &self.target_info)
    }
}

fn new_context<'gcc, 'tcx>(tcx: TyCtxt<'tcx>) -> Context<'gcc> {
    let context = Context::default();
    if tcx.sess.target.arch == "x86" || tcx.sess.target.arch == "x86_64" {
        context.add_command_line_option("-masm=intel");
    }
    #[cfg(feature = "master")]
    {
        context.set_special_chars_allowed_in_func_names("$.*");
        let version = Version::get();
        let version = format!("{}.{}.{}", version.major, version.minor, version.patch);
        context.set_output_ident(&format!(
            "rustc version {} with libgccjit {}",
            rustc_interface::util::rustc_version_str().unwrap_or("unknown version"),
            version,
        ));
    }
    // TODO(antoyo): check if this should only be added when using -Cforce-unwind-tables=n.
    context.add_command_line_option("-fno-asynchronous-unwind-tables");
    context
}

impl ExtraBackendMethods for GccCodegenBackend {
    fn codegen_allocator(
        &self,
        tcx: TyCtxt<'_>,
        module_name: &str,
        kind: AllocatorKind,
        alloc_error_handler_kind: AllocatorKind,
    ) -> Self::Module {
        let mut mods = GccContext {
            context: Arc::new(SyncContext::new(new_context(tcx))),
            relocation_model: tcx.sess.relocation_model(),
            should_combine_object_files: false,
            temp_dir: None,
        };

        unsafe {
            allocator::codegen(tcx, &mut mods, module_name, kind, alloc_error_handler_kind);
        }
        mods
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64) {
        base::compile_codegen_unit(tcx, cgu_name, self.target_info.clone())
    }

    fn target_machine_factory(
        &self,
        _sess: &Session,
        _opt_level: OptLevel,
        _features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        // TODO(antoyo): set opt level.
        Arc::new(|_| Ok(()))
    }
}

pub struct GccContext {
    context: Arc<SyncContext>,
    /// This field is needed in order to be able to set the flag -fPIC when necessary when doing
    /// LTO.
    relocation_model: RelocModel,
    should_combine_object_files: bool,
    // Temporary directory used by LTO. We keep it here so that it's not removed before linking.
    temp_dir: Option<TempDir>,
}

struct SyncContext {
    context: Context<'static>,
}

impl SyncContext {
    fn new(context: Context<'static>) -> Self {
        Self { context }
    }
}

impl Deref for SyncContext {
    type Target = Context<'static>;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

unsafe impl Send for SyncContext {}
// FIXME(antoyo): that shouldn't be Sync. Parallel compilation is currently disabled with "-Zno-parallel-llvm".
// TODO: disable it here by returning false in CodegenBackend::supports_parallel().
unsafe impl Sync for SyncContext {}

impl WriteBackendMethods for GccCodegenBackend {
    type Module = GccContext;
    type TargetMachine = ();
    type TargetMachineError = ();
    type ModuleBuffer = ModuleBuffer;
    type ThinData = ThinData;
    type ThinBuffer = ThinBuffer;

    fn run_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLtoInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<LtoModuleCodegen<Self>, FatalError> {
        back::lto::run_fat(cgcx, modules, cached_modules)
    }

    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError> {
        back::lto::run_thin(cgcx, modules, cached_modules)
    }

    fn print_pass_timings(&self) {
        unimplemented!();
    }

    fn print_statistics(&self) {
        unimplemented!()
    }

    unsafe fn optimize(
        _cgcx: &CodegenContext<Self>,
        _dcx: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError> {
        module.module_llvm.context.set_optimization_level(to_gcc_opt_level(config.opt_level));
        Ok(())
    }

    fn optimize_fat(
        _cgcx: &CodegenContext<Self>,
        _module: &mut ModuleCodegen<Self::Module>,
    ) -> Result<(), FatalError> {
        // TODO(antoyo)
        Ok(())
    }

    unsafe fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::lto::optimize_thin_module(thin, cgcx)
    }

    unsafe fn codegen(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<CompiledModule, FatalError> {
        back::write::codegen(cgcx, dcx, module, config)
    }

    fn prepare_thin(
        module: ModuleCodegen<Self::Module>,
        emit_summary: bool,
    ) -> (String, Self::ThinBuffer) {
        back::lto::prepare_thin(module, emit_summary)
    }

    fn serialize_module(_module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        unimplemented!();
    }

    fn run_link(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        modules: Vec<ModuleCodegen<Self::Module>>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::write::link(cgcx, dcx, modules)
    }
    fn autodiff(
        _cgcx: &CodegenContext<Self>,
        _module: &ModuleCodegen<Self::Module>,
        _diff_fncs: Vec<AutoDiffItem>,
        _config: &ModuleConfig,
    ) -> Result<(), FatalError> {
        unimplemented!()
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_gccjit
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    #[cfg(feature = "master")]
    let info = {
        // Check whether the target supports 128-bit integers, and sized floating point types (like
        // Float16).
        let context = Context::default();
        Arc::new(Mutex::new(IntoDynSyncSend(context.get_target_info())))
    };
    #[cfg(not(feature = "master"))]
    let info = Arc::new(Mutex::new(IntoDynSyncSend(TargetInfo {
        supports_128bit_integers: AtomicBool::new(false),
    })));

    Box::new(GccCodegenBackend { target_info: LockedTargetInfo { info } })
}

fn to_gcc_opt_level(optlevel: Option<OptLevel>) -> OptimizationLevel {
    match optlevel {
        None => OptimizationLevel::None,
        Some(level) => match level {
            OptLevel::No => OptimizationLevel::None,
            OptLevel::Less => OptimizationLevel::Limited,
            OptLevel::More => OptimizationLevel::Standard,
            OptLevel::Aggressive => OptimizationLevel::Aggressive,
            OptLevel::Size | OptLevel::SizeMin => OptimizationLevel::Limited,
        },
    }
}

/// Returns the features that should be set in `cfg(target_feature)`.
fn target_config(sess: &Session, target_info: &LockedTargetInfo) -> TargetConfig {
    // TODO(antoyo): use global_gcc_features.
    let f = |allow_unstable| {
        sess.target
            .rust_target_features()
            .iter()
            .filter_map(|&(feature, gate, _)| {
                if allow_unstable
                    || (gate.in_cfg()
                        && (sess.is_nightly_build() || gate.requires_nightly().is_none()))
                {
                    Some(feature)
                } else {
                    None
                }
            })
            .filter(|feature| {
                // TODO: we disable Neon for now since we don't support the LLVM intrinsics for it.
                if *feature == "neon" {
                    return false;
                }
                target_info.cpu_supports(feature)
                /*
                  adx, aes, avx, avx2, avx512bf16, avx512bitalg, avx512bw, avx512cd, avx512dq, avx512er, avx512f, avx512fp16, avx512ifma,
                  avx512pf, avx512vbmi, avx512vbmi2, avx512vl, avx512vnni, avx512vp2intersect, avx512vpopcntdq,
                  bmi1, bmi2, cmpxchg16b, ermsb, f16c, fma, fxsr, gfni, lzcnt, movbe, pclmulqdq, popcnt, rdrand, rdseed, rtm,
                  sha, sse, sse2, sse3, sse4.1, sse4.2, sse4a, ssse3, tbm, vaes, vpclmulqdq, xsave, xsavec, xsaveopt, xsaves
                */
            })
            .map(Symbol::intern)
            .collect()
    };

    let target_features = f(false);
    let unstable_target_features = f(true);

    TargetConfig {
        target_features,
        unstable_target_features,
        // There are no known bugs with GCC support for f16 or f128
        has_reliable_f16: true,
        has_reliable_f16_math: true,
        has_reliable_f128: true,
        has_reliable_f128_math: true,
    }
}
