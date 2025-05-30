use std::any::Any;
use std::assert_matches::assert_matches;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::{fs, io, mem, str, thread};

use rustc_abi::Size;
use rustc_ast::attr;
use rustc_ast::expand::autodiff_attrs::AutoDiffItem;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::jobserver::{self, Acquired};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::profiling::{SelfProfilerRef, VerboseTimingGuard};
use rustc_errors::emitter::Emitter;
use rustc_errors::translation::Translate;
use rustc_errors::{
    Diag, DiagArgMap, DiagCtxt, DiagMessage, ErrCode, FatalError, FluentBundle, Level, MultiSpan,
    Style, Suggestions,
};
use rustc_fs_util::link_or_copy;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_incremental::{
    copy_cgu_workproduct_to_incr_comp_cache_dir, in_incr_comp_dir, in_incr_comp_dir_sess,
};
use rustc_metadata::fs::copy_to_stdout;
use rustc_middle::bug;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::exported_symbols::SymbolExportInfo;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{
    self, CrateType, Lto, OutFileName, OutputFilenames, OutputType, Passes, SwitchWithOptPath,
};
use rustc_span::source_map::SourceMap;
use rustc_span::{FileName, InnerSpan, Span, SpanData, sym};
use rustc_target::spec::{MergeFunctions, SanitizerSet};
use tracing::debug;

use super::link::{self, ensure_removed};
use super::lto::{self, SerializedModule};
use super::symbol_export::symbol_name_for_instance_in_crate;
use crate::errors::{AutodiffWithoutLto, ErrorCreatingRemarkDir};
use crate::traits::*;
use crate::{
    CachedModuleCodegen, CodegenResults, CompiledModule, CrateInfo, ModuleCodegen, ModuleKind,
    errors,
};

const PRE_LTO_BC_EXT: &str = "pre-lto.bc";

/// What kind of object file to emit.
#[derive(Clone, Copy, PartialEq)]
pub enum EmitObj {
    // No object file.
    None,

    // Just uncompressed llvm bitcode. Provides easy compatibility with
    // emscripten's ecc compiler, when used as the linker.
    Bitcode,

    // Object code, possibly augmented with a bitcode section.
    ObjectCode(BitcodeSection),
}

/// What kind of llvm bitcode section to embed in an object file.
#[derive(Clone, Copy, PartialEq)]
pub enum BitcodeSection {
    // No bitcode section.
    None,

    // A full, uncompressed bitcode section.
    Full,
}

/// Module-specific configuration for `optimize_and_codegen`.
pub struct ModuleConfig {
    /// Names of additional optimization passes to run.
    pub passes: Vec<String>,
    /// Some(level) to optimize at a certain level, or None to run
    /// absolutely no optimizations (used for the metadata module).
    pub opt_level: Option<config::OptLevel>,

    /// Some(level) to optimize binary size, or None to not affect program size.
    pub opt_size: Option<config::OptLevel>,

    pub pgo_gen: SwitchWithOptPath,
    pub pgo_use: Option<PathBuf>,
    pub pgo_sample_use: Option<PathBuf>,
    pub debug_info_for_profiling: bool,
    pub instrument_coverage: bool,

    pub sanitizer: SanitizerSet,
    pub sanitizer_recover: SanitizerSet,
    pub sanitizer_dataflow_abilist: Vec<String>,
    pub sanitizer_memory_track_origins: usize,

    // Flags indicating which outputs to produce.
    pub emit_pre_lto_bc: bool,
    pub emit_no_opt_bc: bool,
    pub emit_bc: bool,
    pub emit_ir: bool,
    pub emit_asm: bool,
    pub emit_obj: EmitObj,
    pub emit_thin_lto: bool,
    pub emit_thin_lto_summary: bool,
    pub bc_cmdline: String,

    // Miscellaneous flags. These are mostly copied from command-line
    // options.
    pub verify_llvm_ir: bool,
    pub lint_llvm_ir: bool,
    pub no_prepopulate_passes: bool,
    pub no_builtins: bool,
    pub time_module: bool,
    pub vectorize_loop: bool,
    pub vectorize_slp: bool,
    pub merge_functions: bool,
    pub emit_lifetime_markers: bool,
    pub llvm_plugins: Vec<String>,
    pub autodiff: Vec<config::AutoDiff>,
}

impl ModuleConfig {
    fn new(kind: ModuleKind, tcx: TyCtxt<'_>, no_builtins: bool) -> ModuleConfig {
        // If it's a regular module, use `$regular`, otherwise use `$other`.
        // `$regular` and `$other` are evaluated lazily.
        macro_rules! if_regular {
            ($regular: expr, $other: expr) => {
                if let ModuleKind::Regular = kind { $regular } else { $other }
            };
        }

        let sess = tcx.sess;
        let opt_level_and_size = if_regular!(Some(sess.opts.optimize), None);

        let save_temps = sess.opts.cg.save_temps;

        let should_emit_obj = sess.opts.output_types.contains_key(&OutputType::Exe)
            || match kind {
                ModuleKind::Regular => sess.opts.output_types.contains_key(&OutputType::Object),
                ModuleKind::Allocator => false,
            };

        let emit_obj = if !should_emit_obj {
            EmitObj::None
        } else if sess.target.obj_is_bitcode
            || (sess.opts.cg.linker_plugin_lto.enabled() && !no_builtins)
        {
            // This case is selected if the target uses objects as bitcode, or
            // if linker plugin LTO is enabled. In the linker plugin LTO case
            // the assumption is that the final link-step will read the bitcode
            // and convert it to object code. This may be done by either the
            // native linker or rustc itself.
            //
            // Note, however, that the linker-plugin-lto requested here is
            // explicitly ignored for `#![no_builtins]` crates. These crates are
            // specifically ignored by rustc's LTO passes and wouldn't work if
            // loaded into the linker. These crates define symbols that LLVM
            // lowers intrinsics to, and these symbol dependencies aren't known
            // until after codegen. As a result any crate marked
            // `#![no_builtins]` is assumed to not participate in LTO and
            // instead goes on to generate object code.
            EmitObj::Bitcode
        } else if need_bitcode_in_object(tcx) {
            EmitObj::ObjectCode(BitcodeSection::Full)
        } else {
            EmitObj::ObjectCode(BitcodeSection::None)
        };

        ModuleConfig {
            passes: if_regular!(sess.opts.cg.passes.clone(), vec![]),

            opt_level: opt_level_and_size,
            opt_size: opt_level_and_size,

            pgo_gen: if_regular!(
                sess.opts.cg.profile_generate.clone(),
                SwitchWithOptPath::Disabled
            ),
            pgo_use: if_regular!(sess.opts.cg.profile_use.clone(), None),
            pgo_sample_use: if_regular!(sess.opts.unstable_opts.profile_sample_use.clone(), None),
            debug_info_for_profiling: sess.opts.unstable_opts.debug_info_for_profiling,
            instrument_coverage: if_regular!(sess.instrument_coverage(), false),

            sanitizer: if_regular!(sess.opts.unstable_opts.sanitizer, SanitizerSet::empty()),
            sanitizer_dataflow_abilist: if_regular!(
                sess.opts.unstable_opts.sanitizer_dataflow_abilist.clone(),
                Vec::new()
            ),
            sanitizer_recover: if_regular!(
                sess.opts.unstable_opts.sanitizer_recover,
                SanitizerSet::empty()
            ),
            sanitizer_memory_track_origins: if_regular!(
                sess.opts.unstable_opts.sanitizer_memory_track_origins,
                0
            ),

            emit_pre_lto_bc: if_regular!(
                save_temps || need_pre_lto_bitcode_for_incr_comp(sess),
                false
            ),
            emit_no_opt_bc: if_regular!(save_temps, false),
            emit_bc: if_regular!(
                save_temps || sess.opts.output_types.contains_key(&OutputType::Bitcode),
                save_temps
            ),
            emit_ir: if_regular!(
                sess.opts.output_types.contains_key(&OutputType::LlvmAssembly),
                false
            ),
            emit_asm: if_regular!(
                sess.opts.output_types.contains_key(&OutputType::Assembly),
                false
            ),
            emit_obj,
            emit_thin_lto: sess.opts.unstable_opts.emit_thin_lto,
            emit_thin_lto_summary: if_regular!(
                sess.opts.output_types.contains_key(&OutputType::ThinLinkBitcode),
                false
            ),
            bc_cmdline: sess.target.bitcode_llvm_cmdline.to_string(),

            verify_llvm_ir: sess.verify_llvm_ir(),
            lint_llvm_ir: sess.opts.unstable_opts.lint_llvm_ir,
            no_prepopulate_passes: sess.opts.cg.no_prepopulate_passes,
            no_builtins: no_builtins || sess.target.no_builtins,

            // Exclude metadata and allocator modules from time_passes output,
            // since they throw off the "LLVM passes" measurement.
            time_module: if_regular!(true, false),

            // Copy what clang does by turning on loop vectorization at O2 and
            // slp vectorization at O3.
            vectorize_loop: !sess.opts.cg.no_vectorize_loops
                && (sess.opts.optimize == config::OptLevel::More
                    || sess.opts.optimize == config::OptLevel::Aggressive),
            vectorize_slp: !sess.opts.cg.no_vectorize_slp
                && sess.opts.optimize == config::OptLevel::Aggressive,

            // Some targets (namely, NVPTX) interact badly with the
            // MergeFunctions pass. This is because MergeFunctions can generate
            // new function calls which may interfere with the target calling
            // convention; e.g. for the NVPTX target, PTX kernels should not
            // call other PTX kernels. MergeFunctions can also be configured to
            // generate aliases instead, but aliases are not supported by some
            // backends (again, NVPTX). Therefore, allow targets to opt out of
            // the MergeFunctions pass, but otherwise keep the pass enabled (at
            // O2 and O3) since it can be useful for reducing code size.
            merge_functions: match sess
                .opts
                .unstable_opts
                .merge_functions
                .unwrap_or(sess.target.merge_functions)
            {
                MergeFunctions::Disabled => false,
                MergeFunctions::Trampolines | MergeFunctions::Aliases => {
                    use config::OptLevel::*;
                    match sess.opts.optimize {
                        Aggressive | More | SizeMin | Size => true,
                        Less | No => false,
                    }
                }
            },

            emit_lifetime_markers: sess.emit_lifetime_markers(),
            llvm_plugins: if_regular!(sess.opts.unstable_opts.llvm_plugins.clone(), vec![]),
            autodiff: if_regular!(sess.opts.unstable_opts.autodiff.clone(), vec![]),
        }
    }

    pub fn bitcode_needed(&self) -> bool {
        self.emit_bc
            || self.emit_thin_lto_summary
            || self.emit_obj == EmitObj::Bitcode
            || self.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full)
    }

    pub fn embed_bitcode(&self) -> bool {
        self.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full)
    }
}

/// Configuration passed to the function returned by the `target_machine_factory`.
pub struct TargetMachineFactoryConfig {
    /// Split DWARF is enabled in LLVM by checking that `TM.MCOptions.SplitDwarfFile` isn't empty,
    /// so the path to the dwarf object has to be provided when we create the target machine.
    /// This can be ignored by backends which do not need it for their Split DWARF support.
    pub split_dwarf_file: Option<PathBuf>,

    /// The name of the output object file. Used for setting OutputFilenames in target options
    /// so that LLVM can emit the CodeView S_OBJNAME record in pdb files
    pub output_obj_file: Option<PathBuf>,
}

impl TargetMachineFactoryConfig {
    pub fn new(
        cgcx: &CodegenContext<impl WriteBackendMethods>,
        module_name: &str,
    ) -> TargetMachineFactoryConfig {
        let split_dwarf_file = if cgcx.target_can_use_split_dwarf {
            cgcx.output_filenames.split_dwarf_path(
                cgcx.split_debuginfo,
                cgcx.split_dwarf_kind,
                module_name,
                cgcx.invocation_temp.as_deref(),
            )
        } else {
            None
        };

        let output_obj_file = Some(cgcx.output_filenames.temp_path_for_cgu(
            OutputType::Object,
            module_name,
            cgcx.invocation_temp.as_deref(),
        ));
        TargetMachineFactoryConfig { split_dwarf_file, output_obj_file }
    }
}

pub type TargetMachineFactoryFn<B> = Arc<
    dyn Fn(
            TargetMachineFactoryConfig,
        ) -> Result<
            <B as WriteBackendMethods>::TargetMachine,
            <B as WriteBackendMethods>::TargetMachineError,
        > + Send
        + Sync,
>;

type ExportedSymbols = FxHashMap<CrateNum, Arc<Vec<(String, SymbolExportInfo)>>>;

/// Additional resources used by optimize_and_codegen (not module specific)
#[derive(Clone)]
pub struct CodegenContext<B: WriteBackendMethods> {
    // Resources needed when running LTO
    pub prof: SelfProfilerRef,
    pub lto: Lto,
    pub save_temps: bool,
    pub fewer_names: bool,
    pub time_trace: bool,
    pub exported_symbols: Option<Arc<ExportedSymbols>>,
    pub opts: Arc<config::Options>,
    pub crate_types: Vec<CrateType>,
    pub each_linked_rlib_for_lto: Vec<(CrateNum, PathBuf)>,
    pub output_filenames: Arc<OutputFilenames>,
    pub invocation_temp: Option<String>,
    pub regular_module_config: Arc<ModuleConfig>,
    pub allocator_module_config: Arc<ModuleConfig>,
    pub tm_factory: TargetMachineFactoryFn<B>,
    pub msvc_imps_needed: bool,
    pub is_pe_coff: bool,
    pub target_can_use_split_dwarf: bool,
    pub target_arch: String,
    pub target_is_like_darwin: bool,
    pub target_is_like_aix: bool,
    pub split_debuginfo: rustc_target::spec::SplitDebuginfo,
    pub split_dwarf_kind: rustc_session::config::SplitDwarfKind,
    pub pointer_size: Size,

    /// All commandline args used to invoke the compiler, with @file args fully expanded.
    /// This will only be used within debug info, e.g. in the pdb file on windows
    /// This is mainly useful for other tools that reads that debuginfo to figure out
    /// how to call the compiler with the same arguments.
    pub expanded_args: Vec<String>,

    /// Emitter to use for diagnostics produced during codegen.
    pub diag_emitter: SharedEmitter,
    /// LLVM optimizations for which we want to print remarks.
    pub remark: Passes,
    /// Directory into which should the LLVM optimization remarks be written.
    /// If `None`, they will be written to stderr.
    pub remark_dir: Option<PathBuf>,
    /// The incremental compilation session directory, or None if we are not
    /// compiling incrementally
    pub incr_comp_session_dir: Option<PathBuf>,
    /// Channel back to the main control thread to send messages to
    pub coordinator_send: Sender<Box<dyn Any + Send>>,
    /// `true` if the codegen should be run in parallel.
    ///
    /// Depends on [`ExtraBackendMethods::supports_parallel()`] and `-Zno_parallel_backend`.
    pub parallel: bool,
}

impl<B: WriteBackendMethods> CodegenContext<B> {
    pub fn create_dcx(&self) -> DiagCtxt {
        DiagCtxt::new(Box::new(self.diag_emitter.clone()))
    }

    pub fn config(&self, kind: ModuleKind) -> &ModuleConfig {
        match kind {
            ModuleKind::Regular => &self.regular_module_config,
            ModuleKind::Allocator => &self.allocator_module_config,
        }
    }
}

fn generate_lto_work<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    autodiff: Vec<AutoDiffItem>,
    needs_fat_lto: Vec<FatLtoInput<B>>,
    needs_thin_lto: Vec<(String, B::ThinBuffer)>,
    import_only_modules: Vec<(SerializedModule<B::ModuleBuffer>, WorkProduct)>,
) -> Vec<(WorkItem<B>, u64)> {
    let _prof_timer = cgcx.prof.generic_activity("codegen_generate_lto_work");

    if !needs_fat_lto.is_empty() {
        assert!(needs_thin_lto.is_empty());
        let mut module =
            B::run_fat_lto(cgcx, needs_fat_lto, import_only_modules).unwrap_or_else(|e| e.raise());
        if cgcx.lto == Lto::Fat && !autodiff.is_empty() {
            let config = cgcx.config(ModuleKind::Regular);
            module = module.autodiff(cgcx, autodiff, config).unwrap_or_else(|e| e.raise());
        }
        // We are adding a single work item, so the cost doesn't matter.
        vec![(WorkItem::LTO(module), 0)]
    } else {
        if !autodiff.is_empty() {
            let dcx = cgcx.create_dcx();
            dcx.handle().emit_fatal(AutodiffWithoutLto {});
        }
        assert!(needs_fat_lto.is_empty());
        let (lto_modules, copy_jobs) = B::run_thin_lto(cgcx, needs_thin_lto, import_only_modules)
            .unwrap_or_else(|e| e.raise());
        lto_modules
            .into_iter()
            .map(|module| {
                let cost = module.cost();
                (WorkItem::LTO(module), cost)
            })
            .chain(copy_jobs.into_iter().map(|wp| {
                (
                    WorkItem::CopyPostLtoArtifacts(CachedModuleCodegen {
                        name: wp.cgu_name.clone(),
                        source: wp,
                    }),
                    0, // copying is very cheap
                )
            }))
            .collect()
    }
}

struct CompiledModules {
    modules: Vec<CompiledModule>,
    allocator_module: Option<CompiledModule>,
}

fn need_bitcode_in_object(tcx: TyCtxt<'_>) -> bool {
    let sess = tcx.sess;
    sess.opts.cg.embed_bitcode
        && tcx.crate_types().contains(&CrateType::Rlib)
        && sess.opts.output_types.contains_key(&OutputType::Exe)
}

fn need_pre_lto_bitcode_for_incr_comp(sess: &Session) -> bool {
    if sess.opts.incremental.is_none() {
        return false;
    }

    match sess.lto() {
        Lto::No => false,
        Lto::Fat | Lto::Thin | Lto::ThinLocal => true,
    }
}

pub(crate) fn start_async_codegen<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'_>,
    target_cpu: String,
) -> OngoingCodegen<B> {
    let (coordinator_send, coordinator_receive) = channel();

    let crate_attrs = tcx.hir_attrs(rustc_hir::CRATE_HIR_ID);
    let no_builtins = attr::contains_name(crate_attrs, sym::no_builtins);

    let crate_info = CrateInfo::new(tcx, target_cpu);

    let regular_config = ModuleConfig::new(ModuleKind::Regular, tcx, no_builtins);
    let allocator_config = ModuleConfig::new(ModuleKind::Allocator, tcx, no_builtins);

    let (shared_emitter, shared_emitter_main) = SharedEmitter::new();
    let (codegen_worker_send, codegen_worker_receive) = channel();

    let coordinator_thread = start_executing_work(
        backend.clone(),
        tcx,
        &crate_info,
        shared_emitter,
        codegen_worker_send,
        coordinator_receive,
        Arc::new(regular_config),
        Arc::new(allocator_config),
        coordinator_send.clone(),
    );

    OngoingCodegen {
        backend,
        crate_info,

        codegen_worker_receive,
        shared_emitter_main,
        coordinator: Coordinator {
            sender: coordinator_send,
            future: Some(coordinator_thread),
            phantom: PhantomData,
        },
        output_filenames: Arc::clone(tcx.output_filenames(())),
    }
}

fn copy_all_cgu_workproducts_to_incr_comp_cache_dir(
    sess: &Session,
    compiled_modules: &CompiledModules,
) -> FxIndexMap<WorkProductId, WorkProduct> {
    let mut work_products = FxIndexMap::default();

    if sess.opts.incremental.is_none() {
        return work_products;
    }

    let _timer = sess.timer("copy_all_cgu_workproducts_to_incr_comp_cache_dir");

    for module in compiled_modules.modules.iter().filter(|m| m.kind == ModuleKind::Regular) {
        let mut files = Vec::new();
        if let Some(object_file_path) = &module.object {
            files.push((OutputType::Object.extension(), object_file_path.as_path()));
        }
        if let Some(dwarf_object_file_path) = &module.dwarf_object {
            files.push(("dwo", dwarf_object_file_path.as_path()));
        }
        if let Some(path) = &module.assembly {
            files.push((OutputType::Assembly.extension(), path.as_path()));
        }
        if let Some(path) = &module.llvm_ir {
            files.push((OutputType::LlvmAssembly.extension(), path.as_path()));
        }
        if let Some(path) = &module.bytecode {
            files.push((OutputType::Bitcode.extension(), path.as_path()));
        }
        if let Some((id, product)) = copy_cgu_workproduct_to_incr_comp_cache_dir(
            sess,
            &module.name,
            files.as_slice(),
            &module.links_from_incr_cache,
        ) {
            work_products.insert(id, product);
        }
    }

    work_products
}

fn produce_final_output_artifacts(
    sess: &Session,
    compiled_modules: &CompiledModules,
    crate_output: &OutputFilenames,
) {
    let mut user_wants_bitcode = false;
    let mut user_wants_objects = false;

    // Produce final compile outputs.
    let copy_gracefully = |from: &Path, to: &OutFileName| match to {
        OutFileName::Stdout if let Err(e) = copy_to_stdout(from) => {
            sess.dcx().emit_err(errors::CopyPath::new(from, to.as_path(), e));
        }
        OutFileName::Real(path) if let Err(e) = fs::copy(from, path) => {
            sess.dcx().emit_err(errors::CopyPath::new(from, path, e));
        }
        _ => {}
    };

    let copy_if_one_unit = |output_type: OutputType, keep_numbered: bool| {
        if let [module] = &compiled_modules.modules[..] {
            // 1) Only one codegen unit. In this case it's no difficulty
            //    to copy `foo.0.x` to `foo.x`.
            let path = crate_output.temp_path_for_cgu(
                output_type,
                &module.name,
                sess.invocation_temp.as_deref(),
            );
            let output = crate_output.path(output_type);
            if !output_type.is_text_output() && output.is_tty() {
                sess.dcx()
                    .emit_err(errors::BinaryOutputToTty { shorthand: output_type.shorthand() });
            } else {
                copy_gracefully(&path, &output);
            }
            if !sess.opts.cg.save_temps && !keep_numbered {
                // The user just wants `foo.x`, not `foo.#module-name#.x`.
                ensure_removed(sess.dcx(), &path);
            }
        } else {
            if crate_output.outputs.contains_explicit_name(&output_type) {
                // 2) Multiple codegen units, with `--emit foo=some_name`. We have
                //    no good solution for this case, so warn the user.
                sess.dcx()
                    .emit_warn(errors::IgnoringEmitPath { extension: output_type.extension() });
            } else if crate_output.single_output_file.is_some() {
                // 3) Multiple codegen units, with `-o some_name`. We have
                //    no good solution for this case, so warn the user.
                sess.dcx().emit_warn(errors::IgnoringOutput { extension: output_type.extension() });
            } else {
                // 4) Multiple codegen units, but no explicit name. We
                //    just leave the `foo.0.x` files in place.
                // (We don't have to do any work in this case.)
            }
        }
    };

    // Flag to indicate whether the user explicitly requested bitcode.
    // Otherwise, we produced it only as a temporary output, and will need
    // to get rid of it.
    for output_type in crate_output.outputs.keys() {
        match *output_type {
            OutputType::Bitcode => {
                user_wants_bitcode = true;
                // Copy to .bc, but always keep the .0.bc. There is a later
                // check to figure out if we should delete .0.bc files, or keep
                // them for making an rlib.
                copy_if_one_unit(OutputType::Bitcode, true);
            }
            OutputType::ThinLinkBitcode => {
                copy_if_one_unit(OutputType::ThinLinkBitcode, false);
            }
            OutputType::LlvmAssembly => {
                copy_if_one_unit(OutputType::LlvmAssembly, false);
            }
            OutputType::Assembly => {
                copy_if_one_unit(OutputType::Assembly, false);
            }
            OutputType::Object => {
                user_wants_objects = true;
                copy_if_one_unit(OutputType::Object, true);
            }
            OutputType::Mir | OutputType::Metadata | OutputType::Exe | OutputType::DepInfo => {}
        }
    }

    // Clean up unwanted temporary files.

    // We create the following files by default:
    //  - #crate#.#module-name#.bc
    //  - #crate#.#module-name#.o
    //  - #crate#.crate.metadata.bc
    //  - #crate#.crate.metadata.o
    //  - #crate#.o (linked from crate.##.o)
    //  - #crate#.bc (copied from crate.##.bc)
    // We may create additional files if requested by the user (through
    // `-C save-temps` or `--emit=` flags).

    if !sess.opts.cg.save_temps {
        // Remove the temporary .#module-name#.o objects. If the user didn't
        // explicitly request bitcode (with --emit=bc), and the bitcode is not
        // needed for building an rlib, then we must remove .#module-name#.bc as
        // well.

        // Specific rules for keeping .#module-name#.bc:
        //  - If the user requested bitcode (`user_wants_bitcode`), and
        //    codegen_units > 1, then keep it.
        //  - If the user requested bitcode but codegen_units == 1, then we
        //    can toss .#module-name#.bc because we copied it to .bc earlier.
        //  - If we're not building an rlib and the user didn't request
        //    bitcode, then delete .#module-name#.bc.
        // If you change how this works, also update back::link::link_rlib,
        // where .#module-name#.bc files are (maybe) deleted after making an
        // rlib.
        let needs_crate_object = crate_output.outputs.contains_key(&OutputType::Exe);

        let keep_numbered_bitcode = user_wants_bitcode && sess.codegen_units().as_usize() > 1;

        let keep_numbered_objects =
            needs_crate_object || (user_wants_objects && sess.codegen_units().as_usize() > 1);

        for module in compiled_modules.modules.iter() {
            if !keep_numbered_objects {
                if let Some(ref path) = module.object {
                    ensure_removed(sess.dcx(), path);
                }

                if let Some(ref path) = module.dwarf_object {
                    ensure_removed(sess.dcx(), path);
                }
            }

            if let Some(ref path) = module.bytecode {
                if !keep_numbered_bitcode {
                    ensure_removed(sess.dcx(), path);
                }
            }
        }

        if !user_wants_bitcode
            && let Some(ref allocator_module) = compiled_modules.allocator_module
            && let Some(ref path) = allocator_module.bytecode
        {
            ensure_removed(sess.dcx(), path);
        }
    }

    if sess.opts.json_artifact_notifications {
        if let [module] = &compiled_modules.modules[..] {
            module.for_each_output(|_path, ty| {
                if sess.opts.output_types.contains_key(&ty) {
                    let descr = ty.shorthand();
                    // for single cgu file is renamed to drop cgu specific suffix
                    // so we regenerate it the same way
                    let path = crate_output.path(ty);
                    sess.dcx().emit_artifact_notification(path.as_path(), descr);
                }
            });
        } else {
            for module in &compiled_modules.modules {
                module.for_each_output(|path, ty| {
                    if sess.opts.output_types.contains_key(&ty) {
                        let descr = ty.shorthand();
                        sess.dcx().emit_artifact_notification(&path, descr);
                    }
                });
            }
        }
    }

    // We leave the following files around by default:
    //  - #crate#.o
    //  - #crate#.crate.metadata.o
    //  - #crate#.bc
    // These are used in linking steps and will be cleaned up afterward.
}

pub(crate) enum WorkItem<B: WriteBackendMethods> {
    /// Optimize a newly codegened, totally unoptimized module.
    Optimize(ModuleCodegen<B::Module>),
    /// Copy the post-LTO artifacts from the incremental cache to the output
    /// directory.
    CopyPostLtoArtifacts(CachedModuleCodegen),
    /// Performs (Thin)LTO on the given module.
    LTO(lto::LtoModuleCodegen<B>),
}

impl<B: WriteBackendMethods> WorkItem<B> {
    fn module_kind(&self) -> ModuleKind {
        match *self {
            WorkItem::Optimize(ref m) => m.kind,
            WorkItem::CopyPostLtoArtifacts(_) | WorkItem::LTO(_) => ModuleKind::Regular,
        }
    }

    /// Generate a short description of this work item suitable for use as a thread name.
    fn short_description(&self) -> String {
        // `pthread_setname()` on *nix ignores anything beyond the first 15
        // bytes. Use short descriptions to maximize the space available for
        // the module name.
        #[cfg(not(windows))]
        fn desc(short: &str, _long: &str, name: &str) -> String {
            // The short label is three bytes, and is followed by a space. That
            // leaves 11 bytes for the CGU name. How we obtain those 11 bytes
            // depends on the CGU name form.
            //
            // - Non-incremental, e.g. `regex.f10ba03eb5ec7975-cgu.0`: the part
            //   before the `-cgu.0` is the same for every CGU, so use the
            //   `cgu.0` part. The number suffix will be different for each
            //   CGU.
            //
            // - Incremental (normal), e.g. `2i52vvl2hco29us0`: use the whole
            //   name because each CGU will have a unique ASCII hash, and the
            //   first 11 bytes will be enough to identify it.
            //
            // - Incremental (with `-Zhuman-readable-cgu-names`), e.g.
            //   `regex.f10ba03eb5ec7975-re_builder.volatile`: use the whole
            //   name. The first 11 bytes won't be enough to uniquely identify
            //   it, but no obvious substring will, and this is a rarely used
            //   option so it doesn't matter much.
            //
            assert_eq!(short.len(), 3);
            let name = if let Some(index) = name.find("-cgu.") {
                &name[index + 1..] // +1 skips the leading '-'.
            } else {
                name
            };
            format!("{short} {name}")
        }

        // Windows has no thread name length limit, so use more descriptive names.
        #[cfg(windows)]
        fn desc(_short: &str, long: &str, name: &str) -> String {
            format!("{long} {name}")
        }

        match self {
            WorkItem::Optimize(m) => desc("opt", "optimize module", &m.name),
            WorkItem::CopyPostLtoArtifacts(m) => desc("cpy", "copy LTO artifacts for", &m.name),
            WorkItem::LTO(m) => desc("lto", "LTO module", m.name()),
        }
    }
}

/// A result produced by the backend.
pub(crate) enum WorkItemResult<B: WriteBackendMethods> {
    /// The backend has finished compiling a CGU, nothing more required.
    Finished(CompiledModule),

    /// The backend has finished compiling a CGU, which now needs linking
    /// because `-Zcombine-cgu` was specified.
    NeedsLink(ModuleCodegen<B::Module>),

    /// The backend has finished compiling a CGU, which now needs to go through
    /// fat LTO.
    NeedsFatLto(FatLtoInput<B>),

    /// The backend has finished compiling a CGU, which now needs to go through
    /// thin LTO.
    NeedsThinLto(String, B::ThinBuffer),
}

pub enum FatLtoInput<B: WriteBackendMethods> {
    Serialized { name: String, buffer: B::ModuleBuffer },
    InMemory(ModuleCodegen<B::Module>),
}

/// Actual LTO type we end up choosing based on multiple factors.
pub(crate) enum ComputedLtoType {
    No,
    Thin,
    Fat,
}

pub(crate) fn compute_per_cgu_lto_type(
    sess_lto: &Lto,
    opts: &config::Options,
    sess_crate_types: &[CrateType],
    module_kind: ModuleKind,
) -> ComputedLtoType {
    // If the linker does LTO, we don't have to do it. Note that we
    // keep doing full LTO, if it is requested, as not to break the
    // assumption that the output will be a single module.
    let linker_does_lto = opts.cg.linker_plugin_lto.enabled();

    // When we're automatically doing ThinLTO for multi-codegen-unit
    // builds we don't actually want to LTO the allocator modules if
    // it shows up. This is due to various linker shenanigans that
    // we'll encounter later.
    let is_allocator = module_kind == ModuleKind::Allocator;

    // We ignore a request for full crate graph LTO if the crate type
    // is only an rlib, as there is no full crate graph to process,
    // that'll happen later.
    //
    // This use case currently comes up primarily for targets that
    // require LTO so the request for LTO is always unconditionally
    // passed down to the backend, but we don't actually want to do
    // anything about it yet until we've got a final product.
    let is_rlib = matches!(sess_crate_types, [CrateType::Rlib]);

    match sess_lto {
        Lto::ThinLocal if !linker_does_lto && !is_allocator => ComputedLtoType::Thin,
        Lto::Thin if !linker_does_lto && !is_rlib => ComputedLtoType::Thin,
        Lto::Fat if !is_rlib => ComputedLtoType::Fat,
        _ => ComputedLtoType::No,
    }
}

fn execute_optimize_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    mut module: ModuleCodegen<B::Module>,
    module_config: &ModuleConfig,
) -> Result<WorkItemResult<B>, FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();

    B::optimize(cgcx, dcx, &mut module, module_config)?;

    // After we've done the initial round of optimizations we need to
    // decide whether to synchronously codegen this module or ship it
    // back to the coordinator thread for further LTO processing (which
    // has to wait for all the initial modules to be optimized).

    let lto_type = compute_per_cgu_lto_type(&cgcx.lto, &cgcx.opts, &cgcx.crate_types, module.kind);

    // If we're doing some form of incremental LTO then we need to be sure to
    // save our module to disk first.
    let bitcode = if cgcx.config(module.kind).emit_pre_lto_bc {
        let filename = pre_lto_bitcode_filename(&module.name);
        cgcx.incr_comp_session_dir.as_ref().map(|path| path.join(&filename))
    } else {
        None
    };

    match lto_type {
        ComputedLtoType::No => finish_intra_module_work(cgcx, module, module_config),
        ComputedLtoType::Thin => {
            let (name, thin_buffer) = B::prepare_thin(module, false);
            if let Some(path) = bitcode {
                fs::write(&path, thin_buffer.data()).unwrap_or_else(|e| {
                    panic!("Error writing pre-lto-bitcode file `{}`: {}", path.display(), e);
                });
            }
            Ok(WorkItemResult::NeedsThinLto(name, thin_buffer))
        }
        ComputedLtoType::Fat => match bitcode {
            Some(path) => {
                let (name, buffer) = B::serialize_module(module);
                fs::write(&path, buffer.data()).unwrap_or_else(|e| {
                    panic!("Error writing pre-lto-bitcode file `{}`: {}", path.display(), e);
                });
                Ok(WorkItemResult::NeedsFatLto(FatLtoInput::Serialized { name, buffer }))
            }
            None => Ok(WorkItemResult::NeedsFatLto(FatLtoInput::InMemory(module))),
        },
    }
}

fn execute_copy_from_cache_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    module: CachedModuleCodegen,
    module_config: &ModuleConfig,
) -> WorkItemResult<B> {
    let incr_comp_session_dir = cgcx.incr_comp_session_dir.as_ref().unwrap();

    let mut links_from_incr_cache = Vec::new();

    let mut load_from_incr_comp_dir = |output_path: PathBuf, saved_path: &str| {
        let source_file = in_incr_comp_dir(incr_comp_session_dir, saved_path);
        debug!(
            "copying preexisting module `{}` from {:?} to {}",
            module.name,
            source_file,
            output_path.display()
        );
        match link_or_copy(&source_file, &output_path) {
            Ok(_) => {
                links_from_incr_cache.push(source_file);
                Some(output_path)
            }
            Err(error) => {
                cgcx.create_dcx().handle().emit_err(errors::CopyPathBuf {
                    source_file,
                    output_path,
                    error,
                });
                None
            }
        }
    };

    let dwarf_object =
        module.source.saved_files.get("dwo").as_ref().and_then(|saved_dwarf_object_file| {
            let dwarf_obj_out = cgcx
                .output_filenames
                .split_dwarf_path(
                    cgcx.split_debuginfo,
                    cgcx.split_dwarf_kind,
                    &module.name,
                    cgcx.invocation_temp.as_deref(),
                )
                .expect(
                    "saved dwarf object in work product but `split_dwarf_path` returned `None`",
                );
            load_from_incr_comp_dir(dwarf_obj_out, saved_dwarf_object_file)
        });

    let mut load_from_incr_cache = |perform, output_type: OutputType| {
        if perform {
            let saved_file = module.source.saved_files.get(output_type.extension())?;
            let output_path = cgcx.output_filenames.temp_path_for_cgu(
                output_type,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );
            load_from_incr_comp_dir(output_path, &saved_file)
        } else {
            None
        }
    };

    let should_emit_obj = module_config.emit_obj != EmitObj::None;
    let assembly = load_from_incr_cache(module_config.emit_asm, OutputType::Assembly);
    let llvm_ir = load_from_incr_cache(module_config.emit_ir, OutputType::LlvmAssembly);
    let bytecode = load_from_incr_cache(module_config.emit_bc, OutputType::Bitcode);
    let object = load_from_incr_cache(should_emit_obj, OutputType::Object);
    if should_emit_obj && object.is_none() {
        cgcx.create_dcx().handle().emit_fatal(errors::NoSavedObjectFile { cgu_name: &module.name })
    }

    WorkItemResult::Finished(CompiledModule {
        links_from_incr_cache,
        name: module.name,
        kind: ModuleKind::Regular,
        object,
        dwarf_object,
        bytecode,
        assembly,
        llvm_ir,
    })
}

fn execute_lto_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    module: lto::LtoModuleCodegen<B>,
    module_config: &ModuleConfig,
) -> Result<WorkItemResult<B>, FatalError> {
    let module = module.optimize(cgcx)?;
    finish_intra_module_work(cgcx, module, module_config)
}

fn finish_intra_module_work<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    module: ModuleCodegen<B::Module>,
    module_config: &ModuleConfig,
) -> Result<WorkItemResult<B>, FatalError> {
    let dcx = cgcx.create_dcx();
    let dcx = dcx.handle();

    if !cgcx.opts.unstable_opts.combine_cgu || module.kind == ModuleKind::Allocator {
        let module = B::codegen(cgcx, dcx, module, module_config)?;
        Ok(WorkItemResult::Finished(module))
    } else {
        Ok(WorkItemResult::NeedsLink(module))
    }
}

/// Messages sent to the coordinator.
pub(crate) enum Message<B: WriteBackendMethods> {
    /// A jobserver token has become available. Sent from the jobserver helper
    /// thread.
    Token(io::Result<Acquired>),

    /// The backend has finished processing a work item for a codegen unit.
    /// Sent from a backend worker thread.
    WorkItem { result: Result<WorkItemResult<B>, Option<WorkerFatalError>>, worker_id: usize },

    /// A vector containing all the AutoDiff tasks that we have to pass to Enzyme.
    AddAutoDiffItems(Vec<AutoDiffItem>),

    /// The frontend has finished generating something (backend IR or a
    /// post-LTO artifact) for a codegen unit, and it should be passed to the
    /// backend. Sent from the main thread.
    CodegenDone { llvm_work_item: WorkItem<B>, cost: u64 },

    /// Similar to `CodegenDone`, but for reusing a pre-LTO artifact
    /// Sent from the main thread.
    AddImportOnlyModule {
        module_data: SerializedModule<B::ModuleBuffer>,
        work_product: WorkProduct,
    },

    /// The frontend has finished generating everything for all codegen units.
    /// Sent from the main thread.
    CodegenComplete,

    /// Some normal-ish compiler error occurred, and codegen should be wound
    /// down. Sent from the main thread.
    CodegenAborted,
}

/// A message sent from the coordinator thread to the main thread telling it to
/// process another codegen unit.
pub struct CguMessage;

// A cut-down version of `rustc_errors::DiagInner` that impls `Send`, which
// can be used to send diagnostics from codegen threads to the main thread.
// It's missing the following fields from `rustc_errors::DiagInner`.
// - `span`: it doesn't impl `Send`.
// - `suggestions`: it doesn't impl `Send`, and isn't used for codegen
//   diagnostics.
// - `sort_span`: it doesn't impl `Send`.
// - `is_lint`: lints aren't relevant during codegen.
// - `emitted_at`: not used for codegen diagnostics.
struct Diagnostic {
    level: Level,
    messages: Vec<(DiagMessage, Style)>,
    code: Option<ErrCode>,
    children: Vec<Subdiagnostic>,
    args: DiagArgMap,
}

// A cut-down version of `rustc_errors::Subdiag` that impls `Send`. It's
// missing the following fields from `rustc_errors::Subdiag`.
// - `span`: it doesn't impl `Send`.
pub(crate) struct Subdiagnostic {
    level: Level,
    messages: Vec<(DiagMessage, Style)>,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MainThreadState {
    /// Doing nothing.
    Idle,

    /// Doing codegen, i.e. MIR-to-LLVM-IR conversion.
    Codegenning,

    /// Idle, but lending the compiler process's Token to an LLVM thread so it can do useful work.
    Lending,
}

fn start_executing_work<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'_>,
    crate_info: &CrateInfo,
    shared_emitter: SharedEmitter,
    codegen_worker_send: Sender<CguMessage>,
    coordinator_receive: Receiver<Box<dyn Any + Send>>,
    regular_config: Arc<ModuleConfig>,
    allocator_config: Arc<ModuleConfig>,
    tx_to_llvm_workers: Sender<Box<dyn Any + Send>>,
) -> thread::JoinHandle<Result<CompiledModules, ()>> {
    let coordinator_send = tx_to_llvm_workers;
    let sess = tcx.sess;

    let mut each_linked_rlib_for_lto = Vec::new();
    drop(link::each_linked_rlib(crate_info, None, &mut |cnum, path| {
        if link::ignored_for_lto(sess, crate_info, cnum) {
            return;
        }
        each_linked_rlib_for_lto.push((cnum, path.to_path_buf()));
    }));

    // Compute the set of symbols we need to retain when doing LTO (if we need to)
    let exported_symbols = {
        let mut exported_symbols = FxHashMap::default();

        let copy_symbols = |cnum| {
            let symbols = tcx
                .exported_symbols(cnum)
                .iter()
                .map(|&(s, lvl)| (symbol_name_for_instance_in_crate(tcx, s, cnum), lvl))
                .collect();
            Arc::new(symbols)
        };

        match sess.lto() {
            Lto::No => None,
            Lto::ThinLocal => {
                exported_symbols.insert(LOCAL_CRATE, copy_symbols(LOCAL_CRATE));
                Some(Arc::new(exported_symbols))
            }
            Lto::Fat | Lto::Thin => {
                exported_symbols.insert(LOCAL_CRATE, copy_symbols(LOCAL_CRATE));
                for &(cnum, ref _path) in &each_linked_rlib_for_lto {
                    exported_symbols.insert(cnum, copy_symbols(cnum));
                }
                Some(Arc::new(exported_symbols))
            }
        }
    };

    // First up, convert our jobserver into a helper thread so we can use normal
    // mpsc channels to manage our messages and such.
    // After we've requested tokens then we'll, when we can,
    // get tokens on `coordinator_receive` which will
    // get managed in the main loop below.
    let coordinator_send2 = coordinator_send.clone();
    let helper = jobserver::client()
        .into_helper_thread(move |token| {
            drop(coordinator_send2.send(Box::new(Message::Token::<B>(token))));
        })
        .expect("failed to spawn helper thread");

    let ol =
        if tcx.sess.opts.unstable_opts.no_codegen || !tcx.sess.opts.output_types.should_codegen() {
            // If we know that we won’t be doing codegen, create target machines without optimisation.
            config::OptLevel::No
        } else {
            tcx.backend_optimization_level(())
        };
    let backend_features = tcx.global_backend_features(());

    let remark_dir = if let Some(ref dir) = sess.opts.unstable_opts.remark_dir {
        let result = fs::create_dir_all(dir).and_then(|_| dir.canonicalize());
        match result {
            Ok(dir) => Some(dir),
            Err(error) => sess.dcx().emit_fatal(ErrorCreatingRemarkDir { error }),
        }
    } else {
        None
    };

    let cgcx = CodegenContext::<B> {
        crate_types: tcx.crate_types().to_vec(),
        each_linked_rlib_for_lto,
        lto: sess.lto(),
        fewer_names: sess.fewer_names(),
        save_temps: sess.opts.cg.save_temps,
        time_trace: sess.opts.unstable_opts.llvm_time_trace,
        opts: Arc::new(sess.opts.clone()),
        prof: sess.prof.clone(),
        exported_symbols,
        remark: sess.opts.cg.remark.clone(),
        remark_dir,
        incr_comp_session_dir: sess.incr_comp_session_dir_opt().map(|r| r.clone()),
        coordinator_send,
        expanded_args: tcx.sess.expanded_args.clone(),
        diag_emitter: shared_emitter.clone(),
        output_filenames: Arc::clone(tcx.output_filenames(())),
        regular_module_config: regular_config,
        allocator_module_config: allocator_config,
        tm_factory: backend.target_machine_factory(tcx.sess, ol, backend_features),
        msvc_imps_needed: msvc_imps_needed(tcx),
        is_pe_coff: tcx.sess.target.is_like_windows,
        target_can_use_split_dwarf: tcx.sess.target_can_use_split_dwarf(),
        target_arch: tcx.sess.target.arch.to_string(),
        target_is_like_darwin: tcx.sess.target.is_like_darwin,
        target_is_like_aix: tcx.sess.target.is_like_aix,
        split_debuginfo: tcx.sess.split_debuginfo(),
        split_dwarf_kind: tcx.sess.opts.unstable_opts.split_dwarf_kind,
        parallel: backend.supports_parallel() && !sess.opts.unstable_opts.no_parallel_backend,
        pointer_size: tcx.data_layout.pointer_size,
        invocation_temp: sess.invocation_temp.clone(),
    };

    // This is the "main loop" of parallel work happening for parallel codegen.
    // It's here that we manage parallelism, schedule work, and work with
    // messages coming from clients.
    //
    // There are a few environmental pre-conditions that shape how the system
    // is set up:
    //
    // - Error reporting can only happen on the main thread because that's the
    //   only place where we have access to the compiler `Session`.
    // - LLVM work can be done on any thread.
    // - Codegen can only happen on the main thread.
    // - Each thread doing substantial work must be in possession of a `Token`
    //   from the `Jobserver`.
    // - The compiler process always holds one `Token`. Any additional `Tokens`
    //   have to be requested from the `Jobserver`.
    //
    // Error Reporting
    // ===============
    // The error reporting restriction is handled separately from the rest: We
    // set up a `SharedEmitter` that holds an open channel to the main thread.
    // When an error occurs on any thread, the shared emitter will send the
    // error message to the receiver main thread (`SharedEmitterMain`). The
    // main thread will periodically query this error message queue and emit
    // any error messages it has received. It might even abort compilation if
    // it has received a fatal error. In this case we rely on all other threads
    // being torn down automatically with the main thread.
    // Since the main thread will often be busy doing codegen work, error
    // reporting will be somewhat delayed, since the message queue can only be
    // checked in between two work packages.
    //
    // Work Processing Infrastructure
    // ==============================
    // The work processing infrastructure knows three major actors:
    //
    // - the coordinator thread,
    // - the main thread, and
    // - LLVM worker threads
    //
    // The coordinator thread is running a message loop. It instructs the main
    // thread about what work to do when, and it will spawn off LLVM worker
    // threads as open LLVM WorkItems become available.
    //
    // The job of the main thread is to codegen CGUs into LLVM work packages
    // (since the main thread is the only thread that can do this). The main
    // thread will block until it receives a message from the coordinator, upon
    // which it will codegen one CGU, send it to the coordinator and block
    // again. This way the coordinator can control what the main thread is
    // doing.
    //
    // The coordinator keeps a queue of LLVM WorkItems, and when a `Token` is
    // available, it will spawn off a new LLVM worker thread and let it process
    // a WorkItem. When a LLVM worker thread is done with its WorkItem,
    // it will just shut down, which also frees all resources associated with
    // the given LLVM module, and sends a message to the coordinator that the
    // WorkItem has been completed.
    //
    // Work Scheduling
    // ===============
    // The scheduler's goal is to minimize the time it takes to complete all
    // work there is, however, we also want to keep memory consumption low
    // if possible. These two goals are at odds with each other: If memory
    // consumption were not an issue, we could just let the main thread produce
    // LLVM WorkItems at full speed, assuring maximal utilization of
    // Tokens/LLVM worker threads. However, since codegen is usually faster
    // than LLVM processing, the queue of LLVM WorkItems would fill up and each
    // WorkItem potentially holds on to a substantial amount of memory.
    //
    // So the actual goal is to always produce just enough LLVM WorkItems as
    // not to starve our LLVM worker threads. That means, once we have enough
    // WorkItems in our queue, we can block the main thread, so it does not
    // produce more until we need them.
    //
    // Doing LLVM Work on the Main Thread
    // ----------------------------------
    // Since the main thread owns the compiler process's implicit `Token`, it is
    // wasteful to keep it blocked without doing any work. Therefore, what we do
    // in this case is: We spawn off an additional LLVM worker thread that helps
    // reduce the queue. The work it is doing corresponds to the implicit
    // `Token`. The coordinator will mark the main thread as being busy with
    // LLVM work. (The actual work happens on another OS thread but we just care
    // about `Tokens`, not actual threads).
    //
    // When any LLVM worker thread finishes while the main thread is marked as
    // "busy with LLVM work", we can do a little switcheroo: We give the Token
    // of the just finished thread to the LLVM worker thread that is working on
    // behalf of the main thread's implicit Token, thus freeing up the main
    // thread again. The coordinator can then again decide what the main thread
    // should do. This allows the coordinator to make decisions at more points
    // in time.
    //
    // Striking a Balance between Throughput and Memory Consumption
    // ------------------------------------------------------------
    // Since our two goals, (1) use as many Tokens as possible and (2) keep
    // memory consumption as low as possible, are in conflict with each other,
    // we have to find a trade off between them. Right now, the goal is to keep
    // all workers busy, which means that no worker should find the queue empty
    // when it is ready to start.
    // How do we do achieve this? Good question :) We actually never know how
    // many `Tokens` are potentially available so it's hard to say how much to
    // fill up the queue before switching the main thread to LLVM work. Also we
    // currently don't have a means to estimate how long a running LLVM worker
    // will still be busy with it's current WorkItem. However, we know the
    // maximal count of available Tokens that makes sense (=the number of CPU
    // cores), so we can take a conservative guess. The heuristic we use here
    // is implemented in the `queue_full_enough()` function.
    //
    // Some Background on Jobservers
    // -----------------------------
    // It's worth also touching on the management of parallelism here. We don't
    // want to just spawn a thread per work item because while that's optimal
    // parallelism it may overload a system with too many threads or violate our
    // configuration for the maximum amount of cpu to use for this process. To
    // manage this we use the `jobserver` crate.
    //
    // Job servers are an artifact of GNU make and are used to manage
    // parallelism between processes. A jobserver is a glorified IPC semaphore
    // basically. Whenever we want to run some work we acquire the semaphore,
    // and whenever we're done with that work we release the semaphore. In this
    // manner we can ensure that the maximum number of parallel workers is
    // capped at any one point in time.
    //
    // LTO and the coordinator thread
    // ------------------------------
    //
    // The final job the coordinator thread is responsible for is managing LTO
    // and how that works. When LTO is requested what we'll do is collect all
    // optimized LLVM modules into a local vector on the coordinator. Once all
    // modules have been codegened and optimized we hand this to the `lto`
    // module for further optimization. The `lto` module will return back a list
    // of more modules to work on, which the coordinator will continue to spawn
    // work for.
    //
    // Each LLVM module is automatically sent back to the coordinator for LTO if
    // necessary. There's already optimizations in place to avoid sending work
    // back to the coordinator if LTO isn't requested.
    return B::spawn_named_thread(cgcx.time_trace, "coordinator".to_string(), move || {
        let mut worker_id_counter = 0;
        let mut free_worker_ids = Vec::new();
        let mut get_worker_id = |free_worker_ids: &mut Vec<usize>| {
            if let Some(id) = free_worker_ids.pop() {
                id
            } else {
                let id = worker_id_counter;
                worker_id_counter += 1;
                id
            }
        };

        // This is where we collect codegen units that have gone all the way
        // through codegen and LLVM.
        let mut autodiff_items = Vec::new();
        let mut compiled_modules = vec![];
        let mut compiled_allocator_module = None;
        let mut needs_link = Vec::new();
        let mut needs_fat_lto = Vec::new();
        let mut needs_thin_lto = Vec::new();
        let mut lto_import_only_modules = Vec::new();
        let mut started_lto = false;

        /// Possible state transitions:
        /// - Ongoing -> Completed
        /// - Ongoing -> Aborted
        /// - Completed -> Aborted
        #[derive(Debug, PartialEq)]
        enum CodegenState {
            Ongoing,
            Completed,
            Aborted,
        }
        use CodegenState::*;
        let mut codegen_state = Ongoing;

        // This is the queue of LLVM work items that still need processing.
        let mut work_items = Vec::<(WorkItem<B>, u64)>::new();

        // This are the Jobserver Tokens we currently hold. Does not include
        // the implicit Token the compiler process owns no matter what.
        let mut tokens = Vec::new();

        let mut main_thread_state = MainThreadState::Idle;

        // How many LLVM worker threads are running while holding a Token. This
        // *excludes* any that the main thread is lending a Token to.
        let mut running_with_own_token = 0;

        // How many LLVM worker threads are running in total. This *includes*
        // any that the main thread is lending a Token to.
        let running_with_any_token = |main_thread_state, running_with_own_token| {
            running_with_own_token
                + if main_thread_state == MainThreadState::Lending { 1 } else { 0 }
        };

        let mut llvm_start_time: Option<VerboseTimingGuard<'_>> = None;

        // Run the message loop while there's still anything that needs message
        // processing. Note that as soon as codegen is aborted we simply want to
        // wait for all existing work to finish, so many of the conditions here
        // only apply if codegen hasn't been aborted as they represent pending
        // work to be done.
        loop {
            // While there are still CGUs to be codegened, the coordinator has
            // to decide how to utilize the compiler processes implicit Token:
            // For codegenning more CGU or for running them through LLVM.
            if codegen_state == Ongoing {
                if main_thread_state == MainThreadState::Idle {
                    // Compute the number of workers that will be running once we've taken as many
                    // items from the work queue as we can, plus one for the main thread. It's not
                    // critically important that we use this instead of just
                    // `running_with_own_token`, but it prevents the `queue_full_enough` heuristic
                    // from fluctuating just because a worker finished up and we decreased the
                    // `running_with_own_token` count, even though we're just going to increase it
                    // right after this when we put a new worker to work.
                    let extra_tokens = tokens.len().checked_sub(running_with_own_token).unwrap();
                    let additional_running = std::cmp::min(extra_tokens, work_items.len());
                    let anticipated_running = running_with_own_token + additional_running + 1;

                    if !queue_full_enough(work_items.len(), anticipated_running) {
                        // The queue is not full enough, process more codegen units:
                        if codegen_worker_send.send(CguMessage).is_err() {
                            panic!("Could not send CguMessage to main thread")
                        }
                        main_thread_state = MainThreadState::Codegenning;
                    } else {
                        // The queue is full enough to not let the worker
                        // threads starve. Use the implicit Token to do some
                        // LLVM work too.
                        let (item, _) =
                            work_items.pop().expect("queue empty - queue_full_enough() broken?");
                        main_thread_state = MainThreadState::Lending;
                        spawn_work(
                            &cgcx,
                            &mut llvm_start_time,
                            get_worker_id(&mut free_worker_ids),
                            item,
                        );
                    }
                }
            } else if codegen_state == Completed {
                if running_with_any_token(main_thread_state, running_with_own_token) == 0
                    && work_items.is_empty()
                {
                    // All codegen work is done. Do we have LTO work to do?
                    if needs_fat_lto.is_empty()
                        && needs_thin_lto.is_empty()
                        && lto_import_only_modules.is_empty()
                    {
                        // Nothing more to do!
                        break;
                    }

                    // We have LTO work to do. Perform the serial work here of
                    // figuring out what we're going to LTO and then push a
                    // bunch of work items onto our queue to do LTO. This all
                    // happens on the coordinator thread but it's very quick so
                    // we don't worry about tokens.
                    assert!(!started_lto);
                    started_lto = true;

                    let needs_fat_lto = mem::take(&mut needs_fat_lto);
                    let needs_thin_lto = mem::take(&mut needs_thin_lto);
                    let import_only_modules = mem::take(&mut lto_import_only_modules);

                    for (work, cost) in generate_lto_work(
                        &cgcx,
                        autodiff_items.clone(),
                        needs_fat_lto,
                        needs_thin_lto,
                        import_only_modules,
                    ) {
                        let insertion_index = work_items
                            .binary_search_by_key(&cost, |&(_, cost)| cost)
                            .unwrap_or_else(|e| e);
                        work_items.insert(insertion_index, (work, cost));
                        if cgcx.parallel {
                            helper.request_token();
                        }
                    }
                }

                // In this branch, we know that everything has been codegened,
                // so it's just a matter of determining whether the implicit
                // Token is free to use for LLVM work.
                match main_thread_state {
                    MainThreadState::Idle => {
                        if let Some((item, _)) = work_items.pop() {
                            main_thread_state = MainThreadState::Lending;
                            spawn_work(
                                &cgcx,
                                &mut llvm_start_time,
                                get_worker_id(&mut free_worker_ids),
                                item,
                            );
                        } else {
                            // There is no unstarted work, so let the main thread
                            // take over for a running worker. Otherwise the
                            // implicit token would just go to waste.
                            // We reduce the `running` counter by one. The
                            // `tokens.truncate()` below will take care of
                            // giving the Token back.
                            assert!(running_with_own_token > 0);
                            running_with_own_token -= 1;
                            main_thread_state = MainThreadState::Lending;
                        }
                    }
                    MainThreadState::Codegenning => bug!(
                        "codegen worker should not be codegenning after \
                              codegen was already completed"
                    ),
                    MainThreadState::Lending => {
                        // Already making good use of that token
                    }
                }
            } else {
                // Don't queue up any more work if codegen was aborted, we're
                // just waiting for our existing children to finish.
                assert!(codegen_state == Aborted);
                if running_with_any_token(main_thread_state, running_with_own_token) == 0 {
                    break;
                }
            }

            // Spin up what work we can, only doing this while we've got available
            // parallelism slots and work left to spawn.
            if codegen_state != Aborted {
                while running_with_own_token < tokens.len()
                    && let Some((item, _)) = work_items.pop()
                {
                    spawn_work(
                        &cgcx,
                        &mut llvm_start_time,
                        get_worker_id(&mut free_worker_ids),
                        item,
                    );
                    running_with_own_token += 1;
                }
            }

            // Relinquish accidentally acquired extra tokens.
            tokens.truncate(running_with_own_token);

            // If a thread exits successfully then we drop a token associated
            // with that worker and update our `running_with_own_token` count.
            // We may later re-acquire a token to continue running more work.
            // We may also not actually drop a token here if the worker was
            // running with an "ephemeral token".
            let mut free_worker = |worker_id| {
                if main_thread_state == MainThreadState::Lending {
                    main_thread_state = MainThreadState::Idle;
                } else {
                    running_with_own_token -= 1;
                }

                free_worker_ids.push(worker_id);
            };

            let msg = coordinator_receive.recv().unwrap();
            match *msg.downcast::<Message<B>>().ok().unwrap() {
                // Save the token locally and the next turn of the loop will use
                // this to spawn a new unit of work, or it may get dropped
                // immediately if we have no more work to spawn.
                Message::Token(token) => {
                    match token {
                        Ok(token) => {
                            tokens.push(token);

                            if main_thread_state == MainThreadState::Lending {
                                // If the main thread token is used for LLVM work
                                // at the moment, we turn that thread into a regular
                                // LLVM worker thread, so the main thread is free
                                // to react to codegen demand.
                                main_thread_state = MainThreadState::Idle;
                                running_with_own_token += 1;
                            }
                        }
                        Err(e) => {
                            let msg = &format!("failed to acquire jobserver token: {e}");
                            shared_emitter.fatal(msg);
                            codegen_state = Aborted;
                        }
                    }
                }

                Message::CodegenDone { llvm_work_item, cost } => {
                    // We keep the queue sorted by estimated processing cost,
                    // so that more expensive items are processed earlier. This
                    // is good for throughput as it gives the main thread more
                    // time to fill up the queue and it avoids scheduling
                    // expensive items to the end.
                    // Note, however, that this is not ideal for memory
                    // consumption, as LLVM module sizes are not evenly
                    // distributed.
                    let insertion_index = work_items.binary_search_by_key(&cost, |&(_, cost)| cost);
                    let insertion_index = match insertion_index {
                        Ok(idx) | Err(idx) => idx,
                    };
                    work_items.insert(insertion_index, (llvm_work_item, cost));

                    if cgcx.parallel {
                        helper.request_token();
                    }
                    assert_eq!(main_thread_state, MainThreadState::Codegenning);
                    main_thread_state = MainThreadState::Idle;
                }

                Message::AddAutoDiffItems(mut items) => {
                    autodiff_items.append(&mut items);
                }

                Message::CodegenComplete => {
                    if codegen_state != Aborted {
                        codegen_state = Completed;
                    }
                    assert_eq!(main_thread_state, MainThreadState::Codegenning);
                    main_thread_state = MainThreadState::Idle;
                }

                // If codegen is aborted that means translation was aborted due
                // to some normal-ish compiler error. In this situation we want
                // to exit as soon as possible, but we want to make sure all
                // existing work has finished. Flag codegen as being done, and
                // then conditions above will ensure no more work is spawned but
                // we'll keep executing this loop until `running_with_own_token`
                // hits 0.
                Message::CodegenAborted => {
                    codegen_state = Aborted;
                }

                Message::WorkItem { result, worker_id } => {
                    free_worker(worker_id);

                    match result {
                        Ok(WorkItemResult::Finished(compiled_module)) => {
                            match compiled_module.kind {
                                ModuleKind::Regular => {
                                    assert!(needs_link.is_empty());
                                    compiled_modules.push(compiled_module);
                                }
                                ModuleKind::Allocator => {
                                    assert!(compiled_allocator_module.is_none());
                                    compiled_allocator_module = Some(compiled_module);
                                }
                            }
                        }
                        Ok(WorkItemResult::NeedsLink(module)) => {
                            assert!(compiled_modules.is_empty());
                            needs_link.push(module);
                        }
                        Ok(WorkItemResult::NeedsFatLto(fat_lto_input)) => {
                            assert!(!started_lto);
                            assert!(needs_thin_lto.is_empty());
                            needs_fat_lto.push(fat_lto_input);
                        }
                        Ok(WorkItemResult::NeedsThinLto(name, thin_buffer)) => {
                            assert!(!started_lto);
                            assert!(needs_fat_lto.is_empty());
                            needs_thin_lto.push((name, thin_buffer));
                        }
                        Err(Some(WorkerFatalError)) => {
                            // Like `CodegenAborted`, wait for remaining work to finish.
                            codegen_state = Aborted;
                        }
                        Err(None) => {
                            // If the thread failed that means it panicked, so
                            // we abort immediately.
                            bug!("worker thread panicked");
                        }
                    }
                }

                Message::AddImportOnlyModule { module_data, work_product } => {
                    assert!(!started_lto);
                    assert_eq!(codegen_state, Ongoing);
                    assert_eq!(main_thread_state, MainThreadState::Codegenning);
                    lto_import_only_modules.push((module_data, work_product));
                    main_thread_state = MainThreadState::Idle;
                }
            }
        }

        if codegen_state == Aborted {
            return Err(());
        }

        let needs_link = mem::take(&mut needs_link);
        if !needs_link.is_empty() {
            assert!(compiled_modules.is_empty());
            let dcx = cgcx.create_dcx();
            let dcx = dcx.handle();
            let module = B::run_link(&cgcx, dcx, needs_link).map_err(|_| ())?;
            let module =
                B::codegen(&cgcx, dcx, module, cgcx.config(ModuleKind::Regular)).map_err(|_| ())?;
            compiled_modules.push(module);
        }

        // Drop to print timings
        drop(llvm_start_time);

        // Regardless of what order these modules completed in, report them to
        // the backend in the same order every time to ensure that we're handing
        // out deterministic results.
        compiled_modules.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(CompiledModules {
            modules: compiled_modules,
            allocator_module: compiled_allocator_module,
        })
    })
    .expect("failed to spawn coordinator thread");

    // A heuristic that determines if we have enough LLVM WorkItems in the
    // queue so that the main thread can do LLVM work instead of codegen
    fn queue_full_enough(items_in_queue: usize, workers_running: usize) -> bool {
        // This heuristic scales ahead-of-time codegen according to available
        // concurrency, as measured by `workers_running`. The idea is that the
        // more concurrency we have available, the more demand there will be for
        // work items, and the fuller the queue should be kept to meet demand.
        // An important property of this approach is that we codegen ahead of
        // time only as much as necessary, so as to keep fewer LLVM modules in
        // memory at once, thereby reducing memory consumption.
        //
        // When the number of workers running is less than the max concurrency
        // available to us, this heuristic can cause us to instruct the main
        // thread to work on an LLVM item (that is, tell it to "LLVM") instead
        // of codegen, even though it seems like it *should* be codegenning so
        // that we can create more work items and spawn more LLVM workers.
        //
        // But this is not a problem. When the main thread is told to LLVM,
        // according to this heuristic and how work is scheduled, there is
        // always at least one item in the queue, and therefore at least one
        // pending jobserver token request. If there *is* more concurrency
        // available, we will immediately receive a token, which will upgrade
        // the main thread's LLVM worker to a real one (conceptually), and free
        // up the main thread to codegen if necessary. On the other hand, if
        // there isn't more concurrency, then the main thread working on an LLVM
        // item is appropriate, as long as the queue is full enough for demand.
        //
        // Speaking of which, how full should we keep the queue? Probably less
        // full than you'd think. A lot has to go wrong for the queue not to be
        // full enough and for that to have a negative effect on compile times.
        //
        // Workers are unlikely to finish at exactly the same time, so when one
        // finishes and takes another work item off the queue, we often have
        // ample time to codegen at that point before the next worker finishes.
        // But suppose that codegen takes so long that the workers exhaust the
        // queue, and we have one or more workers that have nothing to work on.
        // Well, it might not be so bad. Of all the LLVM modules we create and
        // optimize, one has to finish last. It's not necessarily the case that
        // by losing some concurrency for a moment, we delay the point at which
        // that last LLVM module is finished and the rest of compilation can
        // proceed. Also, when we can't take advantage of some concurrency, we
        // give tokens back to the job server. That enables some other rustc to
        // potentially make use of the available concurrency. That could even
        // *decrease* overall compile time if we're lucky. But yes, if no other
        // rustc can make use of the concurrency, then we've squandered it.
        //
        // However, keeping the queue full is also beneficial when we have a
        // surge in available concurrency. Then items can be taken from the
        // queue immediately, without having to wait for codegen.
        //
        // So, the heuristic below tries to keep one item in the queue for every
        // four running workers. Based on limited benchmarking, this appears to
        // be more than sufficient to avoid increasing compilation times.
        let quarter_of_workers = workers_running - 3 * workers_running / 4;
        items_in_queue > 0 && items_in_queue >= quarter_of_workers
    }
}

/// `FatalError` is explicitly not `Send`.
#[must_use]
pub(crate) struct WorkerFatalError;

fn spawn_work<'a, B: ExtraBackendMethods>(
    cgcx: &'a CodegenContext<B>,
    llvm_start_time: &mut Option<VerboseTimingGuard<'a>>,
    worker_id: usize,
    work: WorkItem<B>,
) {
    if cgcx.config(work.module_kind()).time_module && llvm_start_time.is_none() {
        *llvm_start_time = Some(cgcx.prof.verbose_generic_activity("LLVM_passes"));
    }

    let cgcx = cgcx.clone();

    B::spawn_named_thread(cgcx.time_trace, work.short_description(), move || {
        // Set up a destructor which will fire off a message that we're done as
        // we exit.
        struct Bomb<B: ExtraBackendMethods> {
            coordinator_send: Sender<Box<dyn Any + Send>>,
            result: Option<Result<WorkItemResult<B>, FatalError>>,
            worker_id: usize,
        }
        impl<B: ExtraBackendMethods> Drop for Bomb<B> {
            fn drop(&mut self) {
                let worker_id = self.worker_id;
                let msg = match self.result.take() {
                    Some(Ok(result)) => Message::WorkItem::<B> { result: Ok(result), worker_id },
                    Some(Err(FatalError)) => {
                        Message::WorkItem::<B> { result: Err(Some(WorkerFatalError)), worker_id }
                    }
                    None => Message::WorkItem::<B> { result: Err(None), worker_id },
                };
                drop(self.coordinator_send.send(Box::new(msg)));
            }
        }

        let mut bomb =
            Bomb::<B> { coordinator_send: cgcx.coordinator_send.clone(), result: None, worker_id };

        // Execute the work itself, and if it finishes successfully then flag
        // ourselves as a success as well.
        //
        // Note that we ignore any `FatalError` coming out of `execute_work_item`,
        // as a diagnostic was already sent off to the main thread - just
        // surface that there was an error in this worker.
        bomb.result = {
            let module_config = cgcx.config(work.module_kind());

            Some(match work {
                WorkItem::Optimize(m) => {
                    let _timer =
                        cgcx.prof.generic_activity_with_arg("codegen_module_optimize", &*m.name);
                    execute_optimize_work_item(&cgcx, m, module_config)
                }
                WorkItem::CopyPostLtoArtifacts(m) => {
                    let _timer = cgcx.prof.generic_activity_with_arg(
                        "codegen_copy_artifacts_from_incr_cache",
                        &*m.name,
                    );
                    Ok(execute_copy_from_cache_work_item(&cgcx, m, module_config))
                }
                WorkItem::LTO(m) => {
                    let _timer =
                        cgcx.prof.generic_activity_with_arg("codegen_module_perform_lto", m.name());
                    execute_lto_work_item(&cgcx, m, module_config)
                }
            })
        };
    })
    .expect("failed to spawn work thread");
}

enum SharedEmitterMessage {
    Diagnostic(Diagnostic),
    InlineAsmError(SpanData, String, Level, Option<(String, Vec<InnerSpan>)>),
    Fatal(String),
}

#[derive(Clone)]
pub struct SharedEmitter {
    sender: Sender<SharedEmitterMessage>,
}

pub struct SharedEmitterMain {
    receiver: Receiver<SharedEmitterMessage>,
}

impl SharedEmitter {
    fn new() -> (SharedEmitter, SharedEmitterMain) {
        let (sender, receiver) = channel();

        (SharedEmitter { sender }, SharedEmitterMain { receiver })
    }

    pub fn inline_asm_error(
        &self,
        span: SpanData,
        msg: String,
        level: Level,
        source: Option<(String, Vec<InnerSpan>)>,
    ) {
        drop(self.sender.send(SharedEmitterMessage::InlineAsmError(span, msg, level, source)));
    }

    fn fatal(&self, msg: &str) {
        drop(self.sender.send(SharedEmitterMessage::Fatal(msg.to_string())));
    }
}

impl Translate for SharedEmitter {
    fn fluent_bundle(&self) -> Option<&FluentBundle> {
        None
    }

    fn fallback_fluent_bundle(&self) -> &FluentBundle {
        panic!("shared emitter attempted to translate a diagnostic");
    }
}

impl Emitter for SharedEmitter {
    fn emit_diagnostic(
        &mut self,
        mut diag: rustc_errors::DiagInner,
        _registry: &rustc_errors::registry::Registry,
    ) {
        // Check that we aren't missing anything interesting when converting to
        // the cut-down local `DiagInner`.
        assert_eq!(diag.span, MultiSpan::new());
        assert_eq!(diag.suggestions, Suggestions::Enabled(vec![]));
        assert_eq!(diag.sort_span, rustc_span::DUMMY_SP);
        assert_eq!(diag.is_lint, None);
        // No sensible check for `diag.emitted_at`.

        let args = mem::replace(&mut diag.args, DiagArgMap::default());
        drop(
            self.sender.send(SharedEmitterMessage::Diagnostic(Diagnostic {
                level: diag.level(),
                messages: diag.messages,
                code: diag.code,
                children: diag
                    .children
                    .into_iter()
                    .map(|child| Subdiagnostic { level: child.level, messages: child.messages })
                    .collect(),
                args,
            })),
        );
    }

    fn source_map(&self) -> Option<&SourceMap> {
        None
    }
}

impl SharedEmitterMain {
    fn check(&self, sess: &Session, blocking: bool) {
        loop {
            let message = if blocking {
                match self.receiver.recv() {
                    Ok(message) => Ok(message),
                    Err(_) => Err(()),
                }
            } else {
                match self.receiver.try_recv() {
                    Ok(message) => Ok(message),
                    Err(_) => Err(()),
                }
            };

            match message {
                Ok(SharedEmitterMessage::Diagnostic(diag)) => {
                    // The diagnostic has been received on the main thread.
                    // Convert it back to a full `Diagnostic` and emit.
                    let dcx = sess.dcx();
                    let mut d =
                        rustc_errors::DiagInner::new_with_messages(diag.level, diag.messages);
                    d.code = diag.code; // may be `None`, that's ok
                    d.children = diag
                        .children
                        .into_iter()
                        .map(|sub| rustc_errors::Subdiag {
                            level: sub.level,
                            messages: sub.messages,
                            span: MultiSpan::new(),
                        })
                        .collect();
                    d.args = diag.args;
                    dcx.emit_diagnostic(d);
                    sess.dcx().abort_if_errors();
                }
                Ok(SharedEmitterMessage::InlineAsmError(span, msg, level, source)) => {
                    assert_matches!(level, Level::Error | Level::Warning | Level::Note);
                    let mut err = Diag::<()>::new(sess.dcx(), level, msg);
                    if !span.is_dummy() {
                        err.span(span.span());
                    }

                    // Point to the generated assembly if it is available.
                    if let Some((buffer, spans)) = source {
                        let source = sess
                            .source_map()
                            .new_source_file(FileName::inline_asm_source_code(&buffer), buffer);
                        let spans: Vec<_> = spans
                            .iter()
                            .map(|sp| {
                                Span::with_root_ctxt(
                                    source.normalized_byte_pos(sp.start as u32),
                                    source.normalized_byte_pos(sp.end as u32),
                                )
                            })
                            .collect();
                        err.span_note(spans, "instantiated into assembly here");
                    }

                    err.emit();
                }
                Ok(SharedEmitterMessage::Fatal(msg)) => {
                    sess.dcx().fatal(msg);
                }
                Err(_) => {
                    break;
                }
            }
        }
    }
}

pub struct Coordinator<B: ExtraBackendMethods> {
    pub sender: Sender<Box<dyn Any + Send>>,
    future: Option<thread::JoinHandle<Result<CompiledModules, ()>>>,
    // Only used for the Message type.
    phantom: PhantomData<B>,
}

impl<B: ExtraBackendMethods> Coordinator<B> {
    fn join(mut self) -> std::thread::Result<Result<CompiledModules, ()>> {
        self.future.take().unwrap().join()
    }
}

impl<B: ExtraBackendMethods> Drop for Coordinator<B> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            // If we haven't joined yet, signal to the coordinator that it should spawn no more
            // work, and wait for worker threads to finish.
            drop(self.sender.send(Box::new(Message::CodegenAborted::<B>)));
            drop(future.join());
        }
    }
}

pub struct OngoingCodegen<B: ExtraBackendMethods> {
    pub backend: B,
    pub crate_info: CrateInfo,
    pub codegen_worker_receive: Receiver<CguMessage>,
    pub shared_emitter_main: SharedEmitterMain,
    pub output_filenames: Arc<OutputFilenames>,
    pub coordinator: Coordinator<B>,
}

impl<B: ExtraBackendMethods> OngoingCodegen<B> {
    pub fn join(self, sess: &Session) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        self.shared_emitter_main.check(sess, true);
        let compiled_modules = sess.time("join_worker_thread", || match self.coordinator.join() {
            Ok(Ok(compiled_modules)) => compiled_modules,
            Ok(Err(())) => {
                sess.dcx().abort_if_errors();
                panic!("expected abort due to worker thread errors")
            }
            Err(_) => {
                bug!("panic during codegen/LLVM phase");
            }
        });

        sess.dcx().abort_if_errors();

        let work_products =
            copy_all_cgu_workproducts_to_incr_comp_cache_dir(sess, &compiled_modules);
        produce_final_output_artifacts(sess, &compiled_modules, &self.output_filenames);

        // FIXME: time_llvm_passes support - does this use a global context or
        // something?
        if sess.codegen_units().as_usize() == 1 && sess.opts.unstable_opts.time_llvm_passes {
            self.backend.print_pass_timings()
        }

        if sess.print_llvm_stats() {
            self.backend.print_statistics()
        }

        (
            CodegenResults {
                crate_info: self.crate_info,

                modules: compiled_modules.modules,
                allocator_module: compiled_modules.allocator_module,
            },
            work_products,
        )
    }

    pub(crate) fn codegen_finished(&self, tcx: TyCtxt<'_>) {
        self.wait_for_signal_to_codegen_item();
        self.check_for_errors(tcx.sess);
        drop(self.coordinator.sender.send(Box::new(Message::CodegenComplete::<B>)));
    }

    pub(crate) fn submit_autodiff_items(&self, items: Vec<AutoDiffItem>) {
        drop(self.coordinator.sender.send(Box::new(Message::<B>::AddAutoDiffItems(items))));
    }

    pub(crate) fn check_for_errors(&self, sess: &Session) {
        self.shared_emitter_main.check(sess, false);
    }

    pub(crate) fn wait_for_signal_to_codegen_item(&self) {
        match self.codegen_worker_receive.recv() {
            Ok(CguMessage) => {
                // Ok to proceed.
            }
            Err(_) => {
                // One of the LLVM threads must have panicked, fall through so
                // error handling can be reached.
            }
        }
    }
}

pub(crate) fn submit_codegened_module_to_llvm<B: ExtraBackendMethods>(
    _backend: &B,
    tx_to_llvm_workers: &Sender<Box<dyn Any + Send>>,
    module: ModuleCodegen<B::Module>,
    cost: u64,
) {
    let llvm_work_item = WorkItem::Optimize(module);
    drop(tx_to_llvm_workers.send(Box::new(Message::CodegenDone::<B> { llvm_work_item, cost })));
}

pub(crate) fn submit_post_lto_module_to_llvm<B: ExtraBackendMethods>(
    _backend: &B,
    tx_to_llvm_workers: &Sender<Box<dyn Any + Send>>,
    module: CachedModuleCodegen,
) {
    let llvm_work_item = WorkItem::CopyPostLtoArtifacts(module);
    drop(tx_to_llvm_workers.send(Box::new(Message::CodegenDone::<B> { llvm_work_item, cost: 0 })));
}

pub(crate) fn submit_pre_lto_module_to_llvm<B: ExtraBackendMethods>(
    _backend: &B,
    tcx: TyCtxt<'_>,
    tx_to_llvm_workers: &Sender<Box<dyn Any + Send>>,
    module: CachedModuleCodegen,
) {
    let filename = pre_lto_bitcode_filename(&module.name);
    let bc_path = in_incr_comp_dir_sess(tcx.sess, &filename);
    let file = fs::File::open(&bc_path)
        .unwrap_or_else(|e| panic!("failed to open bitcode file `{}`: {}", bc_path.display(), e));

    let mmap = unsafe {
        Mmap::map(file).unwrap_or_else(|e| {
            panic!("failed to mmap bitcode file `{}`: {}", bc_path.display(), e)
        })
    };
    // Schedule the module to be loaded
    drop(tx_to_llvm_workers.send(Box::new(Message::AddImportOnlyModule::<B> {
        module_data: SerializedModule::FromUncompressedFile(mmap),
        work_product: module.source,
    })));
}

fn pre_lto_bitcode_filename(module_name: &str) -> String {
    format!("{module_name}.{PRE_LTO_BC_EXT}")
}

fn msvc_imps_needed(tcx: TyCtxt<'_>) -> bool {
    // This should never be true (because it's not supported). If it is true,
    // something is wrong with commandline arg validation.
    assert!(
        !(tcx.sess.opts.cg.linker_plugin_lto.enabled()
            && tcx.sess.target.is_like_windows
            && tcx.sess.opts.cg.prefer_dynamic)
    );

    // We need to generate _imp__ symbol if we are generating an rlib or we include one
    // indirectly from ThinLTO. In theory these are not needed as ThinLTO could resolve
    // these, but it currently does not do so.
    let can_have_static_objects =
        tcx.sess.lto() == Lto::Thin || tcx.crate_types().contains(&CrateType::Rlib);

    tcx.sess.target.is_like_windows &&
    can_have_static_objects   &&
    // ThinLTO can't handle this workaround in all cases, so we don't
    // emit the `__imp_` symbols. Instead we make them unnecessary by disallowing
    // dynamic linking when linker plugin LTO is enabled.
    !tcx.sess.opts.cg.linker_plugin_lto.enabled()
}
