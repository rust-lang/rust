use crate::{ModuleCodegen, ModuleKind, CachedModuleCodegen, CompiledModule, CrateInfo,
    CodegenResults, RLIB_BYTECODE_EXTENSION};
use super::linker::LinkerInfo;
use super::lto::{self, SerializedModule};
use super::link::{self, remove, get_linker};
use super::command::Command;
use super::symbol_export::ExportedSymbols;

use crate::traits::*;
use rustc_incremental::{copy_cgu_workproducts_to_incr_comp_cache_dir,
                        in_incr_comp_dir, in_incr_comp_dir_sess};
use rustc::dep_graph::{WorkProduct, WorkProductId, WorkProductFileKind};
use rustc::dep_graph::cgu_reuse_tracker::CguReuseTracker;
use rustc::middle::cstore::EncodedMetadata;
use rustc::session::config::{self, OutputFilenames, OutputType, Passes, Lto,
                             Sanitizer, SwitchWithOptPath};
use rustc::session::Session;
use rustc::util::nodemap::FxHashMap;
use rustc::hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc::ty::TyCtxt;
use rustc::util::common::{time_depth, set_time_depth, print_time_passes_entry};
use rustc::util::profiling::SelfProfiler;
use rustc_fs_util::link_or_copy;
use rustc_data_structures::svh::Svh;
use rustc_errors::{Handler, Level, DiagnosticBuilder, FatalError, DiagnosticId};
use rustc_errors::emitter::{Emitter};
use rustc_target::spec::MergeFunctions;
use syntax::attr;
use syntax::ext::hygiene::Mark;
use syntax_pos::MultiSpan;
use syntax_pos::symbol::{Symbol, sym};
use jobserver::{Client, Acquired};

use std::any::Any;
use std::borrow::Cow;
use std::fs;
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::time::Instant;
use std::thread;

const PRE_LTO_BC_EXT: &str = "pre-lto.bc";

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

    // Flags indicating which outputs to produce.
    pub emit_pre_lto_bc: bool,
    pub emit_no_opt_bc: bool,
    pub emit_bc: bool,
    pub emit_bc_compressed: bool,
    pub emit_lto_bc: bool,
    pub emit_ir: bool,
    pub emit_asm: bool,
    pub emit_obj: bool,
    // Miscellaneous flags.  These are mostly copied from command-line
    // options.
    pub verify_llvm_ir: bool,
    pub no_prepopulate_passes: bool,
    pub no_builtins: bool,
    pub time_passes: bool,
    pub vectorize_loop: bool,
    pub vectorize_slp: bool,
    pub merge_functions: bool,
    pub inline_threshold: Option<usize>,
    // Instead of creating an object file by doing LLVM codegen, just
    // make the object file bitcode. Provides easy compatibility with
    // emscripten's ecc compiler, when used as the linker.
    pub obj_is_bitcode: bool,
    pub no_integrated_as: bool,
    pub embed_bitcode: bool,
    pub embed_bitcode_marker: bool,
}

impl ModuleConfig {
    fn new(passes: Vec<String>) -> ModuleConfig {
        ModuleConfig {
            passes,
            opt_level: None,
            opt_size: None,

            pgo_gen: SwitchWithOptPath::Disabled,
            pgo_use: None,

            emit_no_opt_bc: false,
            emit_pre_lto_bc: false,
            emit_bc: false,
            emit_bc_compressed: false,
            emit_lto_bc: false,
            emit_ir: false,
            emit_asm: false,
            emit_obj: false,
            obj_is_bitcode: false,
            embed_bitcode: false,
            embed_bitcode_marker: false,
            no_integrated_as: false,

            verify_llvm_ir: false,
            no_prepopulate_passes: false,
            no_builtins: false,
            time_passes: false,
            vectorize_loop: false,
            vectorize_slp: false,
            merge_functions: false,
            inline_threshold: None
        }
    }

    fn set_flags(&mut self, sess: &Session, no_builtins: bool) {
        self.verify_llvm_ir = sess.verify_llvm_ir();
        self.no_prepopulate_passes = sess.opts.cg.no_prepopulate_passes;
        self.no_builtins = no_builtins || sess.target.target.options.no_builtins;
        self.time_passes = sess.time_extended();
        self.inline_threshold = sess.opts.cg.inline_threshold;
        self.obj_is_bitcode = sess.target.target.options.obj_is_bitcode ||
                              sess.opts.cg.linker_plugin_lto.enabled();
        let embed_bitcode = sess.target.target.options.embed_bitcode ||
                            sess.opts.debugging_opts.embed_bitcode;
        if embed_bitcode {
            match sess.opts.optimize {
                config::OptLevel::No |
                config::OptLevel::Less => {
                    self.embed_bitcode_marker = embed_bitcode;
                }
                _ => self.embed_bitcode = embed_bitcode,
            }
        }

        // Copy what clang does by turning on loop vectorization at O2 and
        // slp vectorization at O3. Otherwise configure other optimization aspects
        // of this pass manager builder.
        // Turn off vectorization for emscripten, as it's not very well supported.
        self.vectorize_loop = !sess.opts.cg.no_vectorize_loops &&
                             (sess.opts.optimize == config::OptLevel::Default ||
                              sess.opts.optimize == config::OptLevel::Aggressive) &&
                             !sess.target.target.options.is_like_emscripten;

        self.vectorize_slp = !sess.opts.cg.no_vectorize_slp &&
                            sess.opts.optimize == config::OptLevel::Aggressive &&
                            !sess.target.target.options.is_like_emscripten;

        // Some targets (namely, NVPTX) interact badly with the MergeFunctions
        // pass. This is because MergeFunctions can generate new function calls
        // which may interfere with the target calling convention; e.g. for the
        // NVPTX target, PTX kernels should not call other PTX kernels.
        // MergeFunctions can also be configured to generate aliases instead,
        // but aliases are not supported by some backends (again, NVPTX).
        // Therefore, allow targets to opt out of the MergeFunctions pass,
        // but otherwise keep the pass enabled (at O2 and O3) since it can be
        // useful for reducing code size.
        self.merge_functions = match sess.opts.debugging_opts.merge_functions
                                     .unwrap_or(sess.target.target.options.merge_functions) {
            MergeFunctions::Disabled => false,
            MergeFunctions::Trampolines |
            MergeFunctions::Aliases => {
                sess.opts.optimize == config::OptLevel::Default ||
                sess.opts.optimize == config::OptLevel::Aggressive
            }
        };
    }

    pub fn bitcode_needed(&self) -> bool {
        self.emit_bc || self.obj_is_bitcode
            || self.emit_bc_compressed || self.embed_bitcode
    }
}

/// Assembler name and command used by codegen when no_integrated_as is enabled
pub struct AssemblerCommand {
    name: PathBuf,
    cmd: Command,
}

// HACK(eddyb) work around `#[derive]` producing wrong bounds for `Clone`.
pub struct TargetMachineFactory<B: WriteBackendMethods>(
    pub Arc<dyn Fn() -> Result<B::TargetMachine, String> + Send + Sync>,
);

impl<B: WriteBackendMethods> Clone for TargetMachineFactory<B> {
    fn clone(&self) -> Self {
        TargetMachineFactory(self.0.clone())
    }
}

pub struct ProfileGenericActivityTimer {
    profiler: Option<Arc<SelfProfiler>>,
    label: Cow<'static, str>,
}

impl ProfileGenericActivityTimer {
    pub fn start(
        profiler: Option<Arc<SelfProfiler>>,
        label: Cow<'static, str>,
    ) -> ProfileGenericActivityTimer {
        if let Some(profiler) = &profiler {
            profiler.start_activity(label.clone());
        }

        ProfileGenericActivityTimer {
            profiler,
            label,
        }
    }
}

impl Drop for ProfileGenericActivityTimer {
    fn drop(&mut self) {
        if let Some(profiler) = &self.profiler {
            profiler.end_activity(self.label.clone());
        }
    }
}

/// Additional resources used by optimize_and_codegen (not module specific)
#[derive(Clone)]
pub struct CodegenContext<B: WriteBackendMethods> {
    // Resources needed when running LTO
    pub backend: B,
    pub time_passes: bool,
    pub profiler: Option<Arc<SelfProfiler>>,
    pub lto: Lto,
    pub no_landing_pads: bool,
    pub save_temps: bool,
    pub fewer_names: bool,
    pub exported_symbols: Option<Arc<ExportedSymbols>>,
    pub opts: Arc<config::Options>,
    pub crate_types: Vec<config::CrateType>,
    pub each_linked_rlib_for_lto: Vec<(CrateNum, PathBuf)>,
    pub output_filenames: Arc<OutputFilenames>,
    pub regular_module_config: Arc<ModuleConfig>,
    pub metadata_module_config: Arc<ModuleConfig>,
    pub allocator_module_config: Arc<ModuleConfig>,
    pub tm_factory: TargetMachineFactory<B>,
    pub msvc_imps_needed: bool,
    pub target_pointer_width: String,
    pub target_arch: String,
    pub debuginfo: config::DebugInfo,

    // Number of cgus excluding the allocator/metadata modules
    pub total_cgus: usize,
    // Handler to use for diagnostics produced during codegen.
    pub diag_emitter: SharedEmitter,
    // LLVM passes added by plugins.
    pub plugin_passes: Vec<String>,
    // LLVM optimizations for which we want to print remarks.
    pub remark: Passes,
    // Worker thread number
    pub worker: usize,
    // The incremental compilation session directory, or None if we are not
    // compiling incrementally
    pub incr_comp_session_dir: Option<PathBuf>,
    // Used to update CGU re-use information during the thinlto phase.
    pub cgu_reuse_tracker: CguReuseTracker,
    // Channel back to the main control thread to send messages to
    pub coordinator_send: Sender<Box<dyn Any + Send>>,
    // The assembler command if no_integrated_as option is enabled, None otherwise
    pub assembler_cmd: Option<Arc<AssemblerCommand>>
}

impl<B: WriteBackendMethods> CodegenContext<B> {
    pub fn create_diag_handler(&self) -> Handler {
        Handler::with_emitter(true, None, Box::new(self.diag_emitter.clone()))
    }

    pub fn config(&self, kind: ModuleKind) -> &ModuleConfig {
        match kind {
            ModuleKind::Regular => &self.regular_module_config,
            ModuleKind::Metadata => &self.metadata_module_config,
            ModuleKind::Allocator => &self.allocator_module_config,
        }
    }

    #[inline(never)]
    #[cold]
    fn profiler_active<F: FnOnce(&SelfProfiler) -> ()>(&self, f: F) {
        match &self.profiler {
            None => bug!("profiler_active() called but there was no profiler active"),
            Some(profiler) => {
                f(&*profiler);
            }
        }
    }

    #[inline(always)]
    pub fn profile<F: FnOnce(&SelfProfiler) -> ()>(&self, f: F) {
        if unlikely!(self.profiler.is_some()) {
            self.profiler_active(f)
        }
    }

    pub fn profile_activity(
        &self,
        label: impl Into<Cow<'static, str>>,
    ) -> ProfileGenericActivityTimer {
        ProfileGenericActivityTimer::start(self.profiler.clone(), label.into())
    }
}

fn generate_lto_work<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    needs_fat_lto: Vec<FatLTOInput<B>>,
    needs_thin_lto: Vec<(String, B::ThinBuffer)>,
    import_only_modules: Vec<(SerializedModule<B::ModuleBuffer>, WorkProduct)>
) -> Vec<(WorkItem<B>, u64)> {
    cgcx.profile(|p| p.start_activity("codegen_run_lto"));

    let (lto_modules, copy_jobs) = if !needs_fat_lto.is_empty() {
        assert!(needs_thin_lto.is_empty());
        let lto_module = B::run_fat_lto(
            cgcx,
            needs_fat_lto,
            import_only_modules,
        )
        .unwrap_or_else(|e| e.raise());
        (vec![lto_module], vec![])
    } else {
        assert!(needs_fat_lto.is_empty());
        B::run_thin_lto(cgcx, needs_thin_lto, import_only_modules)
            .unwrap_or_else(|e| e.raise())
    };

    let result = lto_modules.into_iter().map(|module| {
        let cost = module.cost();
        (WorkItem::LTO(module), cost)
    }).chain(copy_jobs.into_iter().map(|wp| {
        (WorkItem::CopyPostLtoArtifacts(CachedModuleCodegen {
            name: wp.cgu_name.clone(),
            source: wp,
        }), 0)
    })).collect();

    cgcx.profile(|p| p.end_activity("codegen_run_lto"));

    result
}

pub struct CompiledModules {
    pub modules: Vec<CompiledModule>,
    pub metadata_module: Option<CompiledModule>,
    pub allocator_module: Option<CompiledModule>,
}

fn need_crate_bitcode_for_rlib(sess: &Session) -> bool {
    sess.crate_types.borrow().contains(&config::CrateType::Rlib) &&
    sess.opts.output_types.contains_key(&OutputType::Exe)
}

fn need_pre_lto_bitcode_for_incr_comp(sess: &Session) -> bool {
    if sess.opts.incremental.is_none() {
        return false
    }

    match sess.lto() {
        Lto::No => false,
        Lto::Fat |
        Lto::Thin |
        Lto::ThinLocal => true,
    }
}

pub fn start_async_codegen<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'_>,
    metadata: EncodedMetadata,
    coordinator_receive: Receiver<Box<dyn Any + Send>>,
    total_cgus: usize,
) -> OngoingCodegen<B> {
    let sess = tcx.sess;
    let crate_name = tcx.crate_name(LOCAL_CRATE);
    let crate_hash = tcx.crate_hash(LOCAL_CRATE);
    let no_builtins = attr::contains_name(&tcx.hir().krate().attrs, sym::no_builtins);
    let subsystem = attr::first_attr_value_str_by_name(&tcx.hir().krate().attrs,
                                                       sym::windows_subsystem);
    let windows_subsystem = subsystem.map(|subsystem| {
        if subsystem != sym::windows && subsystem != sym::console {
            tcx.sess.fatal(&format!("invalid windows subsystem `{}`, only \
                                     `windows` and `console` are allowed",
                                    subsystem));
        }
        subsystem.to_string()
    });

    let linker_info = LinkerInfo::new(tcx);
    let crate_info = CrateInfo::new(tcx);

    // Figure out what we actually need to build.
    let mut modules_config = ModuleConfig::new(sess.opts.cg.passes.clone());
    let mut metadata_config = ModuleConfig::new(vec![]);
    let mut allocator_config = ModuleConfig::new(vec![]);

    if let Some(ref sanitizer) = sess.opts.debugging_opts.sanitizer {
        match *sanitizer {
            Sanitizer::Address => {
                modules_config.passes.push("asan".to_owned());
                modules_config.passes.push("asan-module".to_owned());
            }
            Sanitizer::Memory => {
                modules_config.passes.push("msan".to_owned())
            }
            Sanitizer::Thread => {
                modules_config.passes.push("tsan".to_owned())
            }
            _ => {}
        }
    }

    if sess.opts.debugging_opts.profile {
        modules_config.passes.push("insert-gcov-profiling".to_owned())
    }

    modules_config.pgo_gen = sess.opts.cg.profile_generate.clone();
    modules_config.pgo_use = sess.opts.cg.profile_use.clone();

    modules_config.opt_level = Some(sess.opts.optimize);
    modules_config.opt_size = Some(sess.opts.optimize);

    // Save all versions of the bytecode if we're saving our temporaries.
    if sess.opts.cg.save_temps {
        modules_config.emit_no_opt_bc = true;
        modules_config.emit_pre_lto_bc = true;
        modules_config.emit_bc = true;
        modules_config.emit_lto_bc = true;
        metadata_config.emit_bc = true;
        allocator_config.emit_bc = true;
    }

    // Emit compressed bitcode files for the crate if we're emitting an rlib.
    // Whenever an rlib is created, the bitcode is inserted into the archive in
    // order to allow LTO against it.
    if need_crate_bitcode_for_rlib(sess) {
        modules_config.emit_bc_compressed = true;
        allocator_config.emit_bc_compressed = true;
    }

    modules_config.emit_pre_lto_bc =
        need_pre_lto_bitcode_for_incr_comp(sess);

    modules_config.no_integrated_as = tcx.sess.opts.cg.no_integrated_as ||
        tcx.sess.target.target.options.no_integrated_as;

    for output_type in sess.opts.output_types.keys() {
        match *output_type {
            OutputType::Bitcode => { modules_config.emit_bc = true; }
            OutputType::LlvmAssembly => { modules_config.emit_ir = true; }
            OutputType::Assembly => {
                modules_config.emit_asm = true;
                // If we're not using the LLVM assembler, this function
                // could be invoked specially with output_type_assembly, so
                // in this case we still want the metadata object file.
                if !sess.opts.output_types.contains_key(&OutputType::Assembly) {
                    metadata_config.emit_obj = true;
                    allocator_config.emit_obj = true;
                }
            }
            OutputType::Object => { modules_config.emit_obj = true; }
            OutputType::Metadata => { metadata_config.emit_obj = true; }
            OutputType::Exe => {
                modules_config.emit_obj = true;
                metadata_config.emit_obj = true;
                allocator_config.emit_obj = true;
            },
            OutputType::Mir => {}
            OutputType::DepInfo => {}
        }
    }

    modules_config.set_flags(sess, no_builtins);
    metadata_config.set_flags(sess, no_builtins);
    allocator_config.set_flags(sess, no_builtins);

    // Exclude metadata and allocator modules from time_passes output, since
    // they throw off the "LLVM passes" measurement.
    metadata_config.time_passes = false;
    allocator_config.time_passes = false;

    let (shared_emitter, shared_emitter_main) = SharedEmitter::new();
    let (codegen_worker_send, codegen_worker_receive) = channel();

    let coordinator_thread = start_executing_work(backend.clone(),
                                                  tcx,
                                                  &crate_info,
                                                  shared_emitter,
                                                  codegen_worker_send,
                                                  coordinator_receive,
                                                  total_cgus,
                                                  sess.jobserver.clone(),
                                                  Arc::new(modules_config),
                                                  Arc::new(metadata_config),
                                                  Arc::new(allocator_config));

    OngoingCodegen {
        backend,
        crate_name,
        crate_hash,
        metadata,
        windows_subsystem,
        linker_info,
        crate_info,

        coordinator_send: tcx.tx_to_llvm_workers.lock().clone(),
        codegen_worker_receive,
        shared_emitter_main,
        future: coordinator_thread,
        output_filenames: tcx.output_filenames(LOCAL_CRATE),
    }
}

fn copy_all_cgu_workproducts_to_incr_comp_cache_dir(
    sess: &Session,
    compiled_modules: &CompiledModules,
) -> FxHashMap<WorkProductId, WorkProduct> {
    let mut work_products = FxHashMap::default();

    if sess.opts.incremental.is_none() {
        return work_products;
    }

    for module in compiled_modules.modules.iter().filter(|m| m.kind == ModuleKind::Regular) {
        let mut files = vec![];

        if let Some(ref path) = module.object {
            files.push((WorkProductFileKind::Object, path.clone()));
        }
        if let Some(ref path) = module.bytecode {
            files.push((WorkProductFileKind::Bytecode, path.clone()));
        }
        if let Some(ref path) = module.bytecode_compressed {
            files.push((WorkProductFileKind::BytecodeCompressed, path.clone()));
        }

        if let Some((id, product)) =
                copy_cgu_workproducts_to_incr_comp_cache_dir(sess, &module.name, &files) {
            work_products.insert(id, product);
        }
    }

    work_products
}

fn produce_final_output_artifacts(sess: &Session,
                                  compiled_modules: &CompiledModules,
                                  crate_output: &OutputFilenames) {
    let mut user_wants_bitcode = false;
    let mut user_wants_objects = false;

    // Produce final compile outputs.
    let copy_gracefully = |from: &Path, to: &Path| {
        if let Err(e) = fs::copy(from, to) {
            sess.err(&format!("could not copy {:?} to {:?}: {}", from, to, e));
        }
    };

    let copy_if_one_unit = |output_type: OutputType,
                            keep_numbered: bool| {
        if compiled_modules.modules.len() == 1 {
            // 1) Only one codegen unit.  In this case it's no difficulty
            //    to copy `foo.0.x` to `foo.x`.
            let module_name = Some(&compiled_modules.modules[0].name[..]);
            let path = crate_output.temp_path(output_type, module_name);
            copy_gracefully(&path,
                            &crate_output.path(output_type));
            if !sess.opts.cg.save_temps && !keep_numbered {
                // The user just wants `foo.x`, not `foo.#module-name#.x`.
                remove(sess, &path);
            }
        } else {
            let ext = crate_output.temp_path(output_type, None)
                                  .extension()
                                  .unwrap()
                                  .to_str()
                                  .unwrap()
                                  .to_owned();

            if crate_output.outputs.contains_key(&output_type) {
                // 2) Multiple codegen units, with `--emit foo=some_name`.  We have
                //    no good solution for this case, so warn the user.
                sess.warn(&format!("ignoring emit path because multiple .{} files \
                                    were produced", ext));
            } else if crate_output.single_output_file.is_some() {
                // 3) Multiple codegen units, with `-o some_name`.  We have
                //    no good solution for this case, so warn the user.
                sess.warn(&format!("ignoring -o because multiple .{} files \
                                    were produced", ext));
            } else {
                // 4) Multiple codegen units, but no explicit name.  We
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
                // Copy to .bc, but always keep the .0.bc.  There is a later
                // check to figure out if we should delete .0.bc files, or keep
                // them for making an rlib.
                copy_if_one_unit(OutputType::Bitcode, true);
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
            OutputType::Mir |
            OutputType::Metadata |
            OutputType::Exe |
            OutputType::DepInfo => {}
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
        // Remove the temporary .#module-name#.o objects.  If the user didn't
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

        let keep_numbered_bitcode = user_wants_bitcode && sess.codegen_units() > 1;

        let keep_numbered_objects = needs_crate_object ||
                (user_wants_objects && sess.codegen_units() > 1);

        for module in compiled_modules.modules.iter() {
            if let Some(ref path) = module.object {
                if !keep_numbered_objects {
                    remove(sess, path);
                }
            }

            if let Some(ref path) = module.bytecode {
                if !keep_numbered_bitcode {
                    remove(sess, path);
                }
            }
        }

        if !user_wants_bitcode {
            if let Some(ref metadata_module) = compiled_modules.metadata_module {
                if let Some(ref path) = metadata_module.bytecode {
                    remove(sess, &path);
                }
            }

            if let Some(ref allocator_module) = compiled_modules.allocator_module {
                if let Some(ref path) = allocator_module.bytecode {
                    remove(sess, path);
                }
            }
        }
    }

    // We leave the following files around by default:
    //  - #crate#.o
    //  - #crate#.crate.metadata.o
    //  - #crate#.bc
    // These are used in linking steps and will be cleaned up afterward.
}

pub fn dump_incremental_data(_codegen_results: &CodegenResults) {
    // FIXME(mw): This does not work at the moment because the situation has
    //            become more complicated due to incremental LTO. Now a CGU
    //            can have more than two caching states.
    // println!("[incremental] Re-using {} out of {} modules",
    //           codegen_results.modules.iter().filter(|m| m.pre_existing).count(),
    //           codegen_results.modules.len());
}

pub enum WorkItem<B: WriteBackendMethods> {
    /// Optimize a newly codegened, totally unoptimized module.
    Optimize(ModuleCodegen<B::Module>),
    /// Copy the post-LTO artifacts from the incremental cache to the output
    /// directory.
    CopyPostLtoArtifacts(CachedModuleCodegen),
    /// Performs (Thin)LTO on the given module.
    LTO(lto::LtoModuleCodegen<B>),
}

impl<B: WriteBackendMethods> WorkItem<B> {
    pub fn module_kind(&self) -> ModuleKind {
        match *self {
            WorkItem::Optimize(ref m) => m.kind,
            WorkItem::CopyPostLtoArtifacts(_) |
            WorkItem::LTO(_) => ModuleKind::Regular,
        }
    }

    pub fn name(&self) -> String {
        match *self {
            WorkItem::Optimize(ref m) => format!("optimize: {}", m.name),
            WorkItem::CopyPostLtoArtifacts(ref m) => format!("copy post LTO artifacts: {}", m.name),
            WorkItem::LTO(ref m) => format!("lto: {}", m.name()),
        }
    }
}

enum WorkItemResult<B: WriteBackendMethods> {
    Compiled(CompiledModule),
    NeedsFatLTO(FatLTOInput<B>),
    NeedsThinLTO(String, B::ThinBuffer),
}

pub enum FatLTOInput<B: WriteBackendMethods> {
    Serialized {
        name: String,
        buffer: B::ModuleBuffer,
    },
    InMemory(ModuleCodegen<B::Module>),
}

fn execute_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    work_item: WorkItem<B>,
) -> Result<WorkItemResult<B>, FatalError> {
    let module_config = cgcx.config(work_item.module_kind());

    match work_item {
        WorkItem::Optimize(module) => {
            execute_optimize_work_item(cgcx, module, module_config)
        }
        WorkItem::CopyPostLtoArtifacts(module) => {
            execute_copy_from_cache_work_item(cgcx, module, module_config)
        }
        WorkItem::LTO(module) => {
            execute_lto_work_item(cgcx, module, module_config)
        }
    }
}

// Actual LTO type we end up chosing based on multiple factors.
enum ComputedLtoType {
    No,
    Thin,
    Fat,
}

fn execute_optimize_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    module: ModuleCodegen<B::Module>,
    module_config: &ModuleConfig,
) -> Result<WorkItemResult<B>, FatalError> {
    let diag_handler = cgcx.create_diag_handler();

    unsafe {
        B::optimize(cgcx, &diag_handler, &module, module_config)?;
    }

    // After we've done the initial round of optimizations we need to
    // decide whether to synchronously codegen this module or ship it
    // back to the coordinator thread for further LTO processing (which
    // has to wait for all the initial modules to be optimized).

    // If the linker does LTO, we don't have to do it. Note that we
    // keep doing full LTO, if it is requested, as not to break the
    // assumption that the output will be a single module.
    let linker_does_lto = cgcx.opts.cg.linker_plugin_lto.enabled();

    // When we're automatically doing ThinLTO for multi-codegen-unit
    // builds we don't actually want to LTO the allocator modules if
    // it shows up. This is due to various linker shenanigans that
    // we'll encounter later.
    let is_allocator = module.kind == ModuleKind::Allocator;

    // We ignore a request for full crate grath LTO if the cate type
    // is only an rlib, as there is no full crate graph to process,
    // that'll happen later.
    //
    // This use case currently comes up primarily for targets that
    // require LTO so the request for LTO is always unconditionally
    // passed down to the backend, but we don't actually want to do
    // anything about it yet until we've got a final product.
    let is_rlib = cgcx.crate_types.len() == 1
        && cgcx.crate_types[0] == config::CrateType::Rlib;

    // Metadata modules never participate in LTO regardless of the lto
    // settings.
    let lto_type = if module.kind == ModuleKind::Metadata {
        ComputedLtoType::No
    } else {
        match cgcx.lto {
            Lto::ThinLocal if !linker_does_lto && !is_allocator
                => ComputedLtoType::Thin,
            Lto::Thin if !linker_does_lto && !is_rlib
                => ComputedLtoType::Thin,
            Lto::Fat if !is_rlib => ComputedLtoType::Fat,
            _ => ComputedLtoType::No,
        }
    };

    // If we're doing some form of incremental LTO then we need to be sure to
    // save our module to disk first.
    let bitcode = if cgcx.config(module.kind).emit_pre_lto_bc {
        let filename = pre_lto_bitcode_filename(&module.name);
        cgcx.incr_comp_session_dir.as_ref().map(|path| path.join(&filename))
    } else {
        None
    };

    Ok(match lto_type {
        ComputedLtoType::No => {
            let module = unsafe {
                B::codegen(cgcx, &diag_handler, module, module_config)?
            };
            WorkItemResult::Compiled(module)
        }
        ComputedLtoType::Thin => {
            let (name, thin_buffer) = B::prepare_thin(module);
            if let Some(path) = bitcode {
                fs::write(&path, thin_buffer.data()).unwrap_or_else(|e| {
                    panic!("Error writing pre-lto-bitcode file `{}`: {}",
                           path.display(),
                           e);
                });
            }
            WorkItemResult::NeedsThinLTO(name, thin_buffer)
        }
        ComputedLtoType::Fat => {
            match bitcode {
                Some(path) => {
                    let (name, buffer) = B::serialize_module(module);
                    fs::write(&path, buffer.data()).unwrap_or_else(|e| {
                        panic!("Error writing pre-lto-bitcode file `{}`: {}",
                               path.display(),
                               e);
                    });
                    WorkItemResult::NeedsFatLTO(FatLTOInput::Serialized { name, buffer })
                }
                None => WorkItemResult::NeedsFatLTO(FatLTOInput::InMemory(module)),
            }
        }
    })
}

fn execute_copy_from_cache_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    module: CachedModuleCodegen,
    module_config: &ModuleConfig,
) -> Result<WorkItemResult<B>, FatalError> {
    let incr_comp_session_dir = cgcx.incr_comp_session_dir
                                    .as_ref()
                                    .unwrap();
    let mut object = None;
    let mut bytecode = None;
    let mut bytecode_compressed = None;
    for (kind, saved_file) in &module.source.saved_files {
        let obj_out = match kind {
            WorkProductFileKind::Object => {
                let path = cgcx.output_filenames.temp_path(OutputType::Object,
                                                           Some(&module.name));
                object = Some(path.clone());
                path
            }
            WorkProductFileKind::Bytecode => {
                let path = cgcx.output_filenames.temp_path(OutputType::Bitcode,
                                                           Some(&module.name));
                bytecode = Some(path.clone());
                path
            }
            WorkProductFileKind::BytecodeCompressed => {
                let path = cgcx.output_filenames.temp_path(OutputType::Bitcode,
                                                           Some(&module.name))
                    .with_extension(RLIB_BYTECODE_EXTENSION);
                bytecode_compressed = Some(path.clone());
                path
            }
        };
        let source_file = in_incr_comp_dir(&incr_comp_session_dir,
                                           &saved_file);
        debug!("copying pre-existing module `{}` from {:?} to {}",
               module.name,
               source_file,
               obj_out.display());
        if let Err(err) = link_or_copy(&source_file, &obj_out) {
            let diag_handler = cgcx.create_diag_handler();
            diag_handler.err(&format!("unable to copy {} to {}: {}",
                                      source_file.display(),
                                      obj_out.display(),
                                      err));
        }
    }

    assert_eq!(object.is_some(), module_config.emit_obj);
    assert_eq!(bytecode.is_some(), module_config.emit_bc);
    assert_eq!(bytecode_compressed.is_some(), module_config.emit_bc_compressed);

    Ok(WorkItemResult::Compiled(CompiledModule {
        name: module.name,
        kind: ModuleKind::Regular,
        object,
        bytecode,
        bytecode_compressed,
    }))
}

fn execute_lto_work_item<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    mut module: lto::LtoModuleCodegen<B>,
    module_config: &ModuleConfig,
) -> Result<WorkItemResult<B>, FatalError> {
    let diag_handler = cgcx.create_diag_handler();

    unsafe {
        let module = module.optimize(cgcx)?;
        let module = B::codegen(cgcx, &diag_handler, module, module_config)?;
        Ok(WorkItemResult::Compiled(module))
    }
}

pub enum Message<B: WriteBackendMethods> {
    Token(io::Result<Acquired>),
    NeedsFatLTO {
        result: FatLTOInput<B>,
        worker_id: usize,
    },
    NeedsThinLTO {
        name: String,
        thin_buffer: B::ThinBuffer,
        worker_id: usize,
    },
    Done {
        result: Result<CompiledModule, ()>,
        worker_id: usize,
    },
    CodegenDone {
        llvm_work_item: WorkItem<B>,
        cost: u64,
    },
    AddImportOnlyModule {
        module_data: SerializedModule<B::ModuleBuffer>,
        work_product: WorkProduct,
    },
    CodegenComplete,
    CodegenItem,
    CodegenAborted,
}

struct Diagnostic {
    msg: String,
    code: Option<DiagnosticId>,
    lvl: Level,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MainThreadWorkerState {
    Idle,
    Codegenning,
    LLVMing,
}

fn start_executing_work<B: ExtraBackendMethods>(
    backend: B,
    tcx: TyCtxt<'_>,
    crate_info: &CrateInfo,
    shared_emitter: SharedEmitter,
    codegen_worker_send: Sender<Message<B>>,
    coordinator_receive: Receiver<Box<dyn Any + Send>>,
    total_cgus: usize,
    jobserver: Client,
    modules_config: Arc<ModuleConfig>,
    metadata_config: Arc<ModuleConfig>,
    allocator_config: Arc<ModuleConfig>,
) -> thread::JoinHandle<Result<CompiledModules, ()>> {
    let coordinator_send = tcx.tx_to_llvm_workers.lock().clone();
    let sess = tcx.sess;

    // Compute the set of symbols we need to retain when doing LTO (if we need to)
    let exported_symbols = {
        let mut exported_symbols = FxHashMap::default();

        let copy_symbols = |cnum| {
            let symbols = tcx.exported_symbols(cnum)
                             .iter()
                             .map(|&(s, lvl)| (s.symbol_name(tcx).to_string(), lvl))
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
                for &cnum in tcx.crates().iter() {
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
    let helper = jobserver.into_helper_thread(move |token| {
        drop(coordinator_send2.send(Box::new(Message::Token::<B>(token))));
    }).expect("failed to spawn helper thread");

    let mut each_linked_rlib_for_lto = Vec::new();
    drop(link::each_linked_rlib(sess, crate_info, &mut |cnum, path| {
        if link::ignored_for_lto(sess, crate_info, cnum) {
            return
        }
        each_linked_rlib_for_lto.push((cnum, path.to_path_buf()));
    }));

    let assembler_cmd = if modules_config.no_integrated_as {
        // HACK: currently we use linker (gcc) as our assembler
        let (linker, flavor) = link::linker_and_flavor(sess);

        let (name, mut cmd) = get_linker(sess, &linker, flavor);
        cmd.args(&sess.target.target.options.asm_args);
        Some(Arc::new(AssemblerCommand {
            name,
            cmd,
        }))
    } else {
        None
    };

    let ol = if tcx.sess.opts.debugging_opts.no_codegen
             || !tcx.sess.opts.output_types.should_codegen() {
        // If we know that we won’t be doing codegen, create target machines without optimisation.
        config::OptLevel::No
    } else {
        tcx.backend_optimization_level(LOCAL_CRATE)
    };
    let cgcx = CodegenContext::<B> {
        backend: backend.clone(),
        crate_types: sess.crate_types.borrow().clone(),
        each_linked_rlib_for_lto,
        lto: sess.lto(),
        no_landing_pads: sess.no_landing_pads(),
        fewer_names: sess.fewer_names(),
        save_temps: sess.opts.cg.save_temps,
        opts: Arc::new(sess.opts.clone()),
        time_passes: sess.time_extended(),
        profiler: sess.self_profiling.clone(),
        exported_symbols,
        plugin_passes: sess.plugin_llvm_passes.borrow().clone(),
        remark: sess.opts.cg.remark.clone(),
        worker: 0,
        incr_comp_session_dir: sess.incr_comp_session_dir_opt().map(|r| r.clone()),
        cgu_reuse_tracker: sess.cgu_reuse_tracker.clone(),
        coordinator_send,
        diag_emitter: shared_emitter.clone(),
        output_filenames: tcx.output_filenames(LOCAL_CRATE),
        regular_module_config: modules_config,
        metadata_module_config: metadata_config,
        allocator_module_config: allocator_config,
        tm_factory: TargetMachineFactory(backend.target_machine_factory(tcx.sess, ol, false)),
        total_cgus,
        msvc_imps_needed: msvc_imps_needed(tcx),
        target_pointer_width: tcx.sess.target.target.target_pointer_width.clone(),
        target_arch: tcx.sess.target.target.arch.clone(),
        debuginfo: tcx.sess.opts.debuginfo,
        assembler_cmd,
    };

    // This is the "main loop" of parallel work happening for parallel codegen.
    // It's here that we manage parallelism, schedule work, and work with
    // messages coming from clients.
    //
    // There are a few environmental pre-conditions that shape how the system
    // is set up:
    //
    // - Error reporting only can happen on the main thread because that's the
    //   only place where we have access to the compiler `Session`.
    // - LLVM work can be done on any thread.
    // - Codegen can only happen on the main thread.
    // - Each thread doing substantial work most be in possession of a `Token`
    //   from the `Jobserver`.
    // - The compiler process always holds one `Token`. Any additional `Tokens`
    //   have to be requested from the `Jobserver`.
    //
    // Error Reporting
    // ===============
    // The error reporting restriction is handled separately from the rest: We
    // set up a `SharedEmitter` the holds an open channel to the main thread.
    // When an error occurs on any thread, the shared emitter will send the
    // error message to the receiver main thread (`SharedEmitterMain`). The
    // main thread will periodically query this error message queue and emit
    // any error messages it has received. It might even abort compilation if
    // has received a fatal error. In this case we rely on all other threads
    // being torn down automatically with the main thread.
    // Since the main thread will often be busy doing codegen work, error
    // reporting will be somewhat delayed, since the message queue can only be
    // checked in between to work packages.
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
    // The job of the main thread is to codegen CGUs into LLVM work package
    // (since the main thread is the only thread that can do this). The main
    // thread will block until it receives a message from the coordinator, upon
    // which it will codegen one CGU, send it to the coordinator and block
    // again. This way the coordinator can control what the main thread is
    // doing.
    //
    // The coordinator keeps a queue of LLVM WorkItems, and when a `Token` is
    // available, it will spawn off a new LLVM worker thread and let it process
    // that a WorkItem. When a LLVM worker thread is done with its WorkItem,
    // it will just shut down, which also frees all resources associated with
    // the given LLVM module, and sends a message to the coordinator that the
    // has been completed.
    //
    // Work Scheduling
    // ===============
    // The scheduler's goal is to minimize the time it takes to complete all
    // work there is, however, we also want to keep memory consumption low
    // if possible. These two goals are at odds with each other: If memory
    // consumption were not an issue, we could just let the main thread produce
    // LLVM WorkItems at full speed, assuring maximal utilization of
    // Tokens/LLVM worker threads. However, since codegen usual is faster
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
    // Since the main thread owns the compiler processes implicit `Token`, it is
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
    // and how that works. When LTO is requested what we'll to is collect all
    // optimized LLVM modules into a local vector on the coordinator. Once all
    // modules have been codegened and optimized we hand this to the `lto`
    // module for further optimization. The `lto` module will return back a list
    // of more modules to work on, which the coordinator will continue to spawn
    // work for.
    //
    // Each LLVM module is automatically sent back to the coordinator for LTO if
    // necessary. There's already optimizations in place to avoid sending work
    // back to the coordinator if LTO isn't requested.
    return thread::spawn(move || {
        // We pretend to be within the top-level LLVM time-passes task here:
        set_time_depth(1);

        let max_workers = ::num_cpus::get();
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
        let mut compiled_modules = vec![];
        let mut compiled_metadata_module = None;
        let mut compiled_allocator_module = None;
        let mut needs_fat_lto = Vec::new();
        let mut needs_thin_lto = Vec::new();
        let mut lto_import_only_modules = Vec::new();
        let mut started_lto = false;
        let mut codegen_aborted = false;

        // This flag tracks whether all items have gone through codegens
        let mut codegen_done = false;

        // This is the queue of LLVM work items that still need processing.
        let mut work_items = Vec::<(WorkItem<B>, u64)>::new();

        // This are the Jobserver Tokens we currently hold. Does not include
        // the implicit Token the compiler process owns no matter what.
        let mut tokens = Vec::new();

        let mut main_thread_worker_state = MainThreadWorkerState::Idle;
        let mut running = 0;

        let mut llvm_start_time = None;

        // Run the message loop while there's still anything that needs message
        // processing. Note that as soon as codegen is aborted we simply want to
        // wait for all existing work to finish, so many of the conditions here
        // only apply if codegen hasn't been aborted as they represent pending
        // work to be done.
        while !codegen_done ||
              running > 0 ||
              (!codegen_aborted && (
                  work_items.len() > 0 ||
                  needs_fat_lto.len() > 0 ||
                  needs_thin_lto.len() > 0 ||
                  lto_import_only_modules.len() > 0 ||
                  main_thread_worker_state != MainThreadWorkerState::Idle
              ))
        {

            // While there are still CGUs to be codegened, the coordinator has
            // to decide how to utilize the compiler processes implicit Token:
            // For codegenning more CGU or for running them through LLVM.
            if !codegen_done {
                if main_thread_worker_state == MainThreadWorkerState::Idle {
                    if !queue_full_enough(work_items.len(), running, max_workers) {
                        // The queue is not full enough, codegen more items:
                        if let Err(_) = codegen_worker_send.send(Message::CodegenItem) {
                            panic!("Could not send Message::CodegenItem to main thread")
                        }
                        main_thread_worker_state = MainThreadWorkerState::Codegenning;
                    } else {
                        // The queue is full enough to not let the worker
                        // threads starve. Use the implicit Token to do some
                        // LLVM work too.
                        let (item, _) = work_items.pop()
                            .expect("queue empty - queue_full_enough() broken?");
                        let cgcx = CodegenContext {
                            worker: get_worker_id(&mut free_worker_ids),
                            .. cgcx.clone()
                        };
                        maybe_start_llvm_timer(cgcx.config(item.module_kind()),
                                               &mut llvm_start_time);
                        main_thread_worker_state = MainThreadWorkerState::LLVMing;
                        spawn_work(cgcx, item);
                    }
                }
            } else if codegen_aborted {
                // don't queue up any more work if codegen was aborted, we're
                // just waiting for our existing children to finish
            } else {
                // If we've finished everything related to normal codegen
                // then it must be the case that we've got some LTO work to do.
                // Perform the serial work here of figuring out what we're
                // going to LTO and then push a bunch of work items onto our
                // queue to do LTO
                if work_items.len() == 0 &&
                   running == 0 &&
                   main_thread_worker_state == MainThreadWorkerState::Idle {
                    assert!(!started_lto);
                    started_lto = true;

                    let needs_fat_lto = mem::take(&mut needs_fat_lto);
                    let needs_thin_lto = mem::take(&mut needs_thin_lto);
                    let import_only_modules = mem::take(&mut lto_import_only_modules);

                    for (work, cost) in generate_lto_work(&cgcx, needs_fat_lto,
                                                          needs_thin_lto, import_only_modules) {
                        let insertion_index = work_items
                            .binary_search_by_key(&cost, |&(_, cost)| cost)
                            .unwrap_or_else(|e| e);
                        work_items.insert(insertion_index, (work, cost));
                        if !cgcx.opts.debugging_opts.no_parallel_llvm {
                            helper.request_token();
                        }
                    }
                }

                // In this branch, we know that everything has been codegened,
                // so it's just a matter of determining whether the implicit
                // Token is free to use for LLVM work.
                match main_thread_worker_state {
                    MainThreadWorkerState::Idle => {
                        if let Some((item, _)) = work_items.pop() {
                            let cgcx = CodegenContext {
                                worker: get_worker_id(&mut free_worker_ids),
                                .. cgcx.clone()
                            };
                            maybe_start_llvm_timer(cgcx.config(item.module_kind()),
                                                   &mut llvm_start_time);
                            main_thread_worker_state = MainThreadWorkerState::LLVMing;
                            spawn_work(cgcx, item);
                        } else {
                            // There is no unstarted work, so let the main thread
                            // take over for a running worker. Otherwise the
                            // implicit token would just go to waste.
                            // We reduce the `running` counter by one. The
                            // `tokens.truncate()` below will take care of
                            // giving the Token back.
                            debug_assert!(running > 0);
                            running -= 1;
                            main_thread_worker_state = MainThreadWorkerState::LLVMing;
                        }
                    }
                    MainThreadWorkerState::Codegenning => {
                        bug!("codegen worker should not be codegenning after \
                              codegen was already completed")
                    }
                    MainThreadWorkerState::LLVMing => {
                        // Already making good use of that token
                    }
                }
            }

            // Spin up what work we can, only doing this while we've got available
            // parallelism slots and work left to spawn.
            while !codegen_aborted && work_items.len() > 0 && running < tokens.len() {
                let (item, _) = work_items.pop().unwrap();

                maybe_start_llvm_timer(cgcx.config(item.module_kind()),
                                       &mut llvm_start_time);

                let cgcx = CodegenContext {
                    worker: get_worker_id(&mut free_worker_ids),
                    .. cgcx.clone()
                };

                spawn_work(cgcx, item);
                running += 1;
            }

            // Relinquish accidentally acquired extra tokens
            tokens.truncate(running);

            // If a thread exits successfully then we drop a token associated
            // with that worker and update our `running` count. We may later
            // re-acquire a token to continue running more work. We may also not
            // actually drop a token here if the worker was running with an
            // "ephemeral token"
            let mut free_worker = |worker_id| {
                if main_thread_worker_state == MainThreadWorkerState::LLVMing {
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                } else {
                    running -= 1;
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

                            if main_thread_worker_state == MainThreadWorkerState::LLVMing {
                                // If the main thread token is used for LLVM work
                                // at the moment, we turn that thread into a regular
                                // LLVM worker thread, so the main thread is free
                                // to react to codegen demand.
                                main_thread_worker_state = MainThreadWorkerState::Idle;
                                running += 1;
                            }
                        }
                        Err(e) => {
                            let msg = &format!("failed to acquire jobserver token: {}", e);
                            shared_emitter.fatal(msg);
                            // Exit the coordinator thread
                            panic!("{}", msg)
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
                    let insertion_index =
                        work_items.binary_search_by_key(&cost, |&(_, cost)| cost);
                    let insertion_index = match insertion_index {
                        Ok(idx) | Err(idx) => idx
                    };
                    work_items.insert(insertion_index, (llvm_work_item, cost));

                    if !cgcx.opts.debugging_opts.no_parallel_llvm {
                        helper.request_token();
                    }
                    assert!(!codegen_aborted);
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }

                Message::CodegenComplete => {
                    codegen_done = true;
                    assert!(!codegen_aborted);
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }

                // If codegen is aborted that means translation was aborted due
                // to some normal-ish compiler error. In this situation we want
                // to exit as soon as possible, but we want to make sure all
                // existing work has finished. Flag codegen as being done, and
                // then conditions above will ensure no more work is spawned but
                // we'll keep executing this loop until `running` hits 0.
                Message::CodegenAborted => {
                    assert!(!codegen_aborted);
                    codegen_done = true;
                    codegen_aborted = true;
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
                }
                Message::Done { result: Ok(compiled_module), worker_id } => {
                    free_worker(worker_id);
                    match compiled_module.kind {
                        ModuleKind::Regular => {
                            compiled_modules.push(compiled_module);
                        }
                        ModuleKind::Metadata => {
                            assert!(compiled_metadata_module.is_none());
                            compiled_metadata_module = Some(compiled_module);
                        }
                        ModuleKind::Allocator => {
                            assert!(compiled_allocator_module.is_none());
                            compiled_allocator_module = Some(compiled_module);
                        }
                    }
                }
                Message::NeedsFatLTO { result, worker_id } => {
                    assert!(!started_lto);
                    free_worker(worker_id);
                    needs_fat_lto.push(result);
                }
                Message::NeedsThinLTO { name, thin_buffer, worker_id } => {
                    assert!(!started_lto);
                    free_worker(worker_id);
                    needs_thin_lto.push((name, thin_buffer));
                }
                Message::AddImportOnlyModule { module_data, work_product } => {
                    assert!(!started_lto);
                    assert!(!codegen_done);
                    assert_eq!(main_thread_worker_state,
                               MainThreadWorkerState::Codegenning);
                    lto_import_only_modules.push((module_data, work_product));
                    main_thread_worker_state = MainThreadWorkerState::Idle;
                }
                // If the thread failed that means it panicked, so we abort immediately.
                Message::Done { result: Err(()), worker_id: _ } => {
                    bug!("worker thread panicked");
                }
                Message::CodegenItem => {
                    bug!("the coordinator should not receive codegen requests")
                }
            }
        }

        if let Some(llvm_start_time) = llvm_start_time {
            let total_llvm_time = Instant::now().duration_since(llvm_start_time);
            // This is the top-level timing for all of LLVM, set the time-depth
            // to zero.
            set_time_depth(0);
            print_time_passes_entry(cgcx.time_passes,
                                    "LLVM passes",
                                    total_llvm_time);
        }

        // Regardless of what order these modules completed in, report them to
        // the backend in the same order every time to ensure that we're handing
        // out deterministic results.
        compiled_modules.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(CompiledModules {
            modules: compiled_modules,
            metadata_module: compiled_metadata_module,
            allocator_module: compiled_allocator_module,
        })
    });

    // A heuristic that determines if we have enough LLVM WorkItems in the
    // queue so that the main thread can do LLVM work instead of codegen
    fn queue_full_enough(items_in_queue: usize,
                         workers_running: usize,
                         max_workers: usize) -> bool {
        // Tune me, plz.
        items_in_queue > 0 &&
        items_in_queue >= max_workers.saturating_sub(workers_running / 2)
    }

    fn maybe_start_llvm_timer(config: &ModuleConfig,
                              llvm_start_time: &mut Option<Instant>) {
        // We keep track of the -Ztime-passes output manually,
        // since the closure-based interface does not fit well here.
        if config.time_passes {
            if llvm_start_time.is_none() {
                *llvm_start_time = Some(Instant::now());
            }
        }
    }
}

pub const CODEGEN_WORKER_ID: usize = ::std::usize::MAX;

fn spawn_work<B: ExtraBackendMethods>(
    cgcx: CodegenContext<B>,
    work: WorkItem<B>
) {
    let depth = time_depth();

    thread::spawn(move || {
        set_time_depth(depth);

        // Set up a destructor which will fire off a message that we're done as
        // we exit.
        struct Bomb<B: ExtraBackendMethods> {
            coordinator_send: Sender<Box<dyn Any + Send>>,
            result: Option<WorkItemResult<B>>,
            worker_id: usize,
        }
        impl<B: ExtraBackendMethods> Drop for Bomb<B> {
            fn drop(&mut self) {
                let worker_id = self.worker_id;
                let msg = match self.result.take() {
                    Some(WorkItemResult::Compiled(m)) => {
                        Message::Done::<B> { result: Ok(m), worker_id }
                    }
                    Some(WorkItemResult::NeedsFatLTO(m)) => {
                        Message::NeedsFatLTO::<B> { result: m, worker_id }
                    }
                    Some(WorkItemResult::NeedsThinLTO(name, thin_buffer)) => {
                        Message::NeedsThinLTO::<B> { name, thin_buffer, worker_id }
                    }
                    None => Message::Done::<B> { result: Err(()), worker_id }
                };
                drop(self.coordinator_send.send(Box::new(msg)));
            }
        }

        let mut bomb = Bomb::<B> {
            coordinator_send: cgcx.coordinator_send.clone(),
            result: None,
            worker_id: cgcx.worker,
        };

        // Execute the work itself, and if it finishes successfully then flag
        // ourselves as a success as well.
        //
        // Note that we ignore any `FatalError` coming out of `execute_work_item`,
        // as a diagnostic was already sent off to the main thread - just
        // surface that there was an error in this worker.
        bomb.result = {
            let label = work.name();
            cgcx.profile(|p| p.start_activity(label.clone()));
            let result = execute_work_item(&cgcx, work).ok();
            cgcx.profile(|p| p.end_activity(label));

            result
        };
    });
}

pub fn run_assembler<B: ExtraBackendMethods>(
    cgcx: &CodegenContext<B>,
    handler: &Handler,
    assembly: &Path,
    object: &Path
) {
    let assembler = cgcx.assembler_cmd
        .as_ref()
        .expect("cgcx.assembler_cmd is missing?");

    let pname = &assembler.name;
    let mut cmd = assembler.cmd.clone();
    cmd.arg("-c").arg("-o").arg(object).arg(assembly);
    debug!("{:?}", cmd);

    match cmd.output() {
        Ok(prog) => {
            if !prog.status.success() {
                let mut note = prog.stderr.clone();
                note.extend_from_slice(&prog.stdout);

                handler.struct_err(&format!("linking with `{}` failed: {}",
                                            pname.display(),
                                            prog.status))
                    .note(&format!("{:?}", &cmd))
                    .note(str::from_utf8(&note[..]).unwrap())
                    .emit();
                handler.abort_if_errors();
            }
        },
        Err(e) => {
            handler.err(&format!("could not exec the linker `{}`: {}", pname.display(), e));
            handler.abort_if_errors();
        }
    }
}


enum SharedEmitterMessage {
    Diagnostic(Diagnostic),
    InlineAsmError(u32, String),
    AbortIfErrors,
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
    pub fn new() -> (SharedEmitter, SharedEmitterMain) {
        let (sender, receiver) = channel();

        (SharedEmitter { sender }, SharedEmitterMain { receiver })
    }

    pub fn inline_asm_error(&self, cookie: u32, msg: String) {
        drop(self.sender.send(SharedEmitterMessage::InlineAsmError(cookie, msg)));
    }

    pub fn fatal(&self, msg: &str) {
        drop(self.sender.send(SharedEmitterMessage::Fatal(msg.to_string())));
    }
}

impl Emitter for SharedEmitter {
    fn emit_diagnostic(&mut self, db: &DiagnosticBuilder<'_>) {
        drop(self.sender.send(SharedEmitterMessage::Diagnostic(Diagnostic {
            msg: db.message(),
            code: db.code.clone(),
            lvl: db.level,
        })));
        for child in &db.children {
            drop(self.sender.send(SharedEmitterMessage::Diagnostic(Diagnostic {
                msg: child.message(),
                code: None,
                lvl: child.level,
            })));
        }
        drop(self.sender.send(SharedEmitterMessage::AbortIfErrors));
    }
}

impl SharedEmitterMain {
    pub fn check(&self, sess: &Session, blocking: bool) {
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
                    let handler = sess.diagnostic();
                    match diag.code {
                        Some(ref code) => {
                            handler.emit_with_code(&MultiSpan::new(),
                                                   &diag.msg,
                                                   code.clone(),
                                                   diag.lvl);
                        }
                        None => {
                            handler.emit(&MultiSpan::new(),
                                         &diag.msg,
                                         diag.lvl);
                        }
                    }
                }
                Ok(SharedEmitterMessage::InlineAsmError(cookie, msg)) => {
                    match Mark::from_u32(cookie).expn_info() {
                        Some(ei) => sess.span_err(ei.call_site, &msg),
                        None     => sess.err(&msg),
                    }
                }
                Ok(SharedEmitterMessage::AbortIfErrors) => {
                    sess.abort_if_errors();
                }
                Ok(SharedEmitterMessage::Fatal(msg)) => {
                    sess.fatal(&msg);
                }
                Err(_) => {
                    break;
                }
            }

        }
    }
}

pub struct OngoingCodegen<B: ExtraBackendMethods> {
    pub backend: B,
    pub crate_name: Symbol,
    pub crate_hash: Svh,
    pub metadata: EncodedMetadata,
    pub windows_subsystem: Option<String>,
    pub linker_info: LinkerInfo,
    pub crate_info: CrateInfo,
    pub coordinator_send: Sender<Box<dyn Any + Send>>,
    pub codegen_worker_receive: Receiver<Message<B>>,
    pub shared_emitter_main: SharedEmitterMain,
    pub future: thread::JoinHandle<Result<CompiledModules, ()>>,
    pub output_filenames: Arc<OutputFilenames>,
}

impl<B: ExtraBackendMethods> OngoingCodegen<B> {
    pub fn join(
        self,
        sess: &Session
    ) -> (CodegenResults, FxHashMap<WorkProductId, WorkProduct>) {
        self.shared_emitter_main.check(sess, true);
        let compiled_modules = match self.future.join() {
            Ok(Ok(compiled_modules)) => compiled_modules,
            Ok(Err(())) => {
                sess.abort_if_errors();
                panic!("expected abort due to worker thread errors")
            },
            Err(_) => {
                bug!("panic during codegen/LLVM phase");
            }
        };

        sess.cgu_reuse_tracker.check_expected_reuse(sess);

        sess.abort_if_errors();

        let work_products =
            copy_all_cgu_workproducts_to_incr_comp_cache_dir(sess,
                                                             &compiled_modules);
        produce_final_output_artifacts(sess,
                                       &compiled_modules,
                                       &self.output_filenames);

        // FIXME: time_llvm_passes support - does this use a global context or
        // something?
        if sess.codegen_units() == 1 && sess.time_llvm_passes() {
            self.backend.print_pass_timings()
        }

        (CodegenResults {
            crate_name: self.crate_name,
            crate_hash: self.crate_hash,
            metadata: self.metadata,
            windows_subsystem: self.windows_subsystem,
            linker_info: self.linker_info,
            crate_info: self.crate_info,

            modules: compiled_modules.modules,
            allocator_module: compiled_modules.allocator_module,
            metadata_module: compiled_modules.metadata_module,
        }, work_products)
    }

    pub fn submit_pre_codegened_module_to_llvm(
        &self,
        tcx: TyCtxt<'_>,
        module: ModuleCodegen<B::Module>,
    ) {
        self.wait_for_signal_to_codegen_item();
        self.check_for_errors(tcx.sess);

        // These are generally cheap and won't throw off scheduling.
        let cost = 0;
        submit_codegened_module_to_llvm(&self.backend, tcx, module, cost);
    }

    pub fn codegen_finished(&self, tcx: TyCtxt<'_>) {
        self.wait_for_signal_to_codegen_item();
        self.check_for_errors(tcx.sess);
        drop(self.coordinator_send.send(Box::new(Message::CodegenComplete::<B>)));
    }

    /// Consumes this context indicating that codegen was entirely aborted, and
    /// we need to exit as quickly as possible.
    ///
    /// This method blocks the current thread until all worker threads have
    /// finished, and all worker threads should have exited or be real close to
    /// exiting at this point.
    pub fn codegen_aborted(self) {
        // Signal to the coordinator it should spawn no more work and start
        // shutdown.
        drop(self.coordinator_send.send(Box::new(Message::CodegenAborted::<B>)));
        drop(self.future.join());
    }

    pub fn check_for_errors(&self, sess: &Session) {
        self.shared_emitter_main.check(sess, false);
    }

    pub fn wait_for_signal_to_codegen_item(&self) {
        match self.codegen_worker_receive.recv() {
            Ok(Message::CodegenItem) => {
                // Nothing to do
            }
            Ok(_) => panic!("unexpected message"),
            Err(_) => {
                // One of the LLVM threads must have panicked, fall through so
                // error handling can be reached.
            }
        }
    }
}

pub fn submit_codegened_module_to_llvm<B: ExtraBackendMethods>(
    _backend: &B,
    tcx: TyCtxt<'_>,
    module: ModuleCodegen<B::Module>,
    cost: u64,
) {
    let llvm_work_item = WorkItem::Optimize(module);
    drop(tcx.tx_to_llvm_workers.lock().send(Box::new(Message::CodegenDone::<B> {
        llvm_work_item,
        cost,
    })));
}

pub fn submit_post_lto_module_to_llvm<B: ExtraBackendMethods>(
    _backend: &B,
    tcx: TyCtxt<'_>,
    module: CachedModuleCodegen,
) {
    let llvm_work_item = WorkItem::CopyPostLtoArtifacts(module);
    drop(tcx.tx_to_llvm_workers.lock().send(Box::new(Message::CodegenDone::<B> {
        llvm_work_item,
        cost: 0,
    })));
}

pub fn submit_pre_lto_module_to_llvm<B: ExtraBackendMethods>(
    _backend: &B,
    tcx: TyCtxt<'_>,
    module: CachedModuleCodegen,
) {
    let filename = pre_lto_bitcode_filename(&module.name);
    let bc_path = in_incr_comp_dir_sess(tcx.sess, &filename);
    let file = fs::File::open(&bc_path).unwrap_or_else(|e| {
        panic!("failed to open bitcode file `{}`: {}", bc_path.display(), e)
    });

    let mmap = unsafe {
        memmap::Mmap::map(&file).unwrap_or_else(|e| {
            panic!("failed to mmap bitcode file `{}`: {}", bc_path.display(), e)
        })
    };
    // Schedule the module to be loaded
    drop(tcx.tx_to_llvm_workers.lock().send(Box::new(Message::AddImportOnlyModule::<B> {
        module_data: SerializedModule::FromUncompressedFile(mmap),
        work_product: module.source,
    })));
}

pub fn pre_lto_bitcode_filename(module_name: &str) -> String {
    format!("{}.{}", module_name, PRE_LTO_BC_EXT)
}

fn msvc_imps_needed(tcx: TyCtxt<'_>) -> bool {
    // This should never be true (because it's not supported). If it is true,
    // something is wrong with commandline arg validation.
    assert!(!(tcx.sess.opts.cg.linker_plugin_lto.enabled() &&
              tcx.sess.target.target.options.is_like_msvc &&
              tcx.sess.opts.cg.prefer_dynamic));

    tcx.sess.target.target.options.is_like_msvc &&
        tcx.sess.crate_types.borrow().iter().any(|ct| *ct == config::CrateType::Rlib) &&
    // ThinLTO can't handle this workaround in all cases, so we don't
    // emit the `__imp_` symbols. Instead we make them unnecessary by disallowing
    // dynamic linking when linker plugin LTO is enabled.
    !tcx.sess.opts.cg.linker_plugin_lto.enabled()
}
