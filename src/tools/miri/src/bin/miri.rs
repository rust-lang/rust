#![feature(rustc_private, stmt_expr_attributes)]
#![allow(
    clippy::manual_range_contains,
    clippy::useless_format,
    clippy::field_reassign_with_default,
    clippy::needless_lifetimes,
    rustc::diagnostic_outside_of_impl,
    rustc::untranslatable_diagnostic
)]

// Some "regular" crates we want to share with rustc
extern crate tracing;

// The rustc crates we need
extern crate rustc_abi;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_hir_analysis;
extern crate rustc_interface;
extern crate rustc_log;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use std::env::{self, VarError};
use std::num::NonZero;
use std::ops::Range;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Once};

use miri::{
    BacktraceStyle, BorrowTrackerMethod, GenmcConfig, GenmcCtx, MiriConfig, MiriEntryFnType,
    ProvenanceMode, RetagFields, ValidationMode,
};
use rustc_abi::ExternAbi;
use rustc_data_structures::sync;
use rustc_driver::Compilation;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::{self as hir, Node};
use rustc_hir_analysis::check::check_function_signature;
use rustc_interface::interface::Config;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::{
    ExportedSymbol, SymbolExportInfo, SymbolExportKind, SymbolExportLevel,
};
use rustc_middle::query::LocalCrate;
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::util::Providers;
use rustc_session::config::{CrateType, ErrorOutputType, OptLevel};
use rustc_session::search_paths::PathKind;
use rustc_session::{CtfeBacktrace, EarlyDiagCtxt};
use rustc_span::def_id::DefId;
use tracing::debug;

struct MiriCompilerCalls {
    miri_config: Option<MiriConfig>,
    many_seeds: Option<ManySeedsConfig>,
    /// Settings for using GenMC with Miri.
    genmc_config: Option<GenmcConfig>,
}

struct ManySeedsConfig {
    seeds: Range<u32>,
    keep_going: bool,
}

impl MiriCompilerCalls {
    fn new(
        miri_config: MiriConfig,
        many_seeds: Option<ManySeedsConfig>,
        genmc_config: Option<GenmcConfig>,
    ) -> Self {
        Self { miri_config: Some(miri_config), many_seeds, genmc_config }
    }
}

fn entry_fn(tcx: TyCtxt<'_>) -> (DefId, MiriEntryFnType) {
    if let Some((def_id, entry_type)) = tcx.entry_fn(()) {
        return (def_id, MiriEntryFnType::Rustc(entry_type));
    }
    // Look for a symbol in the local crate named `miri_start`, and treat that as the entry point.
    let sym = tcx.exported_symbols(LOCAL_CRATE).iter().find_map(|(sym, _)| {
        if sym.symbol_name_for_local_instance(tcx).name == "miri_start" { Some(sym) } else { None }
    });
    if let Some(ExportedSymbol::NonGeneric(id)) = sym {
        let start_def_id = id.expect_local();
        let start_span = tcx.def_span(start_def_id);

        let expected_sig = ty::Binder::dummy(tcx.mk_fn_sig(
            [tcx.types.isize, Ty::new_imm_ptr(tcx, Ty::new_imm_ptr(tcx, tcx.types.u8))],
            tcx.types.isize,
            false,
            hir::Safety::Safe,
            ExternAbi::Rust,
        ));

        let correct_func_sig = check_function_signature(
            tcx,
            ObligationCause::new(start_span, start_def_id, ObligationCauseCode::Misc),
            *id,
            expected_sig,
        )
        .is_ok();

        if correct_func_sig {
            (*id, MiriEntryFnType::MiriStart)
        } else {
            tcx.dcx().fatal(
                "`miri_start` must have the following signature:\n\
                fn miri_start(argc: isize, argv: *const *const u8) -> isize",
            );
        }
    } else {
        tcx.dcx().fatal(
            "Miri can only run programs that have a main function.\n\
            Alternatively, you can export a `miri_start` function:\n\
            \n\
            #[cfg(miri)]\n\
            #[unsafe(no_mangle)]\n\
            fn miri_start(argc: isize, argv: *const *const u8) -> isize {\
            \n    // Call the actual start function that your project implements, based on your target's conventions.\n\
            }"
        );
    }
}

impl rustc_driver::Callbacks for MiriCompilerCalls {
    fn config(&mut self, config: &mut Config) {
        config.override_queries = Some(|_, providers| {
            providers.extern_queries.used_crate_source = |tcx, cnum| {
                let mut providers = Providers::default();
                rustc_metadata::provide(&mut providers);
                let mut crate_source = (providers.extern_queries.used_crate_source)(tcx, cnum);
                // HACK: rustc will emit "crate ... required to be available in rlib format, but
                // was not found in this form" errors once we use `tcx.dependency_formats()` if
                // there's no rlib provided, so setting a dummy path here to workaround those errors.
                Arc::make_mut(&mut crate_source).rlib = Some((PathBuf::new(), PathKind::All));
                crate_source
            };
        });
    }

    fn after_analysis<'tcx>(
        &mut self,
        _: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        if tcx.sess.dcx().has_errors_or_delayed_bugs().is_some() {
            tcx.dcx().fatal("miri cannot be run on programs that fail compilation");
        }

        let early_dcx = EarlyDiagCtxt::new(tcx.sess.opts.error_format);
        init_late_loggers(&early_dcx, tcx);
        if !tcx.crate_types().contains(&CrateType::Executable) {
            tcx.dcx().fatal("miri only makes sense on bin crates");
        }

        let (entry_def_id, entry_type) = entry_fn(tcx);
        let mut config = self.miri_config.take().expect("after_analysis must only be called once");

        // Add filename to `miri` arguments.
        config.args.insert(0, tcx.sess.io.input.filestem().to_string());

        // Adjust working directory for interpretation.
        if let Some(cwd) = env::var_os("MIRI_CWD") {
            env::set_current_dir(cwd).unwrap();
        }

        if tcx.sess.opts.optimize != OptLevel::No {
            tcx.dcx().warn("Miri does not support optimizations: the opt-level is ignored. The only effect \
                    of selecting a Cargo profile that enables optimizations (such as --release) is to apply \
                    its remaining settings, such as whether debug assertions and overflow checks are enabled.");
        }
        if tcx.sess.mir_opt_level() > 0 {
            tcx.dcx().warn("You have explicitly enabled MIR optimizations, overriding Miri's default \
                    which is to completely disable them. Any optimizations may hide UB that Miri would \
                    otherwise detect, and it is not necessarily possible to predict what kind of UB will \
                    be missed. If you are enabling optimizations to make Miri run faster, we advise using \
                    cfg(miri) to shrink your workload instead. The performance benefit of enabling MIR \
                    optimizations is usually marginal at best.");
        }

        if let Some(genmc_config) = &self.genmc_config {
            let _genmc_ctx = Rc::new(GenmcCtx::new(&config, genmc_config));

            todo!("GenMC mode not yet implemented");
        };

        if let Some(many_seeds) = self.many_seeds.take() {
            assert!(config.seed.is_none());
            let exit_code = sync::IntoDynSyncSend(AtomicI32::new(rustc_driver::EXIT_SUCCESS));
            let num_failed = sync::IntoDynSyncSend(AtomicU32::new(0));
            sync::par_for_each_in(many_seeds.seeds.clone(), |seed| {
                let mut config = config.clone();
                config.seed = Some((*seed).into());
                eprintln!("Trying seed: {seed}");
                let return_code = miri::eval_entry(
                    tcx,
                    entry_def_id,
                    entry_type,
                    &config,
                    /* genmc_ctx */ None,
                )
                .unwrap_or(rustc_driver::EXIT_FAILURE);
                if return_code != rustc_driver::EXIT_SUCCESS {
                    eprintln!("FAILING SEED: {seed}");
                    if !many_seeds.keep_going {
                        // `abort_if_errors` would actually not stop, since `par_for_each` waits for the
                        // rest of the to finish, so we just exit immediately.
                        std::process::exit(return_code);
                    }
                    exit_code.store(return_code, Ordering::Relaxed);
                    num_failed.fetch_add(1, Ordering::Relaxed);
                }
            });
            let num_failed = num_failed.0.into_inner();
            if num_failed > 0 {
                eprintln!("{num_failed}/{total} SEEDS FAILED", total = many_seeds.seeds.count());
            }
            std::process::exit(exit_code.0.into_inner());
        } else {
            let return_code = miri::eval_entry(tcx, entry_def_id, entry_type, &config, None)
                .unwrap_or_else(|| {
                    tcx.dcx().abort_if_errors();
                    rustc_driver::EXIT_FAILURE
                });

            std::process::exit(return_code);
        }

        // Unreachable.
    }
}

struct MiriBeRustCompilerCalls {
    target_crate: bool,
}

impl rustc_driver::Callbacks for MiriBeRustCompilerCalls {
    #[allow(rustc::potential_query_instability)] // rustc_codegen_ssa (where this code is copied from) also allows this lint
    fn config(&mut self, config: &mut Config) {
        if config.opts.prints.is_empty() && self.target_crate {
            // Queries overridden here affect the data stored in `rmeta` files of dependencies,
            // which will be used later in non-`MIRI_BE_RUSTC` mode.
            config.override_queries = Some(|_, local_providers| {
                // `exported_symbols` and `reachable_non_generics` provided by rustc always returns
                // an empty result if `tcx.sess.opts.output_types.should_codegen()` is false.
                // In addition we need to add #[used] symbols to exported_symbols for `lookup_link_section`.
                local_providers.exported_symbols = |tcx, LocalCrate| {
                    let reachable_set = tcx.with_stable_hashing_context(|hcx| {
                        tcx.reachable_set(()).to_sorted(&hcx, true)
                    });
                    tcx.arena.alloc_from_iter(
                        // This is based on:
                        // https://github.com/rust-lang/rust/blob/2962e7c0089d5c136f4e9600b7abccfbbde4973d/compiler/rustc_codegen_ssa/src/back/symbol_export.rs#L62-L63
                        // https://github.com/rust-lang/rust/blob/2962e7c0089d5c136f4e9600b7abccfbbde4973d/compiler/rustc_codegen_ssa/src/back/symbol_export.rs#L174
                        reachable_set.into_iter().filter_map(|&local_def_id| {
                            // Do the same filtering that rustc does:
                            // https://github.com/rust-lang/rust/blob/2962e7c0089d5c136f4e9600b7abccfbbde4973d/compiler/rustc_codegen_ssa/src/back/symbol_export.rs#L84-L102
                            // Otherwise it may cause unexpected behaviours and ICEs
                            // (https://github.com/rust-lang/rust/issues/86261).
                            let is_reachable_non_generic = matches!(
                                tcx.hir_node_by_def_id(local_def_id),
                                Node::Item(&hir::Item {
                                    kind: hir::ItemKind::Static(..) | hir::ItemKind::Fn{ .. },
                                    ..
                                }) | Node::ImplItem(&hir::ImplItem {
                                    kind: hir::ImplItemKind::Fn(..),
                                    ..
                                })
                                if !tcx.generics_of(local_def_id).requires_monomorphization(tcx)
                            );
                            if !is_reachable_non_generic {
                                return None;
                            }
                            let codegen_fn_attrs = tcx.codegen_fn_attrs(local_def_id);
                            if codegen_fn_attrs.contains_extern_indicator()
                                || codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER)
                                || codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
                            {
                                Some((
                                    ExportedSymbol::NonGeneric(local_def_id.to_def_id()),
                                    // Some dummy `SymbolExportInfo` here. We only use
                                    // `exported_symbols` in shims/foreign_items.rs and the export info
                                    // is ignored.
                                    SymbolExportInfo {
                                        level: SymbolExportLevel::C,
                                        kind: SymbolExportKind::Text,
                                        used: false,
                                    },
                                ))
                            } else {
                                None
                            }
                        }),
                    )
                }
            });
        }
    }

    fn after_analysis<'tcx>(
        &mut self,
        _: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        if self.target_crate {
            // cargo-miri has patched the compiler flags to make these into check-only builds,
            // but we are still emulating regular rustc builds, which would perform post-mono
            // const-eval during collection. So let's also do that here, even if we might be
            // running with `--emit=metadata`. In particular this is needed to make
            // `compile_fail` doc tests trigger post-mono errors.
            // In general `collect_and_partition_mono_items` is not safe to call in check-only
            // builds, but we are setting `-Zalways-encode-mir` which avoids those issues.
            let _ = tcx.collect_and_partition_mono_items(());
        }
        Compilation::Continue
    }
}

fn show_error(msg: &impl std::fmt::Display) -> ! {
    eprintln!("fatal error: {msg}");
    std::process::exit(1)
}

macro_rules! show_error {
    ($($tt:tt)*) => { show_error(&format_args!($($tt)*)) };
}

fn rustc_logger_config() -> rustc_log::LoggerConfig {
    // Start with the usual env vars.
    let mut cfg = rustc_log::LoggerConfig::from_env("RUSTC_LOG");

    // Overwrite if MIRI_LOG is set.
    if let Ok(var) = env::var("MIRI_LOG") {
        // MIRI_LOG serves as default for RUSTC_LOG, if that is not set.
        if matches!(cfg.filter, Err(VarError::NotPresent)) {
            // We try to be a bit clever here: if `MIRI_LOG` is just a single level
            // used for everything, we only apply it to the parts of rustc that are
            // CTFE-related. Otherwise, we use it verbatim for `RUSTC_LOG`.
            // This way, if you set `MIRI_LOG=trace`, you get only the right parts of
            // rustc traced, but you can also do `MIRI_LOG=miri=trace,rustc_const_eval::interpret=debug`.
            if tracing::Level::from_str(&var).is_ok() {
                cfg.filter = Ok(format!(
                    "rustc_middle::mir::interpret={var},rustc_const_eval::interpret={var},miri={var}"
                ));
            } else {
                cfg.filter = Ok(var);
            }
        }
    }

    cfg
}

/// The global logger can only be set once per process, so track
/// whether that already happened.
static LOGGER_INITED: Once = Once::new();

fn init_early_loggers(early_dcx: &EarlyDiagCtxt) {
    // We only initialize `rustc` if the env var is set (so the user asked for it).
    // If it is not set, we avoid initializing now so that we can initialize later with our custom
    // settings, and *not* log anything for what happens before `miri` starts interpreting.
    if env::var_os("RUSTC_LOG").is_some() {
        LOGGER_INITED.call_once(|| {
            rustc_driver::init_logger(early_dcx, rustc_logger_config());
        });
    }
}

fn init_late_loggers(early_dcx: &EarlyDiagCtxt, tcx: TyCtxt<'_>) {
    // If the logger is not yet initialized, initialize it.
    LOGGER_INITED.call_once(|| {
        rustc_driver::init_logger(early_dcx, rustc_logger_config());
    });

    // If `MIRI_BACKTRACE` is set and `RUSTC_CTFE_BACKTRACE` is not, set `RUSTC_CTFE_BACKTRACE`.
    // Do this late, so we ideally only apply this to Miri's errors.
    if let Some(val) = env::var_os("MIRI_BACKTRACE") {
        let ctfe_backtrace = match &*val.to_string_lossy() {
            "immediate" => CtfeBacktrace::Immediate,
            "0" => CtfeBacktrace::Disabled,
            _ => CtfeBacktrace::Capture,
        };
        *tcx.sess.ctfe_backtrace.borrow_mut() = ctfe_backtrace;
    }
}

/// Execute a compiler with the given CLI arguments and callbacks.
fn run_compiler_and_exit(
    args: &[String],
    callbacks: &mut (dyn rustc_driver::Callbacks + Send),
) -> ! {
    // Invoke compiler, and handle return code.
    let exit_code =
        rustc_driver::catch_with_exit_code(move || rustc_driver::run_compiler(args, callbacks));
    std::process::exit(exit_code)
}

/// Parses a comma separated list of `T` from the given string:
/// `<value1>,<value2>,<value3>,...`
fn parse_comma_list<T: FromStr>(input: &str) -> Result<Vec<T>, T::Err> {
    input.split(',').map(str::parse::<T>).collect()
}

/// Parses the input as a float in the range from 0.0 to 1.0 (inclusive).
fn parse_rate(input: &str) -> Result<f64, &'static str> {
    match input.parse::<f64>() {
        Ok(rate) if rate >= 0.0 && rate <= 1.0 => Ok(rate),
        Ok(_) => Err("must be between `0.0` and `1.0`"),
        Err(_) => Err("requires a `f64` between `0.0` and `1.0`"),
    }
}

/// Parses a seed range
///
/// This function is used for the `-Zmiri-many-seeds` flag. It expects the range in the form
/// `<from>..<to>`. `<from>` is inclusive, `<to>` is exclusive. `<from>` can be omitted,
/// in which case it is assumed to be `0`.
fn parse_range(val: &str) -> Result<Range<u32>, &'static str> {
    let (from, to) = val.split_once("..").ok_or("expected `from..to`")?;
    let from: u32 = if from.is_empty() { 0 } else { from.parse().map_err(|_| "invalid `from`")? };
    let to: u32 = to.parse().map_err(|_| "invalid `to`")?;
    Ok(from..to)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn jemalloc_magic() {
    // These magic runes are copied from
    // <https://github.com/rust-lang/rust/blob/e89bd9428f621545c979c0ec686addc6563a394e/compiler/rustc/src/main.rs#L39>.
    // See there for further comments.
    use std::os::raw::{c_int, c_void};

    use tikv_jemalloc_sys as jemalloc_sys;

    #[used]
    static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = jemalloc_sys::calloc;
    #[used]
    static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int =
        jemalloc_sys::posix_memalign;
    #[used]
    static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void = jemalloc_sys::aligned_alloc;
    #[used]
    static _F4: unsafe extern "C" fn(usize) -> *mut c_void = jemalloc_sys::malloc;
    #[used]
    static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void = jemalloc_sys::realloc;
    #[used]
    static _F6: unsafe extern "C" fn(*mut c_void) = jemalloc_sys::free;

    // On OSX, jemalloc doesn't directly override malloc/free, but instead
    // registers itself with the allocator's zone APIs in a ctor. However,
    // the linker doesn't seem to consider ctors as "used" when statically
    // linking, so we need to explicitly depend on the function.
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn _rjem_je_zone_register();
        }

        #[used]
        static _F7: unsafe extern "C" fn() = _rjem_je_zone_register;
    }
}

fn main() {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    jemalloc_magic();

    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    // Snapshot a copy of the environment before `rustc` starts messing with it.
    // (`install_ice_hook` might change `RUST_BACKTRACE`.)
    let env_snapshot = env::vars_os().collect::<Vec<_>>();

    let args = rustc_driver::catch_fatal_errors(|| rustc_driver::args::raw_args(&early_dcx))
        .unwrap_or_else(|_| std::process::exit(rustc_driver::EXIT_FAILURE));

    // Install the ctrlc handler that sets `rustc_const_eval::CTRL_C_RECEIVED`, even if
    // MIRI_BE_RUSTC is set.
    rustc_driver::install_ctrlc_handler();

    // If the environment asks us to actually be rustc, then do that.
    if let Some(crate_kind) = env::var_os("MIRI_BE_RUSTC") {
        // Earliest rustc setup.
        rustc_driver::install_ice_hook(rustc_driver::DEFAULT_BUG_REPORT_URL, |_| ());
        rustc_driver::init_rustc_env_logger(&early_dcx);

        let target_crate = if crate_kind == "target" {
            true
        } else if crate_kind == "host" {
            false
        } else {
            panic!("invalid `MIRI_BE_RUSTC` value: {crate_kind:?}")
        };

        let mut args = args;
        // Don't insert `MIRI_DEFAULT_ARGS`, in particular, `--cfg=miri`, if we are building
        // a "host" crate. That may cause procedural macros (and probably build scripts) to
        // depend on Miri-only symbols, such as `miri_resolve_frame`:
        // https://github.com/rust-lang/miri/issues/1760
        if target_crate {
            // Splice in the default arguments after the program name.
            // Some options have different defaults in Miri than in plain rustc; apply those by making
            // them the first arguments after the binary name (but later arguments can overwrite them).
            args.splice(1..1, miri::MIRI_DEFAULT_ARGS.iter().map(ToString::to_string));
        }

        // We cannot use `rustc_driver::main` as we want it to use `args` as the CLI arguments.
        run_compiler_and_exit(&args, &mut MiriBeRustCompilerCalls { target_crate })
    }

    // Add an ICE bug report hook.
    rustc_driver::install_ice_hook("https://github.com/rust-lang/miri/issues/new", |_| ());

    // Init loggers the Miri way.
    init_early_loggers(&early_dcx);

    // Parse our arguments and split them across `rustc` and `miri`.
    let mut many_seeds: Option<Range<u32>> = None;
    let mut many_seeds_keep_going = false;
    let mut miri_config = MiriConfig::default();
    miri_config.env = env_snapshot;
    let mut genmc_config = None;

    let mut rustc_args = vec![];
    let mut after_dashdash = false;

    // Note that we require values to be given with `=`, not with a space.
    // This matches how rustc parses `-Z`.
    // However, unlike rustc we do not accept a space after `-Z`.
    for arg in args {
        if rustc_args.is_empty() {
            // Very first arg: binary name.
            rustc_args.push(arg);
            // Also add the default arguments.
            rustc_args.extend(miri::MIRI_DEFAULT_ARGS.iter().map(ToString::to_string));
        } else if after_dashdash {
            // Everything that comes after `--` is forwarded to the interpreted crate.
            miri_config.args.push(arg);
        } else if arg == "--" {
            after_dashdash = true;
        } else if arg == "-Zmiri-disable-validation" {
            miri_config.validation = ValidationMode::No;
        } else if arg == "-Zmiri-recursive-validation" {
            miri_config.validation = ValidationMode::Deep;
        } else if arg == "-Zmiri-disable-stacked-borrows" {
            miri_config.borrow_tracker = None;
        } else if arg == "-Zmiri-tree-borrows" {
            miri_config.borrow_tracker = Some(BorrowTrackerMethod::TreeBorrows);
            miri_config.provenance_mode = ProvenanceMode::Strict;
        } else if arg == "-Zmiri-disable-data-race-detector" {
            miri_config.data_race_detector = false;
            miri_config.weak_memory_emulation = false;
        } else if arg == "-Zmiri-disable-alignment-check" {
            miri_config.check_alignment = miri::AlignmentCheck::None;
        } else if arg == "-Zmiri-symbolic-alignment-check" {
            miri_config.check_alignment = miri::AlignmentCheck::Symbolic;
        } else if arg == "-Zmiri-disable-isolation" {
            miri_config.isolated_op = miri::IsolatedOp::Allow;
        } else if arg == "-Zmiri-disable-leak-backtraces" {
            miri_config.collect_leak_backtraces = false;
        } else if arg == "-Zmiri-disable-weak-memory-emulation" {
            miri_config.weak_memory_emulation = false;
        } else if arg == "-Zmiri-track-weak-memory-loads" {
            miri_config.track_outdated_loads = true;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-isolation-error=") {
            miri_config.isolated_op = match param {
                "abort" => miri::IsolatedOp::Reject(miri::RejectOpWith::Abort),
                "hide" => miri::IsolatedOp::Reject(miri::RejectOpWith::NoWarning),
                "warn" => miri::IsolatedOp::Reject(miri::RejectOpWith::Warning),
                "warn-nobacktrace" =>
                    miri::IsolatedOp::Reject(miri::RejectOpWith::WarningWithoutBacktrace),
                _ =>
                    show_error!(
                        "-Zmiri-isolation-error must be `abort`, `hide`, `warn`, or `warn-nobacktrace`"
                    ),
            };
        } else if arg == "-Zmiri-ignore-leaks" {
            miri_config.ignore_leaks = true;
            miri_config.collect_leak_backtraces = false;
        } else if arg == "-Zmiri-force-intrinsic-fallback" {
            miri_config.force_intrinsic_fallback = true;
        } else if arg == "-Zmiri-strict-provenance" {
            miri_config.provenance_mode = ProvenanceMode::Strict;
        } else if arg == "-Zmiri-permissive-provenance" {
            miri_config.provenance_mode = ProvenanceMode::Permissive;
        } else if arg == "-Zmiri-mute-stdout-stderr" {
            miri_config.mute_stdout_stderr = true;
        } else if arg == "-Zmiri-retag-fields" {
            miri_config.retag_fields = RetagFields::Yes;
        } else if arg == "-Zmiri-fixed-schedule" {
            miri_config.fixed_scheduling = true;
        } else if arg == "-Zmiri-deterministic-concurrency" {
            miri_config.fixed_scheduling = true;
            miri_config.address_reuse_cross_thread_rate = 0.0;
            miri_config.cmpxchg_weak_failure_rate = 0.0;
            miri_config.weak_memory_emulation = false;
        } else if let Some(retag_fields) = arg.strip_prefix("-Zmiri-retag-fields=") {
            miri_config.retag_fields = match retag_fields {
                "all" => RetagFields::Yes,
                "none" => RetagFields::No,
                "scalar" => RetagFields::OnlyScalar,
                _ => show_error!("`-Zmiri-retag-fields` can only be `all`, `none`, or `scalar`"),
            };
        } else if let Some(param) = arg.strip_prefix("-Zmiri-seed=") {
            let seed = param.parse::<u64>().unwrap_or_else(|_| {
                show_error!("-Zmiri-seed must be an integer that fits into u64")
            });
            miri_config.seed = Some(seed);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-many-seeds=") {
            let range = parse_range(param).unwrap_or_else(|err| {
                show_error!(
                    "-Zmiri-many-seeds requires a range in the form `from..to` or `..to`: {err}"
                )
            });
            many_seeds = Some(range);
        } else if arg == "-Zmiri-many-seeds" {
            many_seeds = Some(0..64);
        } else if arg == "-Zmiri-many-seeds-keep-going" {
            many_seeds_keep_going = true;
        } else if let Some(trimmed_arg) = arg.strip_prefix("-Zmiri-genmc") {
            // FIXME(GenMC): Currently, GenMC mode is incompatible with aliasing model checking.
            miri_config.borrow_tracker = None;
            GenmcConfig::parse_arg(&mut genmc_config, trimmed_arg);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-env-forward=") {
            miri_config.forwarded_env_vars.push(param.to_owned());
        } else if let Some(param) = arg.strip_prefix("-Zmiri-env-set=") {
            let Some((name, value)) = param.split_once('=') else {
                show_error!("-Zmiri-env-set requires an argument of the form <name>=<value>");
            };
            miri_config.set_env_vars.insert(name.to_owned(), value.to_owned());
        } else if let Some(param) = arg.strip_prefix("-Zmiri-track-pointer-tag=") {
            let ids: Vec<u64> = parse_comma_list(param).unwrap_or_else(|err| {
                show_error!("-Zmiri-track-pointer-tag requires a comma separated list of valid `u64` arguments: {err}")
            });
            for id in ids.into_iter().map(miri::BorTag::new) {
                if let Some(id) = id {
                    miri_config.tracked_pointer_tags.insert(id);
                } else {
                    show_error!("-Zmiri-track-pointer-tag requires nonzero arguments");
                }
            }
        } else if let Some(param) = arg.strip_prefix("-Zmiri-track-alloc-id=") {
            let ids = parse_comma_list::<NonZero<u64>>(param).unwrap_or_else(|err| {
                show_error!("-Zmiri-track-alloc-id requires a comma separated list of valid non-zero `u64` arguments: {err}")
            });
            miri_config.tracked_alloc_ids.extend(ids.into_iter().map(miri::AllocId));
        } else if arg == "-Zmiri-track-alloc-accesses" {
            miri_config.track_alloc_accesses = true;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-address-reuse-rate=") {
            miri_config.address_reuse_rate = parse_rate(param)
                .unwrap_or_else(|err| show_error!("-Zmiri-address-reuse-rate {err}"));
        } else if let Some(param) = arg.strip_prefix("-Zmiri-address-reuse-cross-thread-rate=") {
            miri_config.address_reuse_cross_thread_rate = parse_rate(param)
                .unwrap_or_else(|err| show_error!("-Zmiri-address-reuse-cross-thread-rate {err}"));
        } else if let Some(param) = arg.strip_prefix("-Zmiri-compare-exchange-weak-failure-rate=") {
            miri_config.cmpxchg_weak_failure_rate = parse_rate(param).unwrap_or_else(|err| {
                show_error!("-Zmiri-compare-exchange-weak-failure-rate {err}")
            });
        } else if let Some(param) = arg.strip_prefix("-Zmiri-preemption-rate=") {
            miri_config.preemption_rate =
                parse_rate(param).unwrap_or_else(|err| show_error!("-Zmiri-preemption-rate {err}"));
        } else if arg == "-Zmiri-report-progress" {
            // This makes it take a few seconds between progress reports on my laptop.
            miri_config.report_progress = Some(1_000_000);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-report-progress=") {
            let interval = param.parse::<u32>().unwrap_or_else(|err| {
                show_error!("-Zmiri-report-progress requires a `u32`: {}", err)
            });
            miri_config.report_progress = Some(interval);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-provenance-gc=") {
            let interval = param.parse::<u32>().unwrap_or_else(|err| {
                show_error!("-Zmiri-provenance-gc requires a `u32`: {}", err)
            });
            miri_config.gc_interval = interval;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-measureme=") {
            miri_config.measureme_out = Some(param.to_string());
        } else if let Some(param) = arg.strip_prefix("-Zmiri-backtrace=") {
            miri_config.backtrace_style = match param {
                "0" => BacktraceStyle::Off,
                "1" => BacktraceStyle::Short,
                "full" => BacktraceStyle::Full,
                _ => show_error!("-Zmiri-backtrace may only be 0, 1, or full"),
            };
        } else if let Some(param) = arg.strip_prefix("-Zmiri-native-lib=") {
            let filename = param.to_string();
            if std::path::Path::new(&filename).exists() {
                if let Some(other_filename) = miri_config.native_lib {
                    show_error!("-Zmiri-native-lib is already set to {}", other_filename.display());
                }
                miri_config.native_lib = Some(filename.into());
            } else {
                show_error!("-Zmiri-native-lib `{}` does not exist", filename);
            }
        } else if let Some(param) = arg.strip_prefix("-Zmiri-num-cpus=") {
            let num_cpus = param
                .parse::<u32>()
                .unwrap_or_else(|err| show_error!("-Zmiri-num-cpus requires a `u32`: {}", err));
            if !(1..=miri::MAX_CPUS).contains(&usize::try_from(num_cpus).unwrap()) {
                show_error!("-Zmiri-num-cpus must be in the range 1..={}", miri::MAX_CPUS);
            }
            miri_config.num_cpus = num_cpus;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-force-page-size=") {
            let page_size = param.parse::<u64>().unwrap_or_else(|err| {
                show_error!("-Zmiri-force-page-size requires a `u64`: {}", err)
            });
            // Convert from kilobytes to bytes.
            let page_size = if page_size.is_power_of_two() {
                page_size * 1024
            } else {
                show_error!("-Zmiri-force-page-size requires a power of 2: {page_size}");
            };
            miri_config.page_size = Some(page_size);
        } else {
            // Forward to rustc.
            rustc_args.push(arg);
        }
    }
    // Tree Borrows implies strict provenance, and is not compatible with native calls.
    if matches!(miri_config.borrow_tracker, Some(BorrowTrackerMethod::TreeBorrows)) {
        if miri_config.provenance_mode != ProvenanceMode::Strict {
            show_error!(
                "Tree Borrows does not support integer-to-pointer casts, and hence requires strict provenance"
            );
        }
        if miri_config.native_lib.is_some() {
            show_error!("Tree Borrows is not compatible with calling native functions");
        }
    }
    // Native calls and strict provenance are not compatible.
    if miri_config.native_lib.is_some() && miri_config.provenance_mode == ProvenanceMode::Strict {
        show_error!("strict provenance is not compatible with calling native functions");
    }
    // You can set either one seed or many.
    if many_seeds.is_some() && miri_config.seed.is_some() {
        show_error!("Only one of `-Zmiri-seed` and `-Zmiri-many-seeds can be set");
    }

    // Ensure we have parallelism for many-seeds mode.
    if many_seeds.is_some() && !rustc_args.iter().any(|arg| arg.starts_with("-Zthreads=")) {
        // Clamp to 20 threads; things get a less efficient beyond that due to lock contention.
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(20);
        rustc_args.push(format!("-Zthreads={threads}"));
    }
    let many_seeds =
        many_seeds.map(|seeds| ManySeedsConfig { seeds, keep_going: many_seeds_keep_going });

    // Validate settings for data race detection and GenMC mode.
    assert_eq!(genmc_config.is_some(), miri_config.genmc_mode);
    if genmc_config.is_some() {
        if !miri_config.data_race_detector {
            show_error!("Cannot disable data race detection in GenMC mode (currently)");
        } else if !miri_config.weak_memory_emulation {
            show_error!("Cannot disable weak memory emulation in GenMC mode");
        }
    } else if miri_config.weak_memory_emulation && !miri_config.data_race_detector {
        show_error!(
            "Weak memory emulation cannot be enabled when the data race detector is disabled"
        );
    };

    debug!("rustc arguments: {:?}", rustc_args);
    debug!("crate arguments: {:?}", miri_config.args);
    run_compiler_and_exit(
        &rustc_args,
        &mut MiriCompilerCalls::new(miri_config, many_seeds, genmc_config),
    )
}
