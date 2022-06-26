#![feature(rustc_private, stmt_expr_attributes)]
#![allow(clippy::manual_range_contains)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;

use std::env;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::str::FromStr;

use log::debug;

use rustc_data_structures::sync::Lrc;
use rustc_driver::Compilation;
use rustc_hir::{self as hir, def_id::LOCAL_CRATE, Node};
use rustc_interface::interface::Config;
use rustc_middle::{
    middle::exported_symbols::{
        ExportedSymbol, SymbolExportInfo, SymbolExportKind, SymbolExportLevel,
    },
    ty::{query::ExternProviders, TyCtxt},
};
use rustc_session::{search_paths::PathKind, CtfeBacktrace};

use miri::{BacktraceStyle, ProvenanceMode};

struct MiriCompilerCalls {
    miri_config: miri::MiriConfig,
}

impl rustc_driver::Callbacks for MiriCompilerCalls {
    fn config(&mut self, config: &mut Config) {
        config.override_queries = Some(|_, _, external_providers| {
            external_providers.used_crate_source = |tcx, cnum| {
                let mut providers = ExternProviders::default();
                rustc_metadata::provide_extern(&mut providers);
                let mut crate_source = (providers.used_crate_source)(tcx, cnum);
                // HACK: rustc will emit "crate ... required to be available in rlib format, but
                // was not found in this form" errors once we use `tcx.dependency_formats()` if
                // there's no rlib provided, so setting a dummy path here to workaround those errors.
                Lrc::make_mut(&mut crate_source).rlib = Some((PathBuf::new(), PathKind::All));
                crate_source
            };
        });
    }

    fn after_analysis<'tcx>(
        &mut self,
        compiler: &rustc_interface::interface::Compiler,
        queries: &'tcx rustc_interface::Queries<'tcx>,
    ) -> Compilation {
        compiler.session().abort_if_errors();

        queries.global_ctxt().unwrap().peek_mut().enter(|tcx| {
            init_late_loggers(tcx);
            let (entry_def_id, entry_type) = if let Some(entry_def) = tcx.entry_fn(()) {
                entry_def
            } else {
                tcx.sess.fatal("miri can only run programs that have a main function");
            };
            let mut config = self.miri_config.clone();

            // Add filename to `miri` arguments.
            config.args.insert(0, compiler.input().filestem().to_string());

            // Adjust working directory for interpretation.
            if let Some(cwd) = env::var_os("MIRI_CWD") {
                env::set_current_dir(cwd).unwrap();
            }

            if let Some(return_code) = miri::eval_entry(tcx, entry_def_id, entry_type, config) {
                std::process::exit(
                    i32::try_from(return_code).expect("Return value was too large!"),
                );
            }
        });

        compiler.session().abort_if_errors();

        Compilation::Stop
    }
}

struct MiriBeRustCompilerCalls {
    target_crate: bool,
}

impl rustc_driver::Callbacks for MiriBeRustCompilerCalls {
    fn config(&mut self, config: &mut Config) {
        if config.opts.prints.is_empty() && self.target_crate {
            // Queries overriden here affect the data stored in `rmeta` files of dependencies,
            // which will be used later in non-`MIRI_BE_RUSTC` mode.
            config.override_queries = Some(|_, local_providers, _| {
                // `exported_symbols()` provided by rustc always returns empty result if
                // `tcx.sess.opts.output_types.should_codegen()` is false.
                local_providers.exported_symbols = |tcx, cnum| {
                    assert_eq!(cnum, LOCAL_CRATE);
                    tcx.arena.alloc_from_iter(
                        // This is based on:
                        // https://github.com/rust-lang/rust/blob/2962e7c0089d5c136f4e9600b7abccfbbde4973d/compiler/rustc_codegen_ssa/src/back/symbol_export.rs#L62-L63
                        // https://github.com/rust-lang/rust/blob/2962e7c0089d5c136f4e9600b7abccfbbde4973d/compiler/rustc_codegen_ssa/src/back/symbol_export.rs#L174
                        tcx.reachable_set(()).iter().filter_map(|&local_def_id| {
                            // Do the same filtering that rustc does:
                            // https://github.com/rust-lang/rust/blob/2962e7c0089d5c136f4e9600b7abccfbbde4973d/compiler/rustc_codegen_ssa/src/back/symbol_export.rs#L84-L102
                            // Otherwise it may cause unexpected behaviours and ICEs
                            // (https://github.com/rust-lang/rust/issues/86261).
                            let is_reachable_non_generic = matches!(
                                tcx.hir().get(tcx.hir().local_def_id_to_hir_id(local_def_id)),
                                Node::Item(&hir::Item {
                                    kind: hir::ItemKind::Static(..) | hir::ItemKind::Fn(..),
                                    ..
                                }) | Node::ImplItem(&hir::ImplItem {
                                    kind: hir::ImplItemKind::Fn(..),
                                    ..
                                })
                                if !tcx.generics_of(local_def_id).requires_monomorphization(tcx)
                            );
                            (is_reachable_non_generic
                                && tcx.codegen_fn_attrs(local_def_id).contains_extern_indicator())
                            .then_some((
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
                        }),
                    )
                }
            });
        }
    }
}

fn init_early_loggers() {
    // Note that our `extern crate log` is *not* the same as rustc's; as a result, we have to
    // initialize them both, and we always initialize `miri`'s first.
    let env = env_logger::Env::new().filter("MIRI_LOG").write_style("MIRI_LOG_STYLE");
    env_logger::init_from_env(env);
    // Enable verbose entry/exit logging by default if MIRI_LOG is set.
    if env::var_os("MIRI_LOG").is_some() && env::var_os("RUSTC_LOG_ENTRY_EXIT").is_none() {
        env::set_var("RUSTC_LOG_ENTRY_EXIT", "1");
    }
    // We only initialize `rustc` if the env var is set (so the user asked for it).
    // If it is not set, we avoid initializing now so that we can initialize
    // later with our custom settings, and *not* log anything for what happens before
    // `miri` gets started.
    if env::var_os("RUSTC_LOG").is_some() {
        rustc_driver::init_rustc_env_logger();
    }
}

fn init_late_loggers(tcx: TyCtxt<'_>) {
    // We initialize loggers right before we start evaluation. We overwrite the `RUSTC_LOG`
    // env var if it is not set, control it based on `MIRI_LOG`.
    // (FIXME: use `var_os`, but then we need to manually concatenate instead of `format!`.)
    if let Ok(var) = env::var("MIRI_LOG") {
        if env::var_os("RUSTC_LOG").is_none() {
            // We try to be a bit clever here: if `MIRI_LOG` is just a single level
            // used for everything, we only apply it to the parts of rustc that are
            // CTFE-related. Otherwise, we use it verbatim for `RUSTC_LOG`.
            // This way, if you set `MIRI_LOG=trace`, you get only the right parts of
            // rustc traced, but you can also do `MIRI_LOG=miri=trace,rustc_const_eval::interpret=debug`.
            if log::Level::from_str(&var).is_ok() {
                env::set_var(
                    "RUSTC_LOG",
                    &format!(
                        "rustc_middle::mir::interpret={0},rustc_const_eval::interpret={0}",
                        var
                    ),
                );
            } else {
                env::set_var("RUSTC_LOG", &var);
            }
            rustc_driver::init_rustc_env_logger();
        }
    }

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

/// Returns the "default sysroot" that Miri will use if no `--sysroot` flag is set.
/// Should be a compile-time constant.
fn compile_time_sysroot() -> Option<String> {
    if option_env!("RUSTC_STAGE").is_some() {
        // This is being built as part of rustc, and gets shipped with rustup.
        // We can rely on the sysroot computation in librustc_session.
        return None;
    }
    // For builds outside rustc, we need to ensure that we got a sysroot
    // that gets used as a default.  The sysroot computation in librustc_session would
    // end up somewhere in the build dir (see `get_or_default_sysroot`).
    // Taken from PR <https://github.com/Manishearth/rust-clippy/pull/911>.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    Some(match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ =>
            option_env!("RUST_SYSROOT")
                .expect(
                    "To build Miri without rustup, set the `RUST_SYSROOT` env var at build time",
                )
                .to_owned(),
    })
}

/// Execute a compiler with the given CLI arguments and callbacks.
fn run_compiler(
    mut args: Vec<String>,
    callbacks: &mut (dyn rustc_driver::Callbacks + Send),
    insert_default_args: bool,
) -> ! {
    // Make sure we use the right default sysroot. The default sysroot is wrong,
    // because `get_or_default_sysroot` in `librustc_session` bases that on `current_exe`.
    //
    // Make sure we always call `compile_time_sysroot` as that also does some sanity-checks
    // of the environment we were built in.
    // FIXME: Ideally we'd turn a bad build env into a compile-time error via CTFE or so.
    if let Some(sysroot) = compile_time_sysroot() {
        let sysroot_flag = "--sysroot";
        if !args.iter().any(|e| e == sysroot_flag) {
            // We need to overwrite the default that librustc_session would compute.
            args.push(sysroot_flag.to_owned());
            args.push(sysroot);
        }
    }

    if insert_default_args {
        // Some options have different defaults in Miri than in plain rustc; apply those by making
        // them the first arguments after the binary name (but later arguments can overwrite them).
        args.splice(1..1, miri::MIRI_DEFAULT_ARGS.iter().map(ToString::to_string));
    }

    // Invoke compiler, and handle return code.
    let exit_code = rustc_driver::catch_with_exit_code(move || {
        rustc_driver::RunCompiler::new(&args, callbacks).run()
    });
    std::process::exit(exit_code)
}

/// Parses a comma separated list of `T` from the given string:
///
/// `<value1>,<value2>,<value3>,...`
fn parse_comma_list<T: FromStr>(input: &str) -> Result<Vec<T>, T::Err> {
    input.split(',').map(str::parse::<T>).collect()
}

fn main() {
    rustc_driver::install_ice_hook();

    // If the environment asks us to actually be rustc, then do that.
    if let Some(crate_kind) = env::var_os("MIRI_BE_RUSTC") {
        rustc_driver::init_rustc_env_logger();

        let target_crate = if crate_kind == "target" {
            true
        } else if crate_kind == "host" {
            false
        } else {
            panic!("invalid `MIRI_BE_RUSTC` value: {:?}", crate_kind)
        };

        // We cannot use `rustc_driver::main` as we need to adjust the CLI arguments.
        run_compiler(
            env::args().collect(),
            &mut MiriBeRustCompilerCalls { target_crate },
            // Don't insert `MIRI_DEFAULT_ARGS`, in particular, `--cfg=miri`, if we are building
            // a "host" crate. That may cause procedural macros (and probably build scripts) to
            // depend on Miri-only symbols, such as `miri_resolve_frame`:
            // https://github.com/rust-lang/miri/issues/1760
            #[rustfmt::skip]
            /* insert_default_args: */ target_crate,
        )
    }

    // Init loggers the Miri way.
    init_early_loggers();

    // Parse our arguments and split them across `rustc` and `miri`.
    let mut miri_config = miri::MiriConfig::default();
    let mut rustc_args = vec![];
    let mut after_dashdash = false;

    // If user has explicitly enabled/disabled isolation
    let mut isolation_enabled: Option<bool> = None;
    for arg in env::args() {
        if rustc_args.is_empty() {
            // Very first arg: binary name.
            rustc_args.push(arg);
        } else if after_dashdash {
            // Everything that comes after `--` is forwarded to the interpreted crate.
            miri_config.args.push(arg);
        } else if arg == "--" {
            after_dashdash = true;
        } else if arg == "-Zmiri-disable-validation" {
            miri_config.validate = false;
        } else if arg == "-Zmiri-disable-stacked-borrows" {
            miri_config.stacked_borrows = false;
        } else if arg == "-Zmiri-disable-data-race-detector" {
            miri_config.data_race_detector = false;
            miri_config.weak_memory_emulation = false;
        } else if arg == "-Zmiri-disable-alignment-check" {
            miri_config.check_alignment = miri::AlignmentCheck::None;
        } else if arg == "-Zmiri-symbolic-alignment-check" {
            miri_config.check_alignment = miri::AlignmentCheck::Symbolic;
        } else if arg == "-Zmiri-check-number-validity" {
            eprintln!(
                "WARNING: the flag `-Zmiri-check-number-validity` no longer has any effect \
                        since it is now enabled by default"
            );
        } else if arg == "-Zmiri-allow-uninit-numbers" {
            eprintln!(
                "WARNING: `-Zmiri-allow-uninit-numbers` is deprecated and planned to be removed. \
                Please let us know at <https://github.com/rust-lang/miri/issues/2187> if you rely on this flag."
            );
            miri_config.allow_uninit_numbers = true;
        } else if arg == "-Zmiri-allow-ptr-int-transmute" {
            eprintln!(
                "WARNING: `-Zmiri-allow-ptr-int-transmute` is deprecated and planned to be removed. \
                Please let us know at <https://github.com/rust-lang/miri/issues/2188> if you rely on this flag."
            );
            miri_config.allow_ptr_int_transmute = true;
        } else if arg == "-Zmiri-disable-abi-check" {
            miri_config.check_abi = false;
        } else if arg == "-Zmiri-disable-isolation" {
            if matches!(isolation_enabled, Some(true)) {
                panic!("-Zmiri-disable-isolation cannot be used along with -Zmiri-isolation-error");
            } else {
                isolation_enabled = Some(false);
            }
            miri_config.isolated_op = miri::IsolatedOp::Allow;
        } else if arg == "-Zmiri-disable-weak-memory-emulation" {
            miri_config.weak_memory_emulation = false;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-isolation-error=") {
            if matches!(isolation_enabled, Some(false)) {
                panic!("-Zmiri-isolation-error cannot be used along with -Zmiri-disable-isolation");
            } else {
                isolation_enabled = Some(true);
            }

            miri_config.isolated_op = match param {
                "abort" => miri::IsolatedOp::Reject(miri::RejectOpWith::Abort),
                "hide" => miri::IsolatedOp::Reject(miri::RejectOpWith::NoWarning),
                "warn" => miri::IsolatedOp::Reject(miri::RejectOpWith::Warning),
                "warn-nobacktrace" =>
                    miri::IsolatedOp::Reject(miri::RejectOpWith::WarningWithoutBacktrace),
                _ =>
                    panic!(
                        "-Zmiri-isolation-error must be `abort`, `hide`, `warn`, or `warn-nobacktrace`"
                    ),
            };
        } else if arg == "-Zmiri-ignore-leaks" {
            miri_config.ignore_leaks = true;
        } else if arg == "-Zmiri-panic-on-unsupported" {
            miri_config.panic_on_unsupported = true;
        } else if arg == "-Zmiri-tag-raw-pointers" {
            miri_config.tag_raw = true;
        } else if arg == "-Zmiri-strict-provenance" {
            miri_config.provenance_mode = ProvenanceMode::Strict;
            miri_config.tag_raw = true;
        } else if arg == "-Zmiri-permissive-provenance" {
            miri_config.provenance_mode = ProvenanceMode::Permissive;
            miri_config.tag_raw = true;
        } else if arg == "-Zmiri-mute-stdout-stderr" {
            miri_config.mute_stdout_stderr = true;
        } else if arg == "-Zmiri-track-raw-pointers" {
            eprintln!(
                "WARNING: -Zmiri-track-raw-pointers has been renamed to -Zmiri-tag-raw-pointers, the old name is deprecated."
            );
            miri_config.tag_raw = true;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-seed=") {
            if miri_config.seed.is_some() {
                panic!("Cannot specify -Zmiri-seed multiple times!");
            }
            let seed = u64::from_str_radix(param, 16)
                        .unwrap_or_else(|_| panic!(
                            "-Zmiri-seed should only contain valid hex digits [0-9a-fA-F] and fit into a u64 (max 16 characters)"
                        ));
            miri_config.seed = Some(seed);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-env-exclude=") {
            miri_config.excluded_env_vars.push(param.to_owned());
        } else if let Some(param) = arg.strip_prefix("-Zmiri-env-forward=") {
            miri_config.forwarded_env_vars.push(param.to_owned());
        } else if let Some(param) = arg.strip_prefix("-Zmiri-track-pointer-tag=") {
            let ids: Vec<u64> = match parse_comma_list(param) {
                Ok(ids) => ids,
                Err(err) =>
                    panic!(
                        "-Zmiri-track-pointer-tag requires a comma separated list of valid `u64` arguments: {}",
                        err
                    ),
            };
            for id in ids.into_iter().map(miri::PtrId::new) {
                if let Some(id) = id {
                    miri_config.tracked_pointer_tags.insert(id);
                } else {
                    panic!("-Zmiri-track-pointer-tag requires nonzero arguments");
                }
            }
        } else if let Some(param) = arg.strip_prefix("-Zmiri-track-call-id=") {
            let ids: Vec<u64> = match parse_comma_list(param) {
                Ok(ids) => ids,
                Err(err) =>
                    panic!(
                        "-Zmiri-track-call-id requires a comma separated list of valid `u64` arguments: {}",
                        err
                    ),
            };
            for id in ids.into_iter().map(miri::CallId::new) {
                if let Some(id) = id {
                    miri_config.tracked_call_ids.insert(id);
                } else {
                    panic!("-Zmiri-track-call-id requires a nonzero argument");
                }
            }
        } else if let Some(param) = arg.strip_prefix("-Zmiri-track-alloc-id=") {
            let ids: Vec<miri::AllocId> = match parse_comma_list::<NonZeroU64>(param) {
                Ok(ids) => ids.into_iter().map(miri::AllocId).collect(),
                Err(err) =>
                    panic!(
                        "-Zmiri-track-alloc-id requires a comma separated list of valid non-zero `u64` arguments: {}",
                        err
                    ),
            };
            miri_config.tracked_alloc_ids.extend(ids);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-compare-exchange-weak-failure-rate=") {
            let rate = match param.parse::<f64>() {
                Ok(rate) if rate >= 0.0 && rate <= 1.0 => rate,
                Ok(_) =>
                    panic!(
                        "-Zmiri-compare-exchange-weak-failure-rate must be between `0.0` and `1.0`"
                    ),
                Err(err) =>
                    panic!(
                        "-Zmiri-compare-exchange-weak-failure-rate requires a `f64` between `0.0` and `1.0`: {}",
                        err
                    ),
            };
            miri_config.cmpxchg_weak_failure_rate = rate;
        } else if let Some(param) = arg.strip_prefix("-Zmiri-preemption-rate=") {
            let rate = match param.parse::<f64>() {
                Ok(rate) if rate >= 0.0 && rate <= 1.0 => rate,
                Ok(_) => panic!("-Zmiri-preemption-rate must be between `0.0` and `1.0`"),
                Err(err) =>
                    panic!(
                        "-Zmiri-preemption-rate requires a `f64` between `0.0` and `1.0`: {}",
                        err
                    ),
            };
            miri_config.preemption_rate = rate;
        } else if arg == "-Zmiri-report-progress" {
            // This makes it take a few seconds between progress reports on my laptop.
            miri_config.report_progress = Some(1_000_000);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-report-progress=") {
            let interval = match param.parse::<u32>() {
                Ok(i) => i,
                Err(err) => panic!("-Zmiri-report-progress requires a `u32`: {}", err),
            };
            miri_config.report_progress = Some(interval);
        } else if let Some(param) = arg.strip_prefix("-Zmiri-measureme=") {
            miri_config.measureme_out = Some(param.to_string());
        } else if let Some(param) = arg.strip_prefix("-Zmiri-backtrace=") {
            miri_config.backtrace_style = match param {
                "0" => BacktraceStyle::Off,
                "1" => BacktraceStyle::Short,
                "full" => BacktraceStyle::Full,
                _ => panic!("-Zmiri-backtrace may only be 0, 1, or full"),
            };
        } else {
            // Forward to rustc.
            rustc_args.push(arg);
        }
    }

    debug!("rustc arguments: {:?}", rustc_args);
    debug!("crate arguments: {:?}", miri_config.args);
    run_compiler(
        rustc_args,
        &mut MiriCompilerCalls { miri_config },
        /* insert_default_args: */ true,
    )
}
