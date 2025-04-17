//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(decl_macro)]
#![feature(let_chains)]
#![feature(panic_backtrace_config)]
#![feature(panic_update_hook)]
#![feature(result_flattening)]
#![feature(rustdoc_internals)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, IsTerminal, Read, Write};
use std::panic::{self, PanicHookInfo, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use std::{env, str};

use rustc_ast as ast;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CodegenErrors, CodegenResults};
use rustc_data_structures::profiling::{
    TimePassesFormat, get_resident_set_size, print_time_passes_entry,
};
use rustc_errors::emitter::stderr_destination;
use rustc_errors::registry::Registry;
use rustc_errors::{ColorConfig, DiagCtxt, ErrCode, FatalError, PResult, markdown};
use rustc_feature::find_gated_cfg;
// This avoids a false positive with `-Wunused_crate_dependencies`.
// `rust_index` isn't used in this crate's code, but it must be named in the
// `Cargo.toml` for the `rustc_randomized_layouts` feature.
use rustc_index as _;
use rustc_interface::util::{self, get_codegen_backend};
use rustc_interface::{Linker, create_and_enter_global_ctxt, interface, passes};
use rustc_lint::unerased_lint_store;
use rustc_metadata::creader::MetadataLoader;
use rustc_metadata::locator;
use rustc_middle::ty::TyCtxt;
use rustc_parse::{new_parser_from_file, new_parser_from_source_str, unwrap_or_emit_fatal};
use rustc_session::config::{
    CG_OPTIONS, ErrorOutputType, Input, OptionDesc, OutFileName, OutputType, UnstableOptions,
    Z_OPTIONS, nightly_options, parse_target_triple,
};
use rustc_session::getopts::{self, Matches};
use rustc_session::lint::{Lint, LintId};
use rustc_session::output::{CRATE_TYPES, collect_crate_types, invalid_output_for_target};
use rustc_session::{EarlyDiagCtxt, Session, config, filesearch};
use rustc_span::FileName;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_target::json::ToJson;
use rustc_target::spec::{Target, TargetTuple};
use tracing::trace;

#[allow(unused_macros)]
macro do_not_use_print($($t:tt)*) {
    std::compile_error!(
        "Don't use `print` or `println` here, use `safe_print` or `safe_println` instead"
    )
}

#[allow(unused_macros)]
macro do_not_use_safe_print($($t:tt)*) {
    std::compile_error!("Don't use `safe_print` or `safe_println` here, use `println_info` instead")
}

// This import blocks the use of panicking `print` and `println` in all the code
// below. Please use `safe_print` and `safe_println` to avoid ICE when
// encountering an I/O error during print.
#[allow(unused_imports)]
use {do_not_use_print as print, do_not_use_print as println};

pub mod args;
pub mod pretty;
#[macro_use]
mod print;
mod session_diagnostics;

// Keep the OS parts of this `cfg` in sync with the `cfg` on the `libc`
// dependency in `compiler/rustc_driver/Cargo.toml`, to keep
// `-Wunused-crated-dependencies` satisfied.
#[cfg(all(not(miri), unix, any(target_env = "gnu", target_os = "macos")))]
mod signal_handler;

#[cfg(not(all(not(miri), unix, any(target_env = "gnu", target_os = "macos"))))]
mod signal_handler {
    /// On platforms which don't support our signal handler's requirements,
    /// simply use the default signal handler provided by std.
    pub(super) fn install() {}
}

use crate::session_diagnostics::{
    CantEmitMIR, RLinkEmptyVersionNumber, RLinkEncodingVersionMismatch, RLinkRustcVersionMismatch,
    RLinkWrongFileType, RlinkCorruptFile, RlinkNotAFile, RlinkUnableToRead, UnstableFeatureUsage,
};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub static DEFAULT_LOCALE_RESOURCES: &[&str] = &[
    // tidy-alphabetical-start
    crate::DEFAULT_LOCALE_RESOURCE,
    rustc_ast_lowering::DEFAULT_LOCALE_RESOURCE,
    rustc_ast_passes::DEFAULT_LOCALE_RESOURCE,
    rustc_attr_parsing::DEFAULT_LOCALE_RESOURCE,
    rustc_borrowck::DEFAULT_LOCALE_RESOURCE,
    rustc_builtin_macros::DEFAULT_LOCALE_RESOURCE,
    rustc_codegen_ssa::DEFAULT_LOCALE_RESOURCE,
    rustc_const_eval::DEFAULT_LOCALE_RESOURCE,
    rustc_errors::DEFAULT_LOCALE_RESOURCE,
    rustc_expand::DEFAULT_LOCALE_RESOURCE,
    rustc_hir_analysis::DEFAULT_LOCALE_RESOURCE,
    rustc_hir_typeck::DEFAULT_LOCALE_RESOURCE,
    rustc_incremental::DEFAULT_LOCALE_RESOURCE,
    rustc_infer::DEFAULT_LOCALE_RESOURCE,
    rustc_interface::DEFAULT_LOCALE_RESOURCE,
    rustc_lint::DEFAULT_LOCALE_RESOURCE,
    rustc_metadata::DEFAULT_LOCALE_RESOURCE,
    rustc_middle::DEFAULT_LOCALE_RESOURCE,
    rustc_mir_build::DEFAULT_LOCALE_RESOURCE,
    rustc_mir_dataflow::DEFAULT_LOCALE_RESOURCE,
    rustc_mir_transform::DEFAULT_LOCALE_RESOURCE,
    rustc_monomorphize::DEFAULT_LOCALE_RESOURCE,
    rustc_parse::DEFAULT_LOCALE_RESOURCE,
    rustc_passes::DEFAULT_LOCALE_RESOURCE,
    rustc_pattern_analysis::DEFAULT_LOCALE_RESOURCE,
    rustc_privacy::DEFAULT_LOCALE_RESOURCE,
    rustc_query_system::DEFAULT_LOCALE_RESOURCE,
    rustc_resolve::DEFAULT_LOCALE_RESOURCE,
    rustc_session::DEFAULT_LOCALE_RESOURCE,
    rustc_trait_selection::DEFAULT_LOCALE_RESOURCE,
    rustc_ty_utils::DEFAULT_LOCALE_RESOURCE,
    // tidy-alphabetical-end
];

/// Exit status code used for successful compilation and help output.
pub const EXIT_SUCCESS: i32 = 0;

/// Exit status code used for compilation failures and invalid flags.
pub const EXIT_FAILURE: i32 = 1;

pub const DEFAULT_BUG_REPORT_URL: &str = "https://github.com/rust-lang/rust/issues/new\
    ?labels=C-bug%2C+I-ICE%2C+T-compiler&template=ice.md";

pub trait Callbacks {
    /// Called before creating the compiler instance
    fn config(&mut self, _config: &mut interface::Config) {}
    /// Called after parsing the crate root. Submodules are not yet parsed when
    /// this callback is called. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_crate_root_parsing(
        &mut self,
        _compiler: &interface::Compiler,
        _krate: &mut ast::Crate,
    ) -> Compilation {
        Compilation::Continue
    }
    /// Called after expansion. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_expansion<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        _tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        Compilation::Continue
    }
    /// Called after analysis. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        _tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        Compilation::Continue
    }
}

#[derive(Default)]
pub struct TimePassesCallbacks {
    time_passes: Option<TimePassesFormat>,
}

impl Callbacks for TimePassesCallbacks {
    // JUSTIFICATION: the session doesn't exist at this point.
    #[allow(rustc::bad_opt_access)]
    fn config(&mut self, config: &mut interface::Config) {
        // If a --print=... option has been given, we don't print the "total"
        // time because it will mess up the --print output. See #64339.
        //
        self.time_passes = (config.opts.prints.is_empty() && config.opts.unstable_opts.time_passes)
            .then(|| config.opts.unstable_opts.time_passes_format);
        config.opts.trimmed_def_paths = true;
    }
}

pub fn diagnostics_registry() -> Registry {
    Registry::new(rustc_errors::codes::DIAGNOSTICS)
}

/// This is the primary entry point for rustc.
pub fn run_compiler(at_args: &[String], callbacks: &mut (dyn Callbacks + Send)) {
    let mut default_early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    // Throw away the first argument, the name of the binary.
    // In case of at_args being empty, as might be the case by
    // passing empty argument array to execve under some platforms,
    // just use an empty slice.
    //
    // This situation was possible before due to arg_expand_all being
    // called before removing the argument, enabling a crash by calling
    // the compiler with @empty_file as argv[0] and no more arguments.
    let at_args = at_args.get(1..).unwrap_or_default();

    let args = args::arg_expand_all(&default_early_dcx, at_args);

    let Some(matches) = handle_options(&default_early_dcx, &args) else {
        return;
    };

    let sopts = config::build_session_options(&mut default_early_dcx, &matches);
    // fully initialize ice path static once unstable options are available as context
    let ice_file = ice_path_with_config(Some(&sopts.unstable_opts)).clone();

    if let Some(ref code) = matches.opt_str("explain") {
        handle_explain(&default_early_dcx, diagnostics_registry(), code, sopts.color);
        return;
    }

    let input = make_input(&default_early_dcx, &matches.free);
    let has_input = input.is_some();
    let (odir, ofile) = make_output(&matches);

    drop(default_early_dcx);

    let mut config = interface::Config {
        opts: sopts,
        crate_cfg: matches.opt_strs("cfg"),
        crate_check_cfg: matches.opt_strs("check-cfg"),
        input: input.unwrap_or(Input::File(PathBuf::new())),
        output_file: ofile,
        output_dir: odir,
        ice_file,
        file_loader: None,
        locale_resources: DEFAULT_LOCALE_RESOURCES.to_vec(),
        lint_caps: Default::default(),
        psess_created: None,
        hash_untracked_state: None,
        register_lints: None,
        override_queries: None,
        extra_symbols: Vec::new(),
        make_codegen_backend: None,
        registry: diagnostics_registry(),
        using_internal_features: &USING_INTERNAL_FEATURES,
        expanded_args: args,
    };

    callbacks.config(&mut config);

    let registered_lints = config.register_lints.is_some();

    interface::run_compiler(config, |compiler| {
        let sess = &compiler.sess;
        let codegen_backend = &*compiler.codegen_backend;

        // This is used for early exits unrelated to errors. E.g. when just
        // printing some information without compiling, or exiting immediately
        // after parsing, etc.
        let early_exit = || {
            sess.dcx().abort_if_errors();
        };

        // This implements `-Whelp`. It should be handled very early, like
        // `--help`/`-Zhelp`/`-Chelp`. This is the earliest it can run, because
        // it must happen after lints are registered, during session creation.
        if sess.opts.describe_lints {
            describe_lints(sess, registered_lints);
            return early_exit();
        }

        if print_crate_info(codegen_backend, sess, has_input) == Compilation::Stop {
            return early_exit();
        }

        if !has_input {
            #[allow(rustc::diagnostic_outside_of_impl)]
            sess.dcx().fatal("no input filename given"); // this is fatal
        }

        if !sess.opts.unstable_opts.ls.is_empty() {
            list_metadata(sess, &*codegen_backend.metadata_loader());
            return early_exit();
        }

        if sess.opts.unstable_opts.link_only {
            process_rlink(sess, compiler);
            return early_exit();
        }

        // Parse the crate root source code (doesn't parse submodules yet)
        // Everything else is parsed during macro expansion.
        let mut krate = passes::parse(sess);

        // If pretty printing is requested: Figure out the representation, print it and exit
        if let Some(pp_mode) = sess.opts.pretty {
            if pp_mode.needs_ast_map() {
                create_and_enter_global_ctxt(compiler, krate, |tcx| {
                    tcx.ensure_ok().early_lint_checks(());
                    pretty::print(sess, pp_mode, pretty::PrintExtra::NeedsAstMap { tcx });
                    passes::write_dep_info(tcx);
                });
            } else {
                pretty::print(sess, pp_mode, pretty::PrintExtra::AfterParsing { krate: &krate });
            }
            trace!("finished pretty-printing");
            return early_exit();
        }

        if callbacks.after_crate_root_parsing(compiler, &mut krate) == Compilation::Stop {
            return early_exit();
        }

        if sess.opts.unstable_opts.parse_crate_root_only {
            return early_exit();
        }

        let linker = create_and_enter_global_ctxt(compiler, krate, |tcx| {
            let early_exit = || {
                sess.dcx().abort_if_errors();
                None
            };

            // Make sure name resolution and macro expansion is run.
            let _ = tcx.resolver_for_lowering();

            if callbacks.after_expansion(compiler, tcx) == Compilation::Stop {
                return early_exit();
            }

            passes::write_dep_info(tcx);

            if sess.opts.output_types.contains_key(&OutputType::DepInfo)
                && sess.opts.output_types.len() == 1
            {
                return early_exit();
            }

            if sess.opts.unstable_opts.no_analysis {
                return early_exit();
            }

            tcx.ensure_ok().analysis(());

            if let Some(metrics_dir) = &sess.opts.unstable_opts.metrics_dir {
                dump_feature_usage_metrics(tcx, metrics_dir);
            }

            if callbacks.after_analysis(compiler, tcx) == Compilation::Stop {
                return early_exit();
            }

            if tcx.sess.opts.output_types.contains_key(&OutputType::Mir) {
                if let Err(error) = rustc_mir_transform::dump_mir::emit_mir(tcx) {
                    tcx.dcx().emit_fatal(CantEmitMIR { error });
                }
            }

            Some(Linker::codegen_and_build_linker(tcx, &*compiler.codegen_backend))
        });

        // Linking is done outside the `compiler.enter()` so that the
        // `GlobalCtxt` within `Queries` can be freed as early as possible.
        if let Some(linker) = linker {
            linker.link(sess, codegen_backend);
        }
    })
}

fn dump_feature_usage_metrics(tcxt: TyCtxt<'_>, metrics_dir: &Path) {
    let hash = tcxt.crate_hash(LOCAL_CRATE);
    let crate_name = tcxt.crate_name(LOCAL_CRATE);
    let metrics_file_name = format!("unstable_feature_usage_metrics-{crate_name}-{hash}.json");
    let metrics_path = metrics_dir.join(metrics_file_name);
    if let Err(error) = tcxt.features().dump_feature_usage_metrics(metrics_path) {
        // FIXME(yaahc): once metrics can be enabled by default we will want "failure to emit
        // default metrics" to only produce a warning when metrics are enabled by default and emit
        // an error only when the user manually enables metrics
        tcxt.dcx().emit_err(UnstableFeatureUsage { error });
    }
}

/// Extract output directory and file from matches.
fn make_output(matches: &getopts::Matches) -> (Option<PathBuf>, Option<OutFileName>) {
    let odir = matches.opt_str("out-dir").map(|o| PathBuf::from(&o));
    let ofile = matches.opt_str("o").map(|o| match o.as_str() {
        "-" => OutFileName::Stdout,
        path => OutFileName::Real(PathBuf::from(path)),
    });
    (odir, ofile)
}

/// Extract input (string or file and optional path) from matches.
/// This handles reading from stdin if `-` is provided.
fn make_input(early_dcx: &EarlyDiagCtxt, free_matches: &[String]) -> Option<Input> {
    match free_matches {
        [] => None, // no input: we will exit early,
        [ifile] if ifile == "-" => {
            // read from stdin as `Input::Str`
            let mut input = String::new();
            if io::stdin().read_to_string(&mut input).is_err() {
                // Immediately stop compilation if there was an issue reading
                // the input (for example if the input stream is not UTF-8).
                early_dcx
                    .early_fatal("couldn't read from stdin, as it did not contain valid UTF-8");
            }

            let name = match env::var("UNSTABLE_RUSTDOC_TEST_PATH") {
                Ok(path) => {
                    let line = env::var("UNSTABLE_RUSTDOC_TEST_LINE").expect(
                        "when UNSTABLE_RUSTDOC_TEST_PATH is set \
                                    UNSTABLE_RUSTDOC_TEST_LINE also needs to be set",
                    );
                    let line = isize::from_str_radix(&line, 10)
                        .expect("UNSTABLE_RUSTDOC_TEST_LINE needs to be an number");
                    FileName::doc_test_source_code(PathBuf::from(path), line)
                }
                Err(_) => FileName::anon_source_code(&input),
            };

            Some(Input::Str { name, input })
        }
        [ifile] => Some(Input::File(PathBuf::from(ifile))),
        [ifile1, ifile2, ..] => early_dcx.early_fatal(format!(
            "multiple input filenames provided (first two filenames are `{}` and `{}`)",
            ifile1, ifile2
        )),
    }
}

/// Whether to stop or continue compilation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Compilation {
    Stop,
    Continue,
}

fn handle_explain(early_dcx: &EarlyDiagCtxt, registry: Registry, code: &str, color: ColorConfig) {
    // Allow "E0123" or "0123" form.
    let upper_cased_code = code.to_ascii_uppercase();
    if let Ok(code) = upper_cased_code.strip_prefix('E').unwrap_or(&upper_cased_code).parse::<u32>()
        && let Ok(description) = registry.try_find_description(ErrCode::from_u32(code))
    {
        let mut is_in_code_block = false;
        let mut text = String::new();
        // Slice off the leading newline and print.
        for line in description.lines() {
            let indent_level =
                line.find(|c: char| !c.is_whitespace()).unwrap_or_else(|| line.len());
            let dedented_line = &line[indent_level..];
            if dedented_line.starts_with("```") {
                is_in_code_block = !is_in_code_block;
                text.push_str(&line[..(indent_level + 3)]);
            } else if is_in_code_block && dedented_line.starts_with("# ") {
                continue;
            } else {
                text.push_str(line);
            }
            text.push('\n');
        }
        if io::stdout().is_terminal() {
            show_md_content_with_pager(&text, color);
        } else {
            safe_print!("{text}");
        }
    } else {
        early_dcx.early_fatal(format!("{code} is not a valid error code"));
    }
}

/// If `color` is `always` or `auto`, try to print pretty (formatted & colorized) markdown. If
/// that fails or `color` is `never`, print the raw markdown.
///
/// Uses a pager if possible, falls back to stdout.
fn show_md_content_with_pager(content: &str, color: ColorConfig) {
    let pager_name = env::var_os("PAGER").unwrap_or_else(|| {
        if cfg!(windows) { OsString::from("more.com") } else { OsString::from("less") }
    });

    let mut cmd = Command::new(&pager_name);
    if pager_name == "less" {
        cmd.arg("-R"); // allows color escape sequences
    }

    let pretty_on_pager = match color {
        ColorConfig::Auto => {
            // Add other pagers that accept color escape sequences here.
            ["less", "bat", "batcat", "delta"].iter().any(|v| *v == pager_name)
        }
        ColorConfig::Always => true,
        ColorConfig::Never => false,
    };

    // Try to prettify the raw markdown text. The result can be used by the pager or on stdout.
    let pretty_data = {
        let mdstream = markdown::MdStream::parse_str(content);
        let bufwtr = markdown::create_stdout_bufwtr();
        let mut mdbuf = bufwtr.buffer();
        if mdstream.write_termcolor_buf(&mut mdbuf).is_ok() { Some((bufwtr, mdbuf)) } else { None }
    };

    // Try to print via the pager, pretty output if possible.
    let pager_res: Option<()> = try {
        let mut pager = cmd.stdin(Stdio::piped()).spawn().ok()?;

        let pager_stdin = pager.stdin.as_mut()?;
        if pretty_on_pager && let Some((_, mdbuf)) = &pretty_data {
            pager_stdin.write_all(mdbuf.as_slice()).ok()?;
        } else {
            pager_stdin.write_all(content.as_bytes()).ok()?;
        };

        pager.wait().ok()?;
    };
    if pager_res.is_some() {
        return;
    }

    // The pager failed. Try to print pretty output to stdout.
    if let Some((bufwtr, mdbuf)) = &pretty_data
        && bufwtr.print(&mdbuf).is_ok()
    {
        return;
    }

    // Everything failed. Print the raw markdown text.
    safe_print!("{content}");
}

fn process_rlink(sess: &Session, compiler: &interface::Compiler) {
    assert!(sess.opts.unstable_opts.link_only);
    let dcx = sess.dcx();
    if let Input::File(file) = &sess.io.input {
        let rlink_data = fs::read(file).unwrap_or_else(|err| {
            dcx.emit_fatal(RlinkUnableToRead { err });
        });
        let (codegen_results, outputs) = match CodegenResults::deserialize_rlink(sess, rlink_data) {
            Ok((codegen, outputs)) => (codegen, outputs),
            Err(err) => {
                match err {
                    CodegenErrors::WrongFileType => dcx.emit_fatal(RLinkWrongFileType),
                    CodegenErrors::EmptyVersionNumber => dcx.emit_fatal(RLinkEmptyVersionNumber),
                    CodegenErrors::EncodingVersionMismatch { version_array, rlink_version } => dcx
                        .emit_fatal(RLinkEncodingVersionMismatch { version_array, rlink_version }),
                    CodegenErrors::RustcVersionMismatch { rustc_version } => {
                        dcx.emit_fatal(RLinkRustcVersionMismatch {
                            rustc_version,
                            current_version: sess.cfg_version,
                        })
                    }
                    CodegenErrors::CorruptFile => {
                        dcx.emit_fatal(RlinkCorruptFile { file });
                    }
                };
            }
        };
        compiler.codegen_backend.link(sess, codegen_results, &outputs);
    } else {
        dcx.emit_fatal(RlinkNotAFile {});
    }
}

fn list_metadata(sess: &Session, metadata_loader: &dyn MetadataLoader) {
    match sess.io.input {
        Input::File(ref ifile) => {
            let path = &(*ifile);
            let mut v = Vec::new();
            locator::list_file_metadata(
                &sess.target,
                path,
                metadata_loader,
                &mut v,
                &sess.opts.unstable_opts.ls,
                sess.cfg_version,
            )
            .unwrap();
            safe_println!("{}", String::from_utf8(v).unwrap());
        }
        Input::Str { .. } => {
            #[allow(rustc::diagnostic_outside_of_impl)]
            sess.dcx().fatal("cannot list metadata for stdin");
        }
    }
}

fn print_crate_info(
    codegen_backend: &dyn CodegenBackend,
    sess: &Session,
    parse_attrs: bool,
) -> Compilation {
    use rustc_session::config::PrintKind::*;
    // This import prevents the following code from using the printing macros
    // used by the rest of the module. Within this function, we only write to
    // the output specified by `sess.io.output_file`.
    #[allow(unused_imports)]
    use {do_not_use_safe_print as safe_print, do_not_use_safe_print as safe_println};

    // NativeStaticLibs and LinkArgs are special - printed during linking
    // (empty iterator returns true)
    if sess.opts.prints.iter().all(|p| p.kind == NativeStaticLibs || p.kind == LinkArgs) {
        return Compilation::Continue;
    }

    let attrs = if parse_attrs {
        let result = parse_crate_attrs(sess);
        match result {
            Ok(attrs) => Some(attrs),
            Err(parse_error) => {
                parse_error.emit();
                return Compilation::Stop;
            }
        }
    } else {
        None
    };

    for req in &sess.opts.prints {
        let mut crate_info = String::new();
        macro println_info($($arg:tt)*) {
            crate_info.write_fmt(format_args!("{}\n", format_args!($($arg)*))).unwrap()
        }

        match req.kind {
            TargetList => {
                let mut targets = rustc_target::spec::TARGETS.to_vec();
                targets.sort_unstable();
                println_info!("{}", targets.join("\n"));
            }
            HostTuple => println_info!("{}", rustc_session::config::host_tuple()),
            Sysroot => println_info!("{}", sess.sysroot.display()),
            TargetLibdir => println_info!("{}", sess.target_tlib_path.dir.display()),
            TargetSpecJson => {
                println_info!("{}", serde_json::to_string_pretty(&sess.target.to_json()).unwrap());
            }
            AllTargetSpecsJson => {
                let mut targets = BTreeMap::new();
                for name in rustc_target::spec::TARGETS {
                    let triple = TargetTuple::from_tuple(name);
                    let target = Target::expect_builtin(&triple);
                    targets.insert(name, target.to_json());
                }
                println_info!("{}", serde_json::to_string_pretty(&targets).unwrap());
            }
            FileNames => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                let t_outputs = rustc_interface::util::build_output_filenames(attrs, sess);
                let crate_name = passes::get_crate_name(sess, attrs);
                let crate_types = collect_crate_types(sess, attrs);
                for &style in &crate_types {
                    let fname = rustc_session::output::filename_for_input(
                        sess, style, crate_name, &t_outputs,
                    );
                    println_info!("{}", fname.as_path().file_name().unwrap().to_string_lossy());
                }
            }
            CrateName => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                println_info!("{}", passes::get_crate_name(sess, attrs));
            }
            CrateRootLintLevels => {
                let Some(attrs) = attrs.as_ref() else {
                    // no crate attributes, print out an error and exit
                    return Compilation::Continue;
                };
                let crate_name = passes::get_crate_name(sess, attrs);
                let lint_store = crate::unerased_lint_store(sess);
                let registered_tools = rustc_resolve::registered_tools_ast(sess.dcx(), attrs);
                let features = rustc_expand::config::features(sess, attrs, crate_name);
                let lint_levels = rustc_lint::LintLevelsBuilder::crate_root(
                    sess,
                    &features,
                    true,
                    lint_store,
                    &registered_tools,
                    attrs,
                );
                for lint in lint_store.get_lints() {
                    if let Some(feature_symbol) = lint.feature_gate
                        && !features.enabled(feature_symbol)
                    {
                        // lint is unstable and feature gate isn't active, don't print
                        continue;
                    }
                    let level = lint_levels.lint_level(lint).level;
                    println_info!("{}={}", lint.name_lower(), level.as_str());
                }
            }
            Cfg => {
                let mut cfgs = sess
                    .psess
                    .config
                    .iter()
                    .filter_map(|&(name, value)| {
                        // On stable, exclude unstable flags.
                        if !sess.is_nightly_build()
                            && find_gated_cfg(|cfg_sym| cfg_sym == name).is_some()
                        {
                            return None;
                        }

                        if let Some(value) = value {
                            Some(format!("{name}=\"{value}\""))
                        } else {
                            Some(name.to_string())
                        }
                    })
                    .collect::<Vec<String>>();

                cfgs.sort();
                for cfg in cfgs {
                    println_info!("{cfg}");
                }
            }
            CheckCfg => {
                let mut check_cfgs: Vec<String> = Vec::with_capacity(410);

                // INSTABILITY: We are sorting the output below.
                #[allow(rustc::potential_query_instability)]
                for (name, expected_values) in &sess.psess.check_config.expecteds {
                    use crate::config::ExpectedValues;
                    match expected_values {
                        ExpectedValues::Any => check_cfgs.push(format!("{name}=any()")),
                        ExpectedValues::Some(values) => {
                            if !values.is_empty() {
                                check_cfgs.extend(values.iter().map(|value| {
                                    if let Some(value) = value {
                                        format!("{name}=\"{value}\"")
                                    } else {
                                        name.to_string()
                                    }
                                }))
                            } else {
                                check_cfgs.push(format!("{name}="))
                            }
                        }
                    }
                }

                check_cfgs.sort_unstable();
                if !sess.psess.check_config.exhaustive_names {
                    if !sess.psess.check_config.exhaustive_values {
                        println_info!("any()=any()");
                    } else {
                        println_info!("any()");
                    }
                }
                for check_cfg in check_cfgs {
                    println_info!("{check_cfg}");
                }
            }
            CallingConventions => {
                let calling_conventions = rustc_abi::all_names();
                println_info!("{}", calling_conventions.join("\n"));
            }
            RelocationModels
            | CodeModels
            | TlsModels
            | TargetCPUs
            | StackProtectorStrategies
            | TargetFeatures => {
                codegen_backend.print(req, &mut crate_info, sess);
            }
            // Any output here interferes with Cargo's parsing of other printed output
            NativeStaticLibs => {}
            LinkArgs => {}
            SplitDebuginfo => {
                use rustc_target::spec::SplitDebuginfo::{Off, Packed, Unpacked};

                for split in &[Off, Packed, Unpacked] {
                    if sess.target.options.supported_split_debuginfo.contains(split) {
                        println_info!("{split}");
                    }
                }
            }
            DeploymentTarget => {
                if sess.target.is_like_darwin {
                    println_info!(
                        "{}={}",
                        rustc_target::spec::apple::deployment_target_env_var(&sess.target.os),
                        sess.apple_deployment_target().fmt_pretty(),
                    )
                } else {
                    #[allow(rustc::diagnostic_outside_of_impl)]
                    sess.dcx().fatal("only Apple targets currently support deployment version info")
                }
            }
            SupportedCrateTypes => {
                let supported_crate_types = CRATE_TYPES
                    .iter()
                    .filter(|(_, crate_type)| !invalid_output_for_target(&sess, *crate_type))
                    .map(|(crate_type_sym, _)| *crate_type_sym)
                    .collect::<BTreeSet<_>>();
                for supported_crate_type in supported_crate_types {
                    println_info!("{}", supported_crate_type.as_str());
                }
            }
        }

        req.out.overwrite(&crate_info, sess);
    }
    Compilation::Stop
}

/// Prints version information
///
/// NOTE: this is a macro to support drivers built at a different time than the main `rustc_driver` crate.
pub macro version($early_dcx: expr, $binary: literal, $matches: expr) {
    fn unw(x: Option<&str>) -> &str {
        x.unwrap_or("unknown")
    }
    $crate::version_at_macro_invocation(
        $early_dcx,
        $binary,
        $matches,
        unw(option_env!("CFG_VERSION")),
        unw(option_env!("CFG_VER_HASH")),
        unw(option_env!("CFG_VER_DATE")),
        unw(option_env!("CFG_RELEASE")),
    )
}

#[doc(hidden)] // use the macro instead
pub fn version_at_macro_invocation(
    early_dcx: &EarlyDiagCtxt,
    binary: &str,
    matches: &getopts::Matches,
    version: &str,
    commit_hash: &str,
    commit_date: &str,
    release: &str,
) {
    let verbose = matches.opt_present("verbose");

    let mut version = version;
    let mut release = release;
    let tmp;
    if let Ok(force_version) = std::env::var("RUSTC_OVERRIDE_VERSION_STRING") {
        tmp = force_version;
        version = &tmp;
        release = &tmp;
    }

    safe_println!("{binary} {version}");

    if verbose {
        safe_println!("binary: {binary}");
        safe_println!("commit-hash: {commit_hash}");
        safe_println!("commit-date: {commit_date}");
        safe_println!("host: {}", config::host_tuple());
        safe_println!("release: {release}");

        get_backend_from_raw_matches(early_dcx, matches).print_version();
    }
}

fn usage(verbose: bool, include_unstable_options: bool, nightly_build: bool) {
    let mut options = getopts::Options::new();
    for option in config::rustc_optgroups()
        .iter()
        .filter(|x| verbose || !x.is_verbose_help_only)
        .filter(|x| include_unstable_options || x.is_stable())
    {
        option.apply(&mut options);
    }
    let message = "Usage: rustc [OPTIONS] INPUT";
    let nightly_help = if nightly_build {
        "\n    -Z help             Print unstable compiler options"
    } else {
        ""
    };
    let verbose_help = if verbose {
        ""
    } else {
        "\n    --help -v           Print the full set of options rustc accepts"
    };
    let at_path = if verbose {
        "    @path               Read newline separated options from `path`\n"
    } else {
        ""
    };
    safe_println!(
        "{options}{at_path}\nAdditional help:
    -C help             Print codegen options
    -W help             \
              Print 'lint' options and default settings{nightly}{verbose}\n",
        options = options.usage(message),
        at_path = at_path,
        nightly = nightly_help,
        verbose = verbose_help
    );
}

fn print_wall_help() {
    safe_println!(
        "
The flag `-Wall` does not exist in `rustc`. Most useful lints are enabled by
default. Use `rustc -W help` to see all available lints. It's more common to put
warning settings in the crate root using `#![warn(LINT_NAME)]` instead of using
the command line flag directly.
"
    );
}

/// Write to stdout lint command options, together with a list of all available lints
pub fn describe_lints(sess: &Session, registered_lints: bool) {
    safe_println!(
        "
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> (deny <foo> and all attempts to override)

"
    );

    fn sort_lints(sess: &Session, mut lints: Vec<&'static Lint>) -> Vec<&'static Lint> {
        // The sort doesn't case-fold but it's doubtful we care.
        lints.sort_by_cached_key(|x: &&Lint| (x.default_level(sess.edition()), x.name));
        lints
    }

    fn sort_lint_groups(
        lints: Vec<(&'static str, Vec<LintId>, bool)>,
    ) -> Vec<(&'static str, Vec<LintId>)> {
        let mut lints: Vec<_> = lints.into_iter().map(|(x, y, _)| (x, y)).collect();
        lints.sort_by_key(|l| l.0);
        lints
    }

    let lint_store = unerased_lint_store(sess);
    let (loaded, builtin): (Vec<_>, _) =
        lint_store.get_lints().iter().cloned().partition(|&lint| lint.is_externally_loaded);
    let loaded = sort_lints(sess, loaded);
    let builtin = sort_lints(sess, builtin);

    let (loaded_groups, builtin_groups): (Vec<_>, _) =
        lint_store.get_lint_groups().partition(|&(.., p)| p);
    let loaded_groups = sort_lint_groups(loaded_groups);
    let builtin_groups = sort_lint_groups(builtin_groups);

    let max_name_len =
        loaded.iter().chain(&builtin).map(|&s| s.name.chars().count()).max().unwrap_or(0);
    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    safe_println!("Lint checks provided by rustc:\n");

    let print_lints = |lints: Vec<&Lint>| {
        safe_println!("    {}  {:7.7}  {}", padded("name"), "default", "meaning");
        safe_println!("    {}  {:7.7}  {}", padded("----"), "-------", "-------");
        for lint in lints {
            let name = lint.name_lower().replace('_', "-");
            safe_println!(
                "    {}  {:7.7}  {}",
                padded(&name),
                lint.default_level(sess.edition()).as_str(),
                lint.desc
            );
        }
        safe_println!("\n");
    };

    print_lints(builtin);

    let max_name_len = max(
        "warnings".len(),
        loaded_groups
            .iter()
            .chain(&builtin_groups)
            .map(|&(s, _)| s.chars().count())
            .max()
            .unwrap_or(0),
    );

    let padded = |x: &str| {
        let mut s = " ".repeat(max_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    safe_println!("Lint groups provided by rustc:\n");

    let print_lint_groups = |lints: Vec<(&'static str, Vec<LintId>)>, all_warnings| {
        safe_println!("    {}  sub-lints", padded("name"));
        safe_println!("    {}  ---------", padded("----"));

        if all_warnings {
            safe_println!("    {}  all lints that are set to issue warnings", padded("warnings"));
        }

        for (name, to) in lints {
            let name = name.to_lowercase().replace('_', "-");
            let desc = to
                .into_iter()
                .map(|x| x.to_string().replace('_', "-"))
                .collect::<Vec<String>>()
                .join(", ");
            safe_println!("    {}  {}", padded(&name), desc);
        }
        safe_println!("\n");
    };

    print_lint_groups(builtin_groups, true);

    match (registered_lints, loaded.len(), loaded_groups.len()) {
        (false, 0, _) | (false, _, 0) => {
            safe_println!("Lint tools like Clippy can load additional lints and lint groups.");
        }
        (false, ..) => panic!("didn't load additional lints but got them anyway!"),
        (true, 0, 0) => {
            safe_println!("This crate does not load any additional lints or lint groups.")
        }
        (true, l, g) => {
            if l > 0 {
                safe_println!("Lint checks loaded by this crate:\n");
                print_lints(loaded);
            }
            if g > 0 {
                safe_println!("Lint groups loaded by this crate:\n");
                print_lint_groups(loaded_groups, false);
            }
        }
    }
}

/// Show help for flag categories shared between rustdoc and rustc.
///
/// Returns whether a help option was printed.
pub fn describe_flag_categories(early_dcx: &EarlyDiagCtxt, matches: &Matches) -> bool {
    // Handle the special case of -Wall.
    let wall = matches.opt_strs("W");
    if wall.iter().any(|x| *x == "all") {
        print_wall_help();
        return true;
    }

    // Don't handle -W help here, because we might first load additional lints.
    let debug_flags = matches.opt_strs("Z");
    if debug_flags.iter().any(|x| *x == "help") {
        describe_debug_flags();
        return true;
    }

    let cg_flags = matches.opt_strs("C");
    if cg_flags.iter().any(|x| *x == "help") {
        describe_codegen_flags();
        return true;
    }

    if cg_flags.iter().any(|x| *x == "passes=list") {
        get_backend_from_raw_matches(early_dcx, matches).print_passes();
        return true;
    }

    false
}

/// Get the codegen backend based on the raw [`Matches`].
///
/// `rustc -vV` and `rustc -Cpasses=list` need to get the codegen backend before we have parsed all
/// arguments and created a [`Session`]. This function reads `-Zcodegen-backend`, `--target` and
/// `--sysroot` without validating any other arguments and loads the codegen backend based on these
/// arguments.
fn get_backend_from_raw_matches(
    early_dcx: &EarlyDiagCtxt,
    matches: &Matches,
) -> Box<dyn CodegenBackend> {
    let debug_flags = matches.opt_strs("Z");
    let backend_name = debug_flags.iter().find_map(|x| x.strip_prefix("codegen-backend="));
    let target = parse_target_triple(early_dcx, matches);
    let sysroot = filesearch::materialize_sysroot(matches.opt_str("sysroot").map(PathBuf::from));
    let target = config::build_target_config(early_dcx, &target, &sysroot);

    get_codegen_backend(early_dcx, &sysroot, backend_name, &target)
}

fn describe_debug_flags() {
    safe_println!("\nAvailable options:\n");
    print_flag_list("-Z", config::Z_OPTIONS);
}

fn describe_codegen_flags() {
    safe_println!("\nAvailable codegen options:\n");
    print_flag_list("-C", config::CG_OPTIONS);
}

fn print_flag_list<T>(cmdline_opt: &str, flag_list: &[OptionDesc<T>]) {
    let max_len =
        flag_list.iter().map(|opt_desc| opt_desc.name().chars().count()).max().unwrap_or(0);

    for opt_desc in flag_list {
        safe_println!(
            "    {} {:>width$}=val -- {}",
            cmdline_opt,
            opt_desc.name().replace('_', "-"),
            opt_desc.desc(),
            width = max_len
        );
    }
}

/// Process command line options. Emits messages as appropriate. If compilation
/// should continue, returns a getopts::Matches object parsed from args,
/// otherwise returns `None`.
///
/// The compiler's handling of options is a little complicated as it ties into
/// our stability story. The current intention of each compiler option is to
/// have one of two modes:
///
/// 1. An option is stable and can be used everywhere.
/// 2. An option is unstable, and can only be used on nightly.
///
/// Like unstable library and language features, however, unstable options have
/// always required a form of "opt in" to indicate that you're using them. This
/// provides the easy ability to scan a code base to check to see if anything
/// unstable is being used. Currently, this "opt in" is the `-Z` "zed" flag.
///
/// All options behind `-Z` are considered unstable by default. Other top-level
/// options can also be considered unstable, and they were unlocked through the
/// `-Z unstable-options` flag. Note that `-Z` remains to be the root of
/// instability in both cases, though.
///
/// So with all that in mind, the comments below have some more detail about the
/// contortions done here to get things to work out correctly.
///
/// This does not need to be `pub` for rustc itself, but @chaosite needs it to
/// be public when using rustc as a library, see
/// <https://github.com/rust-lang/rust/commit/2b4c33817a5aaecabf4c6598d41e190080ec119e>
pub fn handle_options(early_dcx: &EarlyDiagCtxt, args: &[String]) -> Option<getopts::Matches> {
    // Parse with *all* options defined in the compiler, we don't worry about
    // option stability here we just want to parse as much as possible.
    let mut options = getopts::Options::new();
    let optgroups = config::rustc_optgroups();
    for option in &optgroups {
        option.apply(&mut options);
    }
    let matches = options.parse(args).unwrap_or_else(|e| {
        let msg: Option<String> = match e {
            getopts::Fail::UnrecognizedOption(ref opt) => CG_OPTIONS
                .iter()
                .map(|opt_desc| ('C', opt_desc.name()))
                .chain(Z_OPTIONS.iter().map(|opt_desc| ('Z', opt_desc.name())))
                .find(|&(_, name)| *opt == name.replace('_', "-"))
                .map(|(flag, _)| format!("{e}. Did you mean `-{flag} {opt}`?")),
            getopts::Fail::ArgumentMissing(ref opt) => {
                optgroups.iter().find(|option| option.name == opt).map(|option| {
                    // Print the help just for the option in question.
                    let mut options = getopts::Options::new();
                    option.apply(&mut options);
                    // getopt requires us to pass a function for joining an iterator of
                    // strings, even though in this case we expect exactly one string.
                    options.usage_with_format(|it| {
                        it.fold(format!("{e}\nUsage:"), |a, b| a + "\n" + &b)
                    })
                })
            }
            _ => None,
        };
        early_dcx.early_fatal(msg.unwrap_or_else(|| e.to_string()));
    });

    // For all options we just parsed, we check a few aspects:
    //
    // * If the option is stable, we're all good
    // * If the option wasn't passed, we're all good
    // * If `-Z unstable-options` wasn't passed (and we're not a -Z option
    //   ourselves), then we require the `-Z unstable-options` flag to unlock
    //   this option that was passed.
    // * If we're a nightly compiler, then unstable options are now unlocked, so
    //   we're good to go.
    // * Otherwise, if we're an unstable option then we generate an error
    //   (unstable option being used on stable)
    nightly_options::check_nightly_options(early_dcx, &matches, &config::rustc_optgroups());

    if args.is_empty() || matches.opt_present("h") || matches.opt_present("help") {
        // Only show unstable options in --help if we accept unstable options.
        let unstable_enabled = nightly_options::is_unstable_enabled(&matches);
        let nightly_build = nightly_options::match_is_nightly_build(&matches);
        usage(matches.opt_present("verbose"), unstable_enabled, nightly_build);
        return None;
    }

    if describe_flag_categories(early_dcx, &matches) {
        return None;
    }

    if matches.opt_present("version") {
        version!(early_dcx, "rustc", &matches);
        return None;
    }

    Some(matches)
}

fn parse_crate_attrs<'a>(sess: &'a Session) -> PResult<'a, ast::AttrVec> {
    let mut parser = unwrap_or_emit_fatal(match &sess.io.input {
        Input::File(file) => new_parser_from_file(&sess.psess, file, None),
        Input::Str { name, input } => {
            new_parser_from_source_str(&sess.psess, name.clone(), input.clone())
        }
    });
    parser.parse_inner_attributes()
}

/// Runs a closure and catches unwinds triggered by fatal errors.
///
/// The compiler currently unwinds with a special sentinel value to abort
/// compilation on fatal errors. This function catches that sentinel and turns
/// the panic into a `Result` instead.
pub fn catch_fatal_errors<F: FnOnce() -> R, R>(f: F) -> Result<R, FatalError> {
    catch_unwind(panic::AssertUnwindSafe(f)).map_err(|value| {
        if value.is::<rustc_errors::FatalErrorMarker>() {
            FatalError
        } else {
            panic::resume_unwind(value);
        }
    })
}

/// Variant of `catch_fatal_errors` for the `interface::Result` return type
/// that also computes the exit code.
pub fn catch_with_exit_code(f: impl FnOnce()) -> i32 {
    match catch_fatal_errors(f) {
        Ok(()) => EXIT_SUCCESS,
        _ => EXIT_FAILURE,
    }
}

static ICE_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

// This function should only be called from the ICE hook.
//
// The intended behavior is that `run_compiler` will invoke `ice_path_with_config` early in the
// initialization process to properly initialize the ICE_PATH static based on parsed CLI flags.
//
// Subsequent calls to either function will then return the proper ICE path as configured by
// the environment and cli flags
fn ice_path() -> &'static Option<PathBuf> {
    ice_path_with_config(None)
}

fn ice_path_with_config(config: Option<&UnstableOptions>) -> &'static Option<PathBuf> {
    if ICE_PATH.get().is_some() && config.is_some() && cfg!(debug_assertions) {
        tracing::warn!(
            "ICE_PATH has already been initialized -- files may be emitted at unintended paths"
        )
    }

    ICE_PATH.get_or_init(|| {
        if !rustc_feature::UnstableFeatures::from_environment(None).is_nightly_build() {
            return None;
        }
        let mut path = match std::env::var_os("RUSTC_ICE") {
            Some(s) => {
                if s == "0" {
                    // Explicitly opting out of writing ICEs to disk.
                    return None;
                }
                if let Some(unstable_opts) = config && unstable_opts.metrics_dir.is_some() {
                    tracing::warn!("ignoring -Zerror-metrics in favor of RUSTC_ICE for destination of ICE report files");
                }
                PathBuf::from(s)
            }
            None => config
                .and_then(|unstable_opts| unstable_opts.metrics_dir.to_owned())
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_default(),
        };
        // Don't use a standard datetime format because Windows doesn't support `:` in paths
        let file_now = jiff::Zoned::now().strftime("%Y-%m-%dT%H_%M_%S");
        let pid = std::process::id();
        path.push(format!("rustc-ice-{file_now}-{pid}.txt"));
        Some(path)
    })
}

pub static USING_INTERNAL_FEATURES: AtomicBool = AtomicBool::new(false);

/// Installs a panic hook that will print the ICE message on unexpected panics.
///
/// The hook is intended to be useable even by external tools. You can pass a custom
/// `bug_report_url`, or report arbitrary info in `extra_info`. Note that `extra_info` is called in
/// a context where *the thread is currently panicking*, so it must not panic or the process will
/// abort.
///
/// If you have no extra info to report, pass the empty closure `|_| ()` as the argument to
/// extra_info.
///
/// A custom rustc driver can skip calling this to set up a custom ICE hook.
pub fn install_ice_hook(bug_report_url: &'static str, extra_info: fn(&DiagCtxt)) {
    // If the user has not explicitly overridden "RUST_BACKTRACE", then produce
    // full backtraces. When a compiler ICE happens, we want to gather
    // as much information as possible to present in the issue opened
    // by the user. Compiler developers and other rustc users can
    // opt in to less-verbose backtraces by manually setting "RUST_BACKTRACE"
    // (e.g. `RUST_BACKTRACE=1`)
    if env::var_os("RUST_BACKTRACE").is_none() {
        // HACK: this check is extremely dumb, but we don't really need it to be smarter since this should only happen in the test suite anyway.
        let ui_testing = std::env::args().any(|arg| arg == "-Zui-testing");
        if env!("CFG_RELEASE_CHANNEL") == "dev" && !ui_testing {
            panic::set_backtrace_style(panic::BacktraceStyle::Short);
        } else {
            panic::set_backtrace_style(panic::BacktraceStyle::Full);
        }
    }

    panic::update_hook(Box::new(
        move |default_hook: &(dyn Fn(&PanicHookInfo<'_>) + Send + Sync + 'static),
              info: &PanicHookInfo<'_>| {
            // Lock stderr to prevent interleaving of concurrent panics.
            let _guard = io::stderr().lock();
            // If the error was caused by a broken pipe then this is not a bug.
            // Write the error and return immediately. See #98700.
            #[cfg(windows)]
            if let Some(msg) = info.payload().downcast_ref::<String>() {
                if msg.starts_with("failed printing to stdout: ") && msg.ends_with("(os error 232)")
                {
                    // the error code is already going to be reported when the panic unwinds up the stack
                    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());
                    let _ = early_dcx.early_err(msg.clone());
                    return;
                }
            };

            // Invoke the default handler, which prints the actual panic message and optionally a backtrace
            // Don't do this for delayed bugs, which already emit their own more useful backtrace.
            if !info.payload().is::<rustc_errors::DelayedBugPanic>() {
                default_hook(info);
                // Separate the output with an empty line
                eprintln!();

                if let Some(ice_path) = ice_path()
                    && let Ok(mut out) = File::options().create(true).append(true).open(&ice_path)
                {
                    // The current implementation always returns `Some`.
                    let location = info.location().unwrap();
                    let msg = match info.payload().downcast_ref::<&'static str>() {
                        Some(s) => *s,
                        None => match info.payload().downcast_ref::<String>() {
                            Some(s) => &s[..],
                            None => "Box<dyn Any>",
                        },
                    };
                    let thread = std::thread::current();
                    let name = thread.name().unwrap_or("<unnamed>");
                    let _ = write!(
                        &mut out,
                        "thread '{name}' panicked at {location}:\n\
                        {msg}\n\
                        stack backtrace:\n\
                        {:#}",
                        std::backtrace::Backtrace::force_capture()
                    );
                }
            }

            // Print the ICE message
            report_ice(info, bug_report_url, extra_info, &USING_INTERNAL_FEATURES);
        },
    ));
}

/// Prints the ICE message, including query stack, but without backtrace.
///
/// The message will point the user at `bug_report_url` to report the ICE.
///
/// When `install_ice_hook` is called, this function will be called as the panic
/// hook.
fn report_ice(
    info: &panic::PanicHookInfo<'_>,
    bug_report_url: &str,
    extra_info: fn(&DiagCtxt),
    using_internal_features: &AtomicBool,
) {
    let fallback_bundle =
        rustc_errors::fallback_fluent_bundle(crate::DEFAULT_LOCALE_RESOURCES.to_vec(), false);
    let emitter = Box::new(rustc_errors::emitter::HumanEmitter::new(
        stderr_destination(rustc_errors::ColorConfig::Auto),
        fallback_bundle,
    ));
    let dcx = rustc_errors::DiagCtxt::new(emitter);
    let dcx = dcx.handle();

    // a .span_bug or .bug call has already printed what
    // it wants to print.
    if !info.payload().is::<rustc_errors::ExplicitBug>()
        && !info.payload().is::<rustc_errors::DelayedBugPanic>()
    {
        dcx.emit_err(session_diagnostics::Ice);
    }

    if using_internal_features.load(std::sync::atomic::Ordering::Relaxed) {
        dcx.emit_note(session_diagnostics::IceBugReportInternalFeature);
    } else {
        dcx.emit_note(session_diagnostics::IceBugReport { bug_report_url });

        // Only emit update nightly hint for users on nightly builds.
        if rustc_feature::UnstableFeatures::from_environment(None).is_nightly_build() {
            dcx.emit_note(session_diagnostics::UpdateNightlyNote);
        }
    }

    let version = util::version_str!().unwrap_or("unknown_version");
    let tuple = config::host_tuple();

    static FIRST_PANIC: AtomicBool = AtomicBool::new(true);

    let file = if let Some(path) = ice_path() {
        // Create the ICE dump target file.
        match crate::fs::File::options().create(true).append(true).open(&path) {
            Ok(mut file) => {
                dcx.emit_note(session_diagnostics::IcePath { path: path.clone() });
                if FIRST_PANIC.swap(false, Ordering::SeqCst) {
                    let _ = write!(file, "\n\nrustc version: {version}\nplatform: {tuple}");
                }
                Some(file)
            }
            Err(err) => {
                // The path ICE couldn't be written to disk, provide feedback to the user as to why.
                dcx.emit_warn(session_diagnostics::IcePathError {
                    path: path.clone(),
                    error: err.to_string(),
                    env_var: std::env::var_os("RUSTC_ICE")
                        .map(PathBuf::from)
                        .map(|env_var| session_diagnostics::IcePathErrorEnv { env_var }),
                });
                dcx.emit_note(session_diagnostics::IceVersion { version, triple: tuple });
                None
            }
        }
    } else {
        dcx.emit_note(session_diagnostics::IceVersion { version, triple: tuple });
        None
    };

    if let Some((flags, excluded_cargo_defaults)) = rustc_session::utils::extra_compiler_flags() {
        dcx.emit_note(session_diagnostics::IceFlags { flags: flags.join(" ") });
        if excluded_cargo_defaults {
            dcx.emit_note(session_diagnostics::IceExcludeCargoDefaults);
        }
    }

    // If backtraces are enabled, also print the query stack
    let backtrace = env::var_os("RUST_BACKTRACE").is_some_and(|x| &x != "0");

    let limit_frames = if backtrace { None } else { Some(2) };

    interface::try_print_query_stack(dcx, limit_frames, file);

    // We don't trust this callback not to panic itself, so run it at the end after we're sure we've
    // printed all the relevant info.
    extra_info(&dcx);

    #[cfg(windows)]
    if env::var("RUSTC_BREAK_ON_ICE").is_ok() {
        // Trigger a debugger if we crashed during bootstrap
        unsafe { windows::Win32::System::Diagnostics::Debug::DebugBreak() };
    }
}

/// This allows tools to enable rust logging without having to magically match rustc's
/// tracing crate version.
pub fn init_rustc_env_logger(early_dcx: &EarlyDiagCtxt) {
    init_logger(early_dcx, rustc_log::LoggerConfig::from_env("RUSTC_LOG"));
}

/// This allows tools to enable rust logging without having to magically match rustc's
/// tracing crate version. In contrast to `init_rustc_env_logger` it allows you to choose
/// the values directly rather than having to set an environment variable.
pub fn init_logger(early_dcx: &EarlyDiagCtxt, cfg: rustc_log::LoggerConfig) {
    if let Err(error) = rustc_log::init_logger(cfg) {
        early_dcx.early_fatal(error.to_string());
    }
}

/// Install our usual `ctrlc` handler, which sets [`rustc_const_eval::CTRL_C_RECEIVED`].
/// Making this handler optional lets tools can install a different handler, if they wish.
pub fn install_ctrlc_handler() {
    #[cfg(all(not(miri), not(target_family = "wasm")))]
    ctrlc::set_handler(move || {
        // Indicate that we have been signaled to stop, then give the rest of the compiler a bit of
        // time to check CTRL_C_RECEIVED and run its own shutdown logic, but after a short amount
        // of time exit the process. This sleep+exit ensures that even if nobody is checking
        // CTRL_C_RECEIVED, the compiler exits reasonably promptly.
        rustc_const_eval::CTRL_C_RECEIVED.store(true, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(100));
        std::process::exit(1);
    })
    .expect("Unable to install ctrlc handler");
}

pub fn main() -> ! {
    let start_time = Instant::now();
    let start_rss = get_resident_set_size();

    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    init_rustc_env_logger(&early_dcx);
    signal_handler::install();
    let mut callbacks = TimePassesCallbacks::default();
    install_ice_hook(DEFAULT_BUG_REPORT_URL, |_| ());
    install_ctrlc_handler();

    let exit_code =
        catch_with_exit_code(|| run_compiler(&args::raw_args(&early_dcx), &mut callbacks));

    if let Some(format) = callbacks.time_passes {
        let end_rss = get_resident_set_size();
        print_time_passes_entry("total", start_time.elapsed(), start_rss, end_rss, format);
    }

    process::exit(exit_code)
}
