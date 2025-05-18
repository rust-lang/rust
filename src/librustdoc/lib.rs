#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/",
    html_playground_url = "https://play.rust-lang.org/"
)]
#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(debug_closure_helpers)]
#![feature(file_buffered)]
#![feature(format_args_nl)]
#![feature(if_let_guard)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_intersperse)]
#![feature(never_type)]
#![feature(round_char_boundary)]
#![feature(test)]
#![feature(type_alias_impl_trait)]
#![feature(type_ascription)]
#![recursion_limit = "256"]
#![warn(rustc::internal)]
#![allow(clippy::collapsible_if, clippy::collapsible_else_if)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

extern crate thin_vec;

// N.B. these need `extern crate` even in 2018 edition
// because they're loaded implicitly from the sysroot.
// The reason they're loaded from the sysroot is because
// the rustdoc artifacts aren't stored in rustc's cargo target directory.
// So if `rustc` was specified in Cargo.toml, this would spuriously rebuild crates.
//
// Dependencies listed in Cargo.toml do not need `extern crate`.

extern crate pulldown_cmark;
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_attr_parsing;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_expand;
extern crate rustc_feature;
extern crate rustc_hir;
extern crate rustc_hir_analysis;
extern crate rustc_hir_pretty;
extern crate rustc_index;
extern crate rustc_infer;
extern crate rustc_interface;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_lint_defs;
extern crate rustc_log;
extern crate rustc_macros;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_parse;
extern crate rustc_passes;
extern crate rustc_resolve;
extern crate rustc_serialize;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_trait_selection;
extern crate test;

// See docs in https://github.com/rust-lang/rust/blob/master/compiler/rustc/src/main.rs
// about jemalloc.
#[cfg(feature = "jemalloc")]
extern crate tikv_jemalloc_sys as jemalloc_sys;

use std::env::{self, VarError};
use std::io::{self, IsTerminal};
use std::path::Path;
use std::process;

use rustc_errors::DiagCtxtHandle;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{ErrorOutputType, RustcOptGroup, make_crate_type_option};
use rustc_session::{EarlyDiagCtxt, getopts};
use tracing::info;

use crate::clean::utils::DOC_RUST_LANG_ORG_VERSION;

/// A macro to create a FxHashMap.
///
/// Example:
///
/// ```ignore(cannot-test-this-because-non-exported-macro)
/// let letters = map!{"a" => "b", "c" => "d"};
/// ```
///
/// Trailing commas are allowed.
/// Commas between elements are required (even if the expression is a block).
macro_rules! map {
    ($( $key: expr => $val: expr ),* $(,)*) => {{
        let mut map = ::rustc_data_structures::fx::FxIndexMap::default();
        $( map.insert($key, $val); )*
        map
    }}
}

mod clean;
mod config;
mod core;
mod display;
mod docfs;
mod doctest;
mod error;
mod externalfiles;
mod fold;
mod formats;
// used by the error-index generator, so it needs to be public
pub mod html;
mod json;
pub(crate) mod lint;
mod markdown;
mod passes;
mod scrape_examples;
mod theme;
mod visit;
mod visit_ast;
mod visit_lib;

pub fn main() {
    // See docs in https://github.com/rust-lang/rust/blob/master/compiler/rustc/src/main.rs
    // about jemalloc.
    #[cfg(feature = "jemalloc")]
    {
        use std::os::raw::{c_int, c_void};

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

        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn _rjem_je_zone_register();
            }

            #[used]
            static _F7: unsafe extern "C" fn() = _rjem_je_zone_register;
        }
    }

    let mut early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    rustc_driver::install_ice_hook(
        "https://github.com/rust-lang/rust/issues/new\
    ?labels=C-bug%2C+I-ICE%2C+T-rustdoc&template=ice.md",
        |_| (),
    );

    // When using CI artifacts with `download-rustc`, tracing is unconditionally built
    // with `--features=static_max_level_info`, which disables almost all rustdoc logging. To avoid
    // this, compile our own version of `tracing` that logs all levels.
    // NOTE: this compiles both versions of tracing unconditionally, because
    // - The compile time hit is not that bad, especially compared to rustdoc's incremental times, and
    // - Otherwise, there's no warning that logging is being ignored when `download-rustc` is enabled

    crate::init_logging(&early_dcx);
    match rustc_log::init_logger(rustc_log::LoggerConfig::from_env("RUSTDOC_LOG")) {
        Ok(()) => {}
        // With `download-rustc = true` there are definitely 2 distinct tracing crates in the
        // dependency graph: one in the downloaded sysroot and one built just now as a dependency of
        // rustdoc. So the sysroot's tracing is definitely not yet initialized here.
        //
        // But otherwise, depending on link style, there may or may not be 2 tracing crates in play.
        // The one we just initialized in `crate::init_logging` above is rustdoc's direct dependency
        // on tracing. When rustdoc is built by x.py using Cargo, rustc_driver's and rustc_log's
        // tracing dependency is distinct from this one and also needs to be initialized (using the
        // same RUSTDOC_LOG environment variable for both). Other build systems may use just a
        // single tracing crate throughout the rustc and rustdoc build.
        //
        // The reason initializing 2 tracings does not show double logging when `download-rustc =
        // false` and `debug_logging = true` is because all rustc logging goes only to its version
        // of tracing (the one in the sysroot) and all of rustdoc's logging only goes to its version
        // (the one in Cargo.toml).
        Err(rustc_log::Error::AlreadyInit(_)) => {}
        Err(error) => early_dcx.early_fatal(error.to_string()),
    }

    let exit_code = rustc_driver::catch_with_exit_code(|| {
        let at_args = rustc_driver::args::raw_args(&early_dcx);
        main_args(&mut early_dcx, &at_args);
    });
    process::exit(exit_code);
}

fn init_logging(early_dcx: &EarlyDiagCtxt) {
    let color_logs = match env::var("RUSTDOC_LOG_COLOR").as_deref() {
        Ok("always") => true,
        Ok("never") => false,
        Ok("auto") | Err(VarError::NotPresent) => io::stdout().is_terminal(),
        Ok(value) => early_dcx.early_fatal(format!(
            "invalid log color value '{value}': expected one of always, never, or auto",
        )),
        Err(VarError::NotUnicode(value)) => early_dcx.early_fatal(format!(
            "invalid log color value '{}': expected one of always, never, or auto",
            value.to_string_lossy()
        )),
    };
    let filter = tracing_subscriber::EnvFilter::from_env("RUSTDOC_LOG");
    let layer = tracing_tree::HierarchicalLayer::default()
        .with_writer(io::stderr)
        .with_ansi(color_logs)
        .with_targets(true)
        .with_wraparound(10)
        .with_verbose_exit(true)
        .with_verbose_entry(true)
        .with_indent_amount(2);
    #[cfg(debug_assertions)]
    let layer = layer.with_thread_ids(true).with_thread_names(true);

    use tracing_subscriber::layer::SubscriberExt;
    let subscriber = tracing_subscriber::Registry::default().with(filter).with(layer);
    tracing::subscriber::set_global_default(subscriber).unwrap();
}

fn opts() -> Vec<RustcOptGroup> {
    use rustc_session::config::OptionKind::{Flag, FlagMulti, Multi, Opt};
    use rustc_session::config::OptionStability::{Stable, Unstable};
    use rustc_session::config::make_opt as opt;

    vec![
        opt(Stable, FlagMulti, "h", "help", "show this help message", ""),
        opt(Stable, FlagMulti, "V", "version", "print rustdoc's version", ""),
        opt(Stable, FlagMulti, "v", "verbose", "use verbose output", ""),
        opt(Stable, Opt, "w", "output-format", "the output type to write", "[html]"),
        opt(
            Stable,
            Opt,
            "",
            "output",
            "Which directory to place the output. This option is deprecated, use --out-dir instead.",
            "PATH",
        ),
        opt(Stable, Opt, "o", "out-dir", "which directory to place the output", "PATH"),
        opt(Stable, Opt, "", "crate-name", "specify the name of this crate", "NAME"),
        make_crate_type_option(),
        opt(Stable, Multi, "L", "library-path", "directory to add to crate search path", "DIR"),
        opt(Stable, Multi, "", "cfg", "pass a --cfg to rustc", ""),
        opt(Stable, Multi, "", "check-cfg", "pass a --check-cfg to rustc", ""),
        opt(Stable, Multi, "", "extern", "pass an --extern to rustc", "NAME[=PATH]"),
        opt(
            Unstable,
            Multi,
            "",
            "extern-html-root-url",
            "base URL to use for dependencies; for example, \
                \"std=/doc\" links std::vec::Vec to /doc/std/vec/struct.Vec.html",
            "NAME=URL",
        ),
        opt(
            Unstable,
            FlagMulti,
            "",
            "extern-html-root-takes-precedence",
            "give precedence to `--extern-html-root-url`, not `html_root_url`",
            "",
        ),
        opt(Stable, Multi, "C", "codegen", "pass a codegen option to rustc", "OPT[=VALUE]"),
        opt(Stable, FlagMulti, "", "document-private-items", "document private items", ""),
        opt(
            Unstable,
            FlagMulti,
            "",
            "document-hidden-items",
            "document items that have doc(hidden)",
            "",
        ),
        opt(Stable, FlagMulti, "", "test", "run code examples as tests", ""),
        opt(Stable, Multi, "", "test-args", "arguments to pass to the test runner", "ARGS"),
        opt(
            Stable,
            Opt,
            "",
            "test-run-directory",
            "The working directory in which to run tests",
            "PATH",
        ),
        opt(Stable, Opt, "", "target", "target triple to document", "TRIPLE"),
        opt(
            Stable,
            Multi,
            "",
            "markdown-css",
            "CSS files to include via <link> in a rendered Markdown file",
            "FILES",
        ),
        opt(
            Stable,
            Multi,
            "",
            "html-in-header",
            "files to include inline in the <head> section of a rendered Markdown file \
                or generated documentation",
            "FILES",
        ),
        opt(
            Stable,
            Multi,
            "",
            "html-before-content",
            "files to include inline between <body> and the content of a rendered \
                Markdown file or generated documentation",
            "FILES",
        ),
        opt(
            Stable,
            Multi,
            "",
            "html-after-content",
            "files to include inline between the content and </body> of a rendered \
                Markdown file or generated documentation",
            "FILES",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "markdown-before-content",
            "files to include inline between <body> and the content of a rendered \
                Markdown file or generated documentation",
            "FILES",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "markdown-after-content",
            "files to include inline between the content and </body> of a rendered \
                Markdown file or generated documentation",
            "FILES",
        ),
        opt(Stable, Opt, "", "markdown-playground-url", "URL to send code snippets to", "URL"),
        opt(Stable, FlagMulti, "", "markdown-no-toc", "don't include table of contents", ""),
        opt(
            Stable,
            Opt,
            "e",
            "extend-css",
            "To add some CSS rules with a given file to generate doc with your own theme. \
                However, your theme might break if the rustdoc's generated HTML changes, so be careful!",
            "PATH",
        ),
        opt(
            Unstable,
            Multi,
            "Z",
            "",
            "unstable / perma-unstable options (only on nightly build)",
            "FLAG",
        ),
        opt(Stable, Opt, "", "sysroot", "Override the system root", "PATH"),
        opt(
            Unstable,
            Opt,
            "",
            "playground-url",
            "URL to send code snippets to, may be reset by --markdown-playground-url \
                or `#![doc(html_playground_url=...)]`",
            "URL",
        ),
        opt(
            Unstable,
            FlagMulti,
            "",
            "display-doctest-warnings",
            "show warnings that originate in doctests",
            "",
        ),
        opt(
            Stable,
            Opt,
            "",
            "crate-version",
            "crate version to print into documentation",
            "VERSION",
        ),
        opt(
            Unstable,
            FlagMulti,
            "",
            "sort-modules-by-appearance",
            "sort modules by where they appear in the program, rather than alphabetically",
            "",
        ),
        opt(
            Stable,
            Opt,
            "",
            "default-theme",
            "Set the default theme. THEME should be the theme name, generally lowercase. \
                If an unknown default theme is specified, the builtin default is used. \
                The set of themes, and the rustdoc built-in default, are not stable.",
            "THEME",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "default-setting",
            "Default value for a rustdoc setting (used when \"rustdoc-SETTING\" is absent \
                from web browser Local Storage). If VALUE is not supplied, \"true\" is used. \
                Supported SETTINGs and VALUEs are not documented and not stable.",
            "SETTING[=VALUE]",
        ),
        opt(
            Stable,
            Multi,
            "",
            "theme",
            "additional themes which will be added to the generated docs",
            "FILES",
        ),
        opt(Stable, Multi, "", "check-theme", "check if given theme is valid", "FILES"),
        opt(
            Unstable,
            Opt,
            "",
            "resource-suffix",
            "suffix to add to CSS and JavaScript files, \
                e.g., \"search-index.js\" will become \"search-index-suffix.js\"",
            "PATH",
        ),
        opt(
            Stable,
            Opt,
            "",
            "edition",
            "edition to use when compiling rust code (default: 2015)",
            "EDITION",
        ),
        opt(
            Stable,
            Opt,
            "",
            "color",
            "Configure coloring of output:
                                          auto   = colorize, if output goes to a tty (default);
                                          always = always colorize output;
                                          never  = never colorize output",
            "auto|always|never",
        ),
        opt(
            Stable,
            Opt,
            "",
            "error-format",
            "How errors and other messages are produced",
            "human|json|short",
        ),
        opt(
            Stable,
            Opt,
            "",
            "diagnostic-width",
            "Provide width of the output for truncated error messages",
            "WIDTH",
        ),
        opt(Stable, Opt, "", "json", "Configure the structure of JSON diagnostics", "CONFIG"),
        opt(Stable, Multi, "A", "allow", "Set lint allowed", "LINT"),
        opt(Stable, Multi, "W", "warn", "Set lint warnings", "LINT"),
        opt(Stable, Multi, "", "force-warn", "Set lint force-warn", "LINT"),
        opt(Stable, Multi, "D", "deny", "Set lint denied", "LINT"),
        opt(Stable, Multi, "F", "forbid", "Set lint forbidden", "LINT"),
        opt(
            Stable,
            Multi,
            "",
            "cap-lints",
            "Set the most restrictive lint level. \
                More restrictive lints are capped at this level. \
                By default, it is at `forbid` level.",
            "LEVEL",
        ),
        opt(Unstable, Opt, "", "index-page", "Markdown file to be used as index page", "PATH"),
        opt(
            Unstable,
            FlagMulti,
            "",
            "enable-index-page",
            "To enable generation of the index page",
            "",
        ),
        opt(
            Unstable,
            Opt,
            "",
            "static-root-path",
            "Path string to force loading static files from in output pages. \
                If not set, uses combinations of '../' to reach the documentation root.",
            "PATH",
        ),
        opt(
            Unstable,
            Opt,
            "",
            "persist-doctests",
            "Directory to persist doctest executables into",
            "PATH",
        ),
        opt(
            Unstable,
            FlagMulti,
            "",
            "show-coverage",
            "calculate percentage of public items with documentation",
            "",
        ),
        opt(
            Stable,
            Opt,
            "",
            "test-runtool",
            "",
            "The tool to run tests with when building for a different target than host",
        ),
        opt(
            Stable,
            Multi,
            "",
            "test-runtool-arg",
            "",
            "One argument (of possibly many) to pass to the runtool",
        ),
        opt(
            Unstable,
            Opt,
            "",
            "test-builder",
            "The rustc-like binary to use as the test builder",
            "PATH",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "test-builder-wrapper",
            "Wrapper program to pass test-builder and arguments",
            "PATH",
        ),
        opt(Unstable, FlagMulti, "", "check", "Run rustdoc checks", ""),
        opt(
            Unstable,
            FlagMulti,
            "",
            "generate-redirect-map",
            "Generate JSON file at the top level instead of generating HTML redirection files",
            "",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "emit",
            "Comma separated list of types of output for rustdoc to emit",
            "[unversioned-shared-resources,toolchain-shared-resources,invocation-specific,dep-info]",
        ),
        opt(Unstable, FlagMulti, "", "no-run", "Compile doctests without running them", ""),
        opt(
            Unstable,
            Multi,
            "",
            "remap-path-prefix",
            "Remap source names in compiler messages",
            "FROM=TO",
        ),
        opt(
            Unstable,
            FlagMulti,
            "",
            "show-type-layout",
            "Include the memory layout of types in the docs",
            "",
        ),
        opt(Unstable, Flag, "", "nocapture", "Don't capture stdout and stderr of tests", ""),
        opt(
            Unstable,
            Flag,
            "",
            "generate-link-to-definition",
            "Make the identifiers in the HTML source code pages navigable",
            "",
        ),
        opt(
            Unstable,
            Opt,
            "",
            "scrape-examples-output-path",
            "",
            "collect function call information and output at the given path",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "scrape-examples-target-crate",
            "",
            "collect function call information for functions from the target crate",
        ),
        opt(Unstable, Flag, "", "scrape-tests", "Include test code when scraping examples", ""),
        opt(
            Unstable,
            Multi,
            "",
            "with-examples",
            "",
            "path to function call information (for displaying examples in the documentation)",
        ),
        opt(
            Unstable,
            Opt,
            "",
            "merge",
            "Controls how rustdoc handles files from previously documented crates in the doc root\n\
                none = Do not write cross-crate information to the --out-dir\n\
                shared = Append current crate's info to files found in the --out-dir\n\
                finalize = Write current crate's info and --include-parts-dir info to the --out-dir, overwriting conflicting files",
            "none|shared|finalize",
        ),
        opt(
            Unstable,
            Opt,
            "",
            "parts-out-dir",
            "Writes trait implementations and other info for the current crate to provided path. Only use with --merge=none",
            "path/to/doc.parts/<crate-name>",
        ),
        opt(
            Unstable,
            Multi,
            "",
            "include-parts-dir",
            "Includes trait implementations and other crate info from provided path. Only use with --merge=finalize",
            "path/to/doc.parts/<crate-name>",
        ),
        opt(Unstable, Flag, "", "html-no-source", "Disable HTML source code pages generation", ""),
        opt(
            Unstable,
            Multi,
            "",
            "doctest-build-arg",
            "One argument (of possibly many) to be used when compiling doctests",
            "ARG",
        ),
        opt(
            Unstable,
            FlagMulti,
            "",
            "disable-minification",
            "disable the minification of CSS/JS files (perma-unstable, do not use with cached files)",
            "",
        ),
        // deprecated / removed options
        opt(
            Stable,
            Multi,
            "",
            "plugin-path",
            "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information",
            "DIR",
        ),
        opt(
            Stable,
            Multi,
            "",
            "passes",
            "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information",
            "PASSES",
        ),
        opt(
            Stable,
            Multi,
            "",
            "plugins",
            "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information",
            "PLUGINS",
        ),
        opt(
            Stable,
            FlagMulti,
            "",
            "no-defaults",
            "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information",
            "",
        ),
        opt(
            Stable,
            Opt,
            "r",
            "input-format",
            "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information",
            "[rust]",
        ),
    ]
}

fn usage(argv0: &str) {
    let mut options = getopts::Options::new();
    for option in opts() {
        option.apply(&mut options);
    }
    println!("{}", options.usage(&format!("{argv0} [options] <input>")));
    println!("    @path               Read newline separated options from `path`\n");
    println!(
        "More information available at {DOC_RUST_LANG_ORG_VERSION}/rustdoc/what-is-rustdoc.html",
    );
}

pub(crate) fn wrap_return(dcx: DiagCtxtHandle<'_>, res: Result<(), String>) {
    match res {
        Ok(()) => dcx.abort_if_errors(),
        Err(err) => dcx.fatal(err),
    }
}

fn run_renderer<'tcx, T: formats::FormatRenderer<'tcx>>(
    krate: clean::Crate,
    renderopts: config::RenderOptions,
    cache: formats::cache::Cache,
    tcx: TyCtxt<'tcx>,
) {
    match formats::run_format::<T>(krate, renderopts, cache, tcx) {
        Ok(_) => tcx.dcx().abort_if_errors(),
        Err(e) => {
            let mut msg =
                tcx.dcx().struct_fatal(format!("couldn't generate documentation: {}", e.error));
            let file = e.file.display().to_string();
            if !file.is_empty() {
                msg.note(format!("failed to create or modify \"{file}\""));
            }
            msg.emit();
        }
    }
}

/// Renders and writes cross-crate info files, like the search index. This function exists so that
/// we can run rustdoc without a crate root in the `--merge=finalize` mode. Cross-crate info files
/// discovered via `--include-parts-dir` are combined and written to the doc root.
fn run_merge_finalize(opt: config::RenderOptions) -> Result<(), error::Error> {
    assert!(
        opt.should_merge.write_rendered_cci,
        "config.rs only allows us to return InputMode::NoInputMergeFinalize if --merge=finalize"
    );
    assert!(
        !opt.should_merge.read_rendered_cci,
        "config.rs only allows us to return InputMode::NoInputMergeFinalize if --merge=finalize"
    );
    let crates = html::render::CrateInfo::read_many(&opt.include_parts_dir)?;
    let include_sources = !opt.html_no_source;
    html::render::write_not_crate_specific(
        &crates,
        &opt.output,
        &opt,
        &opt.themes,
        opt.extension_css.as_deref(),
        &opt.resource_suffix,
        include_sources,
    )?;
    Ok(())
}

fn main_args(early_dcx: &mut EarlyDiagCtxt, at_args: &[String]) {
    // Throw away the first argument, the name of the binary.
    // In case of at_args being empty, as might be the case by
    // passing empty argument array to execve under some platforms,
    // just use an empty slice.
    //
    // This situation was possible before due to arg_expand_all being
    // called before removing the argument, enabling a crash by calling
    // the compiler with @empty_file as argv[0] and no more arguments.
    let at_args = at_args.get(1..).unwrap_or_default();

    let args = rustc_driver::args::arg_expand_all(early_dcx, at_args);

    let mut options = getopts::Options::new();
    for option in opts() {
        option.apply(&mut options);
    }
    let matches = match options.parse(&args) {
        Ok(m) => m,
        Err(err) => {
            early_dcx.early_fatal(err.to_string());
        }
    };

    // Note that we discard any distinction between different non-zero exit
    // codes from `from_matches` here.
    let (input, options, render_options) =
        match config::Options::from_matches(early_dcx, &matches, args) {
            Some(opts) => opts,
            None => return,
        };

    let dcx =
        core::new_dcx(options.error_format, None, options.diagnostic_width, &options.unstable_opts);
    let dcx = dcx.handle();

    let input = match input {
        config::InputMode::HasFile(input) => input,
        config::InputMode::NoInputMergeFinalize => {
            return wrap_return(
                dcx,
                run_merge_finalize(render_options)
                    .map_err(|e| format!("could not write merged cross-crate info: {e}")),
            );
        }
    };

    let output_format = options.output_format;

    match (
        options.should_test || output_format == config::OutputFormat::Doctest,
        config::markdown_input(&input),
    ) {
        (true, Some(_)) => return wrap_return(dcx, doctest::test_markdown(&input, options)),
        (true, None) => return doctest::run(dcx, input, options),
        (false, Some(md_input)) => {
            let md_input = md_input.to_owned();
            let edition = options.edition;
            let config = core::create_config(input, options, &render_options);

            // `markdown::render` can invoke `doctest::make_test`, which
            // requires session globals and a thread pool, so we use
            // `run_compiler`.
            return wrap_return(
                dcx,
                interface::run_compiler(config, |_compiler| {
                    markdown::render_and_write(&md_input, render_options, edition)
                }),
            );
        }
        (false, None) => {}
    }

    // need to move these items separately because we lose them by the time the closure is called,
    // but we can't create the dcx ahead of time because it's not Send
    let show_coverage = options.show_coverage;
    let run_check = options.run_check;

    // First, parse the crate and extract all relevant information.
    info!("starting to run rustc");

    // Interpret the input file as a rust source file, passing it through the
    // compiler all the way through the analysis passes. The rustdoc output is
    // then generated from the cleaned AST of the crate. This runs all the
    // plug/cleaning passes.
    let crate_version = options.crate_version.clone();

    let scrape_examples_options = options.scrape_examples_options.clone();
    let bin_crate = options.bin_crate;

    let config = core::create_config(input, options, &render_options);

    let registered_lints = config.register_lints.is_some();

    interface::run_compiler(config, |compiler| {
        let sess = &compiler.sess;

        if sess.opts.describe_lints {
            rustc_driver::describe_lints(sess, registered_lints);
            return;
        }

        let krate = rustc_interface::passes::parse(sess);
        rustc_interface::create_and_enter_global_ctxt(compiler, krate, |tcx| {
            if sess.dcx().has_errors().is_some() {
                sess.dcx().fatal("Compilation failed, aborting rustdoc");
            }

            let (krate, render_opts, mut cache) = sess.time("run_global_ctxt", || {
                core::run_global_ctxt(tcx, show_coverage, render_options, output_format)
            });
            info!("finished with rustc");

            if let Some(options) = scrape_examples_options {
                return scrape_examples::run(krate, render_opts, cache, tcx, options, bin_crate);
            }

            cache.crate_version = crate_version;

            if show_coverage {
                // if we ran coverage, bail early, we don't need to also generate docs at this point
                // (also we didn't load in any of the useful passes)
                return;
            }

            if render_opts.dep_info().is_some() {
                rustc_interface::passes::write_dep_info(tcx);
            }

            if let Some(metrics_dir) = &sess.opts.unstable_opts.metrics_dir {
                dump_feature_usage_metrics(tcx, metrics_dir);
            }

            if run_check {
                // Since we're in "check" mode, no need to generate anything beyond this point.
                return;
            }

            info!("going to format");
            match output_format {
                config::OutputFormat::Html => sess.time("render_html", || {
                    run_renderer::<html::render::Context<'_>>(krate, render_opts, cache, tcx)
                }),
                config::OutputFormat::Json => sess.time("render_json", || {
                    run_renderer::<json::JsonRenderer<'_>>(krate, render_opts, cache, tcx)
                }),
                // Already handled above with doctest runners.
                config::OutputFormat::Doctest => unreachable!(),
            }
        })
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
        tcxt.dcx().err(format!("cannot emit feature usage metrics: {error}"));
    }
}
