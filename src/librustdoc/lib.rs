#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/",
    html_playground_url = "https://play.rust-lang.org/"
)]
#![feature(rustc_private)]
#![feature(array_methods)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(box_syntax)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(test)]
#![feature(crate_visibility_modifier)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(type_ascription)]
#![feature(iter_intersperse)]
#![recursion_limit = "256"]
#![warn(rustc::internal)]

#[macro_use]
extern crate tracing;

// N.B. these need `extern crate` even in 2018 edition
// because they're loaded implicitly from the sysroot.
// The reason they're loaded from the sysroot is because
// the rustdoc artifacts aren't stored in rustc's cargo target directory.
// So if `rustc` was specified in Cargo.toml, this would spuriously rebuild crates.
//
// Dependencies listed in Cargo.toml do not need `extern crate`.

extern crate rustc_ast;
extern crate rustc_ast_lowering;
extern crate rustc_ast_pretty;
extern crate rustc_attr;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_expand;
extern crate rustc_feature;
extern crate rustc_hir;
extern crate rustc_hir_pretty;
extern crate rustc_index;
extern crate rustc_infer;
extern crate rustc_interface;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_lint_defs;
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
extern crate rustc_typeck;
extern crate test;

#[cfg(feature = "jemalloc")]
extern crate tikv_jemalloc_sys;
#[cfg(feature = "jemalloc")]
use tikv_jemalloc_sys as jemalloc_sys;
#[cfg(feature = "jemalloc")]
extern crate tikv_jemallocator;
#[cfg(feature = "jemalloc")]
use tikv_jemallocator as jemallocator;

use std::default::Default;
use std::env;
use std::process;

use rustc_driver::{abort_on_err, describe_lints};
use rustc_errors::ErrorReported;
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{make_crate_type_option, ErrorOutputType, RustcOptGroup};
use rustc_session::getopts;
use rustc_session::{early_error, early_warn};

use crate::clean::utils::DOC_RUST_LANG_ORG_CHANNEL;

/// A macro to create a FxHashMap.
///
/// Example:
///
/// ```
/// let letters = map!{"a" => "b", "c" => "d"};
/// ```
///
/// Trailing commas are allowed.
/// Commas between elements are required (even if the expression is a block).
macro_rules! map {
    ($( $key: expr => $val: expr ),* $(,)*) => {{
        let mut map = ::rustc_data_structures::fx::FxHashMap::default();
        $( map.insert($key, $val); )*
        map
    }}
}

#[macro_use]
mod externalfiles;

mod clean;
mod config;
mod core;
mod docfs;
mod doctree;
#[macro_use]
mod error;
mod doctest;
mod fold;
mod formats;
// used by the error-index generator, so it needs to be public
pub mod html;
mod json;
crate mod lint;
mod markdown;
mod passes;
mod scrape_examples;
mod theme;
mod visit_ast;
mod visit_lib;

// See docs in https://github.com/rust-lang/rust/blob/master/compiler/rustc/src/main.rs
// about jemallocator
#[cfg(feature = "jemalloc")]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub fn main() {
    // See docs in https://github.com/rust-lang/rust/blob/master/compiler/rustc/src/main.rs
    // about jemalloc-sys
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

        // On OSX, jemalloc doesn't directly override malloc/free, but instead
        // registers itself with the allocator's zone APIs in a ctor. However,
        // the linker doesn't seem to consider ctors as "used" when statically
        // linking, so we need to explicitly depend on the function.
        #[cfg(target_os = "macos")]
        {
            extern "C" {
                fn _rjem_je_zone_register();
            }

            #[used]
            static _F7: unsafe extern "C" fn() = _rjem_je_zone_register;
        }
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::install_ice_hook();

    // When using CI artifacts (with `download_stage1 = true`), tracing is unconditionally built
    // with `--features=static_max_level_info`, which disables almost all rustdoc logging. To avoid
    // this, compile our own version of `tracing` that logs all levels.
    // NOTE: this compiles both versions of tracing unconditionally, because
    // - The compile time hit is not that bad, especially compared to rustdoc's incremental times, and
    // - Otherwise, there's no warning that logging is being ignored when `download_stage1 = true`.
    // NOTE: The reason this doesn't show double logging when `download_stage1 = false` and
    // `debug_logging = true` is because all rustc logging goes to its version of tracing (the one
    // in the sysroot), and all of rustdoc's logging goes to its version (the one in Cargo.toml).
    init_logging();
    rustc_driver::init_env_logger("RUSTDOC_LOG");

    let exit_code = rustc_driver::catch_with_exit_code(|| match get_args() {
        Some(args) => main_args(&args),
        _ => Err(ErrorReported),
    });
    process::exit(exit_code);
}

fn init_logging() {
    use std::io;

    // FIXME remove these and use winapi 0.3 instead
    // Duplicates: bootstrap/compile.rs, librustc_errors/emitter.rs, rustc_driver/lib.rs
    #[cfg(unix)]
    fn stdout_isatty() -> bool {
        extern crate libc;
        unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 }
    }

    #[cfg(windows)]
    fn stdout_isatty() -> bool {
        extern crate winapi;
        use winapi::um::consoleapi::GetConsoleMode;
        use winapi::um::processenv::GetStdHandle;
        use winapi::um::winbase::STD_OUTPUT_HANDLE;

        unsafe {
            let handle = GetStdHandle(STD_OUTPUT_HANDLE);
            let mut out = 0;
            GetConsoleMode(handle, &mut out) != 0
        }
    }

    let color_logs = match std::env::var("RUSTDOC_LOG_COLOR") {
        Ok(value) => match value.as_ref() {
            "always" => true,
            "never" => false,
            "auto" => stdout_isatty(),
            _ => early_error(
                ErrorOutputType::default(),
                &format!(
                    "invalid log color value '{}': expected one of always, never, or auto",
                    value
                ),
            ),
        },
        Err(std::env::VarError::NotPresent) => stdout_isatty(),
        Err(std::env::VarError::NotUnicode(_value)) => early_error(
            ErrorOutputType::default(),
            "non-Unicode log color value: expected one of always, never, or auto",
        ),
    };
    let filter = tracing_subscriber::EnvFilter::from_env("RUSTDOC_LOG");
    let layer = tracing_tree::HierarchicalLayer::default()
        .with_writer(io::stderr)
        .with_indent_lines(true)
        .with_ansi(color_logs)
        .with_targets(true)
        .with_wraparound(10)
        .with_verbose_exit(true)
        .with_verbose_entry(true)
        .with_indent_amount(2);
    #[cfg(parallel_compiler)]
    let layer = layer.with_thread_ids(true).with_thread_names(true);

    use tracing_subscriber::layer::SubscriberExt;
    let subscriber = tracing_subscriber::Registry::default().with(filter).with(layer);
    tracing::subscriber::set_global_default(subscriber).unwrap();
}

fn get_args() -> Option<Vec<String>> {
    env::args_os()
        .enumerate()
        .map(|(i, arg)| {
            arg.into_string()
                .map_err(|arg| {
                    early_warn(
                        ErrorOutputType::default(),
                        &format!("Argument {} is not valid Unicode: {:?}", i, arg),
                    );
                })
                .ok()
        })
        .collect()
}

fn opts() -> Vec<RustcOptGroup> {
    let stable: fn(_, fn(&mut getopts::Options) -> &mut _) -> _ = RustcOptGroup::stable;
    let unstable: fn(_, fn(&mut getopts::Options) -> &mut _) -> _ = RustcOptGroup::unstable;
    vec![
        stable("h", |o| o.optflagmulti("h", "help", "show this help message")),
        stable("V", |o| o.optflagmulti("V", "version", "print rustdoc's version")),
        stable("v", |o| o.optflagmulti("v", "verbose", "use verbose output")),
        stable("r", |o| {
            o.optopt("r", "input-format", "the input type of the specified file", "[rust]")
        }),
        stable("w", |o| o.optopt("w", "output-format", "the output type to write", "[html]")),
        stable("o", |o| o.optopt("o", "output", "where to place the output", "PATH")),
        stable("crate-name", |o| {
            o.optopt("", "crate-name", "specify the name of this crate", "NAME")
        }),
        make_crate_type_option(),
        stable("L", |o| {
            o.optmulti("L", "library-path", "directory to add to crate search path", "DIR")
        }),
        stable("cfg", |o| o.optmulti("", "cfg", "pass a --cfg to rustc", "")),
        stable("extern", |o| o.optmulti("", "extern", "pass an --extern to rustc", "NAME[=PATH]")),
        unstable("extern-html-root-url", |o| {
            o.optmulti(
                "",
                "extern-html-root-url",
                "base URL to use for dependencies; for example, \
                 \"std=/doc\" links std::vec::Vec to /doc/std/vec/struct.Vec.html",
                "NAME=URL",
            )
        }),
        unstable("extern-html-root-takes-precedence", |o| {
            o.optflagmulti(
                "",
                "extern-html-root-takes-precedence",
                "give precedence to `--extern-html-root-url`, not `html_root_url`",
            )
        }),
        stable("plugin-path", |o| o.optmulti("", "plugin-path", "removed", "DIR")),
        stable("C", |o| {
            o.optmulti("C", "codegen", "pass a codegen option to rustc", "OPT[=VALUE]")
        }),
        stable("passes", |o| {
            o.optmulti(
                "",
                "passes",
                "list of passes to also run, you might want to pass it multiple times; a value of \
                 `list` will print available passes",
                "PASSES",
            )
        }),
        stable("plugins", |o| o.optmulti("", "plugins", "removed", "PLUGINS")),
        stable("no-default", |o| o.optflagmulti("", "no-defaults", "don't run the default passes")),
        stable("document-private-items", |o| {
            o.optflagmulti("", "document-private-items", "document private items")
        }),
        unstable("document-hidden-items", |o| {
            o.optflagmulti("", "document-hidden-items", "document items that have doc(hidden)")
        }),
        stable("test", |o| o.optflagmulti("", "test", "run code examples as tests")),
        stable("test-args", |o| {
            o.optmulti("", "test-args", "arguments to pass to the test runner", "ARGS")
        }),
        unstable("test-run-directory", |o| {
            o.optopt(
                "",
                "test-run-directory",
                "The working directory in which to run tests",
                "PATH",
            )
        }),
        stable("target", |o| o.optopt("", "target", "target triple to document", "TRIPLE")),
        stable("markdown-css", |o| {
            o.optmulti(
                "",
                "markdown-css",
                "CSS files to include via <link> in a rendered Markdown file",
                "FILES",
            )
        }),
        stable("html-in-header", |o| {
            o.optmulti(
                "",
                "html-in-header",
                "files to include inline in the <head> section of a rendered Markdown file \
                 or generated documentation",
                "FILES",
            )
        }),
        stable("html-before-content", |o| {
            o.optmulti(
                "",
                "html-before-content",
                "files to include inline between <body> and the content of a rendered \
                 Markdown file or generated documentation",
                "FILES",
            )
        }),
        stable("html-after-content", |o| {
            o.optmulti(
                "",
                "html-after-content",
                "files to include inline between the content and </body> of a rendered \
                 Markdown file or generated documentation",
                "FILES",
            )
        }),
        unstable("markdown-before-content", |o| {
            o.optmulti(
                "",
                "markdown-before-content",
                "files to include inline between <body> and the content of a rendered \
                 Markdown file or generated documentation",
                "FILES",
            )
        }),
        unstable("markdown-after-content", |o| {
            o.optmulti(
                "",
                "markdown-after-content",
                "files to include inline between the content and </body> of a rendered \
                 Markdown file or generated documentation",
                "FILES",
            )
        }),
        stable("markdown-playground-url", |o| {
            o.optopt("", "markdown-playground-url", "URL to send code snippets to", "URL")
        }),
        stable("markdown-no-toc", |o| {
            o.optflagmulti("", "markdown-no-toc", "don't include table of contents")
        }),
        stable("e", |o| {
            o.optopt(
                "e",
                "extend-css",
                "To add some CSS rules with a given file to generate doc with your \
                 own theme. However, your theme might break if the rustdoc's generated HTML \
                 changes, so be careful!",
                "PATH",
            )
        }),
        unstable("Z", |o| {
            o.optmulti("Z", "", "internal and debugging options (only on nightly build)", "FLAG")
        }),
        stable("sysroot", |o| o.optopt("", "sysroot", "Override the system root", "PATH")),
        unstable("playground-url", |o| {
            o.optopt(
                "",
                "playground-url",
                "URL to send code snippets to, may be reset by --markdown-playground-url \
                 or `#![doc(html_playground_url=...)]`",
                "URL",
            )
        }),
        unstable("display-doctest-warnings", |o| {
            o.optflagmulti(
                "",
                "display-doctest-warnings",
                "show warnings that originate in doctests",
            )
        }),
        stable("crate-version", |o| {
            o.optopt("", "crate-version", "crate version to print into documentation", "VERSION")
        }),
        unstable("sort-modules-by-appearance", |o| {
            o.optflagmulti(
                "",
                "sort-modules-by-appearance",
                "sort modules by where they appear in the program, rather than alphabetically",
            )
        }),
        stable("default-theme", |o| {
            o.optopt(
                "",
                "default-theme",
                "Set the default theme. THEME should be the theme name, generally lowercase. \
                 If an unknown default theme is specified, the builtin default is used. \
                 The set of themes, and the rustdoc built-in default, are not stable.",
                "THEME",
            )
        }),
        unstable("default-setting", |o| {
            o.optmulti(
                "",
                "default-setting",
                "Default value for a rustdoc setting (used when \"rustdoc-SETTING\" is absent \
                 from web browser Local Storage). If VALUE is not supplied, \"true\" is used. \
                 Supported SETTINGs and VALUEs are not documented and not stable.",
                "SETTING[=VALUE]",
            )
        }),
        stable("theme", |o| {
            o.optmulti(
                "",
                "theme",
                "additional themes which will be added to the generated docs",
                "FILES",
            )
        }),
        stable("check-theme", |o| {
            o.optmulti("", "check-theme", "check if given theme is valid", "FILES")
        }),
        unstable("resource-suffix", |o| {
            o.optopt(
                "",
                "resource-suffix",
                "suffix to add to CSS and JavaScript files, e.g., \"light.css\" will become \
                 \"light-suffix.css\"",
                "PATH",
            )
        }),
        stable("edition", |o| {
            o.optopt(
                "",
                "edition",
                "edition to use when compiling rust code (default: 2015)",
                "EDITION",
            )
        }),
        stable("color", |o| {
            o.optopt(
                "",
                "color",
                "Configure coloring of output:
                                          auto   = colorize, if output goes to a tty (default);
                                          always = always colorize output;
                                          never  = never colorize output",
                "auto|always|never",
            )
        }),
        stable("error-format", |o| {
            o.optopt(
                "",
                "error-format",
                "How errors and other messages are produced",
                "human|json|short",
            )
        }),
        stable("json", |o| {
            o.optopt("", "json", "Configure the structure of JSON diagnostics", "CONFIG")
        }),
        unstable("disable-minification", |o| {
            o.optflagmulti("", "disable-minification", "Disable minification applied on JS files")
        }),
        stable("allow", |o| o.optmulti("A", "allow", "Set lint allowed", "LINT")),
        stable("warn", |o| o.optmulti("W", "warn", "Set lint warnings", "LINT")),
        stable("force-warn", |o| o.optmulti("", "force-warn", "Set lint force-warn", "LINT")),
        stable("deny", |o| o.optmulti("D", "deny", "Set lint denied", "LINT")),
        stable("forbid", |o| o.optmulti("F", "forbid", "Set lint forbidden", "LINT")),
        stable("cap-lints", |o| {
            o.optmulti(
                "",
                "cap-lints",
                "Set the most restrictive lint level. \
                 More restrictive lints are capped at this \
                 level. By default, it is at `forbid` level.",
                "LEVEL",
            )
        }),
        unstable("index-page", |o| {
            o.optopt("", "index-page", "Markdown file to be used as index page", "PATH")
        }),
        unstable("enable-index-page", |o| {
            o.optflagmulti("", "enable-index-page", "To enable generation of the index page")
        }),
        unstable("static-root-path", |o| {
            o.optopt(
                "",
                "static-root-path",
                "Path string to force loading static files from in output pages. \
                 If not set, uses combinations of '../' to reach the documentation root.",
                "PATH",
            )
        }),
        unstable("disable-per-crate-search", |o| {
            o.optflagmulti(
                "",
                "disable-per-crate-search",
                "disables generating the crate selector on the search box",
            )
        }),
        unstable("persist-doctests", |o| {
            o.optopt(
                "",
                "persist-doctests",
                "Directory to persist doctest executables into",
                "PATH",
            )
        }),
        unstable("show-coverage", |o| {
            o.optflagmulti(
                "",
                "show-coverage",
                "calculate percentage of public items with documentation",
            )
        }),
        unstable("enable-per-target-ignores", |o| {
            o.optflagmulti(
                "",
                "enable-per-target-ignores",
                "parse ignore-foo for ignoring doctests on a per-target basis",
            )
        }),
        unstable("runtool", |o| {
            o.optopt(
                "",
                "runtool",
                "",
                "The tool to run tests with when building for a different target than host",
            )
        }),
        unstable("runtool-arg", |o| {
            o.optmulti(
                "",
                "runtool-arg",
                "",
                "One (of possibly many) arguments to pass to the runtool",
            )
        }),
        unstable("test-builder", |o| {
            o.optopt("", "test-builder", "The rustc-like binary to use as the test builder", "PATH")
        }),
        unstable("check", |o| o.optflagmulti("", "check", "Run rustdoc checks")),
        unstable("generate-redirect-map", |o| {
            o.optflagmulti(
                "",
                "generate-redirect-map",
                "Generate JSON file at the top level instead of generating HTML redirection files",
            )
        }),
        unstable("emit", |o| {
            o.optmulti(
                "",
                "emit",
                "Comma separated list of types of output for rustdoc to emit",
                "[unversioned-shared-resources,toolchain-shared-resources,invocation-specific]",
            )
        }),
        unstable("no-run", |o| {
            o.optflagmulti("", "no-run", "Compile doctests without running them")
        }),
        unstable("show-type-layout", |o| {
            o.optflagmulti("", "show-type-layout", "Include the memory layout of types in the docs")
        }),
        unstable("nocapture", |o| {
            o.optflag("", "nocapture", "Don't capture stdout and stderr of tests")
        }),
        unstable("generate-link-to-definition", |o| {
            o.optflag(
                "",
                "generate-link-to-definition",
                "Make the identifiers in the HTML source code pages navigable",
            )
        }),
        unstable("scrape-examples-output-path", |o| {
            o.optopt(
                "",
                "scrape-examples-output-path",
                "",
                "collect function call information and output at the given path",
            )
        }),
        unstable("scrape-examples-target-crate", |o| {
            o.optmulti(
                "",
                "scrape-examples-target-crate",
                "",
                "collect function call information for functions from the target crate",
            )
        }),
        unstable("with-examples", |o| {
            o.optmulti(
                "",
                "with-examples",
                "",
                "path to function call information (for displaying examples in the documentation)",
            )
        }),
    ]
}

fn usage(argv0: &str) {
    let mut options = getopts::Options::new();
    for option in opts() {
        (option.apply)(&mut options);
    }
    println!("{}", options.usage(&format!("{} [options] <input>", argv0)));
    println!("    @path               Read newline separated options from `path`\n");
    println!(
        "More information available at {}/rustdoc/what-is-rustdoc.html",
        DOC_RUST_LANG_ORG_CHANNEL
    );
}

/// A result type used by several functions under `main()`.
type MainResult = Result<(), ErrorReported>;

fn main_args(at_args: &[String]) -> MainResult {
    let args = rustc_driver::args::arg_expand_all(at_args);

    let mut options = getopts::Options::new();
    for option in opts() {
        (option.apply)(&mut options);
    }
    let matches = match options.parse(&args[1..]) {
        Ok(m) => m,
        Err(err) => {
            early_error(ErrorOutputType::default(), &err.to_string());
        }
    };

    // Note that we discard any distinction between different non-zero exit
    // codes from `from_matches` here.
    let options = match config::Options::from_matches(&matches) {
        Ok(opts) => opts,
        Err(code) => return if code == 0 { Ok(()) } else { Err(ErrorReported) },
    };
    rustc_interface::util::setup_callbacks_and_run_in_thread_pool_with_globals(
        options.edition,
        1, // this runs single-threaded, even in a parallel compiler
        &None,
        move || main_options(options),
    )
}

fn wrap_return(diag: &rustc_errors::Handler, res: Result<(), String>) -> MainResult {
    match res {
        Ok(()) => Ok(()),
        Err(err) => {
            diag.struct_err(&err).emit();
            Err(ErrorReported)
        }
    }
}

fn run_renderer<'tcx, T: formats::FormatRenderer<'tcx>>(
    krate: clean::Crate,
    renderopts: config::RenderOptions,
    cache: formats::cache::Cache,
    tcx: TyCtxt<'tcx>,
) -> MainResult {
    match formats::run_format::<T>(krate, renderopts, cache, tcx) {
        Ok(_) => Ok(()),
        Err(e) => {
            let mut msg =
                tcx.sess.struct_err(&format!("couldn't generate documentation: {}", e.error));
            let file = e.file.display().to_string();
            if file.is_empty() {
                msg.emit()
            } else {
                msg.note(&format!("failed to create or modify \"{}\"", file)).emit()
            }
            Err(ErrorReported)
        }
    }
}

fn main_options(options: config::Options) -> MainResult {
    let diag = core::new_handler(options.error_format, None, &options.debugging_opts);

    match (options.should_test, options.markdown_input()) {
        (true, true) => return wrap_return(&diag, markdown::test(options)),
        (true, false) => return doctest::run(options),
        (false, true) => {
            return wrap_return(
                &diag,
                markdown::render(&options.input, options.render_options, options.edition),
            );
        }
        (false, false) => {}
    }

    // need to move these items separately because we lose them by the time the closure is called,
    // but we can't create the Handler ahead of time because it's not Send
    let show_coverage = options.show_coverage;
    let run_check = options.run_check;

    // First, parse the crate and extract all relevant information.
    info!("starting to run rustc");

    // Interpret the input file as a rust source file, passing it through the
    // compiler all the way through the analysis passes. The rustdoc output is
    // then generated from the cleaned AST of the crate. This runs all the
    // plug/cleaning passes.
    let crate_version = options.crate_version.clone();

    let default_passes = options.default_passes;
    let output_format = options.output_format;
    // FIXME: fix this clone (especially render_options)
    let manual_passes = options.manual_passes.clone();
    let render_options = options.render_options.clone();
    let scrape_examples_options = options.scrape_examples_options.clone();
    let config = core::create_config(options);

    interface::create_compiler_and_run(config, |compiler| {
        compiler.enter(|queries| {
            let sess = compiler.session();

            if sess.opts.describe_lints {
                let (_, lint_store) = &*queries.register_plugins()?.peek();
                describe_lints(sess, lint_store, true);
                return Ok(());
            }

            // We need to hold on to the complete resolver, so we cause everything to be
            // cloned for the analysis passes to use. Suboptimal, but necessary in the
            // current architecture.
            let resolver = core::create_resolver(queries, sess);

            if sess.has_errors() {
                sess.fatal("Compilation failed, aborting rustdoc");
            }

            let mut global_ctxt = abort_on_err(queries.global_ctxt(), sess).peek_mut();

            global_ctxt.enter(|tcx| {
                let (krate, render_opts, mut cache) = sess.time("run_global_ctxt", || {
                    core::run_global_ctxt(
                        tcx,
                        resolver,
                        default_passes,
                        manual_passes,
                        render_options,
                        output_format,
                    )
                });
                info!("finished with rustc");

                if let Some(options) = scrape_examples_options {
                    return scrape_examples::run(krate, render_opts, cache, tcx, options);
                }

                cache.crate_version = crate_version;

                if show_coverage {
                    // if we ran coverage, bail early, we don't need to also generate docs at this point
                    // (also we didn't load in any of the useful passes)
                    return Ok(());
                } else if run_check {
                    // Since we're in "check" mode, no need to generate anything beyond this point.
                    return Ok(());
                }

                info!("going to format");
                match output_format {
                    config::OutputFormat::Html => sess.time("render_html", || {
                        run_renderer::<html::render::Context<'_>>(krate, render_opts, cache, tcx)
                    }),
                    config::OutputFormat::Json => sess.time("render_json", || {
                        run_renderer::<json::JsonRenderer<'_>>(krate, render_opts, cache, tcx)
                    }),
                }
            })
        })
    })
}
