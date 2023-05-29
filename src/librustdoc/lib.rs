#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/",
    html_playground_url = "https://play.rust-lang.org/"
)]
#![feature(rustc_private)]
#![feature(array_methods)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_intersperse)]
#![feature(lazy_cell)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(round_char_boundary)]
#![feature(test)]
#![feature(type_alias_impl_trait)]
#![feature(type_ascription)]
#![recursion_limit = "256"]
#![warn(rustc::internal)]
#![allow(clippy::collapsible_if, clippy::collapsible_else_if)]
#![allow(rustc::potential_query_instability)]

extern crate thin_vec;
#[macro_use]
extern crate tracing;

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
extern crate rustc_attr;
extern crate rustc_const_eval;
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
extern crate jemalloc_sys;

use std::env::{self, VarError};
use std::io::{self, IsTerminal};
use std::process;

use rustc_driver::abort_on_err;
use rustc_errors::ErrorGuaranteed;
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
/// ```ignore(cannot-test-this-because-non-exported-macro)
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

mod clean;
mod config;
mod core;
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
            extern "C" {
                fn _rjem_je_zone_register();
            }

            #[used]
            static _F7: unsafe extern "C" fn() = _rjem_je_zone_register;
        }
    }

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
    // NOTE: The reason this doesn't show double logging when `download-rustc = false` and
    // `debug_logging = true` is because all rustc logging goes to its version of tracing (the one
    // in the sysroot), and all of rustdoc's logging goes to its version (the one in Cargo.toml).
    init_logging();
    rustc_driver::init_env_logger("RUSTDOC_LOG");

    let exit_code = rustc_driver::catch_with_exit_code(|| match get_args() {
        Some(args) => main_args(&args),
        _ =>
        {
            #[allow(deprecated)]
            Err(ErrorGuaranteed::unchecked_claim_error_was_emitted())
        }
    });
    process::exit(exit_code);
}

fn init_logging() {
    let color_logs = match std::env::var("RUSTDOC_LOG_COLOR").as_deref() {
        Ok("always") => true,
        Ok("never") => false,
        Ok("auto") | Err(VarError::NotPresent) => io::stdout().is_terminal(),
        Ok(value) => early_error(
            ErrorOutputType::default(),
            format!("invalid log color value '{}': expected one of always, never, or auto", value),
        ),
        Err(VarError::NotUnicode(value)) => early_error(
            ErrorOutputType::default(),
            format!(
                "invalid log color value '{}': expected one of always, never, or auto",
                value.to_string_lossy()
            ),
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
    #[cfg(all(parallel_compiler, debug_assertions))]
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
                        format!("Argument {} is not valid Unicode: {:?}", i, arg),
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
        stable("w", |o| o.optopt("w", "output-format", "the output type to write", "[html]")),
        stable("output", |o| {
            o.optopt(
                "",
                "output",
                "Which directory to place the output. \
                 This option is deprecated, use --out-dir instead.",
                "PATH",
            )
        }),
        stable("o", |o| o.optopt("o", "out-dir", "which directory to place the output", "PATH")),
        stable("crate-name", |o| {
            o.optopt("", "crate-name", "specify the name of this crate", "NAME")
        }),
        make_crate_type_option(),
        stable("L", |o| {
            o.optmulti("L", "library-path", "directory to add to crate search path", "DIR")
        }),
        stable("cfg", |o| o.optmulti("", "cfg", "pass a --cfg to rustc", "")),
        unstable("check-cfg", |o| o.optmulti("", "check-cfg", "pass a --check-cfg to rustc", "")),
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
        stable("C", |o| {
            o.optmulti("C", "codegen", "pass a codegen option to rustc", "OPT[=VALUE]")
        }),
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
        stable("test-run-directory", |o| {
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
            o.optmulti("Z", "", "unstable / perma-unstable options (only on nightly build)", "FLAG")
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
        stable("diagnostic-width", |o| {
            o.optopt(
                "",
                "diagnostic-width",
                "Provide width of the output for truncated error messages",
                "WIDTH",
            )
        }),
        stable("json", |o| {
            o.optopt("", "json", "Configure the structure of JSON diagnostics", "CONFIG")
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
        unstable("scrape-tests", |o| {
            o.optflag("", "scrape-tests", "Include test code when scraping examples")
        }),
        unstable("with-examples", |o| {
            o.optmulti(
                "",
                "with-examples",
                "",
                "path to function call information (for displaying examples in the documentation)",
            )
        }),
        // deprecated / removed options
        unstable("disable-minification", |o| o.optflagmulti("", "disable-minification", "removed")),
        stable("plugin-path", |o| {
            o.optmulti(
                "",
                "plugin-path",
                "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                for more information",
                "DIR",
            )
        }),
        stable("passes", |o| {
            o.optmulti(
                "",
                "passes",
                "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                for more information",
                "PASSES",
            )
        }),
        stable("plugins", |o| {
            o.optmulti(
                "",
                "plugins",
                "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                for more information",
                "PLUGINS",
            )
        }),
        stable("no-default", |o| {
            o.optflagmulti(
                "",
                "no-defaults",
                "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                for more information",
            )
        }),
        stable("r", |o| {
            o.optopt(
                "r",
                "input-format",
                "removed, see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                for more information",
                "[rust]",
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
type MainResult = Result<(), ErrorGuaranteed>;

fn wrap_return(diag: &rustc_errors::Handler, res: Result<(), String>) -> MainResult {
    match res {
        Ok(()) => diag.has_errors().map_or(Ok(()), Err),
        Err(err) => {
            let reported = diag.struct_err(err).emit();
            Err(reported)
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
        Ok(_) => tcx.sess.has_errors().map_or(Ok(()), Err),
        Err(e) => {
            let mut msg =
                tcx.sess.struct_err(format!("couldn't generate documentation: {}", e.error));
            let file = e.file.display().to_string();
            if !file.is_empty() {
                msg.note(format!("failed to create or modify \"{}\"", file));
            }
            Err(msg.emit())
        }
    }
}

fn main_args(at_args: &[String]) -> MainResult {
    // Throw away the first argument, the name of the binary.
    // In case of at_args being empty, as might be the case by
    // passing empty argument array to execve under some platforms,
    // just use an empty slice.
    //
    // This situation was possible before due to arg_expand_all being
    // called before removing the argument, enabling a crash by calling
    // the compiler with @empty_file as argv[0] and no more arguments.
    let at_args = at_args.get(1..).unwrap_or_default();

    let args = rustc_driver::args::arg_expand_all(at_args);

    let mut options = getopts::Options::new();
    for option in opts() {
        (option.apply)(&mut options);
    }
    let matches = match options.parse(&args) {
        Ok(m) => m,
        Err(err) => {
            early_error(ErrorOutputType::default(), err.to_string());
        }
    };

    // Note that we discard any distinction between different non-zero exit
    // codes from `from_matches` here.
    let (options, render_options) = match config::Options::from_matches(&matches, args) {
        Ok(opts) => opts,
        Err(code) => {
            return if code == 0 {
                Ok(())
            } else {
                #[allow(deprecated)]
                Err(ErrorGuaranteed::unchecked_claim_error_was_emitted())
            };
        }
    };

    // Set parallel mode before error handler creation, which will create `Lock`s.
    interface::set_thread_safe_mode(&options.unstable_opts);

    let diag = core::new_handler(
        options.error_format,
        None,
        options.diagnostic_width,
        &options.unstable_opts,
    );

    match (options.should_test, options.markdown_input()) {
        (true, true) => return wrap_return(&diag, markdown::test(options)),
        (true, false) => return doctest::run(options),
        (false, true) => {
            let input = options.input.clone();
            let edition = options.edition;
            let config = core::create_config(options, &render_options);

            // `markdown::render` can invoke `doctest::make_test`, which
            // requires session globals and a thread pool, so we use
            // `run_compiler`.
            return wrap_return(
                &diag,
                interface::run_compiler(config, |_compiler| {
                    markdown::render(&input, render_options, edition)
                }),
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

    let output_format = options.output_format;
    let scrape_examples_options = options.scrape_examples_options.clone();
    let bin_crate = options.bin_crate;

    let config = core::create_config(options, &render_options);

    interface::run_compiler(config, |compiler| {
        let sess = compiler.session();

        if sess.opts.describe_lints {
            let mut lint_store = rustc_lint::new_lint_store(sess.enable_internal_lints());
            let registered_lints = if let Some(register_lints) = compiler.register_lints() {
                register_lints(sess, &mut lint_store);
                true
            } else {
                false
            };
            rustc_driver::describe_lints(sess, &lint_store, registered_lints);
            return Ok(());
        }

        compiler.enter(|queries| {
            let mut gcx = abort_on_err(queries.global_ctxt(), sess);
            if sess.diagnostic().has_errors_or_lint_errors().is_some() {
                sess.fatal("Compilation failed, aborting rustdoc");
            }

            gcx.enter(|tcx| {
                let (krate, render_opts, mut cache) = sess.time("run_global_ctxt", || {
                    core::run_global_ctxt(tcx, show_coverage, render_options, output_format)
                });
                info!("finished with rustc");

                if let Some(options) = scrape_examples_options {
                    return scrape_examples::run(
                        krate,
                        render_opts,
                        cache,
                        tcx,
                        options,
                        bin_crate,
                    );
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
