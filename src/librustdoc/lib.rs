#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/",
    html_playground_url = "https://play.rust-lang.org/"
)]
#![feature(rustc_private)]
#![feature(array_methods)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(or_patterns)]
#![feature(peekable_next_if)]
#![feature(test)]
#![feature(crate_visibility_modifier)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(type_ascription)]
#![feature(split_inclusive)]
#![feature(str_split_once)]
#![feature(iter_intersperse)]
#![recursion_limit = "256"]

#[macro_use]
extern crate lazy_static;
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
extern crate rustc_ast_pretty;
extern crate rustc_attr;
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
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_mir;
extern crate rustc_parse;
extern crate rustc_resolve;
extern crate rustc_session;
extern crate rustc_span as rustc_span;
extern crate rustc_target;
extern crate rustc_trait_selection;
extern crate rustc_typeck;
extern crate test as testing;

use std::default::Default;
use std::env;
use std::process;

use rustc_driver::abort_on_err;
use rustc_errors::ErrorReported;
use rustc_interface::interface;
use rustc_middle::ty;
use rustc_session::config::{make_crate_type_option, ErrorOutputType, RustcOptGroup};
use rustc_session::getopts;
use rustc_session::{early_error, early_warn};

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
crate mod formats;
pub mod html;
mod json;
mod markdown;
mod passes;
mod theme;
mod visit_ast;
mod visit_lib;

pub fn main() {
    rustc_driver::set_sigpipe_handler();
    rustc_driver::install_ice_hook();
    rustc_driver::init_env_logger("RUSTDOC_LOG");
    let exit_code = rustc_driver::catch_with_exit_code(|| match get_args() {
        Some(args) => main_args(&args),
        _ => Err(ErrorReported),
    });
    process::exit(exit_code);
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
        stable("h", |o| o.optflag("h", "help", "show this help message")),
        stable("V", |o| o.optflag("V", "version", "print rustdoc's version")),
        stable("v", |o| o.optflag("v", "verbose", "use verbose output")),
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
            o.optmulti("", "extern-html-root-url", "base URL to use for dependencies", "NAME=URL")
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
        stable("no-default", |o| o.optflag("", "no-defaults", "don't run the default passes")),
        stable("document-private-items", |o| {
            o.optflag("", "document-private-items", "document private items")
        }),
        unstable("document-hidden-items", |o| {
            o.optflag("", "document-hidden-items", "document items that have doc(hidden)")
        }),
        stable("test", |o| o.optflag("", "test", "run code examples as tests")),
        stable("test-args", |o| {
            o.optmulti("", "test-args", "arguments to pass to the test runner", "ARGS")
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
            o.optflag("", "markdown-no-toc", "don't include table of contents")
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
        unstable("display-warnings", |o| {
            o.optflag("", "display-warnings", "to print code warnings when testing doc")
        }),
        stable("crate-version", |o| {
            o.optopt("", "crate-version", "crate version to print into documentation", "VERSION")
        }),
        unstable("sort-modules-by-appearance", |o| {
            o.optflag(
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
            o.optflag("", "disable-minification", "Disable minification applied on JS files")
        }),
        stable("warn", |o| o.optmulti("W", "warn", "Set lint warnings", "OPT")),
        stable("allow", |o| o.optmulti("A", "allow", "Set lint allowed", "OPT")),
        stable("deny", |o| o.optmulti("D", "deny", "Set lint denied", "OPT")),
        stable("forbid", |o| o.optmulti("F", "forbid", "Set lint forbidden", "OPT")),
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
            o.optflag("", "enable-index-page", "To enable generation of the index page")
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
            o.optflag(
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
            o.optflag(
                "",
                "show-coverage",
                "calculate percentage of public items with documentation",
            )
        }),
        unstable("enable-per-target-ignores", |o| {
            o.optflag(
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
            o.optflag(
                "",
                "test-builder",
                "specified the rustc-like binary to use as the test builder",
            )
        }),
        unstable("check", |o| o.optflag("", "check", "Run rustdoc checks")),
    ]
}

fn usage(argv0: &str) {
    let mut options = getopts::Options::new();
    for option in opts() {
        (option.apply)(&mut options);
    }
    println!("{}", options.usage(&format!("{} [options] <input>", argv0)));
    println!("More information available at https://doc.rust-lang.org/rustdoc/what-is-rustdoc.html")
}

/// A result type used by several functions under `main()`.
type MainResult = Result<(), ErrorReported>;

fn main_args(args: &[String]) -> MainResult {
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
    render_info: config::RenderInfo,
    diag: &rustc_errors::Handler,
    edition: rustc_span::edition::Edition,
    tcx: ty::TyCtxt<'tcx>,
) -> MainResult {
    match formats::run_format::<T>(krate, renderopts, render_info, &diag, edition, tcx) {
        Ok(_) => Ok(()),
        Err(e) => {
            let mut msg = diag.struct_err(&format!("couldn't generate documentation: {}", e.error));
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
    let diag_opts = (options.error_format, options.edition, options.debugging_opts.clone());
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
    let externs = options.externs.clone();
    let manual_passes = options.manual_passes.clone();
    let render_options = options.render_options.clone();
    let config = core::create_config(options);

    interface::create_compiler_and_run(config, |compiler| {
        compiler.enter(|queries| {
            let sess = compiler.session();

            // We need to hold on to the complete resolver, so we cause everything to be
            // cloned for the analysis passes to use. Suboptimal, but necessary in the
            // current architecture.
            let resolver = core::create_resolver(externs, queries, &sess);

            if sess.has_errors() {
                sess.fatal("Compilation failed, aborting rustdoc");
            }

            let mut global_ctxt = abort_on_err(queries.global_ctxt(), sess).take();

            global_ctxt.enter(|tcx| {
                let (mut krate, render_info, render_opts) = sess.time("run_global_ctxt", || {
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

                krate.version = crate_version;

                if show_coverage {
                    // if we ran coverage, bail early, we don't need to also generate docs at this point
                    // (also we didn't load in any of the useful passes)
                    return Ok(());
                } else if run_check {
                    // Since we're in "check" mode, no need to generate anything beyond this point.
                    return Ok(());
                }

                info!("going to format");
                let (error_format, edition, debugging_options) = diag_opts;
                let diag = core::new_handler(error_format, None, &debugging_options);
                match output_format {
                    None | Some(config::OutputFormat::Html) => sess.time("render_html", || {
                        run_renderer::<html::render::Context<'_>>(
                            krate,
                            render_opts,
                            render_info,
                            &diag,
                            edition,
                            tcx,
                        )
                    }),
                    Some(config::OutputFormat::Json) => sess.time("render_json", || {
                        run_renderer::<json::JsonRenderer<'_>>(
                            krate,
                            render_opts,
                            render_info,
                            &diag,
                            edition,
                            tcx,
                        )
                    }),
                }
            })
        })
    })
}
