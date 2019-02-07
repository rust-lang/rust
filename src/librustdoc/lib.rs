#![deny(rust_2018_idioms)]

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/")]

#![feature(bind_by_move_pattern_guards)]
#![feature(rustc_private)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(nll)]
#![feature(set_stdio)]
#![feature(test)]
#![feature(vec_remove_item)]
#![feature(ptr_offset_from)]
#![feature(crate_visibility_modifier)]
#![feature(const_fn)]
#![feature(drain_filter)]
#![feature(inner_deref)]
#![feature(never_type)]
#![feature(exhaustive_patterns)]

#![recursion_limit="256"]

extern crate getopts;
extern crate env_logger;
extern crate rustc;
extern crate rustc_data_structures;
extern crate rustc_codegen_utils;
extern crate rustc_driver;
extern crate rustc_resolve;
extern crate rustc_lint;
extern crate rustc_metadata;
extern crate rustc_target;
extern crate rustc_typeck;
extern crate rustc_interface;
extern crate serialize;
extern crate syntax;
extern crate syntax_pos;
extern crate test as testing;
#[macro_use] extern crate log;
extern crate rustc_errors as errors;

extern crate serialize as rustc_serialize; // used by deriving

use std::default::Default;
use std::env;
use std::panic;
use std::process;
use std::sync::mpsc::channel;

use rustc::session::{early_warn, early_error};
use rustc::session::config::{ErrorOutputType, RustcOptGroup};

#[macro_use]
mod externalfiles;

mod clean;
mod config;
mod core;
mod doctree;
mod fold;
pub mod html {
    crate mod highlight;
    crate mod escape;
    crate mod item_type;
    crate mod format;
    crate mod layout;
    pub mod markdown;
    crate mod render;
    crate mod static_files;
    crate mod toc;
}
mod markdown;
mod passes;
mod visit_ast;
mod visit_lib;
mod test;
mod theme;

struct Output {
    krate: clean::Crate,
    renderinfo: html::render::RenderInfo,
    renderopts: config::RenderOptions,
    passes: Vec<String>,
}

pub fn main() {
    let thread_stack_size: usize = if cfg!(target_os = "haiku") {
        16_000_000 // 16MB on Haiku
    } else {
        32_000_000 // 32MB on other platforms
    };
    rustc_driver::set_sigpipe_handler();
    env_logger::init();
    let res = std::thread::Builder::new().stack_size(thread_stack_size).spawn(move || {
        syntax::with_globals(move || {
            get_args().map(|args| main_args(&args)).unwrap_or(1)
        })
    }).unwrap().join().unwrap_or(rustc_driver::EXIT_FAILURE);
    process::exit(res as i32);
}

fn get_args() -> Option<Vec<String>> {
    env::args_os().enumerate()
        .map(|(i, arg)| arg.into_string().map_err(|arg| {
             early_warn(ErrorOutputType::default(),
                        &format!("Argument {} is not valid Unicode: {:?}", i, arg));
        }).ok())
        .collect()
}

fn stable<F>(name: &'static str, f: F) -> RustcOptGroup
    where F: Fn(&mut getopts::Options) -> &mut getopts::Options + 'static
{
    RustcOptGroup::stable(name, f)
}

fn unstable<F>(name: &'static str, f: F) -> RustcOptGroup
    where F: Fn(&mut getopts::Options) -> &mut getopts::Options + 'static
{
    RustcOptGroup::unstable(name, f)
}

fn opts() -> Vec<RustcOptGroup> {
    vec![
        stable("h", |o| o.optflag("h", "help", "show this help message")),
        stable("V", |o| o.optflag("V", "version", "print rustdoc's version")),
        stable("v", |o| o.optflag("v", "verbose", "use verbose output")),
        stable("r", |o| {
            o.optopt("r", "input-format", "the input type of the specified file",
                     "[rust]")
        }),
        stable("w", |o| {
            o.optopt("w", "output-format", "the output type to write", "[html]")
        }),
        stable("o", |o| o.optopt("o", "output", "where to place the output", "PATH")),
        stable("crate-name", |o| {
            o.optopt("", "crate-name", "specify the name of this crate", "NAME")
        }),
        stable("L", |o| {
            o.optmulti("L", "library-path", "directory to add to crate search path",
                       "DIR")
        }),
        stable("cfg", |o| o.optmulti("", "cfg", "pass a --cfg to rustc", "")),
        stable("extern", |o| {
            o.optmulti("", "extern", "pass an --extern to rustc", "NAME=PATH")
        }),
        unstable("extern-html-root-url", |o| {
            o.optmulti("", "extern-html-root-url",
                       "base URL to use for dependencies", "NAME=URL")
        }),
        stable("plugin-path", |o| {
            o.optmulti("", "plugin-path", "removed", "DIR")
        }),
        stable("C", |o| {
            o.optmulti("C", "codegen", "pass a codegen option to rustc", "OPT[=VALUE]")
        }),
        stable("passes", |o| {
            o.optmulti("", "passes",
                       "list of passes to also run, you might want \
                        to pass it multiple times; a value of `list` \
                        will print available passes",
                       "PASSES")
        }),
        stable("plugins", |o| {
            o.optmulti("", "plugins", "removed",
                       "PLUGINS")
        }),
        stable("no-default", |o| {
            o.optflag("", "no-defaults", "don't run the default passes")
        }),
        stable("document-private-items", |o| {
            o.optflag("", "document-private-items", "document private items")
        }),
        stable("test", |o| o.optflag("", "test", "run code examples as tests")),
        stable("test-args", |o| {
            o.optmulti("", "test-args", "arguments to pass to the test runner",
                       "ARGS")
        }),
        stable("target", |o| o.optopt("", "target", "target triple to document", "TRIPLE")),
        stable("markdown-css", |o| {
            o.optmulti("", "markdown-css",
                       "CSS files to include via <link> in a rendered Markdown file",
                       "FILES")
        }),
        stable("html-in-header", |o|  {
            o.optmulti("", "html-in-header",
                       "files to include inline in the <head> section of a rendered Markdown file \
                        or generated documentation",
                       "FILES")
        }),
        stable("html-before-content", |o| {
            o.optmulti("", "html-before-content",
                       "files to include inline between <body> and the content of a rendered \
                        Markdown file or generated documentation",
                       "FILES")
        }),
        stable("html-after-content", |o| {
            o.optmulti("", "html-after-content",
                       "files to include inline between the content and </body> of a rendered \
                        Markdown file or generated documentation",
                       "FILES")
        }),
        unstable("markdown-before-content", |o| {
            o.optmulti("", "markdown-before-content",
                       "files to include inline between <body> and the content of a rendered \
                        Markdown file or generated documentation",
                       "FILES")
        }),
        unstable("markdown-after-content", |o| {
            o.optmulti("", "markdown-after-content",
                       "files to include inline between the content and </body> of a rendered \
                        Markdown file or generated documentation",
                       "FILES")
        }),
        stable("markdown-playground-url", |o| {
            o.optopt("", "markdown-playground-url",
                     "URL to send code snippets to", "URL")
        }),
        stable("markdown-no-toc", |o| {
            o.optflag("", "markdown-no-toc", "don't include table of contents")
        }),
        stable("e", |o| {
            o.optopt("e", "extend-css",
                     "To add some CSS rules with a given file to generate doc with your \
                      own theme. However, your theme might break if the rustdoc's generated HTML \
                      changes, so be careful!", "PATH")
        }),
        unstable("Z", |o| {
            o.optmulti("Z", "",
                       "internal and debugging options (only on nightly build)", "FLAG")
        }),
        stable("sysroot", |o| {
            o.optopt("", "sysroot", "Override the system root", "PATH")
        }),
        unstable("playground-url", |o| {
            o.optopt("", "playground-url",
                     "URL to send code snippets to, may be reset by --markdown-playground-url \
                      or `#![doc(html_playground_url=...)]`",
                     "URL")
        }),
        unstable("display-warnings", |o| {
            o.optflag("", "display-warnings", "to print code warnings when testing doc")
        }),
        unstable("crate-version", |o| {
            o.optopt("", "crate-version", "crate version to print into documentation", "VERSION")
        }),
        unstable("linker", |o| {
            o.optopt("", "linker", "linker used for building executable test code", "PATH")
        }),
        unstable("sort-modules-by-appearance", |o| {
            o.optflag("", "sort-modules-by-appearance", "sort modules by where they appear in the \
                                                         program, rather than alphabetically")
        }),
        unstable("themes", |o| {
            o.optmulti("", "themes",
                       "additional themes which will be added to the generated docs",
                       "FILES")
        }),
        unstable("theme-checker", |o| {
            o.optmulti("", "theme-checker",
                       "check if given theme is valid",
                       "FILES")
        }),
        unstable("resource-suffix", |o| {
            o.optopt("",
                     "resource-suffix",
                     "suffix to add to CSS and JavaScript files, e.g., \"light.css\" will become \
                      \"light-suffix.css\"",
                     "PATH")
        }),
        stable("edition", |o| {
            o.optopt("", "edition",
                     "edition to use when compiling rust code (default: 2015)",
                     "EDITION")
        }),
        stable("color", |o| {
            o.optopt("",
                     "color",
                     "Configure coloring of output:
                                          auto   = colorize, if output goes to a tty (default);
                                          always = always colorize output;
                                          never  = never colorize output",
                     "auto|always|never")
        }),
        stable("error-format", |o| {
            o.optopt("",
                     "error-format",
                     "How errors and other messages are produced",
                     "human|json|short")
        }),
        unstable("disable-minification", |o| {
             o.optflag("",
                       "disable-minification",
                       "Disable minification applied on JS files")
        }),
        stable("warn", |o| {
            o.optmulti("W", "warn", "Set lint warnings", "OPT")
        }),
        stable("allow", |o| {
            o.optmulti("A", "allow", "Set lint allowed", "OPT")
        }),
        stable("deny", |o| {
            o.optmulti("D", "deny", "Set lint denied", "OPT")
        }),
        stable("forbid", |o| {
            o.optmulti("F", "forbid", "Set lint forbidden", "OPT")
        }),
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
             o.optopt("",
                      "index-page",
                      "Markdown file to be used as index page",
                      "PATH")
        }),
        unstable("enable-index-page", |o| {
             o.optflag("",
                       "enable-index-page",
                       "To enable generation of the index page")
        }),
        unstable("static-root-path", |o| {
            o.optopt("",
                     "static-root-path",
                     "Path string to force loading static files from in output pages. \
                      If not set, uses combinations of '../' to reach the documentation root.",
                     "PATH")
        }),
        unstable("disable-per-crate-search", |o| {
            o.optflag("",
                      "disable-per-crate-search",
                      "disables generating the crate selector on the search box")
        }),
        unstable("persist-doctests", |o| {
             o.optopt("",
                       "persist-doctests",
                       "Directory to persist doctest executables into",
                       "PATH")
        }),
        unstable("generate-redirect-pages", |o| {
            o.optflag("",
                      "generate-redirect-pages",
                      "Generate extra pages to support legacy URLs and tool links")
        }),
    ]
}

fn usage(argv0: &str) {
    let mut options = getopts::Options::new();
    for option in opts() {
        (option.apply)(&mut options);
    }
    println!("{}", options.usage(&format!("{} [options] <input>", argv0)));
}

fn main_args(args: &[String]) -> isize {
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
    let options = match config::Options::from_matches(&matches) {
        Ok(opts) => opts,
        Err(code) => return code,
    };

    let diag = core::new_handler(options.error_format,
                                 None,
                                 options.debugging_options.treat_err_as_bug,
                                 options.debugging_options.ui_testing);

    match (options.should_test, options.markdown_input()) {
        (true, true) => return markdown::test(options, &diag),
        (true, false) => return test::run(options),
        (false, true) => return markdown::render(options.input, options.render_options, &diag),
        (false, false) => {}
    }

    // need to move these items separately because we lose them by the time the closure is called,
    // but we can't crates the Handler ahead of time because it's not Send
    let diag_opts = (options.error_format,
                     options.debugging_options.treat_err_as_bug,
                     options.debugging_options.ui_testing);
    rust_input(options, move |out| {
        let Output { krate, passes, renderinfo, renderopts } = out;
        info!("going to format");
        let (error_format, treat_err_as_bug, ui_testing) = diag_opts;
        let diag = core::new_handler(error_format, None, treat_err_as_bug, ui_testing);
        match html::render::run(
            krate,
            renderopts,
            passes.into_iter().collect(),
            renderinfo,
            &diag,
        ) {
            Ok(_) => rustc_driver::EXIT_SUCCESS,
            Err(e) => {
                diag.struct_err(&format!("couldn't generate documentation: {}", e.error))
                    .note(&format!("failed to create or modify \"{}\"", e.file.display()))
                    .emit();
                rustc_driver::EXIT_FAILURE
            }
        }
    })
}

/// Interprets the input file as a rust source file, passing it through the
/// compiler all the way through the analysis passes. The rustdoc output is then
/// generated from the cleaned AST of the crate.
///
/// This form of input will run all of the plug/cleaning passes
fn rust_input<R, F>(options: config::Options, f: F) -> R
where R: 'static + Send,
      F: 'static + Send + FnOnce(Output) -> R
{
    // First, parse the crate and extract all relevant information.
    info!("starting to run rustc");

    let (tx, rx) = channel();

    let result = rustc_driver::monitor(move || syntax::with_globals(move || {
        let crate_name = options.crate_name.clone();
        let crate_version = options.crate_version.clone();
        let (mut krate, renderinfo, renderopts, passes) = core::run_core(options);

        info!("finished with rustc");

        if let Some(name) = crate_name {
            krate.name = name
        }

        krate.version = crate_version;

        tx.send(f(Output {
            krate: krate,
            renderinfo: renderinfo,
            renderopts,
            passes: passes
        })).unwrap();
    }));

    match result {
        Ok(()) => rx.recv().unwrap(),
        Err(_) => panic::resume_unwind(Box::new(errors::FatalErrorMarker)),
    }
}
