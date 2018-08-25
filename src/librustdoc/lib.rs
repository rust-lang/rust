// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/")]

#![feature(rustc_private)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![cfg_attr(not(stage0), feature(nll))]
#![feature(set_stdio)]
#![feature(slice_sort_by_cached_key)]
#![feature(test)]
#![feature(vec_remove_item)]
#![feature(ptr_offset_from)]
#![feature(crate_visibility_modifier)]
#![feature(const_fn)]

#![recursion_limit="256"]

extern crate arena;
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
extern crate serialize;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate test as testing;
#[macro_use] extern crate log;
extern crate rustc_errors as errors;
extern crate pulldown_cmark;
extern crate tempfile;
extern crate minifier;

extern crate serialize as rustc_serialize; // used by deriving

use errors::ColorConfig;

use std::collections::{BTreeMap, BTreeSet};
use std::default::Default;
use std::env;
use std::panic;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::mpsc::channel;

use syntax::edition::Edition;
use externalfiles::ExternalHtml;
use rustc::session::{early_warn, early_error};
use rustc::session::search_paths::SearchPaths;
use rustc::session::config::{ErrorOutputType, RustcOptGroup, Externs, CodegenOptions};
use rustc::session::config::{nightly_options, build_codegen_options};
use rustc_target::spec::TargetTriple;
use rustc::session::config::get_cmd_lint_options;

#[macro_use]
mod externalfiles;

mod clean;
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
                     "suffix to add to CSS and JavaScript files, e.g. \"light.css\" will become \
                      \"light-suffix.css\"",
                     "PATH")
        }),
        unstable("edition", |o| {
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
    // Check for unstable options.
    nightly_options::check_nightly_options(&matches, &opts());

    if matches.opt_present("h") || matches.opt_present("help") {
        usage("rustdoc");
        return 0;
    } else if matches.opt_present("version") {
        rustc_driver::version("rustdoc", &matches);
        return 0;
    }

    if matches.opt_strs("passes") == ["list"] {
        println!("Available passes for running rustdoc:");
        for pass in passes::PASSES {
            println!("{:>20} - {}", pass.name(), pass.description());
        }
        println!("\nDefault passes for rustdoc:");
        for &name in passes::DEFAULT_PASSES {
            println!("{:>20}", name);
        }
        println!("\nPasses run with `--document-private-items`:");
        for &name in passes::DEFAULT_PRIVATE_PASSES {
            println!("{:>20}", name);
        }
        return 0;
    }

    let color = match matches.opt_str("color").as_ref().map(|s| &s[..]) {
        Some("auto") => ColorConfig::Auto,
        Some("always") => ColorConfig::Always,
        Some("never") => ColorConfig::Never,
        None => ColorConfig::Auto,
        Some(arg) => {
            early_error(ErrorOutputType::default(),
                        &format!("argument for --color must be `auto`, `always` or `never` \
                                  (instead was `{}`)", arg));
        }
    };
    let error_format = match matches.opt_str("error-format").as_ref().map(|s| &s[..]) {
        Some("human") => ErrorOutputType::HumanReadable(color),
        Some("json") => ErrorOutputType::Json(false),
        Some("pretty-json") => ErrorOutputType::Json(true),
        Some("short") => ErrorOutputType::Short(color),
        None => ErrorOutputType::HumanReadable(color),
        Some(arg) => {
            early_error(ErrorOutputType::default(),
                        &format!("argument for --error-format must be `human`, `json` or \
                                  `short` (instead was `{}`)", arg));
        }
    };

    let diag = core::new_handler(error_format, None);

    // check for deprecated options
    check_deprecated_options(&matches, &diag);

    let to_check = matches.opt_strs("theme-checker");
    if !to_check.is_empty() {
        let paths = theme::load_css_paths(include_bytes!("html/static/themes/light.css"));
        let mut errors = 0;

        println!("rustdoc: [theme-checker] Starting tests!");
        for theme_file in to_check.iter() {
            print!(" - Checking \"{}\"...", theme_file);
            let (success, differences) = theme::test_theme_against(theme_file, &paths, &diag);
            if !differences.is_empty() || !success {
                println!(" FAILED");
                errors += 1;
                if !differences.is_empty() {
                    println!("{}", differences.join("\n"));
                }
            } else {
                println!(" OK");
            }
        }
        if errors != 0 {
            return 1;
        }
        return 0;
    }

    if matches.free.is_empty() {
        diag.struct_err("missing file operand").emit();
        return 1;
    }
    if matches.free.len() > 1 {
        diag.struct_err("too many file operands").emit();
        return 1;
    }
    let input = &matches.free[0];

    let mut libs = SearchPaths::new();
    for s in &matches.opt_strs("L") {
        libs.add_path(s, error_format);
    }
    let externs = match parse_externs(&matches) {
        Ok(ex) => ex,
        Err(err) => {
            diag.struct_err(&err.to_string()).emit();
            return 1;
        }
    };

    let test_args = matches.opt_strs("test-args");
    let test_args: Vec<String> = test_args.iter()
                                          .flat_map(|s| s.split_whitespace())
                                          .map(|s| s.to_string())
                                          .collect();

    let should_test = matches.opt_present("test");
    let markdown_input = Path::new(input).extension()
        .map_or(false, |e| e == "md" || e == "markdown");

    let output = matches.opt_str("o").map(|s| PathBuf::from(&s));
    let css_file_extension = matches.opt_str("e").map(|s| PathBuf::from(&s));
    let cfgs = matches.opt_strs("cfg");

    if let Some(ref p) = css_file_extension {
        if !p.is_file() {
            diag.struct_err("option --extend-css argument must be a file").emit();
            return 1;
        }
    }

    let mut themes = Vec::new();
    if matches.opt_present("themes") {
        let paths = theme::load_css_paths(include_bytes!("html/static/themes/light.css"));

        for (theme_file, theme_s) in matches.opt_strs("themes")
                                            .iter()
                                            .map(|s| (PathBuf::from(&s), s.to_owned())) {
            if !theme_file.is_file() {
                diag.struct_err("option --themes arguments must all be files").emit();
                return 1;
            }
            let (success, ret) = theme::test_theme_against(&theme_file, &paths, &diag);
            if !success || !ret.is_empty() {
                diag.struct_err(&format!("invalid theme: \"{}\"", theme_s))
                    .help("check what's wrong with the --theme-checker option")
                    .emit();
                return 1;
            }
            themes.push(theme_file);
        }
    }

    let mut id_map = html::markdown::IdMap::new();
    id_map.populate(html::render::initial_ids());
    let external_html = match ExternalHtml::load(
            &matches.opt_strs("html-in-header"),
            &matches.opt_strs("html-before-content"),
            &matches.opt_strs("html-after-content"),
            &matches.opt_strs("markdown-before-content"),
            &matches.opt_strs("markdown-after-content"), &diag, &mut id_map) {
        Some(eh) => eh,
        None => return 3,
    };
    let crate_name = matches.opt_str("crate-name");
    let playground_url = matches.opt_str("playground-url");
    let maybe_sysroot = matches.opt_str("sysroot").map(PathBuf::from);
    let display_warnings = matches.opt_present("display-warnings");
    let linker = matches.opt_str("linker").map(PathBuf::from);
    let sort_modules_alphabetically = !matches.opt_present("sort-modules-by-appearance");
    let resource_suffix = matches.opt_str("resource-suffix");
    let enable_minification = !matches.opt_present("disable-minification");

    let edition = matches.opt_str("edition").unwrap_or("2015".to_string());
    let edition = match edition.parse() {
        Ok(e) => e,
        Err(_) => {
            diag.struct_err("could not parse edition").emit();
            return 1;
        }
    };

    let cg = build_codegen_options(&matches, ErrorOutputType::default());

    match (should_test, markdown_input) {
        (true, true) => {
            return markdown::test(input, cfgs, libs, externs, test_args, maybe_sysroot,
                                  display_warnings, linker, edition, cg, &diag)
        }
        (true, false) => {
            return test::run(Path::new(input), cfgs, libs, externs, test_args, crate_name,
                             maybe_sysroot, display_warnings, linker, edition, cg)
        }
        (false, true) => return markdown::render(Path::new(input),
                                                 output.unwrap_or(PathBuf::from("doc")),
                                                 &matches, &external_html,
                                                 !matches.opt_present("markdown-no-toc"), &diag),
        (false, false) => {}
    }

    let output_format = matches.opt_str("w");

    let res = acquire_input(PathBuf::from(input), externs, edition, cg, &matches, error_format,
                            move |out| {
        let Output { krate, passes, renderinfo } = out;
        let diag = core::new_handler(error_format, None);
        info!("going to format");
        match output_format.as_ref().map(|s| &**s) {
            Some("html") | None => {
                html::render::run(krate, &external_html, playground_url,
                                  output.unwrap_or(PathBuf::from("doc")),
                                  resource_suffix.unwrap_or(String::new()),
                                  passes.into_iter().collect(),
                                  css_file_extension,
                                  renderinfo,
                                  sort_modules_alphabetically,
                                  themes,
                                  enable_minification, id_map)
                    .expect("failed to generate documentation");
                0
            }
            Some(s) => {
                diag.struct_err(&format!("unknown output format: {}", s)).emit();
                1
            }
        }
    });
    res.unwrap_or_else(|s| {
        diag.struct_err(&format!("input error: {}", s)).emit();
        1
    })
}

/// Looks inside the command line arguments to extract the relevant input format
/// and files and then generates the necessary rustdoc output for formatting.
fn acquire_input<R, F>(input: PathBuf,
                       externs: Externs,
                       edition: Edition,
                       cg: CodegenOptions,
                       matches: &getopts::Matches,
                       error_format: ErrorOutputType,
                       f: F)
                       -> Result<R, String>
where R: 'static + Send, F: 'static + Send + FnOnce(Output) -> R {
    match matches.opt_str("r").as_ref().map(|s| &**s) {
        Some("rust") => Ok(rust_input(input, externs, edition, cg, matches, error_format, f)),
        Some(s) => Err(format!("unknown input format: {}", s)),
        None => Ok(rust_input(input, externs, edition, cg, matches, error_format, f))
    }
}

/// Extracts `--extern CRATE=PATH` arguments from `matches` and
/// returns a map mapping crate names to their paths or else an
/// error message.
fn parse_externs(matches: &getopts::Matches) -> Result<Externs, String> {
    let mut externs: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
    for arg in &matches.opt_strs("extern") {
        let mut parts = arg.splitn(2, '=');
        let name = parts.next().ok_or("--extern value must not be empty".to_string())?;
        let location = parts.next()
                                 .ok_or("--extern value must be of the format `foo=bar`"
                                    .to_string())?;
        let name = name.to_string();
        externs.entry(name).or_default().insert(location.to_string());
    }
    Ok(Externs::new(externs))
}

/// Interprets the input file as a rust source file, passing it through the
/// compiler all the way through the analysis passes. The rustdoc output is then
/// generated from the cleaned AST of the crate.
///
/// This form of input will run all of the plug/cleaning passes
fn rust_input<R, F>(cratefile: PathBuf,
                    externs: Externs,
                    edition: Edition,
                    cg: CodegenOptions,
                    matches: &getopts::Matches,
                    error_format: ErrorOutputType,
                    f: F) -> R
where R: 'static + Send,
      F: 'static + Send + FnOnce(Output) -> R
{
    let default_passes = if matches.opt_present("no-defaults") {
        passes::DefaultPassOption::None
    } else if matches.opt_present("document-private-items") {
        passes::DefaultPassOption::Private
    } else {
        passes::DefaultPassOption::Default
    };

    let manual_passes = matches.opt_strs("passes");
    let plugins = matches.opt_strs("plugins");

    // First, parse the crate and extract all relevant information.
    let mut paths = SearchPaths::new();
    for s in &matches.opt_strs("L") {
        paths.add_path(s, ErrorOutputType::default());
    }
    let cfgs = matches.opt_strs("cfg");
    let triple = matches.opt_str("target").map(|target| {
        if target.ends_with(".json") {
            TargetTriple::TargetPath(PathBuf::from(target))
        } else {
            TargetTriple::TargetTriple(target)
        }
    });
    let maybe_sysroot = matches.opt_str("sysroot").map(PathBuf::from);
    let crate_name = matches.opt_str("crate-name");
    let crate_version = matches.opt_str("crate-version");
    let plugin_path = matches.opt_str("plugin-path");

    info!("starting to run rustc");
    let display_warnings = matches.opt_present("display-warnings");

    let force_unstable_if_unmarked = matches.opt_strs("Z").iter().any(|x| {
        *x == "force-unstable-if-unmarked"
    });

    let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(matches, error_format);

    let (tx, rx) = channel();

    let result = rustc_driver::monitor(move || syntax::with_globals(move || {
        use rustc::session::config::Input;

        let (mut krate, renderinfo, passes) =
            core::run_core(paths, cfgs, externs, Input::File(cratefile), triple, maybe_sysroot,
                           display_warnings, crate_name.clone(),
                           force_unstable_if_unmarked, edition, cg, error_format,
                           lint_opts, lint_cap, describe_lints, manual_passes, default_passes);

        info!("finished with rustc");

        if let Some(name) = crate_name {
            krate.name = name
        }

        krate.version = crate_version;

        if !plugins.is_empty() {
            eprintln!("WARNING: --plugins no longer functions; see CVE-2018-1000622");
        }

        if !plugin_path.is_none() {
            eprintln!("WARNING: --plugin-path no longer functions; see CVE-2018-1000622");
        }

        info!("Executing passes");

        for pass in &passes {
            // determine if we know about this pass
            let pass = match passes::find_pass(pass) {
                Some(pass) => if let Some(pass) = pass.late_fn() {
                    pass
                } else {
                    // not a late pass, but still valid so don't report the error
                    continue
                }
                None => {
                    error!("unknown pass {}, skipping", *pass);

                    continue
                },
            };

            // run it
            krate = pass(krate);
        }

        tx.send(f(Output { krate: krate, renderinfo: renderinfo, passes: passes })).unwrap();
    }));

    match result {
        Ok(()) => rx.recv().unwrap(),
        Err(_) => panic::resume_unwind(Box::new(errors::FatalErrorMarker)),
    }
}

/// Prints deprecation warnings for deprecated options
fn check_deprecated_options(matches: &getopts::Matches, diag: &errors::Handler) {
    let deprecated_flags = [
       "input-format",
       "output-format",
       "no-defaults",
       "passes",
    ];

    for flag in deprecated_flags.into_iter() {
        if matches.opt_present(flag) {
            let mut err = diag.struct_warn(&format!("the '{}' flag is considered deprecated",
                                                    flag));
            err.warn("please see https://github.com/rust-lang/rust/issues/44136");

            if *flag == "no-defaults" {
                err.help("you may want to use --document-private-items");
            }

            err.emit();
        }
    }
}
