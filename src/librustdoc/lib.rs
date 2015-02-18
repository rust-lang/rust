// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustdoc"]
#![unstable(feature = "rustdoc")]
#![staged_api]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(collections)]
#![feature(core)]
#![feature(env)]
#![feature(int_uint)]
#![feature(old_io)]
#![feature(libc)]
#![feature(os)]
#![feature(old_path)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(std_misc)]
#![feature(test)]
#![feature(unicode)]
#![feature(str_words)]

extern crate arena;
extern crate getopts;
extern crate libc;
extern crate rustc;
extern crate rustc_trans;
extern crate rustc_driver;
extern crate rustc_resolve;
extern crate serialize;
extern crate syntax;
extern crate "test" as testing;
#[macro_use] extern crate log;

extern crate "serialize" as rustc_serialize; // used by deriving

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::old_io::File;
use std::old_io;
use std::rc::Rc;
use std::sync::mpsc::channel;
use externalfiles::ExternalHtml;
use serialize::Decodable;
use serialize::json::{self, Json};
use rustc::session::search_paths::SearchPaths;

// reexported from `clean` so it can be easily updated with the mod itself
pub use clean::SCHEMA_VERSION;

#[macro_use]
pub mod externalfiles;

pub mod clean;
pub mod core;
pub mod doctree;
pub mod fold;
pub mod html {
    pub mod highlight;
    pub mod escape;
    pub mod item_type;
    pub mod format;
    pub mod layout;
    pub mod markdown;
    pub mod render;
    pub mod toc;
}
pub mod markdown;
pub mod passes;
pub mod plugins;
pub mod stability_summary;
pub mod visit_ast;
pub mod test;
mod flock;

type Pass = (&'static str,                                      // name
             fn(clean::Crate) -> plugins::PluginResult,         // fn
             &'static str);                                     // description

static PASSES: &'static [Pass] = &[
    ("strip-hidden", passes::strip_hidden,
     "strips all doc(hidden) items from the output"),
    ("unindent-comments", passes::unindent_comments,
     "removes excess indentation on comments in order for markdown to like it"),
    ("collapse-docs", passes::collapse_docs,
     "concatenates all document attributes into one document attribute"),
    ("strip-private", passes::strip_private,
     "strips all private items from a crate which cannot be seen externally"),
];

static DEFAULT_PASSES: &'static [&'static str] = &[
    "strip-hidden",
    "strip-private",
    "collapse-docs",
    "unindent-comments",
];

thread_local!(pub static ANALYSISKEY: Rc<RefCell<Option<core::CrateAnalysis>>> = {
    Rc::new(RefCell::new(None))
});

struct Output {
    krate: clean::Crate,
    json_plugins: Vec<plugins::PluginJson>,
    passes: Vec<String>,
}

pub fn main() {
    const STACK_SIZE: usize = 32000000; // 32MB
    let res = std::thread::Builder::new().stack_size(STACK_SIZE).scoped(move || {
        let s = env::args().collect::<Vec<_>>();
        main_args(&s)
    }).unwrap().join();
    env::set_exit_status(res as i32);
}

pub fn opts() -> Vec<getopts::OptGroup> {
    use getopts::*;
    vec!(
        optflag("h", "help", "show this help message"),
        optflag("V", "version", "print rustdoc's version"),
        optflag("v", "verbose", "use verbose output"),
        optopt("r", "input-format", "the input type of the specified file",
               "[rust|json]"),
        optopt("w", "output-format", "the output type to write",
               "[html|json]"),
        optopt("o", "output", "where to place the output", "PATH"),
        optopt("", "crate-name", "specify the name of this crate", "NAME"),
        optmulti("L", "library-path", "directory to add to crate search path",
                 "DIR"),
        optmulti("", "cfg", "pass a --cfg to rustc", ""),
        optmulti("", "extern", "pass an --extern to rustc", "NAME=PATH"),
        optmulti("", "plugin-path", "directory to load plugins from", "DIR"),
        optmulti("", "passes", "space separated list of passes to also run, a \
                                value of `list` will print available passes",
                 "PASSES"),
        optmulti("", "plugins", "space separated list of plugins to also load",
                 "PLUGINS"),
        optflag("", "no-defaults", "don't run the default passes"),
        optflag("", "test", "run code examples as tests"),
        optmulti("", "test-args", "arguments to pass to the test runner",
                 "ARGS"),
        optopt("", "target", "target triple to document", "TRIPLE"),
        optmulti("", "markdown-css", "CSS files to include via <link> in a rendered Markdown file",
                 "FILES"),
        optmulti("", "html-in-header",
                 "files to include inline in the <head> section of a rendered Markdown file \
                 or generated documentation",
                 "FILES"),
        optmulti("", "html-before-content",
                 "files to include inline between <body> and the content of a rendered \
                 Markdown file or generated documentation",
                 "FILES"),
        optmulti("", "html-after-content",
                 "files to include inline between the content and </body> of a rendered \
                 Markdown file or generated documentation",
                 "FILES"),
        optopt("", "markdown-playground-url",
               "URL to send code snippets to", "URL"),
        optflag("", "markdown-no-toc", "don't include table of contents")
    )
}

pub fn usage(argv0: &str) {
    println!("{}",
             getopts::usage(&format!("{} [options] <input>", argv0),
                            &opts()));
}

pub fn main_args(args: &[String]) -> int {
    let matches = match getopts::getopts(args.tail(), &opts()) {
        Ok(m) => m,
        Err(err) => {
            println!("{}", err);
            return 1;
        }
    };
    if matches.opt_present("h") || matches.opt_present("help") {
        usage(&args[0]);
        return 0;
    } else if matches.opt_present("version") {
        rustc_driver::version("rustdoc", &matches);
        return 0;
    }

    if matches.opt_strs("passes") == ["list"] {
        println!("Available passes for running rustdoc:");
        for &(name, _, description) in PASSES {
            println!("{:>20} - {}", name, description);
        }
        println!("{}", "\nDefault passes for rustdoc:"); // FIXME: #9970
        for &name in DEFAULT_PASSES {
            println!("{:>20}", name);
        }
        return 0;
    }

    if matches.free.len() == 0 {
        println!("expected an input file to act on");
        return 1;
    } if matches.free.len() > 1 {
        println!("only one input file may be specified");
        return 1;
    }
    let input = &matches.free[0];

    let mut libs = SearchPaths::new();
    for s in &matches.opt_strs("L") {
        libs.add_path(s);
    }
    let externs = match parse_externs(&matches) {
        Ok(ex) => ex,
        Err(err) => {
            println!("{}", err);
            return 1;
        }
    };

    let test_args = matches.opt_strs("test-args");
    let test_args: Vec<String> = test_args.iter()
                                          .flat_map(|s| s.words())
                                          .map(|s| s.to_string())
                                          .collect();

    let should_test = matches.opt_present("test");
    let markdown_input = input.ends_with(".md") || input.ends_with(".markdown");

    let output = matches.opt_str("o").map(|s| Path::new(s));
    let cfgs = matches.opt_strs("cfg");

    let external_html = match ExternalHtml::load(
            &matches.opt_strs("html-in-header"),
            &matches.opt_strs("html-before-content"),
            &matches.opt_strs("html-after-content")) {
        Some(eh) => eh,
        None => return 3
    };
    let crate_name = matches.opt_str("crate-name");

    match (should_test, markdown_input) {
        (true, true) => {
            return markdown::test(input, libs, externs, test_args)
        }
        (true, false) => {
            return test::run(input, cfgs, libs, externs, test_args, crate_name)
        }
        (false, true) => return markdown::render(input, output.unwrap_or(Path::new("doc")),
                                                 &matches, &external_html,
                                                 !matches.opt_present("markdown-no-toc")),
        (false, false) => {}
    }

    let out = match acquire_input(input, externs, &matches) {
        Ok(out) => out,
        Err(s) => {
            println!("input error: {}", s);
            return 1;
        }
    };
    let Output { krate, json_plugins, passes, } = out;
    info!("going to format");
    match matches.opt_str("w").as_ref().map(|s| &**s) {
        Some("html") | None => {
            match html::render::run(krate, &external_html, output.unwrap_or(Path::new("doc")),
                                    passes.into_iter().collect()) {
                Ok(()) => {}
                Err(e) => panic!("failed to generate documentation: {}", e),
            }
        }
        Some("json") => {
            match json_output(krate, json_plugins,
                              output.unwrap_or(Path::new("doc.json"))) {
                Ok(()) => {}
                Err(e) => panic!("failed to write json: {}", e),
            }
        }
        Some(s) => {
            println!("unknown output format: {}", s);
            return 1;
        }
    }

    return 0;
}

/// Looks inside the command line arguments to extract the relevant input format
/// and files and then generates the necessary rustdoc output for formatting.
fn acquire_input(input: &str,
                 externs: core::Externs,
                 matches: &getopts::Matches) -> Result<Output, String> {
    match matches.opt_str("r").as_ref().map(|s| &**s) {
        Some("rust") => Ok(rust_input(input, externs, matches)),
        Some("json") => json_input(input),
        Some(s) => Err(format!("unknown input format: {}", s)),
        None => {
            if input.ends_with(".json") {
                json_input(input)
            } else {
                Ok(rust_input(input, externs, matches))
            }
        }
    }
}

/// Extracts `--extern CRATE=PATH` arguments from `matches` and
/// returns a `HashMap` mapping crate names to their paths or else an
/// error message.
fn parse_externs(matches: &getopts::Matches) -> Result<core::Externs, String> {
    let mut externs = HashMap::new();
    for arg in &matches.opt_strs("extern") {
        let mut parts = arg.splitn(1, '=');
        let name = match parts.next() {
            Some(s) => s,
            None => {
                return Err("--extern value must not be empty".to_string());
            }
        };
        let location = match parts.next() {
            Some(s) => s,
            None => {
                return Err("--extern value must be of the format `foo=bar`".to_string());
            }
        };
        let name = name.to_string();
        let locs = externs.entry(name).get().unwrap_or_else(
            |vacant_entry| vacant_entry.insert(Vec::with_capacity(1)));
        locs.push(location.to_string());
    }
    Ok(externs)
}

/// Interprets the input file as a rust source file, passing it through the
/// compiler all the way through the analysis passes. The rustdoc output is then
/// generated from the cleaned AST of the crate.
///
/// This form of input will run all of the plug/cleaning passes
fn rust_input(cratefile: &str, externs: core::Externs, matches: &getopts::Matches) -> Output {
    let mut default_passes = !matches.opt_present("no-defaults");
    let mut passes = matches.opt_strs("passes");
    let mut plugins = matches.opt_strs("plugins");

    // First, parse the crate and extract all relevant information.
    let mut paths = SearchPaths::new();
    for s in &matches.opt_strs("L") {
        paths.add_path(s);
    }
    let cfgs = matches.opt_strs("cfg");
    let triple = matches.opt_str("target");

    let cr = Path::new(cratefile);
    info!("starting to run rustc");

    let (tx, rx) = channel();
    std::thread::spawn(move || {
        use rustc::session::config::Input;

        let cr = cr;
        tx.send(core::run_core(paths, cfgs, externs, Input::File(cr), triple)).unwrap();
    }).join().map_err(|_| "rustc failed").unwrap();
    let (mut krate, analysis) = rx.recv().unwrap();
    info!("finished with rustc");
    let mut analysis = Some(analysis);
    ANALYSISKEY.with(|s| {
        *s.borrow_mut() = analysis.take();
    });

    match matches.opt_str("crate-name") {
        Some(name) => krate.name = name,
        None => {}
    }

    // Process all of the crate attributes, extracting plugin metadata along
    // with the passes which we are supposed to run.
    match krate.module.as_ref().unwrap().doc_list() {
        Some(nested) => {
            for inner in nested {
                match *inner {
                    clean::Word(ref x)
                            if "no_default_passes" == *x => {
                        default_passes = false;
                    }
                    clean::NameValue(ref x, ref value)
                            if "passes" == *x => {
                        for pass in value.words() {
                            passes.push(pass.to_string());
                        }
                    }
                    clean::NameValue(ref x, ref value)
                            if "plugins" == *x => {
                        for p in value.words() {
                            plugins.push(p.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        None => {}
    }
    if default_passes {
        for name in DEFAULT_PASSES.iter().rev() {
            passes.insert(0, name.to_string());
        }
    }

    // Load all plugins/passes into a PluginManager
    let path = matches.opt_str("plugin-path")
                      .unwrap_or("/tmp/rustdoc/plugins".to_string());
    let mut pm = plugins::PluginManager::new(Path::new(path));
    for pass in &passes {
        let plugin = match PASSES.iter()
                                 .position(|&(p, _, _)| {
                                     p == *pass
                                 }) {
            Some(i) => PASSES[i].1,
            None => {
                error!("unknown pass {}, skipping", *pass);
                continue
            },
        };
        pm.add_plugin(plugin);
    }
    info!("loading plugins...");
    for pname in plugins {
        pm.load_plugin(pname);
    }

    // Run everything!
    info!("Executing passes/plugins");
    let (krate, json) = pm.run_plugins(krate);
    return Output { krate: krate, json_plugins: json, passes: passes, };
}

/// This input format purely deserializes the json output file. No passes are
/// run over the deserialized output.
fn json_input(input: &str) -> Result<Output, String> {
    let mut input = match File::open(&Path::new(input)) {
        Ok(f) => f,
        Err(e) => {
            return Err(format!("couldn't open {}: {}", input, e))
        }
    };
    match json::from_reader(&mut input) {
        Err(s) => Err(format!("{:?}", s)),
        Ok(Json::Object(obj)) => {
            let mut obj = obj;
            // Make sure the schema is what we expect
            match obj.remove(&"schema".to_string()) {
                Some(Json::String(version)) => {
                    if version != SCHEMA_VERSION {
                        return Err(format!(
                                "sorry, but I only understand version {}",
                                SCHEMA_VERSION))
                    }
                }
                Some(..) => return Err("malformed json".to_string()),
                None => return Err("expected a schema version".to_string()),
            }
            let krate = match obj.remove(&"crate".to_string()) {
                Some(json) => {
                    let mut d = json::Decoder::new(json);
                    Decodable::decode(&mut d).unwrap()
                }
                None => return Err("malformed json".to_string()),
            };
            // FIXME: this should read from the "plugins" field, but currently
            //      Json doesn't implement decodable...
            let plugin_output = Vec::new();
            Ok(Output { krate: krate, json_plugins: plugin_output, passes: Vec::new(), })
        }
        Ok(..) => {
            Err("malformed json input: expected an object at the \
                 top".to_string())
        }
    }
}

/// Outputs the crate/plugin json as a giant json blob at the specified
/// destination.
fn json_output(krate: clean::Crate, res: Vec<plugins::PluginJson> ,
               dst: Path) -> old_io::IoResult<()> {
    // {
    //   "schema": version,
    //   "crate": { parsed crate ... },
    //   "plugins": { output of plugins ... }
    // }
    let mut json = std::collections::BTreeMap::new();
    json.insert("schema".to_string(), Json::String(SCHEMA_VERSION.to_string()));
    let plugins_json = res.into_iter()
                          .filter_map(|opt| {
                              match opt {
                                  None => None,
                                  Some((string, json)) => {
                                      Some((string.to_string(), json))
                                  }
                              }
                          }).collect();

    // FIXME #8335: yuck, Rust -> str -> JSON round trip! No way to .encode
    // straight to the Rust JSON representation.
    let crate_json_str = format!("{}", json::as_json(&krate));
    let crate_json = match json::from_str(&crate_json_str) {
        Ok(j) => j,
        Err(e) => panic!("Rust generated JSON is invalid: {:?}", e)
    };

    json.insert("crate".to_string(), crate_json);
    json.insert("plugins".to_string(), Json::Object(plugins_json));

    let mut file = try!(File::create(&dst));
    write!(&mut file, "{}", Json::Object(json))
}
