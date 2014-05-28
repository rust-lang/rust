// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "rustdoc#0.11.0-pre"]
#![desc = "rustdoc, the Rust documentation extractor"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]

#![feature(globs, struct_variant, managed_boxes, macro_rules, phase)]

extern crate collections;
extern crate debug;
extern crate getopts;
extern crate libc;
#[phase(syntax, link)]
extern crate log;
extern crate rustc;
extern crate serialize;
extern crate sync;
extern crate syntax;
extern crate testing = "test";
extern crate time;

use std::io;
use std::io::{File, MemWriter};
use std::str;
use serialize::{json, Decodable, Encodable};

// reexported from `clean` so it can be easily updated with the mod itself
pub use clean::SCHEMA_VERSION;

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

local_data_key!(pub ctxtkey: @core::DocContext)
local_data_key!(pub analysiskey: core::CrateAnalysis)

type Output = (clean::Crate, Vec<plugins::PluginJson> );

pub fn main() {
    std::os::set_exit_status(main_args(std::os::args().iter()
                                                      .map(|x| x.to_string())
                                                      .collect::<Vec<_>>()
                                                      .as_slice()));
}

pub fn opts() -> Vec<getopts::OptGroup> {
    use getopts::*;
    vec!(
        optflag("h", "help", "show this help message"),
        optflag("", "version", "print rustdoc's version"),
        optopt("r", "input-format", "the input type of the specified file",
               "[rust|json]"),
        optopt("w", "output-format", "the output type to write",
               "[html|json]"),
        optopt("o", "output", "where to place the output", "PATH"),
        optmulti("L", "library-path", "directory to add to crate search path",
                 "DIR"),
        optmulti("", "cfg", "pass a --cfg to rustc", ""),
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
        optmulti("", "markdown-css", "CSS files to include via <link> in a rendered Markdown file",
                 "FILES"),
        optmulti("", "markdown-in-header",
                 "files to include inline in the <head> section of a rendered Markdown file",
                 "FILES"),
        optmulti("", "markdown-before-content",
                 "files to include inline between <body> and the content of a rendered \
                 Markdown file",
                 "FILES"),
        optmulti("", "markdown-after-content",
                 "files to include inline between the content and </body> of a rendered \
                 Markdown file",
                 "FILES")
    )
}

pub fn usage(argv0: &str) {
    println!("{}",
             getopts::usage(format!("{} [options] <input>", argv0).as_slice(),
                            opts().as_slice()));
}

pub fn main_args(args: &[String]) -> int {
    let matches = match getopts::getopts(args.tail(), opts().as_slice()) {
        Ok(m) => m,
        Err(err) => {
            println!("{}", err.to_err_msg());
            return 1;
        }
    };
    if matches.opt_present("h") || matches.opt_present("help") {
        usage(args[0].as_slice());
        return 0;
    } else if matches.opt_present("version") {
        rustc::driver::version("rustdoc");
        return 0;
    }

    if matches.free.len() == 0 {
        println!("expected an input file to act on");
        return 1;
    } if matches.free.len() > 1 {
        println!("only one input file may be specified");
        return 1;
    }
    let input = matches.free.get(0).as_slice();

    let libs = matches.opt_strs("L").iter().map(|s| Path::new(s.as_slice())).collect();

    let test_args = matches.opt_strs("test-args");
    let test_args: Vec<String> = test_args.iter()
                                          .flat_map(|s| s.as_slice().words())
                                          .map(|s| s.to_string())
                                          .collect();

    let should_test = matches.opt_present("test");
    let markdown_input = input.ends_with(".md") || input.ends_with(".markdown");

    let output = matches.opt_str("o").map(|s| Path::new(s));
    let cfgs = matches.opt_strs("cfg");

    match (should_test, markdown_input) {
        (true, true) => {
            return markdown::test(input,
                                  libs,
                                  test_args.move_iter().collect())
        }
        (true, false) => {
            return test::run(input,
                             cfgs.move_iter()
                                 .map(|x| x.to_string())
                                 .collect(),
                             libs,
                             test_args)
        }
        (false, true) => return markdown::render(input, output.unwrap_or(Path::new("doc")),
                                                 &matches),
        (false, false) => {}
    }

    if matches.opt_strs("passes").as_slice() == &["list".to_string()] {
        println!("Available passes for running rustdoc:");
        for &(name, _, description) in PASSES.iter() {
            println!("{:>20s} - {}", name, description);
        }
        println!("{}", "\nDefault passes for rustdoc:"); // FIXME: #9970
        for &name in DEFAULT_PASSES.iter() {
            println!("{:>20s}", name);
        }
        return 0;
    }

    let (krate, res) = match acquire_input(input, &matches) {
        Ok(pair) => pair,
        Err(s) => {
            println!("input error: {}", s);
            return 1;
        }
    };

    info!("going to format");
    let started = time::precise_time_ns();
    match matches.opt_str("w").as_ref().map(|s| s.as_slice()) {
        Some("html") | None => {
            match html::render::run(krate, output.unwrap_or(Path::new("doc"))) {
                Ok(()) => {}
                Err(e) => fail!("failed to generate documentation: {}", e),
            }
        }
        Some("json") => {
            match json_output(krate, res, output.unwrap_or(Path::new("doc.json"))) {
                Ok(()) => {}
                Err(e) => fail!("failed to write json: {}", e),
            }
        }
        Some(s) => {
            println!("unknown output format: {}", s);
            return 1;
        }
    }
    let ended = time::precise_time_ns();
    info!("Took {:.03f}s", (ended as f64 - started as f64) / 1e9f64);

    return 0;
}

/// Looks inside the command line arguments to extract the relevant input format
/// and files and then generates the necessary rustdoc output for formatting.
fn acquire_input(input: &str,
                 matches: &getopts::Matches) -> Result<Output, String> {
    match matches.opt_str("r").as_ref().map(|s| s.as_slice()) {
        Some("rust") => Ok(rust_input(input, matches)),
        Some("json") => json_input(input),
        Some(s) => Err(format!("unknown input format: {}", s)),
        None => {
            if input.ends_with(".json") {
                json_input(input)
            } else {
                Ok(rust_input(input, matches))
            }
        }
    }
}

/// Interprets the input file as a rust source file, passing it through the
/// compiler all the way through the analysis passes. The rustdoc output is then
/// generated from the cleaned AST of the crate.
///
/// This form of input will run all of the plug/cleaning passes
fn rust_input(cratefile: &str, matches: &getopts::Matches) -> Output {
    let mut default_passes = !matches.opt_present("no-defaults");
    let mut passes = matches.opt_strs("passes");
    let mut plugins = matches.opt_strs("plugins")
                             .move_iter()
                             .map(|x| x.to_string())
                             .collect::<Vec<_>>();

    // First, parse the crate and extract all relevant information.
    let libs: Vec<Path> = matches.opt_strs("L")
                                 .iter()
                                 .map(|s| Path::new(s.as_slice()))
                                 .collect();
    let cfgs = matches.opt_strs("cfg");
    let cr = Path::new(cratefile);
    info!("starting to run rustc");
    let (krate, analysis) = std::task::try(proc() {
        let cr = cr;
        core::run_core(libs.move_iter().map(|x| x.clone()).collect(),
                       cfgs.move_iter().map(|x| x.to_string()).collect(),
                       &cr)
    }).map_err(|boxed_any|format!("{:?}", boxed_any)).unwrap();
    info!("finished with rustc");
    analysiskey.replace(Some(analysis));

    // Process all of the crate attributes, extracting plugin metadata along
    // with the passes which we are supposed to run.
    match krate.module.get_ref().doc_list() {
        Some(nested) => {
            for inner in nested.iter() {
                match *inner {
                    clean::Word(ref x)
                            if "no_default_passes" == x.as_slice() => {
                        default_passes = false;
                    }
                    clean::NameValue(ref x, ref value)
                            if "passes" == x.as_slice() => {
                        for pass in value.as_slice().words() {
                            passes.push(pass.to_string());
                        }
                    }
                    clean::NameValue(ref x, ref value)
                            if "plugins" == x.as_slice() => {
                        for p in value.as_slice().words() {
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
            passes.unshift(name.to_string());
        }
    }

    // Load all plugins/passes into a PluginManager
    let path = matches.opt_str("plugin-path")
                      .unwrap_or("/tmp/rustdoc/plugins".to_string());
    let mut pm = plugins::PluginManager::new(Path::new(path));
    for pass in passes.iter() {
        let plugin = match PASSES.iter()
                                 .position(|&(p, _, _)| {
                                     p == pass.as_slice()
                                 }) {
            Some(i) => PASSES[i].val1(),
            None => {
                error!("unknown pass {}, skipping", *pass);
                continue
            },
        };
        pm.add_plugin(plugin);
    }
    info!("loading plugins...");
    for pname in plugins.move_iter() {
        pm.load_plugin(pname);
    }

    // Run everything!
    info!("Executing passes/plugins");
    return pm.run_plugins(krate);
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
        Err(s) => Err(s.to_str().to_string()),
        Ok(json::Object(obj)) => {
            let mut obj = obj;
            // Make sure the schema is what we expect
            match obj.pop(&"schema".to_string()) {
                Some(json::String(version)) => {
                    if version.as_slice() != SCHEMA_VERSION {
                        return Err(format!(
                                "sorry, but I only understand version {}",
                                SCHEMA_VERSION))
                    }
                }
                Some(..) => return Err("malformed json".to_string()),
                None => return Err("expected a schema version".to_string()),
            }
            let krate = match obj.pop(&"crate".to_string()) {
                Some(json) => {
                    let mut d = json::Decoder::new(json);
                    Decodable::decode(&mut d).unwrap()
                }
                None => return Err("malformed json".to_string()),
            };
            // FIXME: this should read from the "plugins" field, but currently
            //      Json doesn't implement decodable...
            let plugin_output = Vec::new();
            Ok((krate, plugin_output))
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
               dst: Path) -> io::IoResult<()> {
    // {
    //   "schema": version,
    //   "crate": { parsed crate ... },
    //   "plugins": { output of plugins ... }
    // }
    let mut json = box collections::TreeMap::new();
    json.insert("schema".to_string(),
                json::String(SCHEMA_VERSION.to_string()));
    let plugins_json = box res.move_iter()
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
    let crate_json_str = {
        let mut w = MemWriter::new();
        {
            let mut encoder = json::Encoder::new(&mut w as &mut io::Writer);
            krate.encode(&mut encoder).unwrap();
        }
        str::from_utf8(w.unwrap().as_slice()).unwrap().to_string()
    };
    let crate_json = match json::from_str(crate_json_str.as_slice()) {
        Ok(j) => j,
        Err(e) => fail!("Rust generated JSON is invalid: {:?}", e)
    };

    json.insert("crate".to_string(), crate_json);
    json.insert("plugins".to_string(), json::Object(plugins_json));

    let mut file = try!(File::create(&dst));
    try!(json::Object(json).to_writer(&mut file));
    Ok(())
}
