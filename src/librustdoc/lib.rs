// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id = "rustdoc#0.10-pre"];
#[desc = "rustdoc, the Rust documentation extractor"];
#[license = "MIT/ASL2"];
#[crate_type = "dylib"];

#[feature(globs, struct_variant, managed_boxes)];

extern mod syntax;
extern mod rustc;
extern mod extra;

use std::local_data;
use std::io;
use std::io::{File, MemWriter};
use std::str;
use extra::getopts;
use extra::getopts::groups;
use extra::json;
use extra::serialize::{Decodable, Encodable};
use extra::time;

pub mod clean;
pub mod core;
pub mod doctree;
pub mod fold;
pub mod html {
    pub mod escape;
    pub mod format;
    pub mod layout;
    pub mod markdown;
    pub mod render;
}
pub mod passes;
pub mod plugins;
pub mod visit_ast;
pub mod test;

pub static SCHEMA_VERSION: &'static str = "0.8.1";

type Pass = (&'static str,                                      // name
             extern fn(clean::Crate) -> plugins::PluginResult,  // fn
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

type Output = (clean::Crate, ~[plugins::PluginJson]);

pub fn main() {
    std::os::set_exit_status(main_args(std::os::args()));
}

pub fn opts() -> ~[groups::OptGroup] {
    use extra::getopts::groups::*;
    ~[
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
    ]
}

pub fn usage(argv0: &str) {
    println!("{}", groups::usage(format!("{} [options] <input>", argv0), opts()));
}

pub fn main_args(args: &[~str]) -> int {
    let matches = match groups::getopts(args.tail(), opts()) {
        Ok(m) => m,
        Err(err) => {
            println!("{}", err.to_err_msg());
            return 1;
        }
    };
    if matches.opt_present("h") || matches.opt_present("help") {
        usage(args[0]);
        return 0;
    } else if matches.opt_present("version") {
        rustc::version(args[0]);
        return 0;
    }

    if matches.free.len() == 0 {
        println!("expected an input file to act on");
        return 1;
    } if matches.free.len() > 1 {
        println!("only one input file may be specified");
        return 1;
    }
    let input = matches.free[0].as_slice();

    if matches.opt_present("test") {
        return test::run(input, &matches);
    }

    if matches.opt_strs("passes") == ~[~"list"] {
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

    let (crate, res) = match acquire_input(input, &matches) {
        Ok(pair) => pair,
        Err(s) => {
            println!("input error: {}", s);
            return 1;
        }
    };

    info!("going to format");
    let started = time::precise_time_ns();
    let output = matches.opt_str("o").map(|s| Path::new(s));
    match matches.opt_str("w") {
        Some(~"html") | None => {
            match html::render::run(crate, output.unwrap_or(Path::new("doc"))) {
                Ok(()) => {}
                Err(e) => fail!("failed to generate documentation: {}", e),
            }
        }
        Some(~"json") => {
            match json_output(crate, res, output.unwrap_or(Path::new("doc.json"))) {
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
                 matches: &getopts::Matches) -> Result<Output, ~str> {
    match matches.opt_str("r") {
        Some(~"rust") => Ok(rust_input(input, matches)),
        Some(~"json") => json_input(input),
        Some(s) => Err("unknown input format: " + s),
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
    let mut plugins = matches.opt_strs("plugins");

    // First, parse the crate and extract all relevant information.
    let libs = matches.opt_strs("L").map(|s| Path::new(s.as_slice()));
    let cfgs = matches.opt_strs("cfg");
    let cr = Path::new(cratefile);
    info!("starting to run rustc");
    let (crate, analysis) = std::task::try(proc() {
        let cr = cr;
        core::run_core(libs.move_iter().collect(), cfgs, &cr)
    }).unwrap();
    info!("finished with rustc");
    local_data::set(analysiskey, analysis);

    // Process all of the crate attributes, extracting plugin metadata along
    // with the passes which we are supposed to run.
    match crate.module.get_ref().doc_list() {
        Some(nested) => {
            for inner in nested.iter() {
                match *inner {
                    clean::Word(~"no_default_passes") => {
                        default_passes = false;
                    }
                    clean::NameValue(~"passes", ref value) => {
                        for pass in value.words() {
                            passes.push(pass.to_owned());
                        }
                    }
                    clean::NameValue(~"plugins", ref value) => {
                        for p in value.words() {
                            plugins.push(p.to_owned());
                        }
                    }
                    _ => {}
                }
            }
        }
        None => {}
    }
    if default_passes {
        for name in DEFAULT_PASSES.rev_iter() {
            passes.unshift(name.to_owned());
        }
    }

    // Load all plugins/passes into a PluginManager
    let path = matches.opt_str("plugin-path").unwrap_or(~"/tmp/rustdoc/plugins");
    let mut pm = plugins::PluginManager::new(Path::new(path));
    for pass in passes.iter() {
        let plugin = match PASSES.iter().position(|&(p, _, _)| p == *pass) {
            Some(i) => PASSES[i].n1(),
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
    return pm.run_plugins(crate);
}

/// This input format purely deserializes the json output file. No passes are
/// run over the deserialized output.
fn json_input(input: &str) -> Result<Output, ~str> {
    let mut input = match File::open(&Path::new(input)) {
        Ok(f) => f,
        Err(e) => return Err(format!("couldn't open {}: {}", input, e)),
    };
    match json::from_reader(&mut input) {
        Err(s) => Err(s.to_str()),
        Ok(json::Object(obj)) => {
            let mut obj = obj;
            // Make sure the schema is what we expect
            match obj.pop(&~"schema") {
                Some(json::String(version)) => {
                    if version.as_slice() != SCHEMA_VERSION {
                        return Err(format!("sorry, but I only understand \
                                            version {}", SCHEMA_VERSION))
                    }
                }
                Some(..) => return Err(~"malformed json"),
                None => return Err(~"expected a schema version"),
            }
            let crate = match obj.pop(&~"crate") {
                Some(json) => {
                    let mut d = json::Decoder::new(json);
                    Decodable::decode(&mut d)
                }
                None => return Err(~"malformed json"),
            };
            // FIXME: this should read from the "plugins" field, but currently
            //      Json doesn't implement decodable...
            let plugin_output = ~[];
            Ok((crate, plugin_output))
        }
        Ok(..) => Err(~"malformed json input: expected an object at the top"),
    }
}

/// Outputs the crate/plugin json as a giant json blob at the specified
/// destination.
fn json_output(crate: clean::Crate, res: ~[plugins::PluginJson],
               dst: Path) -> io::IoResult<()> {
    // {
    //   "schema": version,
    //   "crate": { parsed crate ... },
    //   "plugins": { output of plugins ... }
    // }
    let mut json = ~extra::treemap::TreeMap::new();
    json.insert(~"schema", json::String(SCHEMA_VERSION.to_owned()));
    let plugins_json = ~res.move_iter().filter_map(|opt| opt).collect();

    // FIXME #8335: yuck, Rust -> str -> JSON round trip! No way to .encode
    // straight to the Rust JSON representation.
    let crate_json_str = {
        let mut w = MemWriter::new();
        {
            let mut encoder = json::Encoder::new(&mut w as &mut io::Writer);
            crate.encode(&mut encoder);
        }
        str::from_utf8_owned(w.unwrap()).unwrap()
    };
    let crate_json = match json::from_str(crate_json_str) {
        Ok(j) => j,
        Err(_) => fail!("Rust generated JSON is invalid??")
    };

    json.insert(~"crate", crate_json);
    json.insert(~"plugins", json::Object(plugins_json));

    let mut file = if_ok!(File::create(&dst));
    if_ok!(json::Object(json).to_writer(&mut file));
    Ok(())
}
