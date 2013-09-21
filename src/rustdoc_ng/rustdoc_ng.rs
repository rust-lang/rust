// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "rustdoc_ng",
       vers = "0.8",
       uuid = "8c6e4598-1596-4aa5-a24c-b811914bbbc6",
       url = "https://github.com/mozilla/rust/tree/master/src/rustdoc_ng")];

#[desc = "rustdoc, the Rust documentation extractor"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

extern mod syntax;
extern mod rustc;
extern mod extra;

use extra::serialize::Encodable;
use extra::time;
use std::cell::Cell;
use std::rt::io;
use std::rt::io::Writer;
use std::rt::io::file::FileInfo;

pub mod clean;
pub mod core;
pub mod doctree;
pub mod fold;
pub mod html {
    pub mod render;
    pub mod layout;
    pub mod markdown;
    pub mod format;
}
pub mod passes;
pub mod plugins;
pub mod visit_ast;

pub static SCHEMA_VERSION: &'static str = "0.8.0";

local_data_key!(pub ctxtkey: @core::DocContext)

enum OutputFormat {
    HTML, JSON
}

pub fn main() {
    main_args(std::os::args());
}

pub fn main_args(args: &[~str]) {
    use extra::getopts::groups::*;

    let opts = ~[
        optmulti("L", "library-path", "directory to add to crate search path",
                 "DIR"),
        optmulti("", "plugin-path", "directory to load plugins from", "DIR"),
        optmulti("", "passes", "space separated list of passes to also run",
                 "PASSES"),
        optmulti("", "plugins", "space separated list of plugins to also load",
                 "PLUGINS"),
        optflag("h", "help", "show this help message"),
        optflag("", "nodefaults", "don't run the default passes"),
        optopt("o", "output", "where to place the output", "PATH"),
    ];

    let matches = getopts(args.tail(), opts).unwrap();

    let myusage = || {
        println(usage(format!("{} [options] [html|json] <crate>", args[0]), opts));
    };

    if matches.opt_present("h") || matches.opt_present("help") {
        myusage();
        return;
    }

    let (format, cratefile) = match matches.free.clone() {
        [~"json", crate] => (JSON, crate),
        [~"html", crate] => (HTML, crate),
        [s, _] => {
            println!("Unknown output format: `{}`", s);
            myusage();
            exit(1);
        }
        [_, .._] => {
            println!("Expected exactly one crate to process");
            myusage();
            exit(1);
        }
        _ => {
            println!("Expected an output format and then one crate");
            myusage();
            exit(1);
        }
    };

    // First, parse the crate and extract all relevant information.
    let libs = Cell::new(matches.opt_strs("L").map(|s| Path(*s)));
    let cr = Cell::new(Path(cratefile));
    info2!("starting to run rustc");
    let crate = do std::task::try {
        let cr = cr.take();
        core::run_core(libs.take(), &cr)
    }.unwrap();
    info2!("finished with rustc");

    // Process all of the crate attributes, extracting plugin metadata along
    // with the passes which we are supposed to run.
    let mut default_passes = !matches.opt_present("nodefaults");
    let mut passes = matches.opt_strs("passes");
    let mut plugins = matches.opt_strs("plugins");
    match crate.module.get_ref().doc_list() {
        Some(nested) => {
            for inner in nested.iter() {
                match *inner {
                    clean::Word(~"no_default_passes") => {
                        default_passes = false;
                    }
                    clean::NameValue(~"passes", ref value) => {
                        for pass in value.word_iter() {
                            passes.push(pass.to_owned());
                        }
                    }
                    clean::NameValue(~"plugins", ref value) => {
                        for p in value.word_iter() {
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
        passes.unshift(~"collapse-docs");
        passes.unshift(~"unindent-comments");
    }

    // Load all plugins/passes into a PluginManager
    let mut pm = plugins::PluginManager::new(Path("/tmp/rustdoc_ng/plugins"));
    for pass in passes.iter() {
        let plugin = match pass.as_slice() {
            "strip-hidden" => passes::strip_hidden,
            "unindent-comments" => passes::unindent_comments,
            "collapse-docs" => passes::collapse_docs,
            "collapse-privacy" => passes::collapse_privacy,
            s => { error!("unknown pass %s, skipping", s); loop },
        };
        pm.add_plugin(plugin);
    }
    info2!("loading plugins...");
    for pname in plugins.move_iter() {
        pm.load_plugin(pname);
    }

    // Run everything!
    info2!("Executing passes/plugins");
    let (crate, res) = pm.run_plugins(crate);

    info2!("going to format");
    let started = time::precise_time_ns();
    let output = matches.opt_str("o").map(|s| Path(*s));
    match format {
        HTML => { html::render::run(crate, output.unwrap_or(Path("doc"))) }
        JSON => { jsonify(crate, res, output.unwrap_or(Path("doc.json"))) }
    }
    let ended = time::precise_time_ns();
    info2!("Took {:.03f}s", (ended as f64 - started as f64) / 1000000000f64);
}

fn jsonify(crate: clean::Crate, res: ~[plugins::PluginJson], dst: Path) {
    // {
    //   "schema": version,
    //   "crate": { parsed crate ... },
    //   "plugins": { output of plugins ... }
    // }
    let mut json = ~extra::treemap::TreeMap::new();
    json.insert(~"schema", extra::json::String(SCHEMA_VERSION.to_owned()));
    let plugins_json = ~res.move_iter().filter_map(|opt| opt).collect();

    // FIXME #8335: yuck, Rust -> str -> JSON round trip! No way to .encode
    // straight to the Rust JSON representation.
    let crate_json_str = do std::io::with_str_writer |w| {
        crate.encode(&mut extra::json::Encoder(w));
    };
    let crate_json = match extra::json::from_str(crate_json_str) {
        Ok(j) => j,
        Err(_) => fail!("Rust generated JSON is invalid??")
    };

    json.insert(~"crate", crate_json);
    json.insert(~"plugins", extra::json::Object(plugins_json));

    let mut file = dst.open_writer(io::Create).unwrap();
    let output = extra::json::Object(json).to_str();
    file.write(output.as_bytes());
}

fn exit(status: int) -> ! {
    #[fixed_stack_segment]; #[inline(never)];
    use std::libc;
    unsafe { libc::exit(status as libc::c_int) }
}
