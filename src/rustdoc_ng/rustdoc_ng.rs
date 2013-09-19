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
       vers = "0.8-pre",
       uuid = "8c6e4598-1596-4aa5-a24c-b811914bbbc6",
       url = "https://github.com/mozilla/rust/tree/master/src/rustdoc_ng")];

#[desc = "rustdoc, the Rust documentation extractor"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

extern mod syntax;
extern mod rustc;
extern mod extra;

use extra::serialize::Encodable;
use std::cell::Cell;

pub mod core;
pub mod doctree;
pub mod clean;
pub mod visit_ast;
pub mod fold;
pub mod plugins;
pub mod passes;

pub static SCHEMA_VERSION: &'static str = "0.8.0";

local_data_key!(pub ctxtkey: @core::DocContext)

pub fn main() {
    main_args(std::os::args());
}

pub fn main_args(args: &[~str]) {
    use extra::getopts::groups::*;

    let opts = ~[
        optmulti("L", "library-path", "directory to add to crate search path", "DIR"),
        optmulti("p", "plugin", "plugin to load and run", "NAME"),
        optmulti("", "plugin-path", "directory to load plugins from", "DIR"),
        // auxillary pass (defaults to hidden_strip
        optmulti("a", "pass", "auxillary pass to run", "NAME"),
        optflag("n", "no-defult-passes", "do not run the default passes"),
        optflag("h", "help", "show this help message"),
    ];

    let matches = getopts(args.tail(), opts).unwrap();

    if matches.opt_present("h") || matches.opt_present("help") {
        println(usage(args[0], opts));
        return;
    }

    let libs = Cell::new(matches.opt_strs("L").map(|s| Path(*s)));

    let mut passes = if matches.opt_present("n") {
        ~[]
    } else {
        ~[~"collapse-docs", ~"clean-comments", ~"collapse-privacy" ]
    };

    matches.opt_strs("a").map(|x| passes.push(x.clone()));

    if matches.free.len() != 1 {
        println(usage(args[0], opts));
        return;
    }

    let cr = Cell::new(Path(matches.free[0]));

    let crate = do std::task::try {
        let cr = cr.take();
        core::run_core(libs.take(), &cr)
    }.unwrap();

    // { "schema": version, "crate": { parsed crate ... }, "plugins": { output of plugins ... }}
    let mut json = ~extra::treemap::TreeMap::new();
    json.insert(~"schema", extra::json::String(SCHEMA_VERSION.to_owned()));

    let mut pm = plugins::PluginManager::new(Path("/tmp/rustdoc_ng/plugins"));

    for pass in passes.iter() {
        pm.add_plugin(match pass.as_slice() {
            "strip-hidden" => passes::strip_hidden,
            "clean-comments" => passes::clean_comments,
            "collapse-docs" => passes::collapse_docs,
            "collapse-privacy" => passes::collapse_privacy,
            s => { error!("unknown pass %s, skipping", s); passes::noop },
        })
    }

    for pname in matches.opt_strs("p").move_iter() {
        pm.load_plugin(pname);
    }

    let (crate, res) = pm.run_plugins(crate);
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

    println(extra::json::Object(json).to_str());
}
