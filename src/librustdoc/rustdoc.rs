// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rustdoc - The Rust documentation generator

#[link(name = "rustdoc",
       vers = "0.7",
       uuid = "f8abd014-b281-484d-a0c3-26e3de8e2412",
       url = "https://github.com/mozilla/rust/tree/master/src/rustdoc")];

#[comment = "The Rust documentation generator"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

#[allow(non_implicitly_copyable_typarams)];

extern mod extra;
extern mod rustc;
extern mod syntax;

use std::io;
use std::os;

use config::Config;
use doc::Item;
use doc::ItemUtils;

pub mod pass;
pub mod config;
pub mod parse;
pub mod extract;
pub mod attr_parser;
pub mod doc;
pub mod markdown_index_pass;
pub mod markdown_pass;
pub mod markdown_writer;
pub mod fold;
pub mod path_pass;
pub mod attr_pass;
pub mod tystr_pass;
pub mod prune_hidden_pass;
pub mod desc_to_brief_pass;
pub mod text_pass;
pub mod unindent_pass;
pub mod trim_pass;
pub mod astsrv;
pub mod demo;
pub mod sort_pass;
pub mod sort_item_name_pass;
pub mod sort_item_type_pass;
pub mod page_pass;
pub mod sectionalize_pass;
pub mod escape_pass;
pub mod prune_private_pass;
pub mod util;

pub fn main() {
    let args = os::args();

    if args.iter().any_(|x| "-h" == *x) || args.iter().any_(|x| "--help" == *x) {
        config::usage();
        return;
    }

    let config = match config::parse_config(args) {
      Ok(config) => config,
      Err(err) => {
        io::println(fmt!("error: %s", err));
        return;
      }
    };

    run(config);
}

/// Runs rustdoc over the given file
fn run(config: Config) {

    let source_file = copy config.input_crate;

    // Create an AST service from the source code
    do astsrv::from_file(source_file.to_str()) |srv| {

        // Just time how long it takes for the AST to become available
        do time(~"wait_ast") {
            do astsrv::exec(srv.clone()) |_ctxt| { }
        };

        // Extract the initial doc tree from the AST. This contains
        // just names and node ids.
        let doc = time(~"extract", || {
            let default_name = copy source_file;
            extract::from_srv(srv.clone(), default_name.to_str())
        });

        // Refine and publish the document
        pass::run_passes(srv, doc, ~[
            // Generate type and signature strings
            tystr_pass::mk_pass(),
            // Record the full paths to various nodes
            path_pass::mk_pass(),
            // Extract the docs attributes and attach them to doc nodes
            attr_pass::mk_pass(),
            // Perform various text escaping
            escape_pass::mk_pass(),
            // Remove things marked doc(hidden)
            prune_hidden_pass::mk_pass(),
            // Remove things that are private
            prune_private_pass::mk_pass(),
            // Extract brief documentation from the full descriptions
            desc_to_brief_pass::mk_pass(),
            // Massage the text to remove extra indentation
            unindent_pass::mk_pass(),
            // Split text into multiple sections according to headers
            sectionalize_pass::mk_pass(),
            // Trim extra spaces from text
            trim_pass::mk_pass(),
            // Sort items by name
            sort_item_name_pass::mk_pass(),
            // Sort items again by kind
            sort_item_type_pass::mk_pass(),
            // Create indexes appropriate for markdown
            markdown_index_pass::mk_pass(copy config),
            // Break the document into pages if required by the
            // output format
            page_pass::mk_pass(config.output_style),
            // Render
            markdown_pass::mk_pass(
                markdown_writer::make_writer_factory(copy config)
            )
        ]);
    }
}

pub fn time<T>(what: ~str, f: &fn() -> T) -> T {
    let start = extra::time::precise_time_s();
    let rv = f();
    let end = extra::time::precise_time_s();
    info!("time: %3.3f s    %s", end - start, what);
    rv
}
