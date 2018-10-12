// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate clap;
extern crate clippy_dev;
extern crate regex;

use clap::{App, Arg, SubCommand};
use clippy_dev::*;

fn main() {
    let matches = App::new("Clippy developer tooling")
        .subcommand(
            SubCommand::with_name("update_lints")
                .about("Update the lint list")
                .arg(
                    Arg::with_name("print-only")
                        .long("print-only")
                        .short("p")
                        .help("Print a table of lints to STDOUT. Does not modify any files."),
                )
        )
        .get_matches();

    if let Some(matches) = matches.subcommand_matches("update_lints") {
        if matches.is_present("print-only") {
            print_lints();
        }
    }
}

fn print_lints() {
    let lint_list = gather_all().collect::<Vec<Lint>>();
    let grouped_by_lint_group = Lint::by_lint_group(&lint_list);

    for (lint_group, mut lints) in grouped_by_lint_group {
        if lint_group == "Deprecated" { continue; }
        println!("\n## {}", lint_group);

        lints.sort_by(|a, b| a.name.cmp(&b.name));

        for lint in lints {
            println!("* [{}]({}#{}) ({})", lint.name, clippy_dev::DOCS_LINK.clone(), lint.name, lint.desc);
        }
    }

    println!("there are {} lints", Lint::active_lints(lint_list.into_iter()).count());
}
