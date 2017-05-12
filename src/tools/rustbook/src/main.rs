// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
extern crate mdbook;
#[macro_use]
extern crate clap;

use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use clap::{App, ArgMatches, SubCommand, AppSettings};

use mdbook::MDBook;

fn main() {
    let d_message = "-d, --dest-dir=[dest-dir]
'The output directory for your book{n}(Defaults to ./book when omitted)'";
    let dir_message = "[dir]
'A directory for your book{n}(Defaults to Current Directory when omitted)'";

    let matches = App::new("rustbook")
                    .about("Build a book with mdBook")
                    .author("Steve Klabnik <steve@steveklabnik.com>")
                    .version(&*format!("v{}", crate_version!()))
                    .setting(AppSettings::SubcommandRequired)
                    .subcommand(SubCommand::with_name("build")
                        .about("Build the book from the markdown files")
                        .arg_from_usage(d_message)
                        .arg_from_usage(dir_message))
                    .get_matches();

    // Check which subcomamnd the user ran...
    let res = match matches.subcommand() {
        ("build", Some(sub_matches)) => build(sub_matches),
        ("test", Some(sub_matches)) => test(sub_matches),
        (_, _) => unreachable!(),
    };

    if let Err(e) = res {
        writeln!(&mut io::stderr(), "An error occured:\n{}", e).ok();
        ::std::process::exit(101);
    }
}

// Build command implementation
fn build(args: &ArgMatches) -> Result<(), Box<Error>> {
    let book = build_mdbook_struct(args);

    let mut book = match args.value_of("dest-dir") {
        Some(dest_dir) => book.set_dest(Path::new(dest_dir)),
        None => book
    };

    try!(book.build());

    Ok(())
}

fn test(args: &ArgMatches) -> Result<(), Box<Error>> {
    let mut book = build_mdbook_struct(args);

    try!(book.test());

    Ok(())
}

fn build_mdbook_struct(args: &ArgMatches) -> mdbook::MDBook {
    let book_dir = get_book_dir(args);
    let mut book = MDBook::new(&book_dir).read_config();

    // By default mdbook will attempt to create non-existent files referenced
    // from SUMMARY.md files. This is problematic on CI where we mount the
    // source directory as readonly. To avoid any issues, we'll disabled
    // mdbook's implicit file creation feature.
    book.create_missing = false;

    book
}

fn get_book_dir(args: &ArgMatches) -> PathBuf {
    if let Some(dir) = args.value_of("dir") {
        // Check if path is relative from current dir, or absolute...
        let p = Path::new(dir);
        if p.is_relative() {
            env::current_dir().unwrap().join(dir)
        } else {
            p.to_path_buf()
        }
    } else {
        env::current_dir().unwrap()
    }
}
