// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Inspired by Paul Woolcock's cargo-fmt (https://github.com/pwoolcoc/cargo-fmt/)

#![cfg(not(test))]
#![cfg(feature="cargo-fmt")]

extern crate getopts;
extern crate walkdir;
extern crate rustc_serialize;

use std::path::PathBuf;
use std::process::Command;
use std::env;
use std::str;

use getopts::Options;
use walkdir::{WalkDir, DirEntry};
use rustc_serialize::json::Json;

fn main() {
    let mut opts = getopts::Options::new();
    opts.optflag("h", "help", "show this message");

    let matches = match opts.parse(env::args().skip(1)) {
        Ok(m) => m,
        Err(e) => {
            print_usage(&opts, &e.to_string());
            return;
        }
    };

    if matches.opt_present("h") {
        print_usage(&opts, "");
    } else {
        format_crate(&opts);
    }
}

fn print_usage(opts: &Options, reason: &str) {
    let msg = format!("{}\nusage: cargo fmt [options]", reason);
    println!("{}\nThis utility formats all readable .rs files in the src directory of the \
              current crate using rustfmt.",
             opts.usage(&msg));
}

fn format_crate(opts: &Options) {
    let mut root = match locate_root() {
        Ok(r) => r,
        Err(e) => {
            print_usage(opts, &e.to_string());
            return;
        }
    };

    // Currently only files in [root]/src can be formatted
    root.push("src");
    // All unreadable or non .rs files are skipped
    let files: Vec<_> = WalkDir::new(root)
                            .into_iter()
                            .filter(is_rs_file)
                            .filter_map(|f| f.ok())
                            .map(|e| e.path().to_owned())
                            .collect();

    format_files(&files).unwrap_or_else(|e| print_usage(opts, &e.to_string()));
}

fn locate_root() -> Result<PathBuf, std::io::Error> {
    // This seems adequate, as cargo-fmt can only be used systems that have Cargo installed
    let output = try!(Command::new("cargo").arg("locate-project").output());
    if output.status.success() {
        // We assume cargo locate-project is not broken and
        // it will output a valid json document
        let data = &String::from_utf8(output.stdout).unwrap();
        let json = Json::from_str(data).unwrap();
        let root = PathBuf::from(json.find("root").unwrap().as_string().unwrap());

        // root.parent() should never fail if locate-project's output is correct
        Ok(root.parent().unwrap().to_owned())
    } else {
        // This happens when cargo-fmt is not used inside a crate
        Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                                str::from_utf8(&output.stderr).unwrap()))
    }
}

fn is_rs_file(entry: &Result<walkdir::DirEntry, walkdir::Error>) -> bool {
    match *entry {
        Ok(ref file) => {
            match file.path().extension() {
                Some(ext) => ext == "rs",
                None => false,
            }
        }
        Err(_) => false,
    }
}

fn format_files(files: &Vec<PathBuf>) -> Result<(), std::io::Error> {
    let mut command = try!(Command::new("rustfmt")
                               .arg("--write-mode=overwrite")
                               .args(files)
                               .spawn());
    try!(command.wait());

    Ok(())
}
