// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Script to check the validity of `href` links in our HTML documentation.
//!
//! In the past we've been quite error prone to writing in broken links as most
//! of them are manually rather than automatically added. As files move over
//! time or apis change old links become stale or broken. The purpose of this
//! script is to check all relative links in our documentation to make sure they
//! actually point to a valid place.
//!
//! Currently this doesn't actually do any HTML parsing or anything fancy like
//! that, it just has a simple "regex" to search for `href` tags. These values
//! are then translated to file URLs if possible and then the destination is
//! asserted to exist.
//!
//! A few whitelisted exceptions are allowed as there's known bugs in rustdoc,
//! but this should catch the majority of "broken link" cases.

extern crate url;

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use url::{Url, UrlParser};

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

fn main() {
    let docs = env::args().nth(1).unwrap();
    let docs = env::current_dir().unwrap().join(docs);
    let mut url = Url::from_file_path(&docs).unwrap();
    let mut errors = false;
    walk(&docs, &docs, &mut url, &mut errors);
    if errors {
        panic!("found some broken links");
    }
}

fn walk(root: &Path, dir: &Path, url: &mut Url, errors: &mut bool) {
    for entry in t!(dir.read_dir()).map(|e| t!(e)) {
        let path = entry.path();
        let kind = t!(entry.file_type());
        url.path_mut().unwrap().push(entry.file_name().into_string().unwrap());
        if kind.is_dir() {
            walk(root, &path, url, errors);
        } else {
            check(root, &path, url, errors);
        }
        url.path_mut().unwrap().pop();
    }
}

fn check(root: &Path, file: &Path, base: &Url, errors: &mut bool) {
    // ignore js files as they are not prone to errors as the rest of the
    // documentation is and they otherwise bring up false positives.
    if file.extension().and_then(|s| s.to_str()) == Some("js") {
        return
    }

    let pretty_file = file.strip_prefix(root).unwrap_or(file);

    // Unfortunately we're not 100% full of valid links today to we need a few
    // whitelists to get this past `make check` today.
    if let Some(path) = pretty_file.to_str() {
        // FIXME(#32129)
        if path == "std/string/struct.String.html" {
            return
        }
        // FIXME(#32130)
        if path.contains("btree_set/struct.BTreeSet.html") ||
           path == "collections/struct.BTreeSet.html" {
            return
        }
        // FIXME(#31948)
        if path.contains("ParseFloatError") {
            return
        }

        // currently
        if path == "std/sys/ext/index.html" {
            return
        }

        // weird reexports, but this module is on its way out, so chalk it up to
        // "rustdoc weirdness" and move on from there
        if path.contains("scoped_tls") {
            return
        }
    }

    let mut parser = UrlParser::new();
    parser.base_url(base);
    let mut contents = String::new();
    if t!(File::open(file)).read_to_string(&mut contents).is_err() {
        return
    }

    for (i, mut line) in contents.lines().enumerate() {
        // Search for anything that's the regex 'href[ ]*=[ ]*".*?"'
        while let Some(j) = line.find(" href") {
            let rest = &line[j + 5..];
            line = rest;
            let pos_equals = match rest.find("=") {
                Some(i) => i,
                None => continue,
            };
            if rest[..pos_equals].trim_left_matches(" ") != "" {
                continue
            }
            let rest = &rest[pos_equals + 1..];
            let pos_quote = match rest.find("\"").or_else(|| rest.find("'")) {
                Some(i) => i,
                None => continue,
            };
            if rest[..pos_quote].trim_left_matches(" ") != "" {
                continue
            }
            let rest = &rest[pos_quote + 1..];
            let url = match rest.find("\"").or_else(|| rest.find("'")) {
                Some(i) => &rest[..i],
                None => continue,
            };

            // Once we've plucked out the URL, parse it using our base url and
            // then try to extract a file path. If either if these fail then we
            // just keep going.
            let parsed_url = match parser.parse(url) {
                Ok(url) => url,
                Err(..) => continue,
            };
            let path = match parsed_url.to_file_path() {
                Ok(path) => path,
                Err(..) => continue,
            };

            // Alright, if we've found a file name then this file had better
            // exist! If it doesn't then we register and print an error.
            if !path.exists() {
                *errors = true;
                print!("{}:{}: broken link - ", pretty_file.display(), i + 1);
                let pretty_path = path.strip_prefix(root).unwrap_or(&path);
                println!("{}", pretty_path.display());
            }
        }
    }
}
