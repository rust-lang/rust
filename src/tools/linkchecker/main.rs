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
//! that, it just has a simple "regex" to search for `href` and `id` tags.
//! These values are then translated to file URLs if possible and then the
//! destination is asserted to exist.
//!
//! A few whitelisted exceptions are allowed as there's known bugs in rustdoc,
//! but this should catch the majority of "broken link" cases.

extern crate url;

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

use url::{Url, UrlParser};

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {:?}", stringify!($e), e),
    })
}

fn main() {
    let docs = env::args().nth(1).unwrap();
    let docs = env::current_dir().unwrap().join(docs);
    let mut url = Url::from_file_path(&docs).unwrap();
    let mut errors = false;
    walk(&mut HashMap::new(), &docs, &docs, &mut url, &mut errors);
    if errors {
        panic!("found some broken links");
    }
}

#[derive(Debug)]
pub enum LoadError {
    IOError(std::io::Error),
    BrokenRedirect(PathBuf, std::io::Error),
}

struct FileEntry {
    source: String,
    ids: HashSet<String>,
}

type Cache = HashMap<PathBuf, FileEntry>;

fn walk(cache: &mut Cache,
        root: &Path,
        dir: &Path,
        url: &mut Url,
        errors: &mut bool)
{
    for entry in t!(dir.read_dir()).map(|e| t!(e)) {
        let path = entry.path();
        let kind = t!(entry.file_type());
        url.path_mut().unwrap().push(entry.file_name().into_string().unwrap());
        if kind.is_dir() {
            walk(cache, root, &path, url, errors);
        } else {
            check(cache, root, &path, url, errors);
        }
        url.path_mut().unwrap().pop();
    }
}

fn check(cache: &mut Cache,
         root: &Path,
         file: &Path,
         base: &Url,
         errors: &mut bool)
{
    // ignore js files as they are not prone to errors as the rest of the
    // documentation is and they otherwise bring up false positives.
    if file.extension().and_then(|s| s.to_str()) == Some("js") {
        return
    }

    // Unfortunately we're not 100% full of valid links today to we need a few
    // whitelists to get this past `make check` today.
    // FIXME(#32129)
    if file.ends_with("std/string/struct.String.html") ||
       file.ends_with("collections/string/struct.String.html") {
        return
    }
    // FIXME(#32130)
    if file.ends_with("btree_set/struct.BTreeSet.html") ||
       file.ends_with("collections/struct.BTreeSet.html") ||
       file.ends_with("collections/btree_map/struct.BTreeMap.html") ||
       file.ends_with("collections/hash_map/struct.HashMap.html") {
        return
    }

    if file.ends_with("std/sys/ext/index.html") {
        return
    }

    if let Some(file) = file.to_str() {
        // FIXME(#31948)
        if file.contains("ParseFloatError") {
            return
        }
        // weird reexports, but this module is on its way out, so chalk it up to
        // "rustdoc weirdness" and move on from there
        if file.contains("scoped_tls") {
            return
        }
    }

    let mut parser = UrlParser::new();
    parser.base_url(base);

    let res = load_file(cache, root, PathBuf::from(file), false, false);
    let (pretty_file, contents) = match res {
        Ok(res) => res,
        Err(_) => return,
    };

    // Search for anything that's the regex 'href[ ]*=[ ]*".*?"'
    with_attrs_in_source(&contents, " href", |url, i| {
        // Once we've plucked out the URL, parse it using our base url and
        // then try to extract a file path. If either of these fail then we
        // just keep going.
        let (parsed_url, path) = match url_to_file_path(&parser, url) {
            Some((url, path)) => (url, PathBuf::from(path)),
            None => return,
        };

        // Alright, if we've found a file name then this file had better
        // exist! If it doesn't then we register and print an error.
        if path.exists() {
            if path.is_dir() {
                return;
            }
            let res = load_file(cache, root, path.clone(), true, false);
            let (pretty_path, contents) = match res {
                Ok(res) => res,
                Err(LoadError::IOError(err)) => panic!(format!("{}", err)),
                Err(LoadError::BrokenRedirect(target, _)) => {
                    print!("{}:{}: broken redirect to {}",
                           pretty_file.display(), i + 1, target.display());
                    return;
                }
            };

            if let Some(ref fragment) = parsed_url.fragment {
                // Fragments like `#1-6` are most likely line numbers to be
                // interpreted by javascript, so we're ignoring these
                if fragment.splitn(2, '-')
                           .all(|f| f.chars().all(|c| c.is_numeric())) {
                    return;
                }

                let ids = &mut cache.get_mut(&pretty_path).unwrap().ids;
                if ids.is_empty() {
                    // Search for anything that's the regex 'id[ ]*=[ ]*".*?"'
                    with_attrs_in_source(&contents, " id", |fragment, i| {
                        let frag = fragment.trim_left_matches("#").to_owned();
                        if !ids.insert(frag) {
                            *errors = true;
                            println!("{}:{}: id is not unique: `{}`",
                                     pretty_file.display(), i, fragment);
                        }
                    });
                }
                if !ids.contains(fragment) {
                    *errors = true;
                    print!("{}:{}: broken link fragment  ",
                           pretty_file.display(), i + 1);
                    println!("`#{}` pointing to `{}`",
                             fragment, pretty_path.display());
                };
            }
        } else {
            *errors = true;
            print!("{}:{}: broken link - ", pretty_file.display(), i + 1);
            let pretty_path = path.strip_prefix(root).unwrap_or(&path);
            println!("{}", pretty_path.display());
        }
    });
}

fn load_file(cache: &mut Cache,
             root: &Path,
             file: PathBuf,
             follow_redirects: bool,
             is_redirect: bool) -> Result<(PathBuf, String), LoadError> {

    let mut contents = String::new();
    let pretty_file = PathBuf::from(file.strip_prefix(root).unwrap_or(&file));

    let maybe_redirect = match cache.entry(pretty_file.clone()) {
        Entry::Occupied(entry) => {
            contents = entry.get().source.clone();
            None
        },
        Entry::Vacant(entry) => {
            let mut fp = try!(File::open(file.clone()).map_err(|err| {
                if is_redirect {
                    LoadError::BrokenRedirect(file.clone(), err)
                } else {
                    LoadError::IOError(err)
                }
            }));
            try!(fp.read_to_string(&mut contents)
                   .map_err(|err| LoadError::IOError(err)));

            let maybe = if follow_redirects {
                maybe_redirect(&contents)
            } else {
                None
            };
            if maybe.is_none() {
                entry.insert(FileEntry {
                    source: contents.clone(),
                    ids: HashSet::new(),
                });
            }
            maybe
        },
    };
    let base = Url::from_file_path(&file).unwrap();
    let mut parser = UrlParser::new();
    parser.base_url(&base);

    match maybe_redirect.and_then(|url| url_to_file_path(&parser, &url)) {
        Some((_, redirect_file)) => {
            assert!(follow_redirects);
            let path = PathBuf::from(redirect_file);
            load_file(cache, root, path, follow_redirects, true)
        }
        None => Ok((pretty_file, contents))
    }
}

fn maybe_redirect(source: &str) -> Option<String> {
    const REDIRECT: &'static str = "<p>Redirecting to <a href=";

    let mut lines = source.lines();
    let redirect_line = match lines.nth(6) {
        Some(l) => l,
        None => return None,
    };

    redirect_line.find(REDIRECT).map(|i| {
        let rest = &redirect_line[(i + REDIRECT.len() + 1)..];
        let pos_quote = rest.find('"').unwrap();
        rest[..pos_quote].to_owned()
    })
}

fn url_to_file_path(parser: &UrlParser, url: &str) -> Option<(Url, PathBuf)> {
    parser.parse(url).ok().and_then(|parsed_url| {
        parsed_url.to_file_path().ok().map(|f| (parsed_url, f))
    })
}

fn with_attrs_in_source<F: FnMut(&str, usize)>(contents: &str,
                                               attr: &str,
                                               mut f: F)
{
    for (i, mut line) in contents.lines().enumerate() {
        while let Some(j) = line.find(attr) {
            let rest = &line[j + attr.len() ..];
            line = rest;
            let pos_equals = match rest.find("=") {
                Some(i) => i,
                None => continue,
            };
            if rest[..pos_equals].trim_left_matches(" ") != "" {
                continue
            }

            let rest = &rest[pos_equals + 1..];

            let pos_quote = match rest.find(&['"', '\''][..]) {
                Some(i) => i,
                None => continue,
            };
            let quote_delim = rest.as_bytes()[pos_quote] as char;

            if rest[..pos_quote].trim_left_matches(" ") != "" {
                continue
            }
            let rest = &rest[pos_quote + 1..];
            let url = match rest.find(quote_delim) {
                Some(i) => &rest[..i],
                None => continue,
            };
            f(url, i)
        }
    }
}
