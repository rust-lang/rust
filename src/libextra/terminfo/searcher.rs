// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Implement ncurses-compatible database discovery
/// Does not support hashed database, only filesystem!

use core::prelude::*;
use core::{os, str};
use core::os::getenv;
use core::io::{file_reader, Reader};
use path = core::path::Path;

/// Return path to database entry for `term`
pub fn get_dbpath_for_term(term: &str) -> Option<~path> {
    if term.len() == 0 {
        return None;
    }

    let homedir = os::homedir();

    let mut dirs_to_search = ~[];
    let first_char = term.char_at(0);

    // Find search directory
    match getenv("TERMINFO") {
        Some(dir) => dirs_to_search.push(path(dir)),
        None => {
            if homedir.is_some() {
                dirs_to_search.push(homedir.unwrap().push(".terminfo")); // ncurses compatability
            }
            match getenv("TERMINFO_DIRS") {
                Some(dirs) => for dirs.split_iter(':').advance |i| {
                    if i == "" {
                        dirs_to_search.push(path("/usr/share/terminfo"));
                    } else {
                        dirs_to_search.push(path(i.to_owned()));
                    }
                },
                // Found nothing, use the default paths
                // /usr/share/terminfo is the de facto location, but it seems
                // Ubuntu puts it in /lib/terminfo
                None => {
                    dirs_to_search.push(path("/usr/share/terminfo"));
                    dirs_to_search.push(path("/lib/terminfo"));
                }
            }
        }
    };

    // Look for the terminal in all of the search directories
    for dirs_to_search.iter().advance |p| {
        let newp = ~p.push_many(&[str::from_char(first_char), term.to_owned()]);
        if os::path_exists(p) && os::path_exists(newp) {
            return Some(newp);
        }
        // on some installations the dir is named after the hex of the char (e.g. OS X)
        let newp = ~p.push_many(&[fmt!("%x", first_char as uint), term.to_owned()]);
        if os::path_exists(p) && os::path_exists(newp) {
            return Some(newp);
        }
    }
    None
}

/// Return open file for `term`
pub fn open(term: &str) -> Result<@Reader, ~str> {
    match get_dbpath_for_term(term) {
        Some(x) => file_reader(x),
        None => Err(fmt!("could not find terminfo entry for %s", term))
    }
}

#[test]
#[ignore(reason = "buildbots don't have ncurses installed and I can't mock everything I need")]
fn test_get_dbpath_for_term() {
    // woefully inadequate test coverage
    // note: current tests won't work with non-standard terminfo hierarchies (e.g. OS X's)
    use std::os::{setenv, unsetenv};
    fn x(t: &str) -> ~str { get_dbpath_for_term(t).expect("no terminfo entry found").to_str() };
    assert!(x("screen") == ~"/usr/share/terminfo/s/screen");
    assert!(get_dbpath_for_term("") == None);
    setenv("TERMINFO_DIRS", ":");
    assert!(x("screen") == ~"/usr/share/terminfo/s/screen");
    unsetenv("TERMINFO_DIRS");
}

#[test]
#[ignore(reason = "see test_get_dbpath_for_term")]
fn test_open() {
    open("screen");
    let t = open("nonexistent terminal that hopefully does not exist");
    assert!(t.is_err());
}
