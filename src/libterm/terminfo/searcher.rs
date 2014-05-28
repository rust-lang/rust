// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ncurses-compatible database discovery
//!
//! Does not support hashed database, only filesystem!

use std::io::File;
use std::os::getenv;
use std::{os, str};

/// Return path to database entry for `term`
pub fn get_dbpath_for_term(term: &str) -> Option<Box<Path>> {
    if term.len() == 0 {
        return None;
    }

    let homedir = os::homedir();

    let mut dirs_to_search = Vec::new();
    let first_char = term.char_at(0);

    // Find search directory
    match getenv("TERMINFO") {
        Some(dir) => dirs_to_search.push(Path::new(dir)),
        None => {
            if homedir.is_some() {
                // ncurses compatability;
                dirs_to_search.push(homedir.unwrap().join(".terminfo"))
            }
            match getenv("TERMINFO_DIRS") {
                Some(dirs) => for i in dirs.as_slice().split(':') {
                    if i == "" {
                        dirs_to_search.push(Path::new("/usr/share/terminfo"));
                    } else {
                        dirs_to_search.push(Path::new(i.to_string()));
                    }
                },
                // Found nothing in TERMINFO_DIRS, use the default paths:
                // According to  /etc/terminfo/README, after looking at
                // ~/.terminfo, ncurses will search /etc/terminfo, then
                // /lib/terminfo, and eventually /usr/share/terminfo.
                None => {
                    dirs_to_search.push(Path::new("/etc/terminfo"));
                    dirs_to_search.push(Path::new("/lib/terminfo"));
                    dirs_to_search.push(Path::new("/usr/share/terminfo"));
                }
            }
        }
    };

    // Look for the terminal in all of the search directories
    for p in dirs_to_search.iter() {
        if p.exists() {
            let f = str::from_char(first_char);
            let newp = p.join_many([f.as_slice(), term]);
            if newp.exists() {
                return Some(box newp);
            }
            // on some installations the dir is named after the hex of the char (e.g. OS X)
            let f = format!("{:x}", first_char as uint);
            let newp = p.join_many([f.as_slice(), term]);
            if newp.exists() {
                return Some(box newp);
            }
        }
    }
    None
}

/// Return open file for `term`
pub fn open(term: &str) -> Result<File, String> {
    match get_dbpath_for_term(term) {
        Some(x) => {
            match File::open(x) {
                Ok(file) => Ok(file),
                Err(e) => Err(format!("error opening file: {}", e)),
            }
        }
        None => {
            Err(format!("could not find terminfo entry for {}", term))
        }
    }
}

#[test]
#[ignore(reason = "buildbots don't have ncurses installed and I can't mock everything I need")]
fn test_get_dbpath_for_term() {
    // woefully inadequate test coverage
    // note: current tests won't work with non-standard terminfo hierarchies (e.g. OS X's)
    use std::os::{setenv, unsetenv};
    // FIXME (#9639): This needs to handle non-utf8 paths
    fn x(t: &str) -> String {
        let p = get_dbpath_for_term(t).expect("no terminfo entry found");
        p.as_str().unwrap().to_string()
    };
    assert!(x("screen") == "/usr/share/terminfo/s/screen".to_string());
    assert!(get_dbpath_for_term("") == None);
    setenv("TERMINFO_DIRS", ":");
    assert!(x("screen") == "/usr/share/terminfo/s/screen".to_string());
    unsetenv("TERMINFO_DIRS");
}

#[test]
#[ignore(reason = "see test_get_dbpath_for_term")]
fn test_open() {
    open("screen").unwrap();
    let t = open("nonexistent terminal that hopefully does not exist");
    assert!(t.is_err());
}
