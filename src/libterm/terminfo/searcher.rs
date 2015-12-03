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

use std::env;
use std::fs;
use std::path::PathBuf;

/// Return path to database entry for `term`
#[allow(deprecated)]
pub fn get_dbpath_for_term(term: &str) -> Option<PathBuf> {
    let mut dirs_to_search = Vec::new();
    let first_char = match term.chars().next() {
        Some(c) => c,
        None => return None,
    };

    // Find search directory
    match env::var_os("TERMINFO") {
        Some(dir) => dirs_to_search.push(PathBuf::from(dir)),
        None => {
            if let Some(mut homedir) = env::home_dir() {
                // ncurses compatibility;
                homedir.push(".terminfo");
                dirs_to_search.push(homedir)
            }
            match env::var("TERMINFO_DIRS") {
                Ok(dirs) => {
                    for i in dirs.split(':') {
                        if i == "" {
                            dirs_to_search.push(PathBuf::from("/usr/share/terminfo"));
                        } else {
                            dirs_to_search.push(PathBuf::from(i));
                        }
                    }
                }
                // Found nothing in TERMINFO_DIRS, use the default paths:
                // According to  /etc/terminfo/README, after looking at
                // ~/.terminfo, ncurses will search /etc/terminfo, then
                // /lib/terminfo, and eventually /usr/share/terminfo.
                Err(..) => {
                    dirs_to_search.push(PathBuf::from("/etc/terminfo"));
                    dirs_to_search.push(PathBuf::from("/lib/terminfo"));
                    dirs_to_search.push(PathBuf::from("/usr/share/terminfo"));
                }
            }
        }
    };

    // Look for the terminal in all of the search directories
    for mut p in dirs_to_search {
        if fs::metadata(&p).is_ok() {
            p.push(&first_char.to_string());
            p.push(&term);
            if fs::metadata(&p).is_ok() {
                return Some(p);
            }
            p.pop();
            p.pop();

            // on some installations the dir is named after the hex of the char
            // (e.g. OS X)
            p.push(&format!("{:x}", first_char as usize));
            p.push(term);
            if fs::metadata(&p).is_ok() {
                return Some(p);
            }
        }
    }
    None
}

#[test]
#[ignore(reason = "buildbots don't have ncurses installed and I can't mock everything I need")]
fn test_get_dbpath_for_term() {
    // woefully inadequate test coverage
    // note: current tests won't work with non-standard terminfo hierarchies (e.g. OS X's)
    use std::env;
    // FIXME (#9639): This needs to handle non-utf8 paths
    fn x(t: &str) -> String {
        let p = get_dbpath_for_term(t).expect("no terminfo entry found");
        p.to_str().unwrap().to_string()
    }
    assert!(x("screen") == "/usr/share/terminfo/s/screen");
    assert!(get_dbpath_for_term("") == None);
    env::set_var("TERMINFO_DIRS", ":");
    assert!(x("screen") == "/usr/share/terminfo/s/screen");
    env::remove_var("TERMINFO_DIRS");
}
