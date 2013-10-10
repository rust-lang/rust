// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use from_str::from_str;
use libc::exit;
use option::{Some, None, Option};
use rt::crate_map::{ModEntry, CrateMap, iter_crate_map, get_crate_map};
use str::StrSlice;
use u32;
use vec::ImmutableVector;
#[cfg(test)] use cast::transmute;

struct LogDirective {
    name: Option<~str>,
    level: u32
}

static MAX_LOG_LEVEL: u32 = 255;
static DEFAULT_LOG_LEVEL: u32 = 1;
static log_level_names : &'static[&'static str] = &'static["error", "warn", "info", "debug"];

/// Parse an individual log level that is either a number or a symbolic log level
fn parse_log_level(level: &str) -> Option<u32> {
    let num = from_str::<u32>(level);
    let mut log_level;
    match num {
        Some(num) => {
            if num < MAX_LOG_LEVEL {
                log_level = Some(num);
            } else {
                log_level = Some(MAX_LOG_LEVEL);
            }
        }
        _ => {
            let position = log_level_names.iter().position(|&name| name == level);
            match position {
                Some(position) => {
                    log_level = Some(u32::min(MAX_LOG_LEVEL, (position + 1) as u32))
                },
                _ => {
                    log_level = None;
                }
            }
        }
    }
    log_level
}

/// Parse a logging specification string (e.g: "crate1,crate2::mod3,crate3::x=1")
/// and return a vector with log directives.
/// Valid log levels are 0-255, with the most likely ones being 1-4 (defined in std::).
/// Also supports string log levels of error, warn, info, and debug
fn parse_logging_spec(spec: ~str) -> ~[LogDirective]{
    let mut dirs = ~[];
    for s in spec.split_iter(',') {
        let parts: ~[&str] = s.split_iter('=').collect();
        let mut log_level;
        let mut name = Some(parts[0].to_owned());
        match parts.len() {
            1 => {
                //if the single argument is a log-level string or number,
                //treat that as a global fallback
                let possible_log_level = parse_log_level(parts[0]);
                match possible_log_level {
                    Some(num) => {
                        name = None;
                        log_level = num;
                    },
                    _ => {
                        log_level = MAX_LOG_LEVEL
                    }
                }
            }
            2 => {
                let possible_log_level = parse_log_level(parts[1]);
                match possible_log_level {
                    Some(num) => {
                        log_level = num;
                    },
                    _ => {
                        rterrln!("warning: invalid logging spec '{}', \
                                  ignoring it", parts[1]);
                        continue
                    }
                }
            },
            _ => {
                rterrln!("warning: invalid logging spec '{}', \
                          ignoring it", s);
                continue
            }
        }
        let dir = LogDirective {name: name, level: log_level};
        dirs.push(dir);
    }
    return dirs;
}

/// Set the log level of an entry in the crate map depending on the vector
/// of log directives
fn update_entry(dirs: &[LogDirective], entry: &ModEntry) -> u32 {
    let mut new_lvl: u32 = DEFAULT_LOG_LEVEL;
    let mut longest_match = -1i;
    for dir in dirs.iter() {
        match dir.name {
            None => {
                if longest_match == -1 {
                    longest_match = 0;
                    new_lvl = dir.level;
                }
            }
            Some(ref dir_name) => {
                let name = entry.name;
                let len = dir_name.len() as int;
                if name.starts_with(*dir_name) &&
                    len >= longest_match {
                    longest_match = len;
                    new_lvl = dir.level;
                }
            }
        };
    }
    unsafe { *entry.log_level = new_lvl; }
    if longest_match >= 0 { return 1; } else { return 0; }
}

#[fixed_stack_segment] #[inline(never)]
/// Set log level for every entry in crate_map according to the sepecification
/// in settings
fn update_log_settings(crate_map: &CrateMap, settings: ~str) {
    let mut dirs = ~[];
    if settings.len() > 0 {
        if settings == ~"::help" || settings == ~"?" {
            rterrln!("\nCrate log map:\n");
            do iter_crate_map(crate_map) |entry| {
                rterrln!(" {}", entry.name);
            }
            unsafe { exit(1); }
        }
        dirs = parse_logging_spec(settings);
    }

    let mut n_matches: u32 = 0;
    do iter_crate_map(crate_map) |entry| {
        let m = update_entry(dirs, entry);
        n_matches += m;
    }

    if n_matches < (dirs.len() as u32) {
        rterrln!("warning: got {} RUST_LOG specs but only matched\n\
                  {} of them. You may have mistyped a RUST_LOG spec. \n\
                  Use RUST_LOG=::help to see the list of crates and modules.\n",
                 dirs.len(), n_matches);
    }
}

pub trait Logger {
    fn log(&mut self, args: &fmt::Arguments);
}

pub struct StdErrLogger;

impl Logger for StdErrLogger {
    fn log(&mut self, args: &fmt::Arguments) {
        // FIXME(#6846): this should not call the blocking version of println,
        //               or at least the default loggers for tasks shouldn't do
        //               that
        ::rt::util::dumb_println(args);
    }
}

/// Configure logging by traversing the crate map and setting the
/// per-module global logging flags based on the logging spec
pub fn init() {
    use os;

    let log_spec = os::getenv("RUST_LOG");
    match get_crate_map() {
        Some(crate_map) => {
            match log_spec {
                Some(spec) => {
                    update_log_settings(crate_map, spec);
                }
                None => {
                    update_log_settings(crate_map, ~"");
                }
            }
        },
        _ => {
            match log_spec {
                Some(_) => {
                    rterrln!("warning: RUST_LOG set, but no crate map found.");
                },
                None => {}
            }
        }
    }
}

// Tests for parse_logging_spec()
#[test]
fn parse_logging_spec_valid() {
    let dirs = parse_logging_spec(~"crate1::mod1=1,crate1::mod2,crate2=4");
    assert_eq!(dirs.len(), 3);
    assert!(dirs[0].name == Some(~"crate1::mod1"));
    assert_eq!(dirs[0].level, 1);

    assert!(dirs[1].name == Some(~"crate1::mod2"));
    assert_eq!(dirs[1].level, MAX_LOG_LEVEL);

    assert!(dirs[2].name == Some(~"crate2"));
    assert_eq!(dirs[2].level, 4);
}

#[test]
fn parse_logging_spec_invalid_crate() {
    // test parse_logging_spec with multiple = in specification
    let dirs = parse_logging_spec(~"crate1::mod1=1=2,crate2=4");
    assert_eq!(dirs.len(), 1);
    assert!(dirs[0].name == Some(~"crate2"));
    assert_eq!(dirs[0].level, 4);
}

#[test]
fn parse_logging_spec_invalid_log_level() {
    // test parse_logging_spec with 'noNumber' as log level
    let dirs = parse_logging_spec(~"crate1::mod1=noNumber,crate2=4");
    assert_eq!(dirs.len(), 1);
    assert!(dirs[0].name == Some(~"crate2"));
    assert_eq!(dirs[0].level, 4);
}

#[test]
fn parse_logging_spec_string_log_level() {
    // test parse_logging_spec with 'warn' as log level
    let dirs = parse_logging_spec(~"crate1::mod1=wrong,crate2=warn");
    assert_eq!(dirs.len(), 1);
    assert!(dirs[0].name == Some(~"crate2"));
    assert_eq!(dirs[0].level, 2);
}

#[test]
fn parse_logging_spec_global() {
    // test parse_logging_spec with no crate
    let dirs = parse_logging_spec(~"warn,crate2=4");
    assert_eq!(dirs.len(), 2);
    assert!(dirs[0].name == None);
    assert_eq!(dirs[0].level, 2);
    assert!(dirs[1].name == Some(~"crate2"));
    assert_eq!(dirs[1].level, 4);
}

// Tests for update_entry
#[test]
fn update_entry_match_full_path() {
    let dirs = ~[LogDirective {name: Some(~"crate1::mod1"), level: 2 },
                 LogDirective {name: Some(~"crate2"), level: 3}];
    let level = &mut 0;
    unsafe {
        let entry= &ModEntry {name:"crate1::mod1", log_level: level};
        let m = update_entry(dirs, transmute(entry));
        assert!(*entry.log_level == 2);
        assert!(m == 1);
    }
}

#[test]
fn update_entry_no_match() {
    let dirs = ~[LogDirective {name: Some(~"crate1::mod1"), level: 2 },
                 LogDirective {name: Some(~"crate2"), level: 3}];
    let level = &mut 0;
    unsafe {
        let entry= &ModEntry {name: "crate3::mod1", log_level: level};
        let m = update_entry(dirs, transmute(entry));
        assert!(*entry.log_level == DEFAULT_LOG_LEVEL);
        assert!(m == 0);
    }
}

#[test]
fn update_entry_match_beginning() {
    let dirs = ~[LogDirective {name: Some(~"crate1::mod1"), level: 2 },
                 LogDirective {name: Some(~"crate2"), level: 3}];
    let level = &mut 0;
    unsafe {
        let entry= &ModEntry {name: "crate2::mod1", log_level: level};
        let m = update_entry(dirs, transmute(entry));
        assert!(*entry.log_level == 3);
        assert!(m == 1);
    }
}

#[test]
fn update_entry_match_beginning_longest_match() {
    let dirs = ~[LogDirective {name: Some(~"crate1::mod1"), level: 2 },
                 LogDirective {name: Some(~"crate2"), level: 3},
                 LogDirective {name: Some(~"crate2::mod"), level: 4}];
    let level = &mut 0;
    unsafe {
        let entry = &ModEntry {name: "crate2::mod1", log_level: level};
        let m = update_entry(dirs, transmute(entry));
        assert!(*entry.log_level == 4);
        assert!(m == 1);
    }
}

#[test]
fn update_entry_match_default() {
    let dirs = ~[LogDirective {name: Some(~"crate1::mod1"), level: 2 },
                 LogDirective {name: None, level: 3}
                ];
    let level = &mut 0;
    unsafe {
        let entry= &ModEntry {name: "crate1::mod1", log_level: level};
        let m = update_entry(dirs, transmute(entry));
        assert!(*entry.log_level == 2);
        assert!(m == 1);
        let entry= &ModEntry {name: "crate2::mod2", log_level: level};
        let m = update_entry(dirs, transmute(entry));
        assert!(*entry.log_level == 3);
        assert!(m == 1);
    }
}
