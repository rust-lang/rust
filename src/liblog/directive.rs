// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ascii::AsciiExt;
use std::cmp;

#[derive(Debug, Clone)]
pub struct LogDirective {
    pub name: Option<String>,
    pub level: u32,
}

pub const LOG_LEVEL_NAMES: [&'static str; 4] = ["ERROR", "WARN", "INFO", "DEBUG"];

/// Parse an individual log level that is either a number or a symbolic log level
fn parse_log_level(level: &str) -> Option<u32> {
    level.parse::<u32>()
         .ok()
         .or_else(|| {
             let pos = LOG_LEVEL_NAMES.iter().position(|&name| name.eq_ignore_ascii_case(level));
             pos.map(|p| p as u32 + 1)
         })
         .map(|p| cmp::min(p, ::MAX_LOG_LEVEL))
}

/// Parse a logging specification string (e.g: "crate1,crate2::mod3,crate3::x=1/foo")
/// and return a vector with log directives.
///
/// Valid log levels are 0-255, with the most likely ones being 1-4 (defined in
/// std::).  Also supports string log levels of error, warn, info, and debug
pub fn parse_logging_spec(spec: &str) -> (Vec<LogDirective>, Option<String>) {
    let mut dirs = Vec::new();

    let mut parts = spec.split('/');
    let mods = parts.next();
    let filter = parts.next();
    if parts.next().is_some() {
        println!("warning: invalid logging spec '{}', ignoring it (too many '/'s)",
                 spec);
        return (dirs, None);
    }
    if let Some(m) = mods {
        for s in m.split(',') {
            if s.is_empty() {
                continue;
            }
            let mut parts = s.split('=');
            let (log_level, name) = match (parts.next(),
                                           parts.next().map(|s| s.trim()),
                                           parts.next()) {
                (Some(part0), None, None) => {
                    // if the single argument is a log-level string or number,
                    // treat that as a global fallback
                    match parse_log_level(part0) {
                        Some(num) => (num, None),
                        None => (::MAX_LOG_LEVEL, Some(part0)),
                    }
                }
                (Some(part0), Some(""), None) => (::MAX_LOG_LEVEL, Some(part0)),
                (Some(part0), Some(part1), None) => {
                    match parse_log_level(part1) {
                        Some(num) => (num, Some(part0)),
                        _ => {
                            println!("warning: invalid logging spec '{}', ignoring it", part1);
                            continue;
                        }
                    }
                }
                _ => {
                    println!("warning: invalid logging spec '{}', ignoring it", s);
                    continue;
                }
            };
            dirs.push(LogDirective {
                name: name.map(str::to_owned),
                level: log_level,
            });
        }
    }

    (dirs, filter.map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::parse_logging_spec;

    #[test]
    fn parse_logging_spec_valid() {
        let (dirs, filter) = parse_logging_spec("crate1::mod1=1,crate1::mod2,crate2=4");
        assert_eq!(dirs.len(), 3);
        assert_eq!(dirs[0].name, Some("crate1::mod1".to_owned()));
        assert_eq!(dirs[0].level, 1);

        assert_eq!(dirs[1].name, Some("crate1::mod2".to_owned()));
        assert_eq!(dirs[1].level, ::MAX_LOG_LEVEL);

        assert_eq!(dirs[2].name, Some("crate2".to_owned()));
        assert_eq!(dirs[2].level, 4);
        assert!(filter.is_none());
    }

    #[test]
    fn parse_logging_spec_invalid_crate() {
        // test parse_logging_spec with multiple = in specification
        let (dirs, filter) = parse_logging_spec("crate1::mod1=1=2,crate2=4");
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_owned()));
        assert_eq!(dirs[0].level, 4);
        assert!(filter.is_none());
    }

    #[test]
    fn parse_logging_spec_invalid_log_level() {
        // test parse_logging_spec with 'noNumber' as log level
        let (dirs, filter) = parse_logging_spec("crate1::mod1=noNumber,crate2=4");
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_owned()));
        assert_eq!(dirs[0].level, 4);
        assert!(filter.is_none());
    }

    #[test]
    fn parse_logging_spec_string_log_level() {
        // test parse_logging_spec with 'warn' as log level
        let (dirs, filter) = parse_logging_spec("crate1::mod1=wrong,crate2=warn");
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_owned()));
        assert_eq!(dirs[0].level, ::WARN);
        assert!(filter.is_none());
    }

    #[test]
    fn parse_logging_spec_empty_log_level() {
        // test parse_logging_spec with '' as log level
        let (dirs, filter) = parse_logging_spec("crate1::mod1=wrong,crate2=");
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_owned()));
        assert_eq!(dirs[0].level, ::MAX_LOG_LEVEL);
        assert!(filter.is_none());
    }

    #[test]
    fn parse_logging_spec_global() {
        // test parse_logging_spec with no crate
        let (dirs, filter) = parse_logging_spec("warn,crate2=4");
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0].name, None);
        assert_eq!(dirs[0].level, 2);
        assert_eq!(dirs[1].name, Some("crate2".to_owned()));
        assert_eq!(dirs[1].level, 4);
        assert!(filter.is_none());
    }

    #[test]
    fn parse_logging_spec_valid_filter() {
        let (dirs, filter) = parse_logging_spec("crate1::mod1=1,crate1::mod2,crate2=4/abc");
        assert_eq!(dirs.len(), 3);
        assert_eq!(dirs[0].name, Some("crate1::mod1".to_owned()));
        assert_eq!(dirs[0].level, 1);

        assert_eq!(dirs[1].name, Some("crate1::mod2".to_owned()));
        assert_eq!(dirs[1].level, ::MAX_LOG_LEVEL);

        assert_eq!(dirs[2].name, Some("crate2".to_owned()));
        assert_eq!(dirs[2].level, 4);
        assert!(filter.is_some() && filter.unwrap().to_owned() == "abc");
    }

    #[test]
    fn parse_logging_spec_invalid_crate_filter() {
        let (dirs, filter) = parse_logging_spec("crate1::mod1=1=2,crate2=4/a.c");
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_owned()));
        assert_eq!(dirs[0].level, 4);
        assert!(filter.is_some() && filter.unwrap().to_owned() == "a.c");
    }

    #[test]
    fn parse_logging_spec_empty_with_filter() {
        let (dirs, filter) = parse_logging_spec("crate1/a*c");
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate1".to_owned()));
        assert_eq!(dirs[0].level, ::MAX_LOG_LEVEL);
        assert!(filter.is_some() && filter.unwrap().to_owned() == "a*c");
    }
}
