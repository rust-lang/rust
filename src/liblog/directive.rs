// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ascii::StrAsciiExt;
use std::cmp;

#[deriving(Show, Clone)]
pub struct LogDirective {
    pub name: Option<StrBuf>,
    pub level: u32,
}

pub static LOG_LEVEL_NAMES: [&'static str, ..4] = ["ERROR", "WARN", "INFO",
                                               "DEBUG"];

/// Parse an individual log level that is either a number or a symbolic log level
fn parse_log_level(level: &str) -> Option<u32> {
    from_str::<u32>(level).or_else(|| {
        let pos = LOG_LEVEL_NAMES.iter().position(|&name| name.eq_ignore_ascii_case(level));
        pos.map(|p| p as u32 + 1)
    }).map(|p| cmp::min(p, ::MAX_LOG_LEVEL))
}

/// Parse a logging specification string (e.g: "crate1,crate2::mod3,crate3::x=1")
/// and return a vector with log directives.
///
/// Valid log levels are 0-255, with the most likely ones being 1-4 (defined in
/// std::).  Also supports string log levels of error, warn, info, and debug
pub fn parse_logging_spec(spec: &str) -> Vec<LogDirective> {
    let mut dirs = Vec::new();
    for s in spec.split(',') {
        if s.len() == 0 { continue }
        let mut parts = s.split('=');
        let (log_level, name) = match (parts.next(), parts.next(), parts.next()) {
            (Some(part0), None, None) => {
                // if the single argument is a log-level string or number,
                // treat that as a global fallback
                match parse_log_level(part0) {
                    Some(num) => (num, None),
                    None => (::MAX_LOG_LEVEL, Some(part0)),
                }
            }
            (Some(part0), Some(part1), None) => {
                match parse_log_level(part1) {
                    Some(num) => (num, Some(part0)),
                    _ => {
                        println!("warning: invalid logging spec '{}', \
                                 ignoring it", part1);
                        continue
                    }
                }
            },
            _ => {
                println!("warning: invalid logging spec '{}', \
                         ignoring it", s);
                continue
            }
        };
        dirs.push(LogDirective {
            name: name.map(|s| s.to_strbuf()),
            level: log_level,
        });
    }
    return dirs;
}

#[cfg(test)]
mod tests {
    use super::parse_logging_spec;

    #[test]
    fn parse_logging_spec_valid() {
        let dirs = parse_logging_spec("crate1::mod1=1,crate1::mod2,crate2=4");
        let dirs = dirs.as_slice();
        assert_eq!(dirs.len(), 3);
        assert_eq!(dirs[0].name, Some("crate1::mod1".to_strbuf()));
        assert_eq!(dirs[0].level, 1);

        assert_eq!(dirs[1].name, Some("crate1::mod2".to_strbuf()));
        assert_eq!(dirs[1].level, ::MAX_LOG_LEVEL);

        assert_eq!(dirs[2].name, Some("crate2".to_strbuf()));
        assert_eq!(dirs[2].level, 4);
    }

    #[test]
    fn parse_logging_spec_invalid_crate() {
        // test parse_logging_spec with multiple = in specification
        let dirs = parse_logging_spec("crate1::mod1=1=2,crate2=4");
        let dirs = dirs.as_slice();
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_strbuf()));
        assert_eq!(dirs[0].level, 4);
    }

    #[test]
    fn parse_logging_spec_invalid_log_level() {
        // test parse_logging_spec with 'noNumber' as log level
        let dirs = parse_logging_spec("crate1::mod1=noNumber,crate2=4");
        let dirs = dirs.as_slice();
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_strbuf()));
        assert_eq!(dirs[0].level, 4);
    }

    #[test]
    fn parse_logging_spec_string_log_level() {
        // test parse_logging_spec with 'warn' as log level
        let dirs = parse_logging_spec("crate1::mod1=wrong,crate2=warn");
        let dirs = dirs.as_slice();
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].name, Some("crate2".to_strbuf()));
        assert_eq!(dirs[0].level, ::WARN);
    }

    #[test]
    fn parse_logging_spec_global() {
        // test parse_logging_spec with no crate
        let dirs = parse_logging_spec("warn,crate2=4");
        let dirs = dirs.as_slice();
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0].name, None);
        assert_eq!(dirs[0].level, 2);
        assert_eq!(dirs[1].name, Some("crate2".to_strbuf()));
        assert_eq!(dirs[1].level, 4);
    }
}
