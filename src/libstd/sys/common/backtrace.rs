// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use io::prelude::*;

use io;

#[cfg(target_pointer_width = "64")]
pub const HEX_WIDTH: usize = 18;

#[cfg(target_pointer_width = "32")]
pub const HEX_WIDTH: usize = 10;

// All rust symbols are in theory lists of "::"-separated identifiers. Some
// assemblers, however, can't handle these characters in symbol names. To get
// around this, we use C++-style mangling. The mangling method is:
//
// 1. Prefix the symbol with "_ZN"
// 2. For each element of the path, emit the length plus the element
// 3. End the path with "E"
//
// For example, "_ZN4testE" => "test" and "_ZN3foo3bar" => "foo::bar".
//
// We're the ones printing our backtraces, so we can't rely on anything else to
// demangle our symbols. It's *much* nicer to look at demangled symbols, so
// this function is implemented to give us nice pretty output.
//
// Note that this demangler isn't quite as fancy as it could be. We have lots
// of other information in our symbols like hashes, version, type information,
// etc. Additionally, this doesn't handle glue symbols at all.
pub fn demangle(writer: &mut Write, s: &str) -> io::Result<()> {
    // First validate the symbol. If it doesn't look like anything we're
    // expecting, we just print it literally. Note that we must handle non-rust
    // symbols because we could have any function in the backtrace.
    let mut valid = true;
    let mut inner = s;
    if s.len() > 4 && s.starts_with("_ZN") && s.ends_with("E") {
        inner = &s[3 .. s.len() - 1];
    // On Windows, dbghelp strips leading underscores, so we accept "ZN...E" form too.
    } else if s.len() > 3 && s.starts_with("ZN") && s.ends_with("E") {
        inner = &s[2 .. s.len() - 1];
    } else {
        valid = false;
    }

    if valid {
        let mut chars = inner.chars();
        while valid {
            let mut i = 0;
            for c in chars.by_ref() {
                if c.is_numeric() {
                    i = i * 10 + c as usize - '0' as usize;
                } else {
                    break
                }
            }
            if i == 0 {
                valid = chars.next().is_none();
                break
            } else if chars.by_ref().take(i - 1).count() != i - 1 {
                valid = false;
            }
        }
    }

    // Alright, let's do this.
    if !valid {
        try!(writer.write_all(s.as_bytes()));
    } else {
        let mut first = true;
        while !inner.is_empty() {
            if !first {
                try!(writer.write_all(b"::"));
            } else {
                first = false;
            }
            let mut rest = inner;
            while rest.char_at(0).is_numeric() {
                rest = &rest[1..];
            }
            let i: usize = inner[.. (inner.len() - rest.len())].parse().unwrap();
            inner = &rest[i..];
            rest = &rest[..i];
            while !rest.is_empty() {
                if rest.starts_with("$") {
                    macro_rules! demangle {
                        ($($pat:expr, => $demangled:expr),*) => ({
                            $(if rest.starts_with($pat) {
                                try!(writer.write_all($demangled));
                                rest = &rest[$pat.len()..];
                              } else)*
                            {
                                try!(writer.write_all(rest.as_bytes()));
                                break;
                            }

                        })
                    }

                    // see src/librustc/back/link.rs for these mappings
                    demangle! (
                        "$SP$", => b"@",
                        "$BP$", => b"*",
                        "$RF$", => b"&",
                        "$LT$", => b"<",
                        "$GT$", => b">",
                        "$LP$", => b"(",
                        "$RP$", => b")",
                        "$C$", => b",",

                        // in theory we can demangle any Unicode code point, but
                        // for simplicity we just catch the common ones.
                        "$u7e$", => b"~",
                        "$u20$", => b" ",
                        "$u27$", => b"'",
                        "$u5b$", => b"[",
                        "$u5d$", => b"]"
                    )
                } else {
                    let idx = match rest.find('$') {
                        None => rest.len(),
                        Some(i) => i,
                    };
                    try!(writer.write_all(rest[..idx].as_bytes()));
                    rest = &rest[idx..];
                }
            }
        }
    }

    Ok(())
}
