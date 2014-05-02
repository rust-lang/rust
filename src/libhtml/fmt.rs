// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! HTML Escaping
//!
//! This module contains one unit-struct which can be used to HTML-escape a
//! string of text (for use in a format string).

use std::cast;
use std::char;
use std::fmt;
use std::io::{Writer, IoResult};
use std::str;
use entity;

/// Wrapper struct which will emit the HTML-escaped version of the contained
/// `Show` when passed to a format string.
pub struct Escape<T>(pub T);

impl<T: fmt::Show> fmt::Show for Escape<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Escape(ref inner) = *self;
        write!(&mut EscapeWriter::new(&mut fmt.buf), "{}", inner)
    }
}

/// Wrapper struct which will unescape HTML entities in the contained
/// `Show` when passed to a format string.
pub struct Unescape<T>(pub T);

impl<T: fmt::Show> fmt::Show for Unescape<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Unescape(ref inner) = *self;
        let mut w = UnescapeWriter::new(&mut fmt.buf);
        try!(write!(&mut w, "{}", inner));
        w.consume().and_then(|_| Ok(()))
    }
}

struct EscapeWriter<'a, W> {
    inner: &'a mut W
}

impl<'a, W: Writer> EscapeWriter<'a, W> {
    fn new(inner: &'a mut W) -> EscapeWriter<'a, W> {
        EscapeWriter {
            inner: inner
        }
    }
}

impl<'a, W:Writer> Writer for EscapeWriter<'a, W> {
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> {
        // Because the internet is always right, turns out there's not that many
        // characters to escape: http://stackoverflow.com/questions/7381974
        let mut last = 0;
        for (i, b) in bytes.iter().enumerate() {
            let ent = match *b as char {
                '<' => "&lt;",
                '>' => "&gt;",
                '&' => "&amp;",
                '\'' => "&apos;",
                '"' => "&quot;",
                _ => continue
            };
            if last < i {
                try!(self.inner.write(bytes.slice(last, i)));
            }
            try!(self.inner.write(ent.as_bytes()));
            last = i + 1;
        }

        if last < bytes.len() {
            try!(self.inner.write(bytes.slice_from(last)));
        }
        Ok(())
    }
}


struct UnescapeWriter<'a, W> {
    state: UnescapeState,
    inner: &'a mut W
}

enum UnescapeState {
    Normal,
    Entity,
    Named([u8, ..entity::MAX_ENTITY_LENGTH], uint),
    Numeric,
    Hex(u32),
    Dec(u32)
}

impl fmt::Show for UnescapeState {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Normal => write!(fmt.buf, "Normal"),
            Entity => write!(fmt.buf, "Entity"),
            Named(buf, x) => write!(fmt.buf, "Named({})", str::from_utf8(buf.slice_to(x))),
            Numeric => write!(fmt.buf, "Numeric"),
            Hex(x) => write!(fmt.buf, "Hex({})", x),
            Dec(x) => write!(fmt.buf, "Dec({})", x),
        }
    }
}

impl<'a, W: Writer> UnescapeWriter<'a, W> {
    fn new(inner: &'a mut W) -> UnescapeWriter<'a, W> {
        UnescapeWriter {
            state: Normal,
            inner: inner
        }
    }

    fn consume(&mut self) -> IoResult<uint> {
        let state = self.state;
        self.state = Normal;
        match state {
            Named(buf, pos) => {
                // safe because the only way a byte gets added to the buff is
                // if it's alphanumeric
                let buf: &str = unsafe { cast::transmute(buf.slice_to(pos)) };
                let len = pos + 1;
                for i in range(0, len).rev() {
                    let s = buf.slice_to(i);
                    let unescaped = entity::lookup(s);
                    //println!("lookup: {}: {}", s, unescaped);
                    match unescaped {
                        None => {},
                        Some(s) => {
                            try!(self.inner.write_str(s));
                            if i < pos {
                                try!(self.inner.write_str(buf.slice_from(i)));
                            }
                            return Ok(len);
                        }
                    }
                }

                // never found a match. write & and original buf
                try!(self.inner.write_char('&'));
                try!(self.inner.write_str(buf));
                Ok(len)
            },
            Hex(n) | Dec(n) => {
                let n = match n {
                    0x80 .. 0x9f => {
                        COMPAT_TABLE[(n-0x80) as uint]
                    },
                    0 | 0xD800 .. 0xDFFF => {
                        0xDFFF
                    },
                    n => n
                };
                let ch = match char::from_u32(n) {
                    Some(ch) => ch,
                    None => {
                        return Ok(0);
                    }
                };

                try!(self.inner.write_char(ch));
                Ok(1)
            },
            Entity => {
                try!(self.inner.write_char('&'));
                Ok(1)
            },
            _ => Ok(0)
        }
    }
}

#[allow(unused_mut)] //FIXME(#13866)
impl<'a, W:Writer> Writer for UnescapeWriter<'a, W> {
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> {
        let mut last = 0u;
        let mut i = 0u;
        let mut iter = bytes.iter();
        let mut b = 0u8;
        let mut eaten = true;
        loop {
            if eaten {
                b = match iter.next() {
                    Some(&b) => b,
                    None => break
                };
            } else {
                eaten = true;
            }
            let n = try!(match (self.state, b as char) {
                (Normal, '&') => {
                    self.state = Entity;

                    if i > last {
                        try!(self.inner.write(bytes.slice(last, i)));
                    }
                    Ok(i + 1 - last)
                },

                (Normal, _) => Ok(0),

                (Entity, '#') => {
                    self.state = Numeric;
                    Ok(1)
                },

                (Entity, 'a'..'z') |
                (Entity, 'A'..'Z') |
                (Entity, '0'..'9') => {
                    let mut buf = [0, .. entity::MAX_ENTITY_LENGTH];
                    buf[0] = b;
                    self.state = Named(buf, 1);
                    Ok(0)
                },


                (Named(mut buf, x), 'a'..'z') |
                (Named(mut buf, x), 'A'..'Z') |
                (Named(mut buf, x), '0'..'9') => {
                    buf[x] = b;
                    self.state = Named(buf, x + 1);
                    if x + 1 == entity::MAX_ENTITY_LENGTH {
                        self.consume()
                    } else {
                        Ok(0)
                    }
                },

                (Numeric, '0'..'9') => {
                    self.state = Dec(b as u32 - '0' as u32);
                    Ok(1)
                },

                (Numeric, 'x') | (Numeric, 'X') => {
                    self.state = Hex(0);
                    Ok(1)
                },

                (Dec(x), '0'..'9') => {
                    self.state = Dec(10 * x + b as u32 - '0' as u32);
                    Ok(1)
                },

                (Hex(x), '0'..'9') => {
                    self.state = Hex(16 * x + b as u32 - '0' as u32);
                    Ok(1)
                },
                (Hex(x), 'a'..'f') => {
                    self.state = Hex(16 * x + b as u32 - 'a' as u32);
                    Ok(1)
                },
                (Hex(x), 'A'..'F') => {
                    self.state = Hex(16 * x + b as u32 - 'A' as u32);
                    Ok(1)
                },

                (Named(..), ';') | (Dec(..), ';') | (Hex(..), ';') => {
                    let r = self.consume();
                    r
                },

                _ => {
                    // parse error? consume what we have and try b again

                    let r = self.consume();
                    eaten = false;
                    r
                }
            });

            //println!("i: {}, b: {}, n: {}, state: {}", i, b as char, n, self.state);
            last += n;
            if eaten {
                i += 1;
            }
        }

        match (self.state, last < bytes.len()) {
            (Normal, true) => self.inner.write(bytes.slice_from(last)),
            _ => Ok(())
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        self.consume().and_then(|_| self.inner.flush())
    }
}



static COMPAT_TABLE: [u32, ..32] = [
    '\u20AC' as u32, // First entry is what 0x80 should be replaced with.
    '\u0081' as u32,
    '\u201A' as u32,
    '\u0192' as u32,
    '\u201E' as u32,
    '\u2026' as u32,
    '\u2020' as u32,
    '\u2021' as u32,
    '\u02C6' as u32,
    '\u2030' as u32,
    '\u0160' as u32,
    '\u2039' as u32,
    '\u0152' as u32,
    '\u008D' as u32,
    '\u017D' as u32,
    '\u008F' as u32,
    '\u0090' as u32,
    '\u2018' as u32,
    '\u2019' as u32,
    '\u201C' as u32,
    '\u201D' as u32,
    '\u2022' as u32,
    '\u2013' as u32,
    '\u2014' as u32,
    '\u02DC' as u32,
    '\u2122' as u32,
    '\u0161' as u32,
    '\u203A' as u32,
    '\u0153' as u32,
    '\u009D' as u32,
    '\u017E' as u32,
    '\u0178' as u32, // Last entry is 0x9F.
    // 0x00->'\uFFFD' is handled programmatically.
];
