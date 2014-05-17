// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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
//! This module contains `Writer`s for escaping/unescaping HTML.

use std::io::{Writer, IoResult};
use std::char;
use entity::ENTITIES;

/// A `Writer` adaptor that escapes any HTML characters written to it.
pub struct EscapeWriter<W> {
    inner: W,
    mode: EscapeMode
}

/// The mode that controls which characters need escaping.
#[deriving(Eq,Show)]
pub enum EscapeMode {
    /// The general-purpose mode. Escapes ``&<>"'`.
    EscapeDefault,
    /// Escapes characters for text nodes. Escapes `&<>`.
    EscapeText,
    /// Escapes characters for double-quoted attribute values. Escapes `&"`.
    EscapeAttr,
    /// Escapes characters for single-quoted attribute values. Escapes `&'`.
    EscapeSingleQuoteAttr
}

impl<W: Writer> EscapeWriter<W> {
    /// Creates a new `EscapeWriter` with the given mode.
    pub fn new(inner: W, mode: EscapeMode) -> EscapeWriter<W> {
        EscapeWriter {
            inner: inner,
            mode: mode
        }
    }

    /// Gets a reference to the underlying `Writer`.
    pub fn get_ref<'a>(&'a self) -> &'a W {
        &self.inner
    }

    /// Gets a mutable reference to the underlying `Writer`.
    pub fn get_mut_ref<'a>(&'a mut self) -> &'a mut W {
        &mut self.inner
    }

    /// Unwraps this `EscapeWriter`, returning the underlying writer.
    pub fn unwrap(self) -> W {
        self.inner
    }
}

impl<W: Writer> Writer for EscapeWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> {
        let mut last = 0;
        for (i, b) in bytes.iter().enumerate() {
            let ent = match (self.mode, *b as char) {
                (_,'&') => "&amp;",
                (EscapeDefault,'<') |(EscapeText,'<')             => "&lt;",
                (EscapeDefault,'>') |(EscapeText,'>')             => "&gt;",
                (EscapeDefault,'\'')|(EscapeSingleQuoteAttr,'\'') => "&#39;",
                (EscapeDefault,'"') |(EscapeAttr,'"')             => "&quot;",
                _ => continue
            };
            if last < i {
                try!(self.inner.write(bytes.slice(last, i)));
            }
            try!(self.inner.write_str(ent));
            last = i + 1;
        }
        if last < bytes.len() {
            try!(self.inner.write(bytes.slice_from(last)));
        }
        Ok(())
    }

    fn flush(&mut self) -> IoResult<()> {
        self.inner.flush()
    }
}

/// A `Writer` adaptor that decodes any HTML entities in the text written to it.
pub struct UnescapeWriter<W> {
    state: UnescapeState,
    inner: Option<W>,
    allowed: Option<char>
}

enum UnescapeState {
    CharData,
    Begin,
    Named(uint, uint, uint), // index into ENTITIES, prefix len, last non-semi index
    HexStart(bool), // boolean indicates if x is lower or upper case
    Hex(u32),
    DecStart,
    Dec(u32)
}

impl<W: Writer> UnescapeWriter<W> {
    /// Creates a new `UnescapeWriter`.
    pub fn new(inner: W) -> UnescapeWriter<W> {
        UnescapeWriter {
            state: CharData,
            inner: Some(inner),
            allowed: None
        }
    }

    /// Creates a new `UnescapeWriter` with the specified allowed additional character.
    ///
    /// The allowed additional character may occur after `'&'` to indicate that this is not
    /// an entity.
    pub fn with_allowed_char(inner: W, allowed: char) -> UnescapeWriter<W> {
        UnescapeWriter {
            state: CharData,
            inner: Some(inner),
            allowed: Some(allowed)
        }
    }

    /// Gets a reference to the underlying `Writer`.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying `Writer` because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W {
        self.inner.get_ref()
    }

    /// Unwraps this `UnescapeWriter`, returning the underlying `Writer`.
    ///
    /// The `UnescapeWriter` is flushed before returning the `Writer`, but the
    /// `Writer` is not flushed.
    ///
    /// # Failure
    ///
    /// Fails if the outer flush returns an error. Call `flush_outer()`
    /// explicitly to handle this.
    pub fn unwrap(mut self) -> W {
        self.flush_outer().unwrap();
        self.inner.take_unwrap()
    }

    /// Flushes the `UnescapeWriter` without flushing the wrapped `Writer`.
    ///
    /// If the `UnescapeWriter` is in the middle of parsing an entity
    /// reference, it will behave as though EOF were encountered and write the
    /// approprite characters. Otherwise, this does nothing.
    pub fn flush_outer(&mut self) -> IoResult<()> {
        self.abort_entity()
    }

    // Called when a character is encountered that isn't valid
    fn abort_entity(&mut self) -> IoResult<()> {
        let state = self.state;
        self.state = CharData;
        match state {
            CharData => (),
            Begin => {
                try!(self.inner.get_mut_ref().write_str("&"));
            }
            Named(cursor, plen, lastcur) => {
                let (name, chars, needs_semi) = ENTITIES[cursor];
                if !needs_semi && name.len() == plen {
                    try!(self.inner.get_mut_ref().write_str(chars));
                } else if lastcur != -1 {
                    let (lastname, chars, _) = ENTITIES[lastcur];
                    try!(self.inner.get_mut_ref().write_str(chars));
                    try!(self.inner.get_mut_ref().write_str(name.slice(lastname.len(), plen)));
                } else {
                    try!(self.inner.get_mut_ref().write_str(name.slice_to(plen)));
                }
            }
            DecStart => {
                try!(self.inner.get_mut_ref().write_str("&#"));
            }
            HexStart(false) => {
                try!(self.inner.get_mut_ref().write_str("&#x"));
            }
            HexStart(true) => {
                try!(self.inner.get_mut_ref().write_str("&#X"));
            }
            Hex(val) | Dec(val) => {
                let c = match char::from_u32(val) {
                    None|Some('\0') => '\uFFFD',
                    Some(c@'\x80'..'\x9F') => {
                        COMPAT_TABLE[c as uint - 0x80]
                    }
                    Some(c) => c
                };
                let mut buf = [0u8, ..4];
                let n = c.encode_utf8(buf);
                try!(self.inner.get_mut_ref().write(buf.slice_to(n)));
                self.state = CharData;
            }
        }
        Ok(())
    }

    fn inner_write(&mut self, bytes: &[u8]) -> IoResult<()> {
        match self.inner.get_mut_ref().write(bytes) {
            ok@Ok(_) => ok,
            err@Err(_) => {
                self.state = CharData;
                err
            }
        }
    }

    fn inner_write_str(&mut self, s: &str) -> IoResult<()> {
        match self.inner.get_mut_ref().write_str(s) {
            ok@Ok(_) => ok,
            err@Err(_) => {
                self.state = CharData;
                err
            }
        }
    }
}

#[unsafe_destructor]
impl<W: Writer> Drop for UnescapeWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            // Ignore this error, we don't want to fail in Drop
            let _ = self.flush_outer();
        }
    }
}

impl<W:Writer> Writer for UnescapeWriter<W> {
    /// Writes `bytes` to the underlying `Writer`, unescaping any HTML entities.
    ///
    /// If an error is returned, this `UnescapeWriter` discards its internal state,
    /// forgetting any in-progress entities.
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> {
        let mut it = bytes.iter().enumerate().peekable();
        let mut cdata = 0;
        loop {
            let (i, b) = match it.peek() {
                None => break,
                Some(&(i, &b)) => (i, b)
            };
            match (self.state, b as char) {
                (CharData, '&') => {
                    it.next(); // consume &
                    match it.peek().map(|&(_,&b)| b as char) {
                        None|Some('\x09')|Some('\x0A')|Some('\x0C')|
                        Some(' ')|Some('<')|Some('&') => {
                            // This is an allowed character
                            continue;
                        }
                        Some(c) if self.allowed == Some(c) => {
                            // this is the additional allowed character
                            continue;
                        }
                        _ => ()
                    }
                    if i > cdata {
                        try!(self.inner_write(bytes.slice(cdata,i)));
                    }
                    self.state = Begin
                }
                (CharData, _) => {
                    it.next(); // consume character
                }
                (Begin, '#') => {
                    it.next(); // consume #
                    self.state = match it.peek().map(|&(_,&b)| b as char) {
                        Some('x') => {
                            it.next(); // consume x
                            HexStart(false)
                        }
                        Some('X') => {
                            it.next(); // consume X
                            HexStart(true)
                        }
                        _ => DecStart
                    }
                }
                (Begin, 'a'..'z')|(Begin, 'A'..'Z') => {
                    // No entities start with digits, so we don't have to check that
                    it.next(); // consume character
                    // Find the first entity that starts with this character
                    // The array is sorted, so we can binary search.
                    // Unfortunately there's no existing function to find the "insert location"
                    // for a key in a sorted vector, so let's implement it now.
                    let key: &[u8] = [b];
                    let mut base: uint = 0;
                    let mut lim: uint = ENTITIES.len();
                    while lim != 0 {
                        let ix = base + (lim >> 1);
                        let (name, _, _) = ENTITIES[ix];
                        let name = name.slice_from(1); // trim off &
                        if key > name.as_bytes() {
                            base = ix + 1;
                            lim -= 1;
                        }
                        // key will never == name, there are no 1-char entities
                        lim >>= 1;
                    }
                    // base contains the insertion index, which is the first element
                    // with our character as a prefix.
                    // There's at least one entity that starts with every letter, so we don't
                    // have to worry about not finding one.
                    self.state = Named(base, 2, -1); // plen is 2 to include &
                }
                (Named(cursor, plen, _), ';') => {
                    it.next(); // consume ;
                    let (name, chars, _) = ENTITIES[cursor];
                    if name.len() == plen {
                        // valid entity
                        try!(self.inner_write_str(chars));
                        self.state = CharData;
                        cdata = i+1;
                    } else {
                        try!(self.abort_entity());
                        self.state = CharData;
                        cdata = i;
                    }
                }
                (Named(cursor, plen, lastcur), 'a'..'z') |
                (Named(cursor, plen, lastcur), 'A'..'Z') |
                (Named(cursor, plen, lastcur), '0'..'9') => {
                    let mut cursor = cursor;
                    it.next(); // consume character
                    let (mut name, _, mut needs_semi) = ENTITIES[cursor];
                    if name.len() > plen && name[plen] == b {
                        // existing cursor is still a match
                    } else {
                        // search forward to find the next entity with our prefix
                        let prefix = name.slice_to(plen);
                        for ix in range(cursor+1, ENTITIES.len()) {
                            let (name_, _, needs_semi_) = ENTITIES[ix];
                            if !name_.starts_with(prefix) {
                                // no match
                                cursor = -1;
                                break;
                            }
                            if name_.len() > plen && name_[plen] == b {
                                cursor = ix;
                                name = name_;
                                needs_semi = needs_semi_;
                                if name_.len() == plen+1 {
                                    name = name_;
                                    needs_semi = needs_semi_;
                                }
                                break;
                            }
                        }
                    }
                    if cursor == -1 {
                        // no match
                        try!(self.abort_entity());
                        self.state = CharData;
                        cdata = i;
                    } else {
                        let plen = plen+1;
                        let lastcur = if !needs_semi && name.len() == plen {
                            cursor
                        } else {
                            lastcur
                        };
                        self.state = Named(cursor, plen, lastcur);
                    }
                }
                (HexStart(_), 'a'..'f')|(HexStart(_), 'A'..'F')|(HexStart(_), '0'..'9') => {
                    self.state = Hex(0);
                    // don't consume, re-try this digit in the Hex state
                }
                (DecStart, '0'..'9') => {
                    self.state = Dec(0);
                    // don't consume, re-try this digit in the Dec state
                }
                (Hex(val), '0'..'9') => {
                    it.next(); // consume character
                    if val <= char::MAX as u32 {
                        let digit = (b - '0' as u8) as u32;
                        self.state = Hex(val*16 + digit);
                    }
                }
                (Hex(val), 'a'..'f') => {
                    it.next(); // consume character
                    if val <= char::MAX as u32 {
                        let digit = 10 + (b - 'a' as u8) as u32;
                        self.state = Hex(val*16 + digit);
                    }
                }
                (Hex(val), 'A'..'F') => {
                    it.next(); // consume character
                    if val <= char::MAX as u32 {
                        let digit = 10 + (b - 'A' as u8) as u32;
                        self.state = Hex(val*16 + digit);
                    }
                }
                (Dec(val), '0'..'9') => {
                    it.next(); // consume character
                    if val <= char::MAX as u32 {
                        let digit = (b - '0' as u8) as u32;
                        self.state = Dec(val*10 + digit);
                    }
                }
                (Hex(_), ';')|(Dec(_), ';') => {
                    it.next(); // consume character
                    // behavior here is identical to aborting, so let's do that
                    try!(self.abort_entity());
                    cdata = i+1;
                }
                _ => {
                    // parse error that does not emit characters
                    try!(self.abort_entity());
                    self.state = CharData;
                    cdata = i;
                }
            }
        }
        match self.state {
            CharData => {
                if cdata < bytes.len() {
                    try!(self.inner_write(bytes.slice_from(cdata)));
                }
            }
            _ => ()
        }
        Ok(())
    }

    fn flush(&mut self) -> IoResult<()> {
        try!(self.flush_outer());
        self.inner.get_mut_ref().flush()
    }
}



static COMPAT_TABLE: [char, ..32] = [
    '\u20AC', // First entry is what 0x80 should be replaced with.
    '\u0081',
    '\u201A',
    '\u0192',
    '\u201E',
    '\u2026',
    '\u2020',
    '\u2021',
    '\u02C6',
    '\u2030',
    '\u0160',
    '\u2039',
    '\u0152',
    '\u008D',
    '\u017D',
    '\u008F',
    '\u0090',
    '\u2018',
    '\u2019',
    '\u201C',
    '\u201D',
    '\u2022',
    '\u2013',
    '\u2014',
    '\u02DC',
    '\u2122',
    '\u0161',
    '\u203A',
    '\u0153',
    '\u009D',
    '\u017E',
    '\u0178', // Last entry is 0x9F.
    // 0x00->'\uFFFD' is handled programmatically.
];
