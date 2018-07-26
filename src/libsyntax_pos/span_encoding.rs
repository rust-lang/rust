// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Spans are encoded using 1-bit tag and 2 different encoding formats (one for each tag value).
// One format is used for keeping span data inline,
// another contains index into an out-of-line span interner.
// The encoding format for inline spans were obtained by optimizing over crates in rustc/libstd.
// See https://internals.rust-lang.org/t/rfc-compiler-refactoring-spans/1357/28

use GLOBALS;
use {BytePos, SpanData};
use hygiene::SyntaxContext;

use rustc_data_structures::fx::FxHashMap;
use std::hash::{Hash, Hasher};

/// A compressed span.
/// Contains either fields of `SpanData` inline if they are small, or index into span interner.
/// The primary goal of `Span` is to be as small as possible and fit into other structures
/// (that's why it uses `packed` as well). Decoding speed is the second priority.
/// See `SpanData` for the info on span fields in decoded representation.
#[repr(packed)]
pub struct Span(u32);

impl Copy for Span {}
impl Clone for Span {
    #[inline]
    fn clone(&self) -> Span {
        *self
    }
}
impl PartialEq for Span {
    #[inline]
    fn eq(&self, other: &Span) -> bool {
        let a = self.0;
        let b = other.0;
        a == b
    }
}
impl Eq for Span {}
impl Hash for Span {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a = self.0;
        a.hash(state)
    }
}

/// Dummy span, both position and length are zero, syntax context is zero as well.
/// This span is kept inline and encoded with format 0.
pub const DUMMY_SP: Span = Span(0);

impl Span {
    #[inline]
    pub fn new(lo: BytePos, hi: BytePos, ctxt: SyntaxContext) -> Self {
        encode(&match lo <= hi {
            true => SpanData { lo, hi, ctxt },
            false => SpanData { lo: hi, hi: lo, ctxt },
        })
    }

    #[inline]
    pub fn data(self) -> SpanData {
        decode(self)
    }
}

// Tags
const TAG_INLINE: u32 = 0;
const TAG_INTERNED: u32 = 1;
const TAG_MASK: u32 = 1;

// Fields indexes
const BASE_INDEX: usize = 0;
const LEN_INDEX: usize = 1;
const CTXT_INDEX: usize = 2;

// Tag = 0, inline format.
// -------------------------------------------------------------
// | base 31:8  | len 7:1  | ctxt (currently 0 bits) | tag 0:0 |
// -------------------------------------------------------------
// Since there are zero bits for ctxt, only SpanData with a 0 SyntaxContext
// can be inline.
const INLINE_SIZES: [u32; 3] = [24, 7, 0];
const INLINE_OFFSETS: [u32; 3] = [8, 1, 1];

// Tag = 1, interned format.
// ------------------------
// | index 31:1 | tag 0:0 |
// ------------------------
const INTERNED_INDEX_SIZE: u32 = 31;
const INTERNED_INDEX_OFFSET: u32 = 1;

#[inline]
fn encode(sd: &SpanData) -> Span {
    let (base, len, ctxt) = (sd.lo.0, sd.hi.0 - sd.lo.0, sd.ctxt.as_u32());

    let val = if (base >> INLINE_SIZES[BASE_INDEX]) == 0 &&
                 (len >> INLINE_SIZES[LEN_INDEX]) == 0 &&
                 (ctxt >> INLINE_SIZES[CTXT_INDEX]) == 0 {
        (base << INLINE_OFFSETS[BASE_INDEX]) | (len << INLINE_OFFSETS[LEN_INDEX]) |
        (ctxt << INLINE_OFFSETS[CTXT_INDEX]) | TAG_INLINE
    } else {
        let index = with_span_interner(|interner| interner.intern(sd));
        (index << INTERNED_INDEX_OFFSET) | TAG_INTERNED
    };
    Span(val)
}

#[inline]
fn decode(span: Span) -> SpanData {
    let val = span.0;

    // Extract a field at position `pos` having size `size`.
    let extract = |pos: u32, size: u32| {
        let mask = ((!0u32) as u64 >> (32 - size)) as u32; // Can't shift u32 by 32
        (val >> pos) & mask
    };

    let (base, len, ctxt) = if val & TAG_MASK == TAG_INLINE {(
        extract(INLINE_OFFSETS[BASE_INDEX], INLINE_SIZES[BASE_INDEX]),
        extract(INLINE_OFFSETS[LEN_INDEX], INLINE_SIZES[LEN_INDEX]),
        extract(INLINE_OFFSETS[CTXT_INDEX], INLINE_SIZES[CTXT_INDEX]),
    )} else {
        let index = extract(INTERNED_INDEX_OFFSET, INTERNED_INDEX_SIZE);
        return with_span_interner(|interner| *interner.get(index));
    };
    SpanData { lo: BytePos(base), hi: BytePos(base + len), ctxt: SyntaxContext::from_u32(ctxt) }
}

#[derive(Default)]
pub struct SpanInterner {
    spans: FxHashMap<SpanData, u32>,
    span_data: Vec<SpanData>,
}

impl SpanInterner {
    fn intern(&mut self, span_data: &SpanData) -> u32 {
        if let Some(index) = self.spans.get(span_data) {
            return *index;
        }

        let index = self.spans.len() as u32;
        self.span_data.push(*span_data);
        self.spans.insert(*span_data, index);
        index
    }

    #[inline]
    fn get(&self, index: u32) -> &SpanData {
        &self.span_data[index as usize]
    }
}

// If an interner exists, return it. Otherwise, prepare a fresh one.
#[inline]
fn with_span_interner<T, F: FnOnce(&mut SpanInterner) -> T>(f: F) -> T {
    GLOBALS.with(|globals| f(&mut *globals.span_interner.lock()))
}
