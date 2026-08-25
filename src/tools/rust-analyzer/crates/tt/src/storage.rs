//! Spans are memory heavy, and we have a lot of token trees. Storing them straight
//! will waste a lot of memory. So instead we implement a clever compression mechanism:
//!
//! A `TopSubtree` has a list of [`CompressedSpanPart`], which are the parts of a span
//! that tend to be shared between tokens - namely, without the range.
//!
//! The main list of token trees is stored in a variable-length encoding as bytes.
//! The encoding is documented in the [`decode()`] function (which decodes one [`TokenTree`]).

use std::{assert_matches, collections::hash_map, fmt::Debug, hint::cold_path, mem::transmute};

#[cfg(all(debug_assertions, not(miri)))]
use std::cell::Cell;

#[cfg(not(all(debug_assertions, not(miri))))]
use std::mem::MaybeUninit;

use intern::Symbol;
use rustc_hash::FxHashMap;
use span::{Span, SpanAnchor, SyntaxContext, TextRange, TextSize};

use crate::{
    DelimSpan, Delimiter, DelimiterKind, Ident, IdentIsRaw, Leaf, LitKind, Literal, Punct, Spacing,
    Subtree, SubtreeView, TokenTree, TokenTreesView, TtIter,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CompressedSpanPart {
    anchor: SpanAnchor,
    ctx: SyntaxContext,
}

impl CompressedSpanPart {
    #[inline]
    fn from_span(span: &Span) -> Self {
        Self { anchor: span.anchor, ctx: span.ctx }
    }

    #[inline]
    fn recombine(&self, range: TextRange) -> Span {
        Span { range, anchor: self.anchor, ctx: self.ctx }
    }
}

trait Encodable: Sized {
    #[cfg(all(debug_assertions, not(miri)))]
    fn write(self, buffer: &[Cell<u8>]);
    #[cfg(all(debug_assertions, not(miri)))]
    fn read(buffer: &[u8]) -> Self;
}

impl Encodable for u8 {
    #[cfg(all(debug_assertions, not(miri)))]
    fn write(self, buffer: &[Cell<u8>]) {
        buffer[0].set(self);
    }
    #[cfg(all(debug_assertions, not(miri)))]
    fn read(buffer: &[u8]) -> Self {
        buffer[0]
    }
}

impl Encodable for u16 {
    #[cfg(all(debug_assertions, not(miri)))]
    fn write(self, buffer: &[Cell<u8>]) {
        let value = self.to_ne_bytes();
        let buffer: &[Cell<u8>; size_of::<Self>()] = buffer.try_into().unwrap();
        for (b, v) in std::iter::zip(buffer, value) {
            b.set(v);
        }
    }
    #[cfg(all(debug_assertions, not(miri)))]
    fn read(buffer: &[u8]) -> Self {
        Self::from_ne_bytes(buffer.try_into().unwrap())
    }
}

impl Encodable for u32 {
    #[cfg(all(debug_assertions, not(miri)))]
    fn write(self, buffer: &[Cell<u8>]) {
        let value = self.to_ne_bytes();
        let buffer: &[Cell<u8>; size_of::<Self>()] = buffer.try_into().unwrap();
        for (b, v) in std::iter::zip(buffer, value) {
            b.set(v);
        }
    }
    #[cfg(all(debug_assertions, not(miri)))]
    fn read(buffer: &[u8]) -> Self {
        Self::from_ne_bytes(buffer.try_into().unwrap())
    }
}

impl Encodable for char {
    #[cfg(all(debug_assertions, not(miri)))]
    fn write(self, buffer: &[Cell<u8>]) {
        u32::from(self).write(buffer)
    }
    #[cfg(all(debug_assertions, not(miri)))]
    fn read(buffer: &[u8]) -> Self {
        char::from_u32(u32::read(buffer)).unwrap()
    }
}

struct UninitBuffer {
    #[cfg(all(debug_assertions, not(miri)))]
    buffer: Box<[u8]>,
    #[cfg(not(all(debug_assertions, not(miri))))]
    buffer: Box<[MaybeUninit<u8>]>,
}

impl UninitBuffer {
    #[inline]
    fn new(capacity: usize) -> Self {
        Self {
            #[cfg(all(debug_assertions, not(miri)))]
            buffer: vec![0; capacity].into_boxed_slice(),
            #[cfg(not(all(debug_assertions, not(miri))))]
            buffer: Box::new_uninit_slice(capacity),
        }
    }

    #[inline]
    fn writer(&mut self) -> BufferWriter<'_> {
        BufferWriter {
            #[cfg(all(debug_assertions, not(miri)))]
            buffer: Cell::from_mut(&mut *self.buffer).as_slice_of_cells(),
            #[cfg(not(all(debug_assertions, not(miri))))]
            ptr: self.buffer.as_mut_ptr_range().end.cast::<u8>(),
            #[cfg(not(all(debug_assertions, not(miri))))]
            _marker: stdx::variance::PhantomCovariantLifetime::new(),
        }
    }

    #[cfg(all(debug_assertions, not(miri)))]
    unsafe fn finish(self, writer_finish: usize) -> Box<[u8]> {
        self.buffer[writer_finish..].into()
    }

    #[cfg(not(all(debug_assertions, not(miri))))]
    #[inline]
    unsafe fn finish(mut self, writer_finish: *mut u8) -> Box<[u8]> {
        let end = self.buffer.as_mut_ptr_range().end.cast::<u8>();
        unsafe {
            let bytes_len = end.offset_from_unsigned(writer_finish);
            let mut buffer = Box::<[u8]>::new_uninit_slice(bytes_len);
            buffer.as_mut_ptr().cast::<u8>().copy_from_nonoverlapping(writer_finish, bytes_len);
            buffer.assume_init()
        }
    }
}

#[derive(Clone, Copy)]
struct BufferWriter<'a> {
    #[cfg(all(debug_assertions, not(miri)))]
    buffer: &'a [Cell<u8>],
    #[cfg(not(all(debug_assertions, not(miri))))]
    ptr: *mut u8,
    #[cfg(not(all(debug_assertions, not(miri))))]
    _marker: stdx::variance::PhantomCovariantLifetime<'a>,
}

impl<'a> BufferWriter<'a> {
    #[inline]
    unsafe fn new(buffer: &'a mut [u8]) -> Self {
        BufferWriter {
            #[cfg(all(debug_assertions, not(miri)))]
            buffer: Cell::from_mut(buffer).as_slice_of_cells(),
            #[cfg(not(all(debug_assertions, not(miri))))]
            ptr: buffer.as_mut_ptr_range().end,
            #[cfg(not(all(debug_assertions, not(miri))))]
            _marker: stdx::variance::PhantomCovariantLifetime::new(),
        }
    }

    #[cfg_attr(not(all(debug_assertions, not(miri))), inline(always))]
    unsafe fn write<T: Encodable>(&mut self, value: T) {
        #[cfg(all(debug_assertions, not(miri)))]
        {
            let write_at = self.buffer.split_off(self.buffer.len() - size_of::<T>()..).unwrap();
            value.write(write_at);
        }
        #[cfg(not(all(debug_assertions, not(miri))))]
        unsafe {
            self.ptr = self.ptr.sub(size_of::<T>());
            self.ptr.cast::<T>().write_unaligned(value);
        }
    }

    fn len_since(self, other: BufferWriter<'_>) -> usize {
        #[cfg(all(debug_assertions, not(miri)))]
        {
            other.buffer.len() - self.buffer.len()
        }
        #[cfg(not(all(debug_assertions, not(miri))))]
        {
            other.ptr.addr() - self.ptr.addr()
        }
    }

    #[cfg(all(debug_assertions, not(miri)))]
    fn finish(self) -> usize {
        self.buffer.len()
    }

    #[cfg(not(all(debug_assertions, not(miri))))]
    #[inline]
    fn finish(self) -> *mut u8 {
        self.ptr
    }
}

#[inline]
const fn n_bits_mask(n: u32) -> u32 {
    (1 << n) - 1
}

#[inline]
const fn n_bits_mask_u64(n: u32) -> u64 {
    (1 << n) - 1
}

// Encoding is done in reverse, from the end to the beginning. This is in order
// to be able to tell how many bytes the children of a `Subtree` occupy, and store
// *that* efficiently, with the smallest possible type.

#[must_use]
unsafe fn encode_span<'a>(
    ptr: BufferWriter<'a>,
    span: &Span,
    span_parts_map: &FxHashMap<CompressedSpanPart, usize>,
    extra_two_bits: u32,
    force_heavy_encoding: bool,
) -> BufferWriter<'a> {
    let span_parts_index = span_parts_map[&CompressedSpanPart::from_span(span)] as u32;
    let offset = u32::from(span.range.start());
    let len = u32::from(span.range.len());
    unsafe {
        encode_span_no_map(ptr, span_parts_index, offset, len, extra_two_bits, force_heavy_encoding)
    }
}

#[must_use]
unsafe fn encode_span_no_map(
    mut ptr: BufferWriter<'_>,
    mut span_parts_index: u32,
    mut offset: u32,
    mut len: u32,
    extra_two_bits: u32,
    force_heavy_encoding: bool,
) -> BufferWriter<'_> {
    debug_assert!(extra_two_bits & !0b11 == 0);
    let mut first_u32 = extra_two_bits;
    first_u32 |= (span_parts_index & n_bits_mask(4)) << 2;
    span_parts_index >>= 4;
    first_u32 |= (len & n_bits_mask(8)) << (2 + 4);
    len >>= 8;
    first_u32 |= (offset & n_bits_mask(17)) << (2 + 4 + 8 + 1);
    offset >>= 17;

    let extends_to_next = span_parts_index != 0 || len != 0 || offset != 0 || force_heavy_encoding;
    first_u32 |= u32::from(extends_to_next) << (2 + 4 + 8);

    if extends_to_next {
        ptr = unsafe {
            encode_extended_span(ptr, span_parts_index, len, offset, force_heavy_encoding)
        };
    }
    unsafe { ptr.write::<u32>(first_u32) };

    ptr
}

#[cold]
#[must_use]
unsafe fn encode_extended_span(
    mut ptr: BufferWriter<'_>,
    span_parts_index: u32,
    len: u32,
    offset: u32,
    force_heavy_encoding: bool,
) -> BufferWriter<'_> {
    if span_parts_index <= n_bits_mask(11)
        && len <= n_bits_mask(10)
        && offset <= n_bits_mask(10)
        && !force_heavy_encoding
    {
        let mut second_u32 = span_parts_index;
        second_u32 |= len << 11;
        second_u32 |= offset << (11 + 10);
        second_u32 <<= 1;
        unsafe { ptr.write::<u32>(second_u32) };
    } else {
        assert!(span_parts_index <= n_bits_mask(24), "too big `span_parts_index`");

        let mut u64 = u64::from(span_parts_index);
        u64 |= u64::from(len) << 24;
        u64 |= u64::from(offset) << (24 + 24);
        let third_u32 = u64 as u32;
        let mut second_u32 = (u64 >> u32::BITS) as u32;
        second_u32 <<= 1;
        second_u32 |= 0b1;
        unsafe {
            ptr.write::<u32>(third_u32);
            ptr.write::<u32>(second_u32);
        }
    }

    ptr
}

#[must_use]
unsafe fn encode_symbol<'a>(
    mut ptr: BufferWriter<'a>,
    symbol: &Symbol,
    tag: u32,
    symbols_map: &FxHashMap<Symbol, usize>,
) -> BufferWriter<'a> {
    let symbol_idx = symbols_map[symbol] as u32;
    unsafe {
        if symbol_idx <= n_bits_mask(6) {
            ptr.write::<u8>(((symbol_idx << 2) | tag) as u8);
        } else if symbol_idx <= n_bits_mask(13) {
            ptr.write::<u8>(((symbol_idx >> 6) << 1) as u8);
            ptr.write::<u8>(((symbol_idx << 2) | 0b10 | tag) as u8);
        } else {
            ptr.write::<u16>((symbol_idx >> 13) as u16);
            ptr.write::<u8>((((symbol_idx >> 6) << 1) | 0b1) as u8);
            ptr.write::<u8>(((symbol_idx << 2) | 0b10 | tag) as u8);
        }
    }
    ptr
}

#[must_use]
unsafe fn encode<'a>(
    mut ptr: BufferWriter<'a>,
    tt: TokenTree,
    span_parts_map: &FxHashMap<CompressedSpanPart, usize>,
    byte_size_after: &mut [u32],
    symbols_map: &FxHashMap<Symbol, usize>,
) -> BufferWriter<'a> {
    let before_ptr = ptr;
    unsafe {
        match tt {
            TokenTree::Leaf(Leaf::Punct(Punct { char, spacing, span })) => {
                if char.is_ascii() {
                    let spacing = spacing as u8;
                    let span_extra = 0b1 | (u32::from(spacing) & 0b10);
                    let char = ((char as u8) << 1) | (spacing & 0b1);
                    ptr.write::<u8>(char);
                    ptr = encode_span(ptr, &span, span_parts_map, span_extra, false);
                } else {
                    let mut control_byte = 0b110;
                    control_byte |= (spacing as u8) << 3;
                    ptr.write::<char>(char);
                    ptr.write::<u8>(control_byte);
                    ptr = encode_span(ptr, &span, span_parts_map, 0b00, false);
                }
            }
            TokenTree::Leaf(Leaf::Ident(Ident { sym, span, is_raw })) => {
                ptr = encode_symbol(ptr, &sym, is_raw as u32, symbols_map);
                ptr = encode_span(ptr, &span, span_parts_map, 0b10, false);
            }
            TokenTree::Leaf(Leaf::Literal(Literal { text_and_suffix, span, kind, suffix_len })) => {
                if matches!(kind, LitKind::Str | LitKind::StrRaw(0 | 1) | LitKind::Integer)
                    && u32::from(suffix_len) <= n_bits_mask(4)
                {
                    // Literal, format 1.
                    let mut control_byte = match kind {
                        LitKind::Str => 0b0_011,
                        LitKind::StrRaw(0) => 0b0_100,
                        LitKind::StrRaw(1) => 0b1_011,
                        LitKind::Integer => 0b1_100,
                        _ => unreachable!(),
                    };
                    control_byte |= suffix_len << 4;
                    ptr = encode_symbol(ptr, &text_and_suffix, 0, symbols_map);
                    ptr.write::<u8>(control_byte);
                } else {
                    // Literal, format 2.
                    let mut control_byte = 0b111;
                    let (kind, raw_count) = match kind {
                        LitKind::Byte => (0, None),
                        LitKind::Char => (1, None),
                        LitKind::Integer => (2, None),
                        LitKind::Float => (3, None),
                        LitKind::Str => (4, None),
                        LitKind::StrRaw(count) => (5, Some(count)),
                        LitKind::ByteStr => (6, None),
                        LitKind::ByteStrRaw(count) => (7, Some(count)),
                        LitKind::CStr => (8, None),
                        LitKind::CStrRaw(count) => (9, Some(count)),
                        LitKind::Err(()) => (10, None),
                    };
                    control_byte |= kind << 3;
                    ptr = encode_symbol(ptr, &text_and_suffix, 0, symbols_map);
                    ptr.write::<u8>(suffix_len);
                    if let Some(raw_count) = raw_count {
                        ptr.write::<u8>(raw_count);
                    }
                    ptr.write::<u8>(control_byte);
                }
                ptr = encode_span(ptr, &span, span_parts_map, 0b00, false);
            }
            TokenTree::Subtree(Subtree { delimiter, len }) => {
                let open_span_parts_index =
                    span_parts_map[&CompressedSpanPart::from_span(&delimiter.open)] as u32;
                let close_span_parts_index =
                    span_parts_map[&CompressedSpanPart::from_span(&delimiter.close)] as u32;
                let close_span_offset_from_open = delimiter
                    .close
                    .range
                    .start()
                    .checked_sub(delimiter.open.range.start())
                    .map_or(u32::MAX, u32::from);
                let children_byte_len = byte_size_after[1] - byte_size_after[1 + len as usize];
                if open_span_parts_index == close_span_parts_index
                    && len <= n_bits_mask(u8::BITS)
                    && children_byte_len <= n_bits_mask(u8::BITS)
                    && delimiter.open.range.len() == TextSize::new(1)
                    && delimiter.close.range.len() == TextSize::new(1)
                    && close_span_offset_from_open <= n_bits_mask(11)
                {
                    // Subtree, format 1.
                    let span = Span {
                        range: TextRange::at(
                            delimiter.open.range.start(),
                            TextSize::new(close_span_offset_from_open),
                        ),
                        anchor: delimiter.open.anchor,
                        ctx: delimiter.open.ctx,
                    };
                    let mut control_byte = 0b000;
                    control_byte |= (delimiter.kind as u8) << 3;
                    control_byte |= ((close_span_offset_from_open >> 8) << (3 + 2)) as u8;
                    ptr.write::<u8>(children_byte_len as u8);
                    ptr.write::<u8>(len as u8);
                    ptr.write::<u8>(control_byte);
                    ptr = encode_span(ptr, &span, span_parts_map, 0b00, false);
                } else if open_span_parts_index == close_span_parts_index
                    && len <= n_bits_mask(u8::BITS)
                    && children_byte_len <= n_bits_mask(u8::BITS)
                    && delimiter.open.range.len() == delimiter.close.range.len()
                    && close_span_offset_from_open <= n_bits_mask(3)
                {
                    // Subtree, format 2.
                    let mut control_byte = 0b001;
                    control_byte |= (delimiter.kind as u8) << 3;
                    control_byte |= (close_span_offset_from_open << (3 + 2)) as u8;
                    ptr.write::<u8>(children_byte_len as u8);
                    ptr.write::<u8>(len as u8);
                    ptr.write::<u8>(control_byte);
                    ptr = encode_span(ptr, &delimiter.open, span_parts_map, 0b00, false);
                } else if len <= n_bits_mask(u8::BITS)
                    && children_byte_len <= n_bits_mask(12)
                    && delimiter.open.range.len() == delimiter.close.range.len()
                    && close_span_offset_from_open <= n_bits_mask(8)
                    && close_span_parts_index <= n_bits_mask(7)
                {
                    // Subtree, format 3.
                    let mut control_byte = 0b010;
                    control_byte |= (delimiter.kind as u8) << 3;
                    control_byte |= (close_span_parts_index << (3 + 2)) as u8;
                    let mut children_byte_len = children_byte_len << 4;
                    children_byte_len |= close_span_parts_index >> 3;
                    ptr.write::<u16>(children_byte_len as u16);
                    ptr.write::<u8>(len as u8);
                    ptr.write::<u8>(close_span_offset_from_open as u8);
                    ptr.write::<u8>(control_byte);
                    ptr = encode_span(ptr, &delimiter.open, span_parts_map, 0b00, false);
                } else {
                    // Subtree, format 4.
                    let mut control_byte = 0b101;
                    control_byte |= (delimiter.kind as u8) << 3;
                    ptr.write::<u32>(children_byte_len);
                    ptr.write::<u32>(len);
                    ptr = encode_span(ptr, &delimiter.close, span_parts_map, 0b00, false);
                    ptr.write::<u8>(control_byte);
                    ptr = encode_span(ptr, &delimiter.open, span_parts_map, 0b00, false);
                }
            }
        }

        let element_byte_size: u32 = ptr.len_since(before_ptr).try_into().unwrap();
        byte_size_after[0] = byte_size_after[1] + element_byte_size;
    }

    ptr
}

/// We always encode the top subtree with the heaviest encoding because we sometimes want to change it.
unsafe fn encode_top_subtree<'a>(
    mut ptr: BufferWriter<'a>,
    top_subtree: Subtree,
    open_span_parts_index: u32,
    byte_size_after: &[u32],
) -> BufferWriter<'a> {
    unsafe {
        let Subtree { delimiter, len } = top_subtree;
        let children_byte_len = byte_size_after[1] - byte_size_after[1 + len as usize];

        let mut control_byte = 0b101;
        control_byte |= (delimiter.kind as u8) << 3;
        ptr.write::<u32>(children_byte_len);
        ptr.write::<u32>(len);
        ptr = encode_span_no_map(
            ptr,
            open_span_parts_index + 1,
            delimiter.close.range.start().into(),
            delimiter.close.range.len().into(),
            0b00,
            true,
        );
        ptr.write::<u8>(control_byte);
        ptr = encode_span_no_map(
            ptr,
            open_span_parts_index,
            delimiter.open.range.start().into(),
            delimiter.open.range.len().into(),
            0b00,
            true,
        );
    }
    ptr
}

fn change_root_delimiter(buffer: &mut [u8], new_delim: DelimiterKind) {
    // The span is 3*u32 and then the control byte, in which the delimiter comes.
    let control_byte_index = 3 * size_of::<u32>();
    let mut control_byte = buffer[control_byte_index];
    control_byte &= 0b111; // Remove previous delimiter.
    control_byte |= (new_delim as u8) << 3;
    buffer[control_byte_index] = control_byte;
}

unsafe fn change_root_spans(
    buffer: &mut [u8],
    open_span_parts_index: u32,
    close_span_parts_index: u32,
    open_range: TextRange,
    close_range: TextRange,
) {
    // Remember we write in reverse, so we add `3 * size_of::<u32>()`.
    unsafe {
        _ = encode_span_no_map(
            BufferWriter::new(&mut buffer[..3 * size_of::<u32>()]),
            open_span_parts_index,
            u32::from(open_range.start()),
            u32::from(open_range.len()),
            0b00,
            true,
        );
        _ = encode_span_no_map(
            BufferWriter::new(
                &mut buffer[..3 * size_of::<u32>() + size_of::<u8>() + 3 * size_of::<u32>()],
            ),
            close_span_parts_index,
            u32::from(close_range.start()),
            u32::from(close_range.len()),
            0b00,
            true,
        );
    }
}

/// This is subtree in format 4: two spans, each at most 3*u32, a u8 control byte, a u32 length and a u32 bytes length.
const BIGGEST_POSSIBLE_TT_ENCODING: usize =
    2 * 3 * size_of::<u32>() + size_of::<u8>() + size_of::<u32>() + size_of::<u32>();

fn encode_all(
    tts: std::vec::IntoIter<TokenTree>,
    mut compressed_span_frequencies: FxHashMap<CompressedSpanPart, usize>,
    mut symbol_frequencies: FxHashMap<Symbol, usize>,
) -> TopSubtree {
    let tts_len = tts.len();
    let mut token_trees = tts.enumerate();
    let Some((_, TokenTree::Subtree(top_subtree))) = token_trees.next() else {
        panic!("must always have a top subtree");
    };

    let (span_parts, span_parts_map) = {
        let mut compressed_spans = compressed_span_frequencies
            .keys()
            .copied()
            .chain([
                CompressedSpanPart::from_span(&top_subtree.delimiter.open),
                CompressedSpanPart::from_span(&top_subtree.delimiter.close),
            ])
            .collect::<Box<[_]>>();
        {
            // For this purpose, do not consider the top delimiters. They should stay last and not affect the other spans,
            // since we might want to change them.
            let len = compressed_spans.len();
            let compressed_spans = &mut compressed_spans[..len - 2];
            // No need sort if there is already enough space for everyone to be encoded efficiently.
            if compressed_span_frequencies.len() > n_bits_mask(4) as usize {
                // We want more used spans to have lower indices, so they can be encoded more efficiently.
                compressed_spans.sort_unstable_by_key(|span| {
                    std::cmp::Reverse(compressed_span_frequencies[span])
                });
            }
            for (index, span) in compressed_spans.iter().enumerate() {
                *compressed_span_frequencies.get_mut(span).unwrap() = index;
            }
        }
        (compressed_spans, compressed_span_frequencies)
    };

    let (symbols, symbols_map) = {
        let mut symbols = symbol_frequencies.keys().cloned().collect::<Box<[_]>>();
        // No need sort if there is already enough space for everyone to be encoded efficiently.
        if symbol_frequencies.len() > n_bits_mask(6) as usize {
            // We want more used spans to have lower indices, so they can be encoded more efficiently.
            symbols.sort_unstable_by_key(|symbol| std::cmp::Reverse(symbol_frequencies[symbol]));
        }
        for (index, span) in symbols.iter().enumerate() {
            *symbol_frequencies.get_mut(span).unwrap() = index;
        }
        (symbols, symbol_frequencies)
    };

    // +1 because each `encode()` calls reads the previous value.
    let mut byte_size_after = vec![0u32; tts_len + 1];
    let bytes_capacity = tts_len * BIGGEST_POSSIBLE_TT_ENCODING;
    unsafe {
        let mut temp_buffer = UninitBuffer::new(bytes_capacity);
        let mut ptr = temp_buffer.writer();
        for (index, tt) in token_trees.rev() {
            ptr = encode(ptr, tt, &span_parts_map, &mut byte_size_after[index..], &symbols_map);
        }
        ptr = encode_top_subtree(ptr, top_subtree, (span_parts.len() - 2) as u32, &byte_size_after);

        let writer_finish = ptr.finish();
        let buffer = temp_buffer.finish(writer_finish);

        TopSubtree { buffer, span_parts, len: tts_len, symbols }
    }
}

fn compute_span_frequencies_and_symbols(
    tts: &[TokenTree],
) -> (FxHashMap<CompressedSpanPart, usize>, FxHashMap<Symbol, usize>) {
    let mut span_frequencies = FxHashMap::default();
    let mut symbols = FxHashMap::default();
    for tt in tts {
        match tt {
            TokenTree::Leaf(leaf) => {
                if let Some(symbol) = leaf.symbol() {
                    *symbols.entry(symbol.clone()).or_insert(0) += 1;
                }

                *span_frequencies.entry(CompressedSpanPart::from_span(leaf.span())).or_insert(0) +=
                    1;
            }
            TokenTree::Subtree(subtree) => {
                *span_frequencies
                    .entry(CompressedSpanPart::from_span(&subtree.delimiter.open))
                    .or_insert(0) += 1;
                *span_frequencies
                    .entry(CompressedSpanPart::from_span(&subtree.delimiter.close))
                    .or_insert(0) += 1;
            }
        }
    }
    (span_frequencies, symbols)
}

#[derive(Clone, Copy)]
struct BufferReader<'a> {
    #[cfg(all(debug_assertions, not(miri)))]
    buffer: &'a [u8],
    #[cfg(not(all(debug_assertions, not(miri))))]
    ptr: *const u8,
    #[cfg(not(all(debug_assertions, not(miri))))]
    _marker: stdx::variance::PhantomCovariantLifetime<'a>,
}

impl<'a> BufferReader<'a> {
    #[inline]
    fn start_end(slice: &'a [u8]) -> (Self, Self) {
        #[cfg(all(debug_assertions, not(miri)))]
        {
            let start = Self { buffer: slice };
            let end = Self { buffer: &slice[slice.len()..] };
            (start, end)
        }
        #[cfg(not(all(debug_assertions, not(miri))))]
        {
            let ptrs = slice.as_ptr_range();
            let start = Self {
                ptr: ptrs.start,
                #[cfg(not(all(debug_assertions, not(miri))))]
                _marker: stdx::variance::PhantomCovariantLifetime::new(),
            };
            let end = Self {
                ptr: ptrs.end,
                #[cfg(not(all(debug_assertions, not(miri))))]
                _marker: stdx::variance::PhantomCovariantLifetime::new(),
            };
            (start, end)
        }
    }

    #[cfg_attr(not(all(debug_assertions, not(miri))), inline(always))]
    unsafe fn read<T: Encodable>(&mut self) -> T {
        #[cfg(all(debug_assertions, not(miri)))]
        {
            let read_at = self.buffer.split_off(..size_of::<T>()).unwrap();
            T::read(read_at)
        }
        #[cfg(not(all(debug_assertions, not(miri))))]
        unsafe {
            let result = self.ptr.cast::<T>().read_unaligned();
            self.ptr = self.ptr.add(size_of::<T>());
            result
        }
    }

    #[cfg_attr(not(all(debug_assertions, not(miri))), inline(always))]
    unsafe fn skip(&mut self, amount: usize) {
        #[cfg(all(debug_assertions, not(miri)))]
        {
            self.buffer = &self.buffer[amount..];
        }
        #[cfg(not(all(debug_assertions, not(miri))))]
        unsafe {
            self.ptr = self.ptr.add(amount);
        }
    }
}

impl PartialEq for BufferReader<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(all(debug_assertions, not(miri)))]
        let self_ptr = self.buffer.as_ptr();
        #[cfg(not(all(debug_assertions, not(miri))))]
        let self_ptr = self.ptr;
        #[cfg(all(debug_assertions, not(miri)))]
        let other_ptr = other.buffer.as_ptr();
        #[cfg(not(all(debug_assertions, not(miri))))]
        let other_ptr = other.ptr;

        self_ptr == other_ptr
    }
}

struct InlineSpanParts {
    span_parts_index: usize,
    text_range: TextRange,
}

#[inline]
unsafe fn decode_span(
    ptr: BufferReader<'_>,
    mut first_u32: u32,
) -> (BufferReader<'_>, InlineSpanParts) {
    first_u32 >>= 2;
    let span_parts_index = first_u32 & n_bits_mask(4);
    let len = (first_u32 >> 4) & n_bits_mask(8);
    let extends_to_next = (first_u32 & (1 << (4 + 8))) != 0;
    let offset = first_u32 >> (4 + 8 + 1);
    if !extends_to_next {
        let result = InlineSpanParts {
            span_parts_index: span_parts_index as usize,
            text_range: TextRange::at(TextSize::new(offset), TextSize::new(len)),
        };
        (ptr, result)
    } else {
        unsafe { decode_extended_span(ptr, span_parts_index, len, offset) }
    }
}

#[cold]
unsafe fn decode_extended_span(
    mut ptr: BufferReader<'_>,
    mut span_parts_index: u32,
    mut len: u32,
    mut offset: u32,
) -> (BufferReader<'_>, InlineSpanParts) {
    unsafe {
        let mut second_u32 = ptr.read::<u32>();
        let extends_to_next = (second_u32 & 0b1) != 0;
        second_u32 >>= 1;
        if extends_to_next {
            let third_u32 = ptr.read::<u32>();
            let u64 = u64::from(third_u32) | (u64::from(second_u32) << u32::BITS);
            let rest_span_parts_index = (u64 & n_bits_mask_u64(24)) as u32;
            span_parts_index |= rest_span_parts_index << 4;
            let rest_len = ((u64 >> 24) & n_bits_mask_u64(24)) as u32;
            len |= rest_len << 8;
            let rest_offset = (u64 >> (24 + 24)) as u32;
            offset |= rest_offset << 17;
        } else {
            let rest_span_parts_index = second_u32 & n_bits_mask(11);
            span_parts_index |= rest_span_parts_index << 4;
            let rest_len = (second_u32 >> 11) & n_bits_mask(10);
            len |= rest_len << 8;
            let rest_offset = second_u32 >> (11 + 10);
            offset |= rest_offset << 17;
        };
        let result = InlineSpanParts {
            span_parts_index: span_parts_index as usize,
            text_range: TextRange::at(TextSize::new(offset), TextSize::new(len)),
        };
        (ptr, result)
    }
}

// FIXME: It'll probably be better to ensure this ourselves via a `#[repr(C, align(4))]` wrapper, even though practically
// this holds for all 32- and 64-bit targets (Rust does not guarantee this).
const _: () = assert!(align_of::<*const *const str>() >= 4); // Needed for the tagging of idents.

unsafe fn decode_symbol<'a>(
    mut ptr: BufferReader<'a>,
    first_byte: u8,
    symbols: &[Symbol],
) -> (BufferReader<'a>, Symbol) {
    let mut symbol_idx = u32::from(first_byte) >> 1;
    let extends_to_next = symbol_idx & 0b1 == 0b1;
    symbol_idx >>= 1;
    if extends_to_next {
        let mut second_byte = unsafe { ptr.read::<u8>() };
        let rest_bytes =
            if second_byte & 0b1 == 0b1 { u32::from(unsafe { ptr.read::<u16>() }) } else { 0 };
        second_byte >>= 1;
        symbol_idx |= u32::from(second_byte) << 6;
        symbol_idx |= rest_bytes << 13;
    }
    (ptr, symbols[symbol_idx as usize].clone())
}

/// We need `MaybeUninit` to preserve provenance.
///
/// The returned `u32` is the length of the children *in bytes*, if we read a subtree. Otherwise it's zero.
unsafe fn decode<'a>(
    mut ptr: BufferReader<'a>,
    compressed: &[CompressedSpanPart],
    symbols: &[Symbol],
) -> (BufferReader<'a>, TokenTree, u32) {
    unsafe {
        let span_and_extra = ptr.read::<u32>();
        let span;
        (ptr, span) = decode_span(ptr, span_and_extra);
        let span = compressed[span.span_parts_index].recombine(span.text_range);

        if span_and_extra & 0b1 == 0b1 {
            // An ASCII punct.
            let char_and_half_spacing = ptr.read::<u8>();

            let spacing = (span_and_extra & 0b10) | (u32::from(char_and_half_spacing) & 0b1);
            let spacing = transmute::<u8, Spacing>(spacing as u8);

            let char = char::from(char_and_half_spacing >> 1);

            return (ptr, TokenTree::Leaf(Leaf::Punct(Punct { char, spacing, span })), 0);
        } else if span_and_extra & 0b10 == 0b10 {
            // An ident.
            let symbol_first_byte = ptr.read::<u8>();
            let is_raw = symbol_first_byte & 0b1;
            let symbol;
            (ptr, symbol) = decode_symbol(ptr, symbol_first_byte, symbols);
            let is_raw = transmute::<u8, IdentIsRaw>(is_raw);

            return (ptr, TokenTree::Leaf(Leaf::Ident(Ident { sym: symbol, span, is_raw })), 0);
        }

        let mut children_byte_len = 0;

        let control_byte = u32::from(ptr.read::<u8>());
        let control_byte_extra_data = control_byte >> 3;
        let result = match control_byte & 0b111 {
            0b000 => {
                // Subtree, format 1:
                //  - Same span_parts_index for open and close span.
                //  - Subtree length is a u8.
                //  - Subtree length in bytes is a u8.
                //  - The length of both the open and close span is 1 - so we use the already-parsed length for the open span
                //    for other things (we can only assume it has 8 bits available, the minimum format for a length).
                //  - The offset between the open span's start and the close span's start is stored in 11 bits.
                let kind = transmute::<u8, DelimiterKind>((control_byte_extra_data & 0b11) as u8);
                let mut open_span = span;
                let mut close_span_offset_from_open = u32::from(span.range.len());
                open_span.range = TextRange::at(open_span.range.start(), TextSize::new(1));
                close_span_offset_from_open |= (control_byte_extra_data >> 2) << 8;
                let close_span = Span {
                    range: open_span.range + TextSize::new(close_span_offset_from_open),
                    anchor: open_span.anchor,
                    ctx: open_span.ctx,
                };
                let len = u32::from(ptr.read::<u8>());
                children_byte_len = u32::from(ptr.read::<u8>());

                TokenTree::Subtree(Subtree {
                    delimiter: Delimiter { open: open_span, close: close_span, kind },
                    len,
                })
            }
            0b001 => {
                // Subtree, format 2:
                //  - Same span_parts_index for open and close span.
                //  - Subtree length is a u8.
                //  - Subtree length in bytes is a u8.
                //  - The open and close span have the same length. This covers many cases because most cases either give
                //    both length 1 (the brackets themselves) or the same span (usually, the whole range they encompass).
                //  - The offset between the open span's start and the close span's start is stored in 3 bits.
                let kind = transmute::<u8, DelimiterKind>((control_byte_extra_data & 0b11) as u8);
                let open_span = span;
                let close_span_offset_from_open = control_byte_extra_data >> 2;
                let close_span = Span {
                    range: open_span.range + TextSize::new(close_span_offset_from_open),
                    anchor: open_span.anchor,
                    ctx: open_span.ctx,
                };
                let len = u32::from(ptr.read::<u8>());
                children_byte_len = u32::from(ptr.read::<u8>());

                TokenTree::Subtree(Subtree {
                    delimiter: Delimiter { open: open_span, close: close_span, kind },
                    len,
                })
            }
            0b010 => {
                // Subtree, format 3:
                //  - Subtree length is a u8.
                //  - Subtree length in bytes is 12 bits.
                //  - The open and close span have the same length.
                //  - The offset between the open span's start and the close span's start is stored in 8 bits.
                //  - The close span's span_parts_index is stored in 7 bits.
                // This is less efficient than formats 1 and 2 (requires two more bytes), but more efficient than the general format.
                let kind = transmute::<u8, DelimiterKind>((control_byte_extra_data & 0b11) as u8);
                let open_span = span;
                let close_span_offset_from_open = u32::from(ptr.read::<u8>());
                let len = u32::from(ptr.read::<u8>());
                children_byte_len = u32::from(ptr.read::<u16>());
                let mut close_span_parts_index = control_byte_extra_data >> 2;
                close_span_parts_index |= (children_byte_len & 0b1111) << 3;
                children_byte_len >>= 4;
                let close_span = compressed[close_span_parts_index as usize]
                    .recombine(open_span.range + TextSize::new(close_span_offset_from_open));

                TokenTree::Subtree(Subtree {
                    delimiter: Delimiter { open: open_span, close: close_span, kind },
                    len,
                })
            }
            0b101 => {
                cold_path();

                // Subtree, format 4 - most general format: subtree length is u32, subtree length in bytes is u32, full decoded
                // span for close span, delimiter kind is 5 bits (not needed now, will be needed when we have different kinds of
                // invisible delimiters).
                let kind = transmute::<u8, DelimiterKind>(control_byte_extra_data as u8);
                let open_span = span;
                let close_span_start = ptr.read::<u32>();
                let close_span;
                (ptr, close_span) = decode_span(ptr, close_span_start);
                let close_span =
                    compressed[close_span.span_parts_index].recombine(close_span.text_range);
                let len = ptr.read::<u32>();
                children_byte_len = ptr.read::<u32>();

                TokenTree::Subtree(Subtree {
                    delimiter: Delimiter { open: open_span, close: close_span, kind },
                    len,
                })
            }
            0b110 => {
                cold_path();

                // Non-ASCII punct. Extremely rare but technically possible.
                let spacing = transmute::<u8, Spacing>(control_byte_extra_data as u8);
                let char = ptr.read::<char>();

                TokenTree::Leaf(Leaf::Punct(Punct { char, spacing, span }))
            }
            0b011 | 0b100 => {
                // Literal, format 1: the 6 bits remaining from `control_byte` decide the kind and the suffix len from a constant set
                // (of the most common).
                let control_byte_extra_data = control_byte >> 2;
                let text_and_suffix_first_byte = ptr.read::<u8>();
                let text_and_suffix;
                (ptr, text_and_suffix) = decode_symbol(ptr, text_and_suffix_first_byte, symbols);
                let kind = match control_byte_extra_data & 0b11 {
                    0b00 => LitKind::Str,
                    0b01 => LitKind::StrRaw(0),
                    0b10 => LitKind::StrRaw(1),
                    0b11 => LitKind::Integer,
                    _ => unreachable!(),
                };
                let suffix_len = (control_byte_extra_data >> 2) as u8;

                TokenTree::Leaf(Leaf::Literal(Literal { text_and_suffix, span, kind, suffix_len }))
            }
            0b111 => {
                cold_path();

                // Literal, format 2: most general format.
                let kind = match control_byte_extra_data {
                    0 => LitKind::Byte,
                    1 => LitKind::Char,
                    2 => LitKind::Integer,
                    3 => LitKind::Float,
                    4 => LitKind::Str,
                    5 => LitKind::StrRaw(ptr.read::<u8>()),
                    6 => LitKind::ByteStr,
                    7 => LitKind::ByteStrRaw(ptr.read::<u8>()),
                    8 => LitKind::CStr,
                    9 => LitKind::CStrRaw(ptr.read::<u8>()),
                    10 => LitKind::Err(()),
                    _ => unreachable!(),
                };
                let suffix_len = ptr.read::<u8>();
                let text_and_suffix_first_byte = ptr.read::<u8>();
                let text_and_suffix;
                (ptr, text_and_suffix) = decode_symbol(ptr, text_and_suffix_first_byte, symbols);

                TokenTree::Leaf(Leaf::Literal(Literal { text_and_suffix, span, kind, suffix_len }))
            }
            _ => unreachable!(),
        };
        (ptr, result, children_byte_len)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TokenTreesSlice<'a> {
    current: BufferReader<'a>,
    end: BufferReader<'a>,
    span_parts: &'a [CompressedSpanPart],
    symbols: &'a [Symbol],
}

unsafe impl Send for TokenTreesSlice<'_> {}
unsafe impl Sync for TokenTreesSlice<'_> {}

impl<'a> TokenTreesSlice<'a> {
    #[inline]
    fn new(top_subtree: &'a TopSubtree) -> Self {
        let (current, end) = BufferReader::start_end(&top_subtree.buffer);
        Self { current, end, span_parts: &top_subtree.span_parts, symbols: &top_subtree.symbols }
    }

    #[inline]
    pub(crate) fn empty() -> Self {
        let (current, end) = BufferReader::start_end(&[]);
        Self { current, end, span_parts: &[], symbols: &[] }
    }

    pub(crate) fn advance(&mut self) -> Option<TokenTree> {
        if self.current == self.end {
            return None;
        }

        let (new_current, token_tree, _children_byte_len) =
            unsafe { decode(self.current, self.span_parts, self.symbols) };
        self.current = new_current;
        Some(token_tree)
    }

    /// This is like `advance()`, but when encountering a subtree, it changes `self` to skip it and returns a
    /// slice into it (what `advance()` would have done to `self`).
    pub(crate) fn advance_skip_subtree(&mut self) -> Option<(TokenTree, TokenTreesSlice<'a>)> {
        if self.current == self.end {
            return None;
        }

        let (new_current, token_tree, children_byte_len) =
            unsafe { decode(self.current, self.span_parts, self.symbols) };
        self.current = new_current;
        let subtree_slice = *self;
        unsafe { self.current.skip(children_byte_len as usize) };
        Some((token_tree, subtree_slice))
    }

    pub(crate) fn iter(mut self) -> impl Iterator<Item = TokenTree> {
        std::iter::from_fn(move || self.advance())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TopSubtree {
    buffer: Box<[u8]>,
    /// The last two are the top subtree's open and close span, in this order.
    span_parts: Box<[CompressedSpanPart]>,
    symbols: Box<[Symbol]>,
    len: usize,
}

impl TopSubtree {
    pub fn empty(span: DelimSpan) -> Self {
        encode_all(
            vec![TokenTree::Subtree(Subtree {
                delimiter: Delimiter::invisible_delim_spanned(span),
                len: 0,
            })]
            .into_iter(),
            FxHashMap::default(),
            FxHashMap::default(),
        )
    }

    pub fn invisible_from_leaves<const N: usize>(delim_span: Span, leaves: [Leaf; N]) -> Self {
        Self::from_serialized(
            std::iter::chain(
                [TokenTree::Subtree(Subtree {
                    delimiter: Delimiter::invisible_spanned(delim_span),
                    len: leaves.len() as u32,
                })],
                leaves.into_iter().map(TokenTree::Leaf),
            )
            .collect(),
        )
    }

    pub fn from_token_trees(delimiter: Delimiter, token_trees: TokenTreesView<'_>) -> Self {
        let mut builder = TopSubtreeBuilder::new(delimiter);
        builder.extend_with_tt(token_trees);
        builder.build()
    }

    pub fn from_serialized(tts: Vec<TokenTree>) -> Self {
        let (span_frequencies, symbols) = compute_span_frequencies_and_symbols(&tts[1..]); // Do not include the top subtree.
        encode_all(tts.into_iter(), span_frequencies, symbols)
    }

    pub fn from_subtree(subtree: SubtreeView<'_>) -> Self {
        let mut builder = TopSubtreeBuilder::new(subtree.top_subtree().delimiter);
        builder.extend_with_tt(subtree.token_trees());
        builder.build()
    }

    pub fn view(&self) -> SubtreeView<'_> {
        let slice = TokenTreesSlice::new(self);
        SubtreeView(TokenTreesView { slice, len: self.len })
    }

    pub fn iter(&self) -> TtIter<'_> {
        self.view().iter()
    }

    pub fn top_subtree(&self) -> Subtree {
        self.view().top_subtree()
    }

    pub fn set_top_subtree_delimiter_kind(&mut self, kind: DelimiterKind) {
        change_root_delimiter(&mut self.buffer, kind);
    }

    pub fn set_top_subtree_delimiter_span(&mut self, span: DelimSpan) {
        let open_span_idx = self.span_parts.len() - 2;
        let close_span_idx = open_span_idx + 1;
        unsafe {
            change_root_spans(
                &mut self.buffer,
                open_span_idx as u32,
                close_span_idx as u32,
                span.open.range,
                span.close.range,
            );
        }
        self.span_parts[open_span_idx] = CompressedSpanPart::from_span(&span.open);
        self.span_parts[close_span_idx] = CompressedSpanPart::from_span(&span.close);
    }

    /// **Warning**: This is very expensive, this rebuilds the whole tree. Avoid using this if you can.
    pub fn set_token(&mut self, idx: usize, leaf: Leaf) {
        let mut tts = TokenTreesSlice::new(self).iter().collect::<Vec<_>>();
        assert_matches!(tts[idx], TokenTree::Leaf(_), "cannot change a subtree to a leaf");
        tts[idx] = leaf.into();
        *self = TopSubtree::from_serialized(tts);
    }

    pub fn token_trees(&self) -> TokenTreesView<'_> {
        self.view().token_trees()
    }

    pub fn as_token_trees(&self) -> TokenTreesView<'_> {
        self.view().as_token_trees()
    }

    pub fn change_every_ast_id(&mut self, mut callback: impl FnMut(&mut span::ErasedFileAstId)) {
        for span_part in &mut self.span_parts {
            callback(&mut span_part.anchor.ast_id);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TopSubtreeBuilder {
    unclosed_subtree_indices: Vec<usize>,
    token_trees: Vec<TokenTree>,
    span_parts_frequencies: FxHashMap<CompressedSpanPart, usize>,
    last_closed_subtree: Option<usize>,
    symbol_frequencies: FxHashMap<Symbol, usize>,
}

impl TopSubtreeBuilder {
    fn insert_span(&mut self, span: &Span) {
        *self.span_parts_frequencies.entry(CompressedSpanPart::from_span(span)).or_insert(0) += 1;
    }

    fn remove_span(span_parts_frequencies: &mut FxHashMap<CompressedSpanPart, usize>, span: &Span) {
        let hash_map::Entry::Occupied(mut entry) =
            span_parts_frequencies.entry(CompressedSpanPart::from_span(span))
        else {
            panic!("span not present");
        };
        *entry.get_mut() -= 1;
        if *entry.get() == 0 {
            entry.remove();
        }
    }

    fn insert_symbol(&mut self, symbol: Symbol) {
        *self.symbol_frequencies.entry(symbol).or_insert(0) += 1;
    }

    fn remove_symbol(symbol_frequencies: &mut FxHashMap<Symbol, usize>, symbol: Symbol) {
        let hash_map::Entry::Occupied(mut entry) = symbol_frequencies.entry(symbol) else {
            panic!("span not present");
        };
        *entry.get_mut() -= 1;
        if *entry.get() == 0 {
            entry.remove();
        }
    }

    pub fn new(top_delimiter: Delimiter) -> Self {
        let mut result = Self {
            unclosed_subtree_indices: Vec::new(),
            token_trees: Vec::new(),
            span_parts_frequencies: FxHashMap::default(),
            last_closed_subtree: None,
            symbol_frequencies: FxHashMap::default(),
        };
        // Do not insert the top delimiters, they have their own place because we sometimes need to change them.
        result.token_trees.push(TokenTree::Subtree(Subtree { delimiter: top_delimiter, len: 0 }));
        result
    }

    /// Not to be exposed, this assumes the subtree's children will be filled in immediately.
    fn push_subtree(&mut self, subtree: Subtree) {
        self.insert_span(&subtree.delimiter.open);
        self.insert_span(&subtree.delimiter.close);
        self.token_trees.push(subtree.into());
    }

    pub fn open(&mut self, delimiter_kind: DelimiterKind, open_span: Span) {
        self.insert_span(&open_span);
        let subtree_idx = self.token_trees.len();
        self.token_trees.push(TokenTree::Subtree(Subtree {
            delimiter: Delimiter { open: open_span, close: open_span, kind: delimiter_kind },
            len: 0, // Will be overwritten on close.
        }));
        self.unclosed_subtree_indices.push(subtree_idx);
    }

    pub fn close(&mut self, close_span: Span) {
        self.insert_span(&close_span);

        let last_unclosed_index = self
            .unclosed_subtree_indices
            .pop()
            .expect("attempt to close a `tt::Subtree` when none is open");
        let token_trees_len = self.token_trees.len();
        let TokenTree::Subtree(Subtree { delimiter: Delimiter { open: _, close, kind: _ }, len }) =
            &mut self.token_trees[last_unclosed_index]
        else {
            unreachable!("unclosed token tree is always a subtree");
        };
        *len = (token_trees_len - last_unclosed_index - 1) as u32;
        *close = close_span;
        self.last_closed_subtree = Some(last_unclosed_index);
    }

    /// You cannot call this consecutively, it will only work once after close.
    pub fn remove_last_subtree_if_invisible(&mut self) {
        let Some(last_subtree_idx) = self.last_closed_subtree else { return };
        if let TokenTree::Subtree(Subtree {
            delimiter: Delimiter { kind: DelimiterKind::Invisible, .. },
            ..
        }) = self.token_trees[last_subtree_idx]
        {
            self.token_trees.remove(last_subtree_idx);
        }
        self.last_closed_subtree = None;
    }

    pub fn push(&mut self, leaf: Leaf) {
        self.insert_span(leaf.span());
        if let Some(symbol) = leaf.symbol() {
            self.insert_symbol(symbol.clone());
        }
        self.token_trees.push(leaf.into());
    }

    fn push_token_tree(&mut self, tt: TokenTree) {
        match tt {
            TokenTree::Leaf(leaf) => self.push(leaf),
            TokenTree::Subtree(subtree) => self.push_subtree(subtree),
        }
    }

    pub fn extend(&mut self, leaves: impl IntoIterator<Item = Leaf>) {
        leaves.into_iter().for_each(|leaf| self.push(leaf));
    }

    pub fn extend_with_tt(&mut self, tt: TokenTreesView<'_>) {
        tt.iter_flat_tokens().for_each(|tt| self.push_token_tree(tt));
    }

    /// Like [`Self::extend_with_tt()`], but makes sure the new tokens will never be
    /// joint with whatever comes after them.
    pub fn extend_with_tt_alone(&mut self, tt: TokenTreesView<'_>) {
        self.extend_with_tt(tt);
        if !tt.is_empty()
            && let Some(TokenTree::Leaf(Leaf::Punct(Punct { spacing, .. }))) =
                self.token_trees.last_mut()
        {
            *spacing = Spacing::Alone;
        }
    }

    pub fn expected_delimiters(&self) -> impl Iterator<Item = DelimiterKind> {
        self.unclosed_subtree_indices.iter().rev().map(|&subtree_idx| {
            let TokenTree::Subtree(Subtree { delimiter, .. }) = self.token_trees[subtree_idx]
            else {
                unreachable!("unclosed token tree is always a subtree")
            };
            delimiter.kind
        })
    }

    /// Builds, and remove the top subtree if it has only one subtree child.
    pub fn build_skip_top_subtree(mut self) -> TopSubtree {
        assert!(
            self.unclosed_subtree_indices.is_empty(),
            "attempt to build an unbalanced `TopSubtreeBuilder`"
        );
        let tt_len = self.token_trees.len();
        if let Some(&TokenTree::Subtree(Subtree { len, delimiter })) = self.token_trees.get(1)
            && (len as usize) == (tt_len - 2)
        {
            // The top subtree's delimiters should not be included.
            Self::remove_span(&mut self.span_parts_frequencies, &delimiter.open);
            Self::remove_span(&mut self.span_parts_frequencies, &delimiter.close);

            let mut token_trees = self.token_trees.into_iter();
            token_trees.next(); // Remove the first subtree.
            encode_all(token_trees, self.span_parts_frequencies, self.symbol_frequencies)
        } else {
            self.build()
        }
    }

    pub fn build(mut self) -> TopSubtree {
        assert!(
            self.unclosed_subtree_indices.is_empty(),
            "attempt to build an unbalanced `TopSubtreeBuilder`"
        );
        let tts_len = self.token_trees.len();
        let TokenTree::Subtree(top_subtree) = &mut self.token_trees[0] else {
            panic!("first token tree must be a subtree");
        };
        top_subtree.len = (tts_len - 1).try_into().unwrap();
        encode_all(
            self.token_trees.into_iter(),
            self.span_parts_frequencies,
            self.symbol_frequencies,
        )
    }

    pub fn restore_point(&mut self) -> SubtreeBuilderRestorePoint {
        // We reset the `last_closed_subtree`, since restoring from a restore point doesn't play well with removing the last subtree.
        self.last_closed_subtree = None;
        SubtreeBuilderRestorePoint {
            unclosed_subtree_indices_len: self.unclosed_subtree_indices.len(),
            token_trees_len: self.token_trees.len(),
        }
    }

    pub fn restore(&mut self, restore_point: SubtreeBuilderRestorePoint) {
        if restore_point.token_trees_len >= self.token_trees.len() {
            // This means we restored twice, potentially with an earlier restore point first.
            return;
        }

        for tt in &self.token_trees[restore_point.token_trees_len..] {
            match tt {
                TokenTree::Leaf(leaf) => {
                    Self::remove_span(&mut self.span_parts_frequencies, leaf.span());

                    if let Some(symbol) = leaf.symbol() {
                        Self::remove_symbol(&mut self.symbol_frequencies, symbol.clone());
                    }
                }
                TokenTree::Subtree(subtree) => {
                    Self::remove_span(&mut self.span_parts_frequencies, &subtree.delimiter.open);
                    Self::remove_span(&mut self.span_parts_frequencies, &subtree.delimiter.close);
                }
            }
        }

        self.unclosed_subtree_indices.truncate(restore_point.unclosed_subtree_indices_len);
        self.token_trees.truncate(restore_point.token_trees_len);
        self.last_closed_subtree = None;
    }
}

#[derive(Clone, Copy)]
pub struct SubtreeBuilderRestorePoint {
    unclosed_subtree_indices_len: usize,
    token_trees_len: usize,
}
