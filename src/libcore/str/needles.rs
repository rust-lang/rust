use crate::needle::*;
use crate::ops::Range;
use crate::slice::needles::{SliceSearcher, NaiveSearcher, TwoWaySearcher};
use crate::slice::memchr::{memchr, memrchr};
use crate::fmt;

//------------------------------------------------------------------------------
// Character function searcher
//------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct MultiCharEq<'p>(&'p [char]);

impl<'p> FnOnce<(char,)> for MultiCharEq<'p> {
    type Output = bool;
    #[inline]
    extern "rust-call" fn call_once(self, args: (char,)) -> bool {
        self.call(args)
    }
}

impl<'p> FnMut<(char,)> for MultiCharEq<'p> {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, args: (char,)) -> bool {
        self.call(args)
    }
}

impl<'p> Fn<(char,)> for MultiCharEq<'p> {
    #[inline]
    extern "rust-call" fn call(&self, (c,): (char,)) -> bool {
        self.0.iter().any(|ch| *ch == c)
    }
}

#[derive(Clone)]
pub struct MultiCharSearcher<F> {
    predicate: F,
}

// we need to impl Debug for everything due to stability guarantee.
impl<F> fmt::Debug for MultiCharSearcher<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiCharSearcher").finish()
    }
}

unsafe impl<F: FnMut(char) -> bool> Searcher<str> for MultiCharSearcher<F> {
    #[inline]
    fn search(&mut self, span: Span<&str>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        let st = range.start;
        let h = &hay[range];
        let mut chars = h.chars();
        let c = chars.find(|c| (self.predicate)(*c))?;
        let end = chars.as_str().as_ptr();
        let end = unsafe { end.offset_from(h.as_ptr()) as usize } + st;
        Some((end - c.len_utf8())..end)
    }
}

unsafe impl<F: FnMut(char) -> bool> Consumer<str> for MultiCharSearcher<F> {
    #[inline]
    fn consume(&mut self, hay: Span<&str>) -> Option<usize> {
        let (hay, range) = hay.into_parts();
        let start = range.start;
        if start == range.end {
            return None;
        }
        let c = unsafe { hay.get_unchecked(start..) }.chars().next().unwrap();
        if (self.predicate)(c) {
            Some(start + c.len_utf8())
        } else {
            None
        }
    }

    #[inline]
    fn trim_start(&mut self, hay: &str) -> usize {
        let mut chars = hay.chars();
        let unconsume_amount = chars
            .find_map(|c| if !(self.predicate)(c) { Some(c.len_utf8()) } else { None })
            .unwrap_or(0);
        let consumed = unsafe { chars.as_str().as_ptr().offset_from(hay.as_ptr()) as usize };
        consumed.wrapping_sub(unconsume_amount)
    }
}

unsafe impl<F: FnMut(char) -> bool> ReverseSearcher<str> for MultiCharSearcher<F> {
    #[inline]
    fn rsearch(&mut self, span: Span<&str>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        let st = range.start;
        let h = &hay[range];
        let mut chars = h.chars();
        let c = chars.rfind(|c| (self.predicate)(*c))?;
        let start = chars.as_str().len() + st;
        Some(start..(start + c.len_utf8()))
    }
}

unsafe impl<F: FnMut(char) -> bool> ReverseConsumer<str> for MultiCharSearcher<F> {
    #[inline]
    fn rconsume(&mut self, hay: Span<&str>) -> Option<usize> {
        let (hay, range) = hay.into_parts();
        let end = range.end;
        if range.start == end {
            return None;
        }
        let c = unsafe { hay.get_unchecked(..end) }.chars().next_back().unwrap();
        if (self.predicate)(c) {
            Some(end - c.len_utf8())
        } else {
            None
        }
    }

    #[inline]
    fn trim_end(&mut self, hay: &str) -> usize {
        // `find.map_or` is faster in trim_end in the microbenchmark, while
        // `find.unwrap_or` is faster in trim_start. Don't ask me why.
        let mut chars = hay.chars();
        let unconsume_amount = chars
            .by_ref()
            .rev() // btw, `rev().find()` is faster than `rfind()`
            .find(|c| !(self.predicate)(*c))
            .map_or(0, |c| c.len_utf8());
        chars.as_str().len() + unconsume_amount
    }
}

unsafe impl<F: FnMut(char) -> bool> DoubleEndedSearcher<str> for MultiCharSearcher<F> {}
unsafe impl<F: FnMut(char) -> bool> DoubleEndedConsumer<str> for MultiCharSearcher<F> {}

macro_rules! impl_needle_with_multi_char_searcher {
    ($ty:ty) => {
        impl<'h, F: FnMut(char) -> bool> Needle<$ty> for F {
            type Searcher = MultiCharSearcher<F>;
            type Consumer = MultiCharSearcher<F>;

            #[inline]
            fn into_searcher(self) -> Self::Searcher {
                MultiCharSearcher { predicate: self }
            }

            #[inline]
            fn into_consumer(self) -> Self::Consumer {
                MultiCharSearcher { predicate: self }
            }
        }

        impl<'h, 'p> Needle<$ty> for &'p [char] {
            type Searcher = MultiCharSearcher<MultiCharEq<'p>>;
            type Consumer = MultiCharSearcher<MultiCharEq<'p>>;

            #[inline]
            fn into_searcher(self) -> Self::Searcher {
                MultiCharSearcher { predicate: MultiCharEq(self) }
            }

            #[inline]
            fn into_consumer(self) -> Self::Consumer {
                MultiCharSearcher { predicate: MultiCharEq(self) }
            }
        }
    }
}

impl_needle_with_multi_char_searcher!(&'h str);
impl_needle_with_multi_char_searcher!(&'h mut str);

//------------------------------------------------------------------------------
// Character searcher
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CharSearcher {
    // safety invariant: `utf8_size` must be less than 5
    utf8_size: usize,

    /// A utf8 encoded copy of the `needle`
    utf8_encoded: [u8; 4],

    /// The character currently being searched.
    c: char,
}

impl CharSearcher {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.utf8_encoded[..self.utf8_size]
    }

    #[inline]
    fn last_byte(&self) -> u8 {
        self.utf8_encoded[self.utf8_size - 1]
    }

    #[inline]
    fn new(c: char) -> Self {
        let mut utf8_encoded = [0u8; 4];
        let utf8_size = c.encode_utf8(&mut utf8_encoded).len();
        CharSearcher {
            utf8_size,
            utf8_encoded,
            c,
        }
    }
}

unsafe impl Searcher<str> for CharSearcher {
    #[inline]
    fn search(&mut self, span: Span<&str>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        let mut finger = range.start;
        let bytes = hay.as_bytes();
        loop {
            let index = memchr(self.last_byte(), &bytes[finger..range.end])?;
            finger += index + 1;
            if finger >= self.utf8_size {
                let found = &bytes[(finger - self.utf8_size)..finger];
                if found == self.as_bytes() {
                    return Some((finger - self.utf8_size)..finger);
                }
            }
        }
    }
}

unsafe impl Consumer<str> for CharSearcher {
    #[inline]
    fn consume(&mut self, span: Span<&str>) -> Option<usize> {
        let mut consumer = Needle::<&[u8]>::into_consumer(self.as_bytes());
        consumer.consume(span.as_bytes())
    }

    #[inline]
    fn trim_start(&mut self, hay: &str) -> usize {
        let mut consumer = Needle::<&str>::into_consumer(|c: char| c == self.c);
        consumer.trim_start(hay)
    }
}

unsafe impl ReverseSearcher<str> for CharSearcher {
    #[inline]
    fn rsearch(&mut self, span: Span<&str>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        let start = range.start;
        let mut bytes = hay[range].as_bytes();
        loop {
            let index = memrchr(self.last_byte(), bytes)? + 1;
            if index >= self.utf8_size {
                let found = &bytes[(index - self.utf8_size)..index];
                if found == self.as_bytes() {
                    let index = index + start;
                    return Some((index - self.utf8_size)..index);
                }
            }
            bytes = &bytes[..(index - 1)];
        }
    }
}

unsafe impl ReverseConsumer<str> for CharSearcher {
    #[inline]
    fn rconsume(&mut self, span: Span<&str>) -> Option<usize> {
        if self.utf8_size == 1 {
            let mut consumer = Needle::<&[u8]>::into_consumer(|b: &u8| *b == self.c as u8);
            consumer.rconsume(span.as_bytes())
        } else {
            let mut consumer = Needle::<&str>::into_consumer(|c: char| c == self.c);
            consumer.rconsume(span)
        }
    }

    #[inline]
    fn trim_end(&mut self, haystack: &str) -> usize {
        let mut consumer = Needle::<&str>::into_consumer(|c: char| c == self.c);
        consumer.trim_end(haystack)
    }
}

unsafe impl DoubleEndedSearcher<str> for CharSearcher {}
unsafe impl DoubleEndedConsumer<str> for CharSearcher {}

impl<H: Haystack<Target = str>> Needle<H> for char {
    type Searcher = CharSearcher;
    type Consumer = CharSearcher;

    #[inline]
    fn into_searcher(self) -> Self::Searcher {
        CharSearcher::new(self)
    }

    #[inline]
    fn into_consumer(self) -> Self::Consumer {
        CharSearcher::new(self)
    }
}

//------------------------------------------------------------------------------
// String searcher
//------------------------------------------------------------------------------

unsafe impl<'p> Searcher<str> for TwoWaySearcher<'p, u8> {
    #[inline]
    fn search(&mut self, span: Span<&str>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        self.next(hay.as_bytes(), range)
    }
}

unsafe impl<'p> ReverseSearcher<str> for TwoWaySearcher<'p, u8> {
    #[inline]
    fn rsearch(&mut self, span: Span<&str>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        self.next_back(hay.as_bytes(), range)
    }
}

unsafe impl<'p> Consumer<str> for NaiveSearcher<'p, u8> {
    #[inline]
    fn consume(&mut self, span: Span<&str>) -> Option<usize> {
        self.consume(span.as_bytes())
    }

    #[inline]
    fn trim_start(&mut self, hay: &str) -> usize {
        self.trim_start(hay.as_bytes())
    }
}

unsafe impl<'p> ReverseConsumer<str> for NaiveSearcher<'p, u8> {
    #[inline]
    fn rconsume(&mut self, span: Span<&str>) -> Option<usize> {
        self.rconsume(span.as_bytes())
    }

    #[inline]
    fn trim_end(&mut self, hay: &str) -> usize {
        self.trim_end(hay.as_bytes())
    }
}

macro_rules! impl_needle_for_str_searcher {
    (<[$($gen:tt)*]> for $pat:ty) => {
        impl<$($gen)*, H: Haystack<Target = str>> Needle<H> for $pat {
            type Searcher = SliceSearcher<'p, u8>;
            type Consumer = NaiveSearcher<'p, u8>;

            #[inline]
            fn into_searcher(self) -> Self::Searcher {
                SliceSearcher::new(self.as_bytes())
            }

            #[inline]
            fn into_consumer(self) -> Self::Consumer {
                NaiveSearcher::new(self.as_bytes())
            }
        }
    }
}

impl_needle_for_str_searcher!(<['p]> for &'p str);
impl_needle_for_str_searcher!(<['q, 'p]> for &'q &'p str);
