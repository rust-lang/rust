use std::char;
use std::cmp::Ordering;
use std::fmt;
use std::ops;
use std::u32;

use crate::literal::LiteralSearcher;
use crate::prog::InstEmptyLook;
use crate::utf8::{decode_last_utf8, decode_utf8};

/// Represents a location in the input.
#[derive(Clone, Copy, Debug)]
pub struct InputAt {
    pos: usize,
    c: Char,
    byte: Option<u8>,
    len: usize,
}

impl InputAt {
    /// Returns true iff this position is at the beginning of the input.
    pub fn is_start(&self) -> bool {
        self.pos == 0
    }

    /// Returns true iff this position is past the end of the input.
    pub fn is_end(&self) -> bool {
        self.c.is_none() && self.byte.is_none()
    }

    /// Returns the character at this position.
    ///
    /// If this position is just before or after the input, then an absent
    /// character is returned.
    pub fn char(&self) -> Char {
        self.c
    }

    /// Returns the byte at this position.
    pub fn byte(&self) -> Option<u8> {
        self.byte
    }

    /// Returns the UTF-8 width of the character at this position.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the UTF-8 width of the character at this position
    /// is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the byte offset of this position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Returns the byte offset of the next position in the input.
    pub fn next_pos(&self) -> usize {
        self.pos + self.len
    }
}

/// An abstraction over input used in the matching engines.
pub trait Input: fmt::Debug {
    /// Return an encoding of the position at byte offset `i`.
    fn at(&self, i: usize) -> InputAt;

    /// Return the Unicode character occurring next to `at`.
    ///
    /// If no such character could be decoded, then `Char` is absent.
    fn next_char(&self, at: InputAt) -> Char;

    /// Return the Unicode character occurring previous to `at`.
    ///
    /// If no such character could be decoded, then `Char` is absent.
    fn previous_char(&self, at: InputAt) -> Char;

    /// Return true if the given empty width instruction matches at the
    /// input position given.
    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool;

    /// Scan the input for a matching prefix.
    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt>;

    /// The number of bytes in the input.
    fn len(&self) -> usize;

    /// Whether the input is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the given input as a sequence of bytes.
    fn as_bytes(&self) -> &[u8];
}

impl<'a, T: Input> Input for &'a T {
    fn at(&self, i: usize) -> InputAt {
        (**self).at(i)
    }

    fn next_char(&self, at: InputAt) -> Char {
        (**self).next_char(at)
    }

    fn previous_char(&self, at: InputAt) -> Char {
        (**self).previous_char(at)
    }

    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool {
        (**self).is_empty_match(at, empty)
    }

    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt> {
        (**self).prefix_at(prefixes, at)
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn as_bytes(&self) -> &[u8] {
        (**self).as_bytes()
    }
}

/// An input reader over characters.
#[derive(Clone, Copy, Debug)]
pub struct CharInput<'t>(&'t [u8]);

impl<'t> CharInput<'t> {
    /// Return a new character input reader for the given string.
    pub fn new(s: &'t [u8]) -> CharInput<'t> {
        CharInput(s)
    }
}

impl<'t> ops::Deref for CharInput<'t> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0
    }
}

impl<'t> Input for CharInput<'t> {
    fn at(&self, i: usize) -> InputAt {
        if i >= self.len() {
            InputAt { pos: self.len(), c: None.into(), byte: None, len: 0 }
        } else {
            let c = decode_utf8(&self[i..]).map(|(c, _)| c).into();
            InputAt { pos: i, c, byte: None, len: c.len_utf8() }
        }
    }

    fn next_char(&self, at: InputAt) -> Char {
        at.char()
    }

    fn previous_char(&self, at: InputAt) -> Char {
        decode_last_utf8(&self[..at.pos()]).map(|(c, _)| c).into()
    }

    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool {
        use crate::prog::EmptyLook::*;
        match empty.look {
            StartLine => {
                let c = self.previous_char(at);
                at.pos() == 0 || c == '\n'
            }
            EndLine => {
                let c = self.next_char(at);
                at.pos() == self.len() || c == '\n'
            }
            StartText => at.pos() == 0,
            EndText => at.pos() == self.len(),
            WordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() != c2.is_word_char()
            }
            NotWordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() == c2.is_word_char()
            }
            WordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_byte() != c2.is_word_byte()
            }
            NotWordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_byte() == c2.is_word_byte()
            }
        }
    }

    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt> {
        prefixes.find(&self[at.pos()..]).map(|(s, _)| self.at(at.pos() + s))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_bytes(&self) -> &[u8] {
        self.0
    }
}

/// An input reader over bytes.
#[derive(Clone, Copy, Debug)]
pub struct ByteInput<'t> {
    text: &'t [u8],
    only_utf8: bool,
}

impl<'t> ByteInput<'t> {
    /// Return a new byte-based input reader for the given string.
    pub fn new(text: &'t [u8], only_utf8: bool) -> ByteInput<'t> {
        ByteInput { text, only_utf8 }
    }
}

impl<'t> ops::Deref for ByteInput<'t> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.text
    }
}

impl<'t> Input for ByteInput<'t> {
    fn at(&self, i: usize) -> InputAt {
        if i >= self.len() {
            InputAt { pos: self.len(), c: None.into(), byte: None, len: 0 }
        } else {
            InputAt {
                pos: i,
                c: None.into(),
                byte: self.get(i).cloned(),
                len: 1,
            }
        }
    }

    fn next_char(&self, at: InputAt) -> Char {
        decode_utf8(&self[at.pos()..]).map(|(c, _)| c).into()
    }

    fn previous_char(&self, at: InputAt) -> Char {
        decode_last_utf8(&self[..at.pos()]).map(|(c, _)| c).into()
    }

    fn is_empty_match(&self, at: InputAt, empty: &InstEmptyLook) -> bool {
        use crate::prog::EmptyLook::*;
        match empty.look {
            StartLine => {
                let c = self.previous_char(at);
                at.pos() == 0 || c == '\n'
            }
            EndLine => {
                let c = self.next_char(at);
                at.pos() == self.len() || c == '\n'
            }
            StartText => at.pos() == 0,
            EndText => at.pos() == self.len(),
            WordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() != c2.is_word_char()
            }
            NotWordBoundary => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                c1.is_word_char() == c2.is_word_char()
            }
            WordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                if self.only_utf8 {
                    // If we must match UTF-8, then we can't match word
                    // boundaries at invalid UTF-8.
                    if c1.is_none() && !at.is_start() {
                        return false;
                    }
                    if c2.is_none() && !at.is_end() {
                        return false;
                    }
                }
                c1.is_word_byte() != c2.is_word_byte()
            }
            NotWordBoundaryAscii => {
                let (c1, c2) = (self.previous_char(at), self.next_char(at));
                if self.only_utf8 {
                    // If we must match UTF-8, then we can't match word
                    // boundaries at invalid UTF-8.
                    if c1.is_none() && !at.is_start() {
                        return false;
                    }
                    if c2.is_none() && !at.is_end() {
                        return false;
                    }
                }
                c1.is_word_byte() == c2.is_word_byte()
            }
        }
    }

    fn prefix_at(
        &self,
        prefixes: &LiteralSearcher,
        at: InputAt,
    ) -> Option<InputAt> {
        prefixes.find(&self[at.pos()..]).map(|(s, _)| self.at(at.pos() + s))
    }

    fn len(&self) -> usize {
        self.text.len()
    }

    fn as_bytes(&self) -> &[u8] {
        self.text
    }
}

/// An inline representation of `Option<char>`.
///
/// This eliminates the need to do case analysis on `Option<char>` to determine
/// ordinality with other characters.
///
/// (The `Option<char>` is not related to encoding. Instead, it is used in the
/// matching engines to represent the beginning and ending boundaries of the
/// search text.)
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Char(u32);

impl fmt::Debug for Char {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match char::from_u32(self.0) {
            None => write!(f, "Empty"),
            Some(c) => write!(f, "{:?}", c),
        }
    }
}

impl Char {
    /// Returns true iff the character is absent.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns the length of the character's UTF-8 encoding.
    ///
    /// If the character is absent, then `1` is returned.
    #[inline]
    pub fn len_utf8(self) -> usize {
        char::from_u32(self.0).map_or(1, |c| c.len_utf8())
    }

    /// Returns true iff the character is a word character.
    ///
    /// If the character is absent, then false is returned.
    pub fn is_word_char(self) -> bool {
        // is_word_character can panic if the Unicode data for \w isn't
        // available. However, our compiler ensures that if a Unicode word
        // boundary is used, then the data must also be available. If it isn't,
        // then the compiler returns an error.
        char::from_u32(self.0).map_or(false, regex_syntax::is_word_character)
    }

    /// Returns true iff the byte is a word byte.
    ///
    /// If the byte is absent, then false is returned.
    pub fn is_word_byte(self) -> bool {
        match char::from_u32(self.0) {
            Some(c) if c <= '\u{7F}' => regex_syntax::is_word_byte(c as u8),
            None | Some(_) => false,
        }
    }
}

impl From<char> for Char {
    fn from(c: char) -> Char {
        Char(c as u32)
    }
}

impl From<Option<char>> for Char {
    fn from(c: Option<char>) -> Char {
        c.map_or(Char(u32::MAX), |c| c.into())
    }
}

impl PartialEq<char> for Char {
    #[inline]
    fn eq(&self, other: &char) -> bool {
        self.0 == *other as u32
    }
}

impl PartialEq<Char> for char {
    #[inline]
    fn eq(&self, other: &Char) -> bool {
        *self as u32 == other.0
    }
}

impl PartialOrd<char> for Char {
    #[inline]
    fn partial_cmp(&self, other: &char) -> Option<Ordering> {
        self.0.partial_cmp(&(*other as u32))
    }
}

impl PartialOrd<Char> for char {
    #[inline]
    fn partial_cmp(&self, other: &Char) -> Option<Ordering> {
        (*self as u32).partial_cmp(&other.0)
    }
}
