use std::str::Chars;

pub enum FrontmatterAllowed {
    Yes,
    No,
}

/// Peekable iterator over a char sequence.
///
/// Next characters can be peeked via `first` method,
/// and position can be shifted forward via `bump` method.
pub struct Cursor<'a> {
    len_remaining: usize,
    /// Iterator over chars. Slightly faster than a &str.
    chars: Chars<'a>,
    pub(crate) frontmatter_allowed: FrontmatterAllowed,
    #[cfg(debug_assertions)]
    prev: char,
}

pub(crate) const EOF_CHAR: char = '\0';

impl<'a> Cursor<'a> {
    pub fn new(input: &'a str, frontmatter_allowed: FrontmatterAllowed) -> Cursor<'a> {
        Cursor {
            len_remaining: input.len(),
            chars: input.chars(),
            frontmatter_allowed,
            #[cfg(debug_assertions)]
            prev: EOF_CHAR,
        }
    }

    pub fn as_str(&self) -> &'a str {
        self.chars.as_str()
    }

    /// Returns the last eaten symbol (or `'\0'` in release builds).
    /// (For debug assertions only.)
    pub(crate) fn prev(&self) -> char {
        #[cfg(debug_assertions)]
        {
            self.prev
        }

        #[cfg(not(debug_assertions))]
        {
            EOF_CHAR
        }
    }

    /// Peeks the next symbol from the input stream without consuming it.
    /// If requested position doesn't exist, `EOF_CHAR` is returned.
    /// However, getting `EOF_CHAR` doesn't always mean actual end of file,
    /// it should be checked with `is_eof` method.
    pub fn first(&self) -> char {
        // `.next()` optimizes better than `.nth(0)`
        self.chars.clone().next().unwrap_or(EOF_CHAR)
    }

    /// Peeks the second symbol from the input stream without consuming it.
    pub(crate) fn second(&self) -> char {
        // `.next()` optimizes better than `.nth(1)`
        let mut iter = self.chars.clone();
        iter.next();
        iter.next().unwrap_or(EOF_CHAR)
    }

    /// Peeks the third symbol from the input stream without consuming it.
    pub fn third(&self) -> char {
        // `.next()` optimizes better than `.nth(2)`
        let mut iter = self.chars.clone();
        iter.next();
        iter.next();
        iter.next().unwrap_or(EOF_CHAR)
    }

    /// Checks if there is nothing more to consume.
    pub(crate) fn is_eof(&self) -> bool {
        self.chars.as_str().is_empty()
    }

    /// Returns amount of already consumed symbols.
    pub(crate) fn pos_within_token(&self) -> u32 {
        (self.len_remaining - self.chars.as_str().len()) as u32
    }

    /// Resets the number of bytes consumed to 0.
    pub(crate) fn reset_pos_within_token(&mut self) {
        self.len_remaining = self.chars.as_str().len();
    }

    /// Moves to the next character.
    pub(crate) fn bump(&mut self) -> Option<char> {
        let c = self.chars.next()?;

        #[cfg(debug_assertions)]
        {
            self.prev = c;
        }

        Some(c)
    }

    pub(crate) fn bump_if(&mut self, byte: char) -> bool {
        let mut chars = self.chars.clone();
        if chars.next() == Some(byte) {
            self.chars = chars;
            true
        } else {
            false
        }
    }

    /// Bumps the cursor if the next character is either of the two expected characters.
    pub(crate) fn bump_if_either(&mut self, byte1: char, byte2: char) -> bool {
        let mut chars = self.chars.clone();
        if let Some(c) = chars.next()
            && (c == byte1 || c == byte2)
        {
            self.chars = chars;
            return true;
        }
        false
    }

    /// Moves to a substring by a number of bytes.
    pub(crate) fn bump_bytes(&mut self, n: usize) {
        self.chars = self.as_str()[n..].chars();
    }

    /// Eats symbols while predicate returns true or until the end of file is reached.
    pub(crate) fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
        // It was tried making optimized version of this for eg. line comments, but
        // LLVM can inline all of this and compile it down to fast iteration over bytes.
        while predicate(self.first()) && !self.is_eof() {
            self.bump();
        }
    }
    /// Eats characters until the given byte is found.
    /// Returns true if the byte was found, false if end of file was reached.
    pub(crate) fn eat_until(&mut self, byte: u8) -> bool {
        match memchr::memchr(byte, self.as_str().as_bytes()) {
            Some(index) => {
                self.bump_bytes(index);
                true
            }
            None => {
                self.chars = "".chars();
                false
            }
        }
    }

    /// Eats characters until any of the given bytes is found, then consumes past it.
    /// Returns the found byte if any, or None if end of file was reached.
    pub(crate) fn eat_past_either(&mut self, byte1: u8, byte2: u8) -> Option<u8> {
        let bytes = self.as_str().as_bytes();
        match memchr::memchr2(byte1, byte2, bytes) {
            Some(index) => {
                let found = bytes[index];
                self.bump_bytes(index + 1);
                Some(found)
            }
            None => {
                self.chars = "".chars();
                None
            }
        }
    }
}
