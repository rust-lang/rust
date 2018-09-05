use TextUnit;

use std::str::Chars;

/// A simple view into the characters of a string.
pub(crate) struct Ptr<'s> {
    text: &'s str,
    len: TextUnit,
}

impl<'s> Ptr<'s> {
    /// Creates a new `Ptr` from a string.
    pub fn new(text: &'s str) -> Ptr<'s> {
        Ptr {
            text,
            len: 0.into(),
        }
    }

    /// Gets the length of the remaining string.
    pub fn into_len(self) -> TextUnit {
        self.len
    }

    /// Gets the current character, if one exists.
    pub fn current(&self) -> Option<char> {
        self.chars().next()
    }

    /// Gets the nth character from the current.
    /// For example, 0 will return the current token, 1 will return the next, etc.
    pub fn nth(&self, n: u32) -> Option<char> {
        let mut chars = self.chars().peekable();
        chars.by_ref().skip(n as usize).next()
    }

    /// Checks whether the current character is `c`.
    pub fn at(&self, c: char) -> bool {
        self.current() == Some(c)
    }

    /// Checks whether the next characters match `s`.
    pub fn at_str(&self, s: &str) -> bool {
        let chars = self.chars();
        chars.as_str().starts_with(s)
    }

    /// Checks whether the current character satisfies the predicate `p`.
    pub fn at_p<P: Fn(char) -> bool>(&self, p: P) -> bool {
        self.current().map(p) == Some(true)
    }

    /// Checks whether the nth character satisfies the predicate `p`.
    pub fn nth_is_p<P: Fn(char) -> bool>(&self, n: u32, p: P) -> bool {
        self.nth(n).map(p) == Some(true)
    }

    /// Moves to the next character.
    pub fn bump(&mut self) -> Option<char> {
        let ch = self.chars().next()?;
        self.len += TextUnit::of_char(ch);
        Some(ch)
    }

    /// Moves to the next character as long as `pred` is satisfied.
    pub fn bump_while<F: Fn(char) -> bool>(&mut self, pred: F) {
        loop {
            match self.current() {
                Some(c) if pred(c) => {
                    self.bump();
                }
                _ => return,
            }
        }
    }

    /// Returns the text up to the current point.
    pub fn current_token_text(&self) -> &str {
        let len: u32 = self.len.into();
        &self.text[..len as usize]
    }

    /// Returns an iterator over the remaining characters.
    fn chars(&self) -> Chars {
        let len: u32 = self.len.into();
        self.text[len as usize..].chars()
    }
}
