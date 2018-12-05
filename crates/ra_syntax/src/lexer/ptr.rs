use crate::TextUnit;

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
    /// For example, 0 will return the current character, 1 will return the next, etc.
    pub fn nth(&self, n: u32) -> Option<char> {
        self.chars().nth(n as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current() {
        let ptr = Ptr::new("test");
        assert_eq!(ptr.current(), Some('t'));
    }

    #[test]
    fn test_nth() {
        let ptr = Ptr::new("test");
        assert_eq!(ptr.nth(0), Some('t'));
        assert_eq!(ptr.nth(1), Some('e'));
        assert_eq!(ptr.nth(2), Some('s'));
        assert_eq!(ptr.nth(3), Some('t'));
        assert_eq!(ptr.nth(4), None);
    }

    #[test]
    fn test_at() {
        let ptr = Ptr::new("test");
        assert!(ptr.at('t'));
        assert!(!ptr.at('a'));
    }

    #[test]
    fn test_at_str() {
        let ptr = Ptr::new("test");
        assert!(ptr.at_str("t"));
        assert!(ptr.at_str("te"));
        assert!(ptr.at_str("test"));
        assert!(!ptr.at_str("tests"));
        assert!(!ptr.at_str("rust"));
    }

    #[test]
    fn test_at_p() {
        let ptr = Ptr::new("test");
        assert!(ptr.at_p(|c| c == 't'));
        assert!(!ptr.at_p(|c| c == 'e'));
    }

    #[test]
    fn test_nth_is_p() {
        let ptr = Ptr::new("test");
        assert!(ptr.nth_is_p(0, |c| c == 't'));
        assert!(!ptr.nth_is_p(1, |c| c == 't'));
        assert!(ptr.nth_is_p(3, |c| c == 't'));
        assert!(!ptr.nth_is_p(150, |c| c == 't'));
    }

    #[test]
    fn test_bump() {
        let mut ptr = Ptr::new("test");
        assert_eq!(ptr.current(), Some('t'));
        ptr.bump();
        assert_eq!(ptr.current(), Some('e'));
        ptr.bump();
        assert_eq!(ptr.current(), Some('s'));
        ptr.bump();
        assert_eq!(ptr.current(), Some('t'));
        ptr.bump();
        assert_eq!(ptr.current(), None);
        ptr.bump();
        assert_eq!(ptr.current(), None);
    }

    #[test]
    fn test_bump_while() {
        let mut ptr = Ptr::new("test");
        assert_eq!(ptr.current(), Some('t'));
        ptr.bump_while(|c| c != 's');
        assert_eq!(ptr.current(), Some('s'));
    }
}
