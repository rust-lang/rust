//! Yet another version of owned string, backed by a syntax tree token.

use std::{cmp::Ordering, fmt, ops};

pub enum TokenText<'a> {
    Borrowed(&'a str),
    Owned(rowan::GreenToken),
}

impl TokenText<'_> {
    pub fn as_str(&self) -> &str {
        match self {
            TokenText::Borrowed(it) => *it,
            TokenText::Owned(green) => green.text(),
        }
    }
}

impl ops::Deref for TokenText<'_> {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl AsRef<str> for TokenText<'_> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<TokenText<'_>> for String {
    fn from(token_text: TokenText) -> Self {
        token_text.as_str().into()
    }
}

impl PartialEq<&'_ str> for TokenText<'_> {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
impl PartialEq<TokenText<'_>> for &'_ str {
    fn eq(&self, other: &TokenText) -> bool {
        other == self
    }
}
impl PartialEq<String> for TokenText<'_> {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}
impl PartialEq<TokenText<'_>> for String {
    fn eq(&self, other: &TokenText) -> bool {
        other == self
    }
}
impl PartialEq for TokenText<'_> {
    fn eq(&self, other: &TokenText) -> bool {
        self.as_str() == other.as_str()
    }
}
impl Eq for TokenText<'_> {}
impl Ord for TokenText<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl PartialOrd for TokenText<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl fmt::Display for TokenText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}
impl fmt::Debug for TokenText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}
