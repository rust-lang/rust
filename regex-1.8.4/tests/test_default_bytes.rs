macro_rules! regex_new {
    ($re:expr) => {{
        use regex::bytes::Regex;
        Regex::new($re)
    }};
}

macro_rules! regex_set_new {
    ($res:expr) => {{
        use regex::bytes::RegexSet;
        RegexSet::new($res)
    }};
}

macro_rules! regex {
    ($re:expr) => {
        regex_new!($re).unwrap()
    };
}

macro_rules! regex_set {
    ($res:expr) => {
        regex_set_new!($res).unwrap()
    };
}

// Must come before other module definitions.
include!("macros_bytes.rs");
include!("macros.rs");

// A silly wrapper to make it possible to write and match raw bytes.
struct R<'a>(&'a [u8]);
impl<'a> R<'a> {
    fn as_bytes(&self) -> &'a [u8] {
        self.0
    }
}

// See: https://github.com/rust-lang/regex/issues/321
//
// These tests are here because they do not have the same behavior in every
// regex engine.
mat!(invalid_utf8_nfa1, r".", R(b"\xD4\xC2\x65\x2B\x0E\xFE"), Some((2, 3)));
mat!(invalid_utf8_nfa2, r"${2}ä", R(b"\xD4\xC2\x65\x2B\x0E\xFE"), None);
mat!(
    invalid_utf8_nfa3,
    r".",
    R(b"\x0A\xDB\x82\x6E\x33\x01\xDD\x33\xCD"),
    Some((1, 3))
);
mat!(
    invalid_utf8_nfa4,
    r"${2}ä",
    R(b"\x0A\xDB\x82\x6E\x33\x01\xDD\x33\xCD"),
    None
);

mod api;
mod bytes;
mod crazy;
mod flags;
mod fowler;
mod multiline;
mod noparse;
mod regression;
mod replace;
mod set;
mod shortest_match;
mod suffix_reverse;
#[cfg(feature = "unicode")]
mod unicode;
#[cfg(feature = "unicode-perl")]
mod word_boundary;
#[cfg(feature = "unicode-perl")]
mod word_boundary_unicode;
