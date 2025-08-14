#![warn(clippy::doc_suspicious_footnotes)]
#![allow(clippy::needless_raw_string_hashes)]
//! This is not a footnote[^1].
//~^ doc_suspicious_footnotes
//!
//! This is not a footnote[^either], but it doesn't warn.
//!
//! This is not a footnote\[^1], but it also doesn't warn.
//!
//! This is not a footnote[^1\], but it also doesn't warn.
//!
//! This is not a `footnote[^1]`, but it also doesn't warn.
//!
//! This is a footnote[^2].
//!
//! [^2]: hello world

/// This is not a footnote[^1].
//~^ doc_suspicious_footnotes
///
/// This is not a footnote[^either], but it doesn't warn.
///
/// This is not a footnote\[^1], but it also doesn't warn.
///
/// This is not a footnote[^1\], but it also doesn't warn.
///
/// This is not a `footnote[^1]`, but it also doesn't warn.
///
/// This is a footnote[^2].
///
/// [^2]: hello world
pub fn footnotes() {
    // test code goes here
}

pub struct Foo;
#[rustfmt::skip]
impl Foo {
    #[doc = r#"This is not a footnote[^1]."#]
    //~^ doc_suspicious_footnotes
    #[doc = r#""#]
    #[doc = r#"This is not a footnote[^either], but it doesn't warn."#]
    #[doc = r#""#]
    #[doc = r#"This is not a footnote\[^1], but it also doesn't warn."#]
    #[doc = r#""#]
    #[doc = r#"This is not a footnote[^1\], but it also doesn't warn."#]
    #[doc = r#""#]
    #[doc = r#"This is not a `footnote[^1]`, but it also doesn't warn."#]
    #[doc = r#""#]
    #[doc = r#"This is a footnote[^2]."#]
    #[doc = r#""#]
    #[doc = r#"[^2]: hello world"#]
    pub fn footnotes() {
        // test code goes here
    }
    #[doc = "This is not a footnote[^1].

    This is not a footnote[^either], but it doesn't warn.

    This is not a footnote\\[^1], but it also doesn't warn.

    This is not a footnote[^1\\], but it also doesn't warn.

    This is not a `footnote[^1]`, but it also doesn't warn.

    This is a footnote[^2].

    [^2]: hello world
    "]
    //~^^^^^^^^^^^^^^ doc_suspicious_footnotes
    pub fn footnotes2() {
        // test code goes here
    }
    #[cfg_attr(
        not(FALSE),
        doc = "This is not a footnote[^1].\n\nThis is not a footnote[^either], but it doesn't warn."
    //~^ doc_suspicious_footnotes
    )]
    pub fn footnotes3() {
        // test code goes here
    }
    #[doc = "My footnote [^foot\note]"]
    pub fn footnote4() {
        // test code goes here
    }
    #[doc = "Hihi"]pub fn footnote5() {
        // test code goes here
    }
}

#[doc = r"This is not a footnote[^1]."]
//~^ doc_suspicious_footnotes
#[doc = r""]
#[doc = r"This is not a footnote[^either], but it doesn't warn."]
#[doc = r""]
#[doc = r"This is not a footnote\[^1], but it also doesn't warn."]
#[doc = r""]
#[doc = r"This is not a footnote[^1\], but it also doesn't warn."]
#[doc = r""]
#[doc = r"This is not a `footnote[^1]`, but it also doesn't warn."]
#[doc = r""]
#[doc = r"This is a footnote[^2]."]
#[doc = r""]
#[doc = r"[^2]: hello world"]
pub fn footnotes_attrs() {
    // test code goes here
}

pub mod multiline {
    /*!
     * This is not a footnote[^1]. //~ doc_suspicious_footnotes
     *
     * This is not a footnote\[^1], but it doesn't warn.
     *
     * This is a footnote[^2].
     *
     * These give weird results, but correct ones, so it works.
     *
     * [^2]: hello world
     */
    /**
     * This is not a footnote[^1]. //~ doc_suspicious_footnotes
     *
     * This is not a footnote\[^1], but it doesn't warn.
     *
     * This is a footnote[^2].
     *
     * These give weird results, but correct ones, so it works.
     *
     * [^2]: hello world
     */
    pub fn foo() {}
}

/// This is not a footnote [^1]
//~^ doc_suspicious_footnotes
///
/// This one is [^2]
///
/// [^2]: contents
#[doc = "This is not a footnote [^3]"]
//~^ doc_suspicious_footnotes
#[doc = ""]
#[doc = "This one is [^4]"]
#[doc = ""]
#[doc = "[^4]: contents"]
pub struct MultiFragmentFootnote;

#[doc(inline)]
/// This is not a footnote [^5]
//~^ doc_suspicious_footnotes
///
/// This one is [^6]
///
/// [^6]: contents
#[doc = "This is not a footnote [^7]"]
//~^ doc_suspicious_footnotes
#[doc = ""]
#[doc = "This one is [^8]"]
#[doc = ""]
#[doc = "[^8]: contents"]
pub use MultiFragmentFootnote as OtherInlinedFootnote;
