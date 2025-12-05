//! HTML escaping.
//!
//! This module contains one unit struct, which can be used to HTML-escape a
//! string of text (for use in a format string).

use std::fmt;

use pulldown_cmark_escape::FmtWriter;
use unicode_segmentation::UnicodeSegmentation;

/// Wrapper struct which will emit the HTML-escaped version of the contained
/// string when passed to a format string.
pub(crate) struct Escape<'a>(pub &'a str);

impl fmt::Display for Escape<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        pulldown_cmark_escape::escape_html(FmtWriter(fmt), self.0)
    }
}

/// Wrapper struct which will emit the HTML-escaped version of the contained
/// string when passed to a format string.
///
/// This is only safe to use for text nodes. If you need your output to be
/// safely contained in an attribute, use [`Escape`]. If you don't know the
/// difference, use [`Escape`].
pub(crate) struct EscapeBodyText<'a>(pub &'a str);

impl fmt::Display for EscapeBodyText<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        pulldown_cmark_escape::escape_html_body_text(FmtWriter(fmt), self.0)
    }
}

/// Wrapper struct which will emit the HTML-escaped version of the contained
/// string when passed to a format string. This function also word-breaks
/// CamelCase and snake_case word names.
///
/// This is only safe to use for text nodes. If you need your output to be
/// safely contained in an attribute, use [`Escape`]. If you don't know the
/// difference, use [`Escape`].
pub(crate) struct EscapeBodyTextWithWbr<'a>(pub &'a str);

impl fmt::Display for EscapeBodyTextWithWbr<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let EscapeBodyTextWithWbr(text) = *self;
        if text.len() < 8 {
            return EscapeBodyText(text).fmt(fmt);
        }
        let mut last = 0;
        let mut it = text.grapheme_indices(true).peekable();
        let _ = it.next(); // don't insert wbr before first char
        while let Some((i, s)) = it.next() {
            let pk = it.peek();
            if s.chars().all(|c| c.is_whitespace()) {
                // don't need "First <wbr>Second"; the space is enough
                EscapeBodyText(&text[last..i]).fmt(fmt)?;
                last = i;
                continue;
            }
            let is_uppercase = || s.chars().any(|c| c.is_uppercase());
            let next_is_uppercase = || pk.is_none_or(|(_, t)| t.chars().any(|c| c.is_uppercase()));
            let next_is_underscore = || pk.is_none_or(|(_, t)| t.contains('_'));
            let next_is_colon = || pk.is_none_or(|(_, t)| t.contains(':'));
            // Check for CamelCase.
            //
            // `i - last > 3` avoids turning FmRadio into Fm<wbr>Radio, which is technically
            // correct, but needlessly bloated.
            //
            // is_uppercase && !next_is_uppercase checks for camelCase. HTTPSProxy,
            // for example, should become HTTPS<wbr>Proxy.
            //
            // !next_is_underscore avoids turning TEST_RUN into TEST<wbr>_<wbr>RUN, which is also
            // needlessly bloated.
            if i - last > 3 && is_uppercase() && !next_is_uppercase() && !next_is_underscore() {
                EscapeBodyText(&text[last..i]).fmt(fmt)?;
                fmt.write_str("<wbr>")?;
                last = i;
            } else if (s.contains(':') && !next_is_colon())
                || (s.contains('_') && !next_is_underscore())
            {
                EscapeBodyText(&text[last..i + 1]).fmt(fmt)?;
                fmt.write_str("<wbr>")?;
                last = i + 1;
            }
        }
        if last < text.len() {
            EscapeBodyText(&text[last..]).fmt(fmt)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
