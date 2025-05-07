//! See [`HtmlWithLimit`].

use std::fmt::Write;
use std::ops::ControlFlow;

use crate::html::escape::Escape;

/// A buffer that allows generating HTML with a length limit.
///
/// This buffer ensures that:
///
/// * all tags are closed,
/// * tags are closed in the reverse order of when they were opened (i.e., the correct HTML order),
/// * no tags are left empty (e.g., `<em></em>`) due to the length limit being reached,
/// * all text is escaped.
#[derive(Debug)]
pub(super) struct HtmlWithLimit {
    buf: String,
    len: usize,
    limit: usize,
    /// A list of tags that have been requested to be opened via [`Self::open_tag()`]
    /// but have not actually been pushed to `buf` yet. This ensures that tags are not
    /// left empty (e.g., `<em></em>`) due to the length limit being reached.
    queued_tags: Vec<&'static str>,
    /// A list of all tags that have been opened but not yet closed.
    unclosed_tags: Vec<&'static str>,
}

impl HtmlWithLimit {
    /// Create a new buffer, with a limit of `length_limit`.
    pub(super) fn new(length_limit: usize) -> Self {
        let buf = if length_limit > 1000 {
            // If the length limit is really large, don't preallocate tons of memory.
            String::new()
        } else {
            // The length limit is actually a good heuristic for initial allocation size.
            // Measurements showed that using it as the initial capacity ended up using less memory
            // than `String::new`.
            // See https://github.com/rust-lang/rust/pull/88173#discussion_r692531631 for more.
            String::with_capacity(length_limit)
        };
        Self {
            buf,
            len: 0,
            limit: length_limit,
            unclosed_tags: Vec::new(),
            queued_tags: Vec::new(),
        }
    }

    /// Finish using the buffer and get the written output.
    /// This function will close all unclosed tags for you.
    pub(super) fn finish(mut self) -> String {
        self.close_all_tags();
        self.buf
    }

    /// Write some plain text to the buffer, escaping as needed.
    ///
    /// This function skips writing the text if the length limit was reached
    /// and returns [`ControlFlow::Break`].
    pub(super) fn push(&mut self, text: &str) -> ControlFlow<(), ()> {
        if self.len + text.len() > self.limit {
            return ControlFlow::Break(());
        }

        self.flush_queue();
        write!(self.buf, "{}", Escape(text)).unwrap();
        self.len += text.len();

        ControlFlow::Continue(())
    }

    /// Open an HTML tag.
    ///
    /// **Note:** HTML attributes have not yet been implemented.
    /// This function will panic if called with a non-alphabetic `tag_name`.
    pub(super) fn open_tag(&mut self, tag_name: &'static str) {
        assert!(
            tag_name.chars().all(|c: char| c.is_ascii_lowercase()),
            "tag_name contained non-alphabetic chars: {tag_name:?}",
        );
        self.queued_tags.push(tag_name);
    }

    /// Close the most recently opened HTML tag.
    pub(super) fn close_tag(&mut self) {
        if let Some(tag_name) = self.unclosed_tags.pop() {
            // Close the most recently opened tag.
            write!(self.buf, "</{tag_name}>").unwrap()
        }
        // There are valid cases where `close_tag()` is called without
        // there being any tags to close. For example, this occurs when
        // a tag is opened after the length limit is exceeded;
        // `flush_queue()` will never be called, and thus, the tag will
        // not end up being added to `unclosed_tags`.
    }

    /// Write all queued tags and add them to the `unclosed_tags` list.
    fn flush_queue(&mut self) {
        for tag_name in self.queued_tags.drain(..) {
            write!(self.buf, "<{tag_name}>").unwrap();

            self.unclosed_tags.push(tag_name);
        }
    }

    /// Close all unclosed tags.
    fn close_all_tags(&mut self) {
        while !self.unclosed_tags.is_empty() {
            self.close_tag();
        }
    }
}

#[cfg(test)]
mod tests;
