use std::fmt::{self, Write};

use rustc_span::Symbol;

/// A builder that allows efficiently and easily constructing the part of a URL
/// after the domain: `nightly/core/str/struct.Bytes.html`.
///
/// This type is a wrapper around the final `String` buffer,
/// but its API is like that of a `Vec` of URL components.
#[derive(Debug)]
pub(crate) struct UrlPartsBuilder {
    buf: String,
}

impl UrlPartsBuilder {
    /// Create an empty buffer.
    pub(crate) fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Create an empty buffer with capacity for the specified number of bytes.
    fn with_capacity_bytes(count: usize) -> Self {
        Self { buf: String::with_capacity(count) }
    }

    /// Create a buffer with one URL component.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```ignore (private-type)
    /// let builder = UrlPartsBuilder::singleton("core");
    /// assert_eq!(builder.finish(), "core");
    /// ```
    ///
    /// Adding more components afterward.
    ///
    /// ```ignore (private-type)
    /// let mut builder = UrlPartsBuilder::singleton("core");
    /// builder.push("str");
    /// builder.push_front("nightly");
    /// assert_eq!(builder.finish(), "nightly/core/str");
    /// ```
    pub(crate) fn singleton(part: &str) -> Self {
        Self { buf: part.to_owned() }
    }

    /// Push a component onto the buffer.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```ignore (private-type)
    /// let mut builder = UrlPartsBuilder::new();
    /// builder.push("core");
    /// builder.push("str");
    /// builder.push("struct.Bytes.html");
    /// assert_eq!(builder.finish(), "core/str/struct.Bytes.html");
    /// ```
    pub(crate) fn push(&mut self, part: &str) {
        if !self.buf.is_empty() {
            self.buf.push('/');
        }
        self.buf.push_str(part);
    }

    /// Push a component onto the buffer, using [`format!`]'s formatting syntax.
    ///
    /// # Examples
    ///
    /// Basic usage (equivalent to the example for [`UrlPartsBuilder::push`]):
    ///
    /// ```ignore (private-type)
    /// let mut builder = UrlPartsBuilder::new();
    /// builder.push("core");
    /// builder.push("str");
    /// builder.push_fmt(format_args!("{}.{}.html", "struct", "Bytes"));
    /// assert_eq!(builder.finish(), "core/str/struct.Bytes.html");
    /// ```
    pub(crate) fn push_fmt(&mut self, args: fmt::Arguments<'_>) {
        if !self.buf.is_empty() {
            self.buf.push('/');
        }
        self.buf.write_fmt(args).unwrap()
    }

    /// Push a component onto the front of the buffer.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```ignore (private-type)
    /// let mut builder = UrlPartsBuilder::new();
    /// builder.push("core");
    /// builder.push("str");
    /// builder.push_front("nightly");
    /// builder.push("struct.Bytes.html");
    /// assert_eq!(builder.finish(), "nightly/core/str/struct.Bytes.html");
    /// ```
    pub(crate) fn push_front(&mut self, part: &str) {
        let is_empty = self.buf.is_empty();
        self.buf.reserve(part.len() + if !is_empty { 1 } else { 0 });
        self.buf.insert_str(0, part);
        if !is_empty {
            self.buf.insert(part.len(), '/');
        }
    }

    /// Get the final `String` buffer.
    pub(crate) fn finish(self) -> String {
        self.buf
    }
}

/// This is just a guess at the average length of a URL part,
/// used for [`String::with_capacity`] calls in the [`FromIterator`]
/// and [`Extend`] impls, and for [estimating item path lengths].
///
/// The value `8` was chosen for two main reasons:
///
/// * It seems like a good guess for the average part length.
/// * jemalloc's size classes are all multiples of eight,
///   which means that the amount of memory it allocates will often match
///   the amount requested, avoiding wasted bytes.
///
/// [estimating item path lengths]: estimate_item_path_byte_length
const AVG_PART_LENGTH: usize = 8;

/// Estimate the number of bytes in an item's path, based on how many segments it has.
///
/// **Note:** This is only to be used with, e.g., [`String::with_capacity()`];
/// the return value is just a rough estimate.
pub(crate) const fn estimate_item_path_byte_length(segment_count: usize) -> usize {
    AVG_PART_LENGTH * segment_count
}

impl<'a> FromIterator<&'a str> for UrlPartsBuilder {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut builder = Self::with_capacity_bytes(AVG_PART_LENGTH * iter.size_hint().0);
        iter.for_each(|part| builder.push(part));
        builder
    }
}

impl<'a> Extend<&'a str> for UrlPartsBuilder {
    fn extend<T: IntoIterator<Item = &'a str>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.buf.reserve(AVG_PART_LENGTH * iter.size_hint().0);
        iter.for_each(|part| self.push(part));
    }
}

impl FromIterator<Symbol> for UrlPartsBuilder {
    fn from_iter<T: IntoIterator<Item = Symbol>>(iter: T) -> Self {
        // This code has to be duplicated from the `&str` impl because of
        // `Symbol::as_str`'s lifetimes.
        let iter = iter.into_iter();
        let mut builder = Self::with_capacity_bytes(AVG_PART_LENGTH * iter.size_hint().0);
        iter.for_each(|part| builder.push(part.as_str()));
        builder
    }
}

impl Extend<Symbol> for UrlPartsBuilder {
    fn extend<T: IntoIterator<Item = Symbol>>(&mut self, iter: T) {
        // This code has to be duplicated from the `&str` impl because of
        // `Symbol::as_str`'s lifetimes.
        let iter = iter.into_iter();
        self.buf.reserve(AVG_PART_LENGTH * iter.size_hint().0);
        iter.for_each(|part| self.push(part.as_str()));
    }
}

#[cfg(test)]
mod tests;
