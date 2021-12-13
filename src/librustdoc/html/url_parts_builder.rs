/// A builder that allows efficiently and easily constructing the part of a URL
/// after the domain: `nightly/core/str/struct.Bytes.html`.
///
/// This type is a wrapper around the final `String` buffer,
/// but its API is like that of a `Vec` of URL components.
#[derive(Debug)]
crate struct UrlPartsBuilder {
    buf: String,
}

impl UrlPartsBuilder {
    /// Create an empty buffer.
    crate fn new() -> Self {
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
    crate fn singleton(part: &str) -> Self {
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
    crate fn push(&mut self, part: &str) {
        if !self.buf.is_empty() {
            self.buf.push('/');
        }
        self.buf.push_str(part);
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
    crate fn push_front(&mut self, part: &str) {
        let is_empty = self.buf.is_empty();
        self.buf.reserve(part.len() + if !is_empty { 1 } else { 0 });
        self.buf.insert_str(0, part);
        if !is_empty {
            self.buf.insert(part.len(), '/');
        }
    }

    /// Get the final `String` buffer.
    crate fn finish(self) -> String {
        self.buf
    }
}

/// This is just a guess at the average length of a URL part,
/// used for [`String::with_capacity`] calls in the [`FromIterator`]
/// and [`Extend`] impls.
///
/// This is intentionally on the lower end to avoid overallocating.
const AVG_PART_LENGTH: usize = 5;

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

#[cfg(test)]
mod tests;
