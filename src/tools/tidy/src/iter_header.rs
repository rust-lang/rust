const COMMENT: &str = "//@";

/// A header line, like `//@name: value` consists of the prefix `//@` and the directive
/// `name: value`. It is also possibly revisioned, e.g. `//@[revision] name: value`.
pub(crate) struct HeaderLine<'ln> {
    pub(crate) line_number: usize,
    pub(crate) revision: Option<&'ln str>,
    pub(crate) directive: &'ln str,
}

/// Iterate through compiletest headers in a test contents.
///
/// Adjusted from compiletest/src/header.rs.
pub(crate) fn iter_header<'ln>(contents: &'ln str, it: &mut dyn FnMut(HeaderLine<'ln>)) {
    for (line_number, ln) in (1..).zip(contents.lines()) {
        let ln = ln.trim();

        // We're left with potentially `[rev]name: value`.
        let Some(remainder) = ln.strip_prefix(COMMENT) else {
            continue;
        };

        if let Some(remainder) = remainder.trim_start().strip_prefix('[') {
            let Some((revision, remainder)) = remainder.split_once(']') else {
                panic!("malformed revision directive: expected `//@[rev]`, found `{ln}`");
            };
            // We trimmed off the `[rev]` portion, left with `name: value`.
            it(HeaderLine { line_number, revision: Some(revision), directive: remainder.trim() });
        } else {
            it(HeaderLine { line_number, revision: None, directive: remainder.trim() });
        }
    }
}
