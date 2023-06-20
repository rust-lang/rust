use crate::comment::CommentStyle;
use std::fmt::{self, Display};

/// Formats a string as a doc comment using the given [`CommentStyle`].
pub(super) struct DocCommentFormatter<'a> {
    literal: &'a str,
    style: CommentStyle<'a>,
}

impl<'a> DocCommentFormatter<'a> {
    pub(super) const fn new(literal: &'a str, style: CommentStyle<'a>) -> Self {
        Self { literal, style }
    }
}

impl Display for DocCommentFormatter<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opener = self.style.opener().trim_end();
        let mut lines = self.literal.lines().peekable();

        // Handle `#[doc = ""]`.
        if lines.peek().is_none() {
            return write!(formatter, "{}", opener);
        }

        while let Some(line) = lines.next() {
            let is_last_line = lines.peek().is_none();
            if is_last_line {
                write!(formatter, "{}{}", opener, line)?;
            } else {
                writeln!(formatter, "{}{}", opener, line)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_controls_leading_spaces() {
        test_doc_comment_is_formatted_correctly(
            "    Lorem ipsum",
            "///    Lorem ipsum",
            CommentStyle::TripleSlash,
        );
    }

    #[test]
    fn single_line_doc_comment_is_formatted_correctly() {
        test_doc_comment_is_formatted_correctly(
            "Lorem ipsum",
            "///Lorem ipsum",
            CommentStyle::TripleSlash,
        );
    }

    #[test]
    fn multi_line_doc_comment_is_formatted_correctly() {
        test_doc_comment_is_formatted_correctly(
            "Lorem ipsum\nDolor sit amet",
            "///Lorem ipsum\n///Dolor sit amet",
            CommentStyle::TripleSlash,
        );
    }

    #[test]
    fn whitespace_within_lines_is_preserved() {
        test_doc_comment_is_formatted_correctly(
            " Lorem ipsum \n Dolor sit amet ",
            "/// Lorem ipsum \n/// Dolor sit amet ",
            CommentStyle::TripleSlash,
        );
    }

    fn test_doc_comment_is_formatted_correctly(
        literal: &str,
        expected_comment: &str,
        style: CommentStyle<'_>,
    ) {
        assert_eq!(
            expected_comment,
            format!("{}", DocCommentFormatter::new(literal, style))
        );
    }
}
