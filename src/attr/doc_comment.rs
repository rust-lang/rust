use crate::comment::CommentStyle;
use std::fmt::{self, Display};
use syntax_pos::symbol::Symbol;

pub(super) struct DocCommentFormatter<'a> {
    literal: &'a Symbol,
    style: CommentStyle<'a>,
}

impl<'a> DocCommentFormatter<'a> {
    pub(super) fn new(literal: &'a Symbol, style: CommentStyle<'a>) -> Self {
        Self { literal, style }
    }
}

impl Display for DocCommentFormatter<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opener = self.style.opener().trim_end();

        let literal_as_str = self.literal.as_str().get();
        let line_count = literal_as_str.lines().count();
        let last_line_index = line_count - 1;
        let lines = literal_as_str.lines().enumerate();

        for (index, line) in lines {
            if index == last_line_index {
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
    use syntax_pos::{Globals, GLOBALS};

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
        GLOBALS.set(&Globals::new(), || {
            let literal = Symbol::gensym(literal);

            assert_eq!(
                expected_comment,
                format!("{}", DocCommentFormatter::new(&literal, style))
            );
        });
    }
}
