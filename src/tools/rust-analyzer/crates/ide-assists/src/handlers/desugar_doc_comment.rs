use either::Either;
use itertools::Itertools;
use syntax::{
    ast::{self, edit::IndentLevel, CommentPlacement, Whitespace},
    AstToken, TextRange,
};

use crate::{
    handlers::convert_comment_block::{line_comment_text, relevant_line_comments},
    utils::required_hashes,
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: desugar_doc_comment
//
// Desugars doc-comments to the attribute form.
//
// ```
// /// Multi-line$0
// /// comment
// ```
// ->
// ```
// #[doc = r"Multi-line
// comment"]
// ```
pub(crate) fn desugar_doc_comment(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let comment = ctx.find_token_at_offset::<ast::Comment>()?;
    // Only allow doc comments
    let Some(placement) = comment.kind().doc else { return None; };

    // Only allow comments which are alone on their line
    if let Some(prev) = comment.syntax().prev_token() {
        if Whitespace::cast(prev).filter(|w| w.text().contains('\n')).is_none() {
            return None;
        }
    }

    let indentation = IndentLevel::from_token(comment.syntax()).to_string();

    let (target, comments) = match comment.kind().shape {
        ast::CommentShape::Block => (comment.syntax().text_range(), Either::Left(comment)),
        ast::CommentShape::Line => {
            // Find all the comments we'll be desugaring
            let comments = relevant_line_comments(&comment);

            // Establish the target of our edit based on the comments we found
            (
                TextRange::new(
                    comments[0].syntax().text_range().start(),
                    comments.last().unwrap().syntax().text_range().end(),
                ),
                Either::Right(comments),
            )
        }
    };

    acc.add(
        AssistId("desugar_doc_comment", AssistKind::RefactorRewrite),
        "Desugar doc-comment to attribute macro",
        target,
        |edit| {
            let text = match comments {
                Either::Left(comment) => {
                    let text = comment.text();
                    text[comment.prefix().len()..(text.len() - "*/".len())]
                        .trim()
                        .lines()
                        .map(|l| l.strip_prefix(&indentation).unwrap_or(l))
                        .join("\n")
                }
                Either::Right(comments) => {
                    comments.into_iter().map(|c| line_comment_text(IndentLevel(0), c)).join("\n")
                }
            };

            let hashes = "#".repeat(required_hashes(&text));

            let prefix = match placement {
                CommentPlacement::Inner => "#!",
                CommentPlacement::Outer => "#",
            };

            let output = format!(r#"{prefix}[doc = r{hashes}"{text}"{hashes}]"#);

            edit.replace(target, output)
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn single_line() {
        check_assist(
            desugar_doc_comment,
            r#"
/// line$0 comment
fn main() {
    foo();
}
"#,
            r#"
#[doc = r"line comment"]
fn main() {
    foo();
}
"#,
        );
        check_assist(
            desugar_doc_comment,
            r#"
//! line$0 comment
fn main() {
    foo();
}
"#,
            r#"
#![doc = r"line comment"]
fn main() {
    foo();
}
"#,
        );
    }

    #[test]
    fn single_line_indented() {
        check_assist(
            desugar_doc_comment,
            r#"
fn main() {
    /// line$0 comment
    struct Foo;
}
"#,
            r#"
fn main() {
    #[doc = r"line comment"]
    struct Foo;
}
"#,
        );
    }

    #[test]
    fn multiline() {
        check_assist(
            desugar_doc_comment,
            r#"
fn main() {
    /// above
    /// line$0 comment
    ///
    /// below
    struct Foo;
}
"#,
            r#"
fn main() {
    #[doc = r"above
line comment

below"]
    struct Foo;
}
"#,
        );
    }

    #[test]
    fn end_of_line() {
        check_assist_not_applicable(
            desugar_doc_comment,
            r#"
fn main() { /// end-of-line$0 comment
    struct Foo;
}
"#,
        );
    }

    #[test]
    fn single_line_different_kinds() {
        check_assist(
            desugar_doc_comment,
            r#"
fn main() {
    //! different prefix
    /// line$0 comment
    /// below
    struct Foo;
}
"#,
            r#"
fn main() {
    //! different prefix
    #[doc = r"line comment
below"]
    struct Foo;
}
"#,
        );
    }

    #[test]
    fn single_line_separate_chunks() {
        check_assist(
            desugar_doc_comment,
            r#"
/// different chunk

/// line$0 comment
/// below
"#,
            r#"
/// different chunk

#[doc = r"line comment
below"]
"#,
        );
    }

    #[test]
    fn block_comment() {
        check_assist(
            desugar_doc_comment,
            r#"
/**
 hi$0 there
*/
"#,
            r#"
#[doc = r"hi there"]
"#,
        );
    }

    #[test]
    fn inner_doc_block() {
        check_assist(
            desugar_doc_comment,
            r#"
/*!
 hi$0 there
*/
"#,
            r#"
#![doc = r"hi there"]
"#,
        );
    }

    #[test]
    fn block_indent() {
        check_assist(
            desugar_doc_comment,
            r#"
fn main() {
    /*!
    hi$0 there

    ```
      code_sample
    ```
    */
}
"#,
            r#"
fn main() {
    #![doc = r"hi there

```
  code_sample
```"]
}
"#,
        );
    }

    #[test]
    fn end_of_line_block() {
        check_assist_not_applicable(
            desugar_doc_comment,
            r#"
fn main() {
    foo(); /** end-of-line$0 comment */
}
"#,
        );
    }

    #[test]
    fn regular_comment() {
        check_assist_not_applicable(desugar_doc_comment, r#"// some$0 comment"#);
        check_assist_not_applicable(desugar_doc_comment, r#"/* some$0 comment*/"#);
    }

    #[test]
    fn quotes_and_escapes() {
        check_assist(
            desugar_doc_comment,
            r###"/// some$0 "\ "## comment"###,
            r####"#[doc = r###"some "\ "## comment"###]"####,
        );
    }
}
