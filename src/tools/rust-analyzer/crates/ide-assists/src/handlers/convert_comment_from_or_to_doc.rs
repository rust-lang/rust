use itertools::Itertools;
use syntax::{
    AstToken, Direction, SyntaxElement, TextRange,
    ast::{self, Comment, CommentPlacement, Whitespace, edit::IndentLevel},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: comment_to_doc
//
// Converts comments to documentation.
//
// ```
// // Wow what $0a nice module
// // I sure hope this shows up when I hover over it
// ```
// ->
// ```
// //! Wow what a nice module
// //! I sure hope this shows up when I hover over it
// ```
pub(crate) fn convert_comment_from_or_to_doc(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let comment = ctx.find_token_at_offset::<ast::Comment>()?;

    match comment.kind().doc {
        Some(_) => doc_to_comment(acc, comment),
        None => can_be_doc_comment(&comment).and_then(|style| comment_to_doc(acc, comment, style)),
    }
}

fn doc_to_comment(acc: &mut Assists, comment: ast::Comment) -> Option<()> {
    let target = if comment.kind().shape.is_line() {
        line_comments_text_range(&comment)?
    } else {
        comment.syntax().text_range()
    };

    acc.add(
        AssistId::refactor_rewrite("doc_to_comment"),
        "Replace doc comment with comment",
        target,
        |edit| {
            // We need to either replace the first occurrence of /* with /***, or we need to replace
            // the occurrences // at the start of each line with ///
            let output = match comment.kind().shape {
                ast::CommentShape::Line => {
                    let indentation = IndentLevel::from_token(comment.syntax());
                    let line_start = comment.prefix();
                    let prefix = format!("{indentation}//");
                    relevant_line_comments(&comment)
                        .iter()
                        .map(|comment| comment.text())
                        .flat_map(|text| text.lines())
                        .map(|line| line.replacen(line_start, &prefix, 1))
                        .join("\n")
                }
                ast::CommentShape::Block => {
                    let block_start = comment.prefix();
                    comment
                        .text()
                        .lines()
                        .enumerate()
                        .map(|(idx, line)| {
                            if idx == 0 {
                                line.replacen(block_start, "/*", 1)
                            } else {
                                line.replacen("*  ", "* ", 1)
                            }
                        })
                        .join("\n")
                }
            };
            edit.replace(target, output)
        },
    )
}

fn comment_to_doc(acc: &mut Assists, comment: ast::Comment, style: CommentPlacement) -> Option<()> {
    let target = if comment.kind().shape.is_line() {
        line_comments_text_range(&comment)?
    } else {
        comment.syntax().text_range()
    };

    acc.add(
        AssistId::refactor_rewrite("comment_to_doc"),
        "Replace comment with doc comment",
        target,
        |edit| {
            // We need to either replace the first occurrence of /* with /***, or we need to replace
            // the occurrences // at the start of each line with ///
            let output = match comment.kind().shape {
                ast::CommentShape::Line => {
                    let indentation = IndentLevel::from_token(comment.syntax());
                    let line_start = match style {
                        CommentPlacement::Inner => format!("{indentation}//!"),
                        CommentPlacement::Outer => format!("{indentation}///"),
                    };
                    relevant_line_comments(&comment)
                        .iter()
                        .map(|comment| comment.text())
                        .flat_map(|text| text.lines())
                        .map(|line| line.replacen("//", &line_start, 1))
                        .join("\n")
                }
                ast::CommentShape::Block => {
                    let block_start = match style {
                        CommentPlacement::Inner => "/*!",
                        CommentPlacement::Outer => "/**",
                    };
                    comment
                        .text()
                        .lines()
                        .enumerate()
                        .map(|(idx, line)| {
                            if idx == 0 {
                                // On the first line we replace the comment start with a doc comment
                                // start.
                                line.replacen("/*", block_start, 1)
                            } else {
                                // put one extra space after each * since we moved the first line to
                                // the right by one column as well.
                                line.replacen("* ", "*  ", 1)
                            }
                        })
                        .join("\n")
                }
            };
            edit.replace(target, output)
        },
    )
}

/// Not all comments are valid candidates for conversion into doc comments. For example, the
/// comments in the code:
/// ```ignore
/// // Brilliant module right here
///
/// // Really good right
/// fn good_function(foo: Foo) -> Bar {
///     foo.into_bar()
/// }
///
/// // So nice
/// mod nice_module {}
/// ```
/// can be converted to doc comments. However, the comments in this example:
/// ```ignore
/// fn foo_bar(foo: Foo /* not bar yet */) -> Bar {
///     foo.into_bar()
///     // Nicely done
/// }
/// // end of function
///
/// struct S {
///     // The S struct
/// }
/// ```
/// are not allowed to become doc comments. Moreover, some comments _are_ allowed, but aren't common
/// style in Rust. For example, the following comments are allowed to be doc comments, but it is not
/// common style for them to be:
/// ```ignore
/// fn foo_bar(foo: Foo) -> Bar {
///     // this could be an inner comment with //!
///     foo.into_bar()
/// }
///
/// trait T {
///     // The T struct could also be documented from within
/// }
///
/// mod mymod {
///     // Modules only normally get inner documentation when they are defined as a separate file.
/// }
/// ```
fn can_be_doc_comment(comment: &ast::Comment) -> Option<CommentPlacement> {
    use syntax::SyntaxKind::*;

    // if the comment is not on its own line, then we do not propose anything.
    match comment.syntax().prev_token() {
        Some(prev) => {
            // There was a previous token, now check if it was a newline
            Whitespace::cast(prev).filter(|w| w.text().contains('\n'))?;
        }
        // There is no previous token, this is the start of the file.
        None => return Some(CommentPlacement::Inner),
    }

    // check if comment is followed by: `struct`, `trait`, `mod`, `fn`, `type`, `extern crate`,
    // `use` or `const`.
    let parent = comment.syntax().parent();
    let par_kind = parent.as_ref().map(|parent| parent.kind());
    matches!(par_kind, Some(STRUCT | TRAIT | MODULE | FN | TYPE_ALIAS | EXTERN_CRATE | USE | CONST))
        .then_some(CommentPlacement::Outer)
}

/// The line -> block assist can  be invoked from anywhere within a sequence of line comments.
/// relevant_line_comments crawls backwards and forwards finding the complete sequence of comments that will
/// be joined.
pub(crate) fn relevant_line_comments(comment: &ast::Comment) -> Vec<Comment> {
    // The prefix identifies the kind of comment we're dealing with
    let prefix = comment.prefix();
    let same_prefix = |c: &ast::Comment| c.prefix() == prefix;

    // These tokens are allowed to exist between comments
    let skippable = |not: &SyntaxElement| {
        not.clone()
            .into_token()
            .and_then(Whitespace::cast)
            .map(|w| !w.spans_multiple_lines())
            .unwrap_or(false)
    };

    // Find all preceding comments (in reverse order) that have the same prefix
    let prev_comments = comment
        .syntax()
        .siblings_with_tokens(Direction::Prev)
        .filter(|s| !skippable(s))
        .map(|not| not.into_token().and_then(Comment::cast).filter(same_prefix))
        .take_while(|opt_com| opt_com.is_some())
        .flatten()
        .skip(1); // skip the first element so we don't duplicate it in next_comments

    let next_comments = comment
        .syntax()
        .siblings_with_tokens(Direction::Next)
        .filter(|s| !skippable(s))
        .map(|not| not.into_token().and_then(Comment::cast).filter(same_prefix))
        .take_while(|opt_com| opt_com.is_some())
        .flatten();

    let mut comments: Vec<_> = prev_comments.collect();
    comments.reverse();
    comments.extend(next_comments);
    comments
}

fn line_comments_text_range(comment: &ast::Comment) -> Option<TextRange> {
    let comments = relevant_line_comments(comment);
    let first = comments.first()?;
    let indentation = IndentLevel::from_token(first.syntax());
    let start =
        first.syntax().text_range().start().checked_sub((indentation.0 as u32 * 4).into())?;
    let end = comments.last()?.syntax().text_range().end();
    Some(TextRange::new(start, end))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn module_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            // such a nice module$0
            fn main() {
                foo();
            }
            "#,
            r#"
            //! such a nice module
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_line_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            // unseen$0 docs
            fn main() {
                foo();
            }
            "#,
            r#"

            /// unseen docs
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn multi_line_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            // unseen$0 docs
            // make me seen!
            fn main() {
                foo();
            }
            "#,
            r#"

            /// unseen docs
            /// make me seen!
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_line_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            /// visible$0 docs
            fn main() {
                foo();
            }
            "#,
            r#"

            // visible docs
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn multi_line_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            /// visible$0 docs
            /// Hide me!
            fn main() {
                foo();
            }
            "#,
            r#"

            // visible docs
            // Hide me!
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_line_block_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            /* unseen$0 docs */
            fn main() {
                foo();
            }
            "#,
            r#"

            /** unseen docs */
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn multi_line_block_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            /* unseen$0 docs
            *  make me seen!
            */
            fn main() {
                foo();
            }
            "#,
            r#"

            /** unseen docs
            *   make me seen!
            */
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_line_block_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            /** visible$0 docs */
            fn main() {
                foo();
            }
            "#,
            r#"

            /* visible docs */
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn multi_line_block_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"

            /** visible$0 docs
            *   Hide me!
            */
            fn main() {
                foo();
            }
            "#,
            r#"

            /* visible docs
            *  Hide me!
            */
            fn main() {
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_inner_line_comment_to_doc() {
        check_assist_not_applicable(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                // unseen$0 docs
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_inner_line_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                //! visible$0 docs
                foo();
            }
            "#,
            r#"
            mod mymod {
                // visible docs
                foo();
            }
            "#,
        );
    }

    #[test]
    fn multi_inner_line_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                //! visible$0 docs
                //! Hide me!
                foo();
            }
            "#,
            r#"
            mod mymod {
                // visible docs
                // Hide me!
                foo();
            }
            "#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                /// visible$0 docs
                /// Hide me!
                foo();
            }
            "#,
            r#"
            mod mymod {
                // visible docs
                // Hide me!
                foo();
            }
            "#,
        );
    }

    #[test]
    fn single_inner_line_block_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                /*! visible$0 docs */
                type Int = i32;
            }
            "#,
            r#"
            mod mymod {
                /* visible docs */
                type Int = i32;
            }
            "#,
        );
    }

    #[test]
    fn multi_inner_line_block_doc_to_comment() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                /*! visible$0 docs
                *   Hide me!
                */
                type Int = i32;
            }
            "#,
            r#"
            mod mymod {
                /* visible docs
                *  Hide me!
                */
                type Int = i32;
            }
            "#,
        );
    }

    #[test]
    fn not_overeager() {
        check_assist_not_applicable(
            convert_comment_from_or_to_doc,
            r#"
            fn main() {
                foo();
                // $0well that settles main
            }
            // $1 nicely done
            "#,
        );
    }

    #[test]
    fn all_possible_items() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice struct$0 */
                struct S {}
            }"#,
            r#"mod m {
                /** Nice struct */
                struct S {}
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice trait$0 */
                trait T {}
            }"#,
            r#"mod m {
                /** Nice trait */
                trait T {}
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice module$0 */
                mod module {}
            }"#,
            r#"mod m {
                /** Nice module */
                mod module {}
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice function$0 */
                fn function() {}
            }"#,
            r#"mod m {
                /** Nice function */
                fn function() {}
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice type$0 */
                type Type Int = i32;
            }"#,
            r#"mod m {
                /** Nice type */
                type Type Int = i32;
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice crate$0 */
                extern crate rust_analyzer;
            }"#,
            r#"mod m {
                /** Nice crate */
                extern crate rust_analyzer;
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice import$0 */
                use ide_assists::convert_comment_from_or_to_doc::tests
            }"#,
            r#"mod m {
                /** Nice import */
                use ide_assists::convert_comment_from_or_to_doc::tests
            }"#,
        );
        check_assist(
            convert_comment_from_or_to_doc,
            r#"mod m {
                /* Nice constant$0 */
                const CONST: &str = "very const";
            }"#,
            r#"mod m {
                /** Nice constant */
                const CONST: &str = "very const";
            }"#,
        );
    }

    #[test]
    fn no_inner_comments() {
        check_assist_not_applicable(
            convert_comment_from_or_to_doc,
            r#"
            mod mymod {
                // aaa$0aa
            }
            "#,
        );
    }
}
